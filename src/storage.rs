use crate::index::TextIndex;
use crate::{schema, text, types::*, Error, Result};
use fs4::fs_std::FileExt;
use parking_lot::{Mutex, RwLock};
use rusqlite::{params, params_from_iter, types::Value as SqlValue, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::{collections::{BTreeMap, BTreeSet, HashMap, HashSet}, fs::{File, OpenOptions}, path::{Path, PathBuf}, sync::{atomic::{AtomicU64, Ordering}, Arc}};

/// 读路径自愈索引时最多试几次拿写锁（每次退避 1ms，合计约 1s）。
/// 只在库内补向量那条线程正在补齐时才试：它提交的正是读者要看的索引增删，
/// 而那份提交一落地索引就干净了，循环随即结束。别的时候（例如批量导入的写者占着写锁）
/// 一律不试，读路径绝不为索引排队——那正是当初吞吐塌方的成因。


/// 写者：独占的写连接 + 跨进程文件锁。只挡其他写者，不挡读。
pub(crate) struct Writer { pub conn: Connection, _file_lock: File }

/// 只读连接池。`rusqlite::Connection` 不是 `Sync`，并发读必须各持一条独立连接；
/// 池锁只在取出与归还时短暂持有，读的整个过程不占任何全局锁。
pub(crate) struct Readers { pub idle: Vec<Connection> }

/// 向量分区缓存。失效按知识领域分开做：一个领域写了，别的领域已经载入的分区照旧留着。
/// 条目留在表里就意味着可用——失效是就地删条目，不靠条目自带的版本号比对。
/// 版本号仍要按领域记：载入期间该领域若被写过，手上这份数据已经过期，不能再写进缓存。
/// 版本号也必须独立于全局 revision：向量落盘不推进 revision，只盯 revision 会漏失效。
pub(crate) struct VectorCache {
    /// 全局代次，整体失效时 +1。所有领域的版本号随之上移，因此它只增不减。
    generation: AtomicU64,
    /// 领域自己的递增计数，只在这个领域被写时 +1。
    bumps: Mutex<HashMap<String, u64>>,
    entries: Mutex<HashMap<(String, String, String), Option<Arc<crate::embeddings::Partition>>>>,
}

impl VectorCache {
    fn new() -> Self {
        Self { generation: AtomicU64::new(0), bumps: Mutex::new(HashMap::new()), entries: Mutex::new(HashMap::new()) }
    }
    /// 某个领域当前的版本号。没写过的领域就是全局代次本身。
    pub fn epoch_of(&self, namespace: &str) -> u64 {
        self.generation.load(Ordering::SeqCst) + self.bumps.lock().get(namespace).copied().unwrap_or(0)
    }
    /// 整体失效：清空所有条目，并让在途的载入结果不再被采用。
    /// 清空是为了让内存有界——重新载入本来就是按需的。
    pub fn invalidate(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.entries.lock().clear();
    }
    /// 只失效这些领域：删掉它们的条目、各自推进版本号，别的领域的缓存原样留着。
    pub fn invalidate_namespaces(&self, namespaces: &HashSet<String>) {
        {
            let mut bumps = self.bumps.lock();
            for namespace in namespaces { *bumps.entry(namespace.clone()).or_insert(0) += 1; }
        }
        self.entries.lock().retain(|(_, namespace, _), _| !namespaces.contains(namespace));
    }
}

thread_local! {
    /// 本次写入事务动过的知识领域。写路径自己登记（`touch_namespace`），
    /// `mutate` 装上它、提交后取走，用来精准失效对应领域的向量分区缓存。
    /// 事务之外的写入（迁移、导入进度回写）登记不生效，那些路径本来就走整体失效。
    static TOUCHED_NAMESPACES: std::cell::RefCell<Option<HashSet<String>>> = const { std::cell::RefCell::new(None) };
}

/// 登记本次事务动过的领域。只在事务里有效，事务外是空操作。
/// 名字用归一化后的写法：检索侧的缓存键就是归一化后的领域名。
pub(crate) fn touch_namespace(namespace: &str) {
    TOUCHED_NAMESPACES.with(|slot| {
        if let Some(touched) = slot.borrow_mut().as_mut() { touched.insert(text::normalized_tag(namespace)); }
    });
}

/// 登记某条记录所属的领域。记录还在库里时才取得出名字。
pub(crate) fn touch_record_namespace(conn: &Connection, record_id: i64) -> Result<()> {
    if let Some(namespace) = namespace_of(conn, record_id)? { touch_namespace(&namespace); }
    Ok(())
}

/// 某条记录所属的领域名；记录不存在时是 `None`。
pub(crate) fn namespace_of(conn: &Connection, record_id: i64) -> Result<Option<String>> {
    Ok(conn.query_row("SELECT s.text FROM records r JOIN strings s ON s.id=r.namespace_id WHERE r.id=?1",
        [record_id], |r| r.get(0)).optional()?)
}

/// 事务期间装上的领域登记表。装与还原都走它，中途报错提前返回也不会把登记表留在外面。
struct TouchLog(Option<HashSet<String>>);

impl TouchLog {
    fn install() -> Self {
        Self(TOUCHED_NAMESPACES.with(|slot| slot.borrow_mut().replace(HashSet::new())))
    }
    /// 取走这次事务登记的领域；登记表本身由 `Drop` 还原。
    fn take(&self) -> HashSet<String> {
        TOUCHED_NAMESPACES.with(|slot| slot.borrow_mut().take()).unwrap_or_default()
    }
}

impl Drop for TouchLog {
    fn drop(&mut self) {
        let previous = self.0.take();
        TOUCHED_NAMESPACES.with(|slot| *slot.borrow_mut() = previous);
    }
}

pub(crate) struct Engine {
    pub writer: Mutex<Option<Writer>>,
    pub readers: Mutex<Option<Readers>>,
    /// 向量外挂库 `vectors.sqlite3` 的写连接与读连接池：与主库同目录、独立 WAL、独立锁。
    /// 向量是派生检索索引，读写不走主库通道，向量的写不会占用业务写的写锁。
    pub vector_writer: Mutex<Option<Writer>>,
    pub vector_readers: Mutex<Option<Readers>>,
    /// `TextIndex` 自带内部写锁、`IndexReader` 可并发检索，用 `Arc` 共享给所有读线程。
    /// 放进 `Option` 是为了 `close` 时能真正销毁它——Tantivy 的写锁由 `IndexWriter` 持有，
    /// 不销毁就无法释放，目录也重开不了。
    pub index: RwLock<Option<Arc<TextIndex>>>,
    pub vectors: VectorCache,
    /// 宿主注册的模型能力。它们是运行时状态（闭包 / Python 函数无法序列化），不落盘。
    pub embedders: crate::embeddings::EmbedderRegistry,
    pub rerankers: crate::search::RerankerRegistry,
    /// 宿主注册的事件接收位。不注册就什么都不产出。
    pub events: crate::events::EventRegistry,
    /// 最近观察到的降级档位，供健康检查读出「结果为什么变差」。
    pub degraded: Mutex<Vec<Degrade>>,
    pub root: PathBuf,
}

/// 只读连接的新建：`journal_mode` 是库文件上的持久属性，无需在每条连接上重设。
fn open_reader(root: &Path) -> Result<Connection> {
    let conn = Connection::open(root.join("store.sqlite3"))?;
    conn.execute_batch("PRAGMA busy_timeout=5000; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;")?;
    // 向量库外挂：读路径也要看得见它（缺口核对、指纹比对都靠 ATTACH 读）。
    conn.execute("ATTACH DATABASE ?1 AS vectors", [root.join("vectors.sqlite3").to_string_lossy().to_string()])?;
    Ok(conn)
}

/// 向量外挂库的连接：独立文件、独立 WAL；`foreign_keys` 关掉——它只存自己的路由快照，
/// 不跟主库做任何外键级联。
fn open_vector_writer(root: &Path) -> Result<Connection> {
    let conn = Connection::open(root.join("vectors.sqlite3"))?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;")?;
    conn.execute_batch(include_str!("vectors_schema.sql"))?;
    Ok(conn)
}
fn open_vector_reader(root: &Path) -> Result<Connection> {
    let conn = Connection::open(root.join("vectors.sqlite3"))?;
    conn.execute_batch("PRAGMA busy_timeout=5000; PRAGMA synchronous=NORMAL;")?;
    Ok(conn)
}

/// Clone shares one process-local engine. Close invalidates all its handles.
#[derive(Clone)]
pub struct KnowledgeBase { pub(crate) engine: Arc<Engine> }

/// 一次只读访问：独占一条连接，析构时归还池中。
pub(crate) struct ReadGuard<'a> { engine: &'a Engine, conn: Option<Connection> }

impl ReadGuard<'_> {
    pub fn conn(&self) -> &Connection { self.conn.as_ref().expect("read connection lives until drop") }
}

impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        let Some(conn) = self.conn.take() else { return };
        // 关闭后归还无处安放，直接丢弃即可。
        if let Some(readers) = self.engine.readers.lock().as_mut() { readers.idle.push(conn); }
    }
}

impl KnowledgeBase {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        std::fs::create_dir_all(directory.as_ref())?;
        let root = std::fs::canonicalize(directory.as_ref())?;
        let file_lock = OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("writer.lock"))?;
        if !file_lock.try_lock_exclusive()? { return Err(Error::Locked(root.display().to_string())); }
        let mut write_conn = Connection::open(root.join("store.sqlite3"))?;
        schema::initialize(&mut write_conn)?;
        // 向量外挂库：主库有 embeddings 表说明是旧库，先把它整表搬进 vectors.sqlite3。
        // 搬迁在主库写连接上做，搬完再 ATTACH，避免挂一个还不存在的文件。
        let needs_vector_migration: bool = write_conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='embeddings'", [], |r| r.get::<_, i64>(0)).map(|n| n > 0)?;
        if needs_vector_migration { crate::embeddings::migrate_vectors(&root, &mut write_conn)?; }
        // 向量库文件此刻必须已存在（迁移会建，新库也要建）：否则 ATTACH 查不到表。
        let vector_writer = open_vector_writer(&root)?;
        // 向量库外挂：写路径的删除与指纹核对也要看得见它。
        write_conn.execute("ATTACH DATABASE ?1 AS vectors", [root.join("vectors.sqlite3").to_string_lossy().to_string()])?;
        let index = Arc::new(TextIndex::open(&root)?);
        // 打开即对账：索引与主库对齐，缺的补、多的删。没有「重建」这条路。
        index.reconcile(&write_conn)?;
        // 只读连接在 schema 建好之后再开，保证它看到的是完整结构。
        let reader = open_reader(&root)?;
        let vector_reader = open_vector_reader(&root)?;
        let engine = Arc::new(Engine {
            writer: Mutex::new(Some(Writer { conn: write_conn, _file_lock: file_lock })),
            readers: Mutex::new(Some(Readers { idle: vec![reader] })),
            vector_writer: Mutex::new(Some(Writer { conn: vector_writer, _file_lock: OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("vector_writer.lock"))? })),
            vector_readers: Mutex::new(Some(Readers { idle: vec![vector_reader] })),
            index: RwLock::new(Some(index)), vectors: VectorCache::new(),
            embedders: crate::embeddings::EmbedderRegistry::new(),
            rerankers: crate::search::RerankerRegistry::new(),
            events: crate::events::EventRegistry::default(),
            degraded: Mutex::new(Vec::new()), root,
        });
        Ok(Self { engine })
    }

    pub fn directory(&self) -> &Path { &self.engine.root }

    /// 取文本索引的共享句柄。只在这一瞬间持有索引锁，拿到 `Arc` 后即可并发使用。
    pub(crate) fn index(&self) -> Result<Arc<TextIndex>> {
        self.engine.index.read().clone().ok_or(Error::Closed)
    }

    /// 把写入流程就地交过来的索引文档写进 Tantivy：此刻只是写进 writer，
    /// 对搜索不可见，commit 仍由 `update_index`（或关闭时的收尾）一次做完。
    pub(crate) fn index_documents(&self, docs: &[crate::index::IndexDocument]) -> Result<()> {
        if docs.is_empty() { return Ok(()); }
        self.index()?.stage(docs)?;
        Ok(())
    }

    pub fn close(&self) -> Result<()> {
        let mut guard = self.engine.writer.lock();
        let result = match guard.as_ref() {
            Some(writer) => self.index()?.sync(&writer.conn),
            None => Ok(()),
        };
        *guard = None;
        *self.engine.vector_writer.lock() = None;
        *self.engine.vector_readers.lock() = None;
        // 必须真正销毁索引：Tantivy 的目录写锁由 IndexWriter 持有，不销毁就释放不掉。
        *self.engine.index.write() = None;
        *self.engine.readers.lock() = None;
        result
    }

    /// 取一条独占的只读连接。并发读各拿各的，互不等待。
    pub(crate) fn read(&self) -> Result<ReadGuard<'_>> {
        let conn = {
            let mut readers = self.engine.readers.lock();
            match readers.as_mut() {
                Some(readers) => match readers.idle.pop() {
                    Some(conn) => conn,
                    None => open_reader(&self.engine.root)?,
                },
                None => return Err(Error::Closed),
            }
        };
        Ok(ReadGuard { engine: &self.engine, conn: Some(conn) })
    }

    /// 取一个向量分区：缓存里有就直接复用，否则从向量外挂库按需载入。
    /// 载入过程不持缓存锁——否则一个慢分区会挡住所有其他分区的查询。
    /// 向量库是独立文件、独立连接，载入期间不碰主库。
    pub(crate) fn partition(&self, space: &crate::embeddings::EmbeddingSpace,
        namespace: &str, scope: &str) -> Result<Option<Arc<crate::embeddings::Partition>>> {
        let key = (space.id.clone(), namespace.to_string(), scope.to_string());
        let epoch = self.engine.vectors.epoch_of(namespace);
        let cached = self.engine.vectors.entries.lock().get(&key).cloned();
        if let Some(partition) = cached { return Ok(partition); }
        let loaded = {
            let conn = {
                let mut readers = self.engine.vector_readers.lock();
                match readers.as_mut() {
                    Some(readers) => readers.idle.pop().unwrap_or_else(|| open_vector_reader(&self.engine.root).unwrap_or_else(|_| unreachable!())),
                    None => return Err(Error::Closed),
                }
            };
            let result = crate::embeddings::Partition::load(&conn, space, namespace, scope)?;
            if let Some(readers) = self.engine.vector_readers.lock().as_mut() { readers.idle.push(conn); }
            result.map(Arc::new)
        };
        // 载入期间这个领域可能被写过：版本号变了就说明手上这份已经过期，索性不写缓存。
        // 复查版本号与写条目要在同一把条目锁里：否则失效正好落在两步之间时，
        // 一份过期分区会被永久留在表里，直到这个领域下次被写。
        {
            let mut entries = self.engine.vectors.entries.lock();
            if self.engine.vectors.epoch_of(namespace) == epoch { entries.insert(key, loaded.clone()); }
        }
        Ok(loaded)
    }

    /// 索引可能落后时（写路径攒了增删、或删过记录）才尝试对齐。
    /// 干净时这次预检让读完全不碰写锁；需要对齐时也**只尝试、不等待**——
    /// 抢不到写锁说明写者正在提交，读路径绝不为它排队，等下一次读取再补平。
    /// `conn` 只用于判定，不写任何东西；真正对齐走写连接。
    pub(crate) fn sync_index_if_behind(&self, _conn: &Connection) -> Result<()> {
        let index = self.index()?;
        if !index.is_dirty() { return Ok(()); }
        if let Some(mut guard) = self.engine.writer.try_lock() {
            if let Some(writer) = guard.as_mut() { index.sync(&writer.conn)?; }
        }
        Ok(())
    }

    /// 业务写入：改记录、改向量。提交后按领域精准失效向量分区缓存。
    pub(crate) fn mutate<T>(&self, f: impl FnOnce(&Transaction<'_>) -> Result<T>) -> Result<WriteReceipt<T>> {
        let mut guard = self.engine.writer.lock();
        let writer = guard.as_mut().ok_or(Error::Closed)?;
        let changed_before = writer.conn.total_changes();
        let log = TouchLog::install();
        let tx = writer.conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let value = f(&tx)?;
        let revision = current_revision(&tx)?;
        tx.commit()?;
        let touched = log.take();
        drop(log);
        // 索引不在写入路径上提交——写入只把文档/删除攒在 writer 里，
        // 由使用方稍后调用 `update_index` 一趟对齐（提交 + 对账）。
        let rows_changed = writer.conn.total_changes() > changed_before;
        if rows_changed {
            // 索引可能已落后主库（有记录增删）：标脏，等 `update_index` 或读取侧自愈时对账。
            self.index()?.mark_dirty();
            // 向量缓存按领域失效：动过的领域就地清掉，没动过的照旧留着。
            if touched.is_empty() { self.engine.vectors.invalidate(); }
            else { self.engine.vectors.invalidate_namespaces(&touched); }
            // 记录一变，「应向量化集合」就变了：受影响领域的就绪标记就地作废，
            // 等补齐核对过再重新标。这样检索侧读到的永远是「核对过的那一份」。
            self.invalidate_readiness(&touched);
        }
        drop(guard);
        Ok(WriteReceipt { value, revision })
    }

    /// 只写那些不喂向量分区的表的写入：就绪标记、向量空间登记、各档开关、谓词规则。
    /// 这类写入既不改记录也不改向量，向量分区缓存因此不动。
    pub(crate) fn mutate_meta<T>(&self, f: impl FnOnce(&Transaction<'_>) -> Result<T>) -> Result<WriteReceipt<T>> {
        let mut guard = self.engine.writer.lock();
        let writer = guard.as_mut().ok_or(Error::Closed)?;
        let tx = writer.conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let value = f(&tx)?;
        let revision = current_revision(&tx)?;
        tx.commit()?;
        Ok(WriteReceipt { value, revision })
    }

    /// 作废这些领域（本次事务实际动过的）的向量就绪标记。
    /// 领域集合由写路径就地登记，不再依赖任何待办表推算。
    fn invalidate_readiness(&self, namespaces: &HashSet<String>) {
        if namespaces.is_empty() { return; }
        let mut guard = self.engine.vector_writer.lock();
        let Some(vector_writer) = guard.as_mut() else { return; };
        for namespace in namespaces {
            let _ = crate::embeddings::clear_vector_ready(&vector_writer.conn, namespace);
        }
    }

    pub fn memories(&self) -> crate::memory::MemoryStore { crate::memory::MemoryStore(self.clone()) }
    pub fn graph(&self) -> crate::graph::GraphStore { crate::graph::GraphStore(self.clone()) }
    pub fn notes(&self) -> crate::notes::NoteStore { crate::notes::NoteStore(self.clone()) }
    pub fn embeddings(&self) -> crate::embeddings::EmbeddingStore { crate::embeddings::EmbeddingStore(self.clone()) }

    /// 需要写连接但不走事务的极少数场景（如导入进度回写）。
    /// 走写锁，并保守失效向量缓存：调用方改了库里的东西，缓存不能不知情。
    pub(crate) fn write<T>(&self, f: impl FnOnce(&Writer) -> Result<T>) -> Result<T> {
        let mut guard = self.engine.writer.lock();
        let value = f(guard.as_mut().ok_or(Error::Closed)?)?;
        self.engine.vectors.invalidate();
        // 这条路径可能改了记录（例如导入进度回写），保守地把索引标脏，等下一次对账。
        let _ = self.index().map(|index| index.mark_dirty());
        Ok(value)
    }

    /// 记下一个降级档位，去重保留少量，供健康检查读出。
    pub(crate) fn note_degrade(&self, degrade: Degrade) {
        let mut observed = self.engine.degraded.lock();
        if !observed.contains(&degrade) {
            observed.push(degrade);
            if observed.len() > 8 { observed.remove(0); }
        }
    }

    /// 收敛索引：把写入攒下的增删提交一次，再对主库算差集对齐、只提交一次。
    /// 写入不再就地索引，使用方（尤其批量导入）在合适时机调用本方法即可；
    /// 期间读取走 `sync_index_if_behind` 自愈兜底。
    pub fn update_index(&self) -> Result<HealthReport> {
        self.catch_up_index()?;
        self.health()
    }

    /// 收敛索引：提交写入攒下的增删、再对主库算差集对齐，全程在写连接上一次做完。
    /// 只由写入侧（`update_index`）触发——索引提交是写入路径的职责，
    /// 读取侧的自愈走 `sync_index_if_behind`，向量化则完全不碰这里。
    fn catch_up_index(&self) -> Result<()> {
        let mut guard = self.engine.writer.lock();
        let Some(writer) = guard.as_mut() else { return Ok(()) };
        self.index()?.sync(&writer.conn)
    }

    /// 注册事件接收回调。库在检索线程里同步调用它，所以回调必须非阻塞——
    /// 在里面做同步 IO 或网络上报，会把检索拖住，和慢的重排回调一样。
    /// 回调抛错只丢这一条事件，不影响检索。不注册就完全不产出事件。
    pub fn register_event_sink<F: Fn(&crate::events::LogEvent) + Send + Sync + 'static>(&self, sink: F) {
        self.engine.events.set(Arc::new(sink));
    }

    /// 注销事件接收回调，返回此前是否有注册。
    pub fn unregister_event_sink(&self) -> bool { self.engine.events.clear() }

    /// 当前是否注册了事件接收回调。
    pub fn event_sink_registered(&self) -> bool { self.engine.events.is_registered() }

    pub fn health(&self) -> Result<HealthReport> {
        let state = self.read()?;
        let conn = state.conn();
        let mut counts = BTreeMap::new();
        let mut stmt = conn.prepare("SELECT kind, COUNT(*) FROM records GROUP BY kind")?;
        for row in stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))? {
            let (code, count) = row?;
            let name = RecordKind::from_code(code).map(|k| k.as_str().to_string()).unwrap_or_else(|| code.to_string());
            counts.insert(name, count as usize);
        }
        let record_count = counts.values().sum();
        let mut foreign = conn.prepare("PRAGMA foreign_key_check")?;
        let mut foreign_key_errors = 0;
        let mut rows = foreign.query([])?;
        while rows.next()?.is_some() { foreign_key_errors += 1; }
        Ok(HealthReport {
            schema_version: schema::SCHEMA_VERSION,
            revision: current_revision(conn)?,
            indexed_revision: meta(conn, "indexed_revision")?,
            index_document_count: self.index()?.document_count(),
            record_count,
            sqlite_integrity: conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?,
            foreign_key_errors, counts,
            embedder_spaces: self.engine.embedders.space_ids(),
            reranker_registered: self.engine.rerankers.is_registered(),
            last_degraded: self.engine.degraded.lock().clone(),
        })
    }

    /// Consistent SQLite snapshot, including embeddings. Refuses to overwrite a file.
    pub fn backup(&self, target: impl AsRef<Path>) -> Result<()> {
        let target = target.as_ref();
        let state = self.read()?;
        let reservation = OpenOptions::new().write(true).create_new(true).open(target)?;
        drop(reservation);
        if let Err(err) = state.conn().backup(rusqlite::MAIN_DB, target, None) {
            let _ = std::fs::remove_file(target);
            return Err(err.into());
        }
        // 向量外挂库也备份：它是派生索引，但备份恢复后不该要求重新跑模型。
        let vectors_target = target.with_file_name(format!("{}.vectors", target.file_name().unwrap().to_string_lossy()));
        if let Err(err) = state.conn().backup("vectors", &vectors_target, None) {
            let _ = std::fs::remove_file(&vectors_target);
            return Err(err.into());
        }
        Ok(())
    }

    /// Restores to a new directory; search indexes are rebuilt from the snapshot.
    pub fn restore(snapshot: impl AsRef<Path>, directory: impl AsRef<Path>) -> Result<Self> {
        let snapshot = snapshot.as_ref();
        let source = Connection::open_with_flags(snapshot, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let app: i64 = source.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let version: i64 = source.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if app != schema::APPLICATION_ID { return Err(Error::Validation("snapshot is not a p-memory database".into())); }
        if version != schema::SCHEMA_VERSION { return Err(Error::SchemaVersion { found: version, supported: schema::SCHEMA_VERSION }); }
        std::fs::create_dir(directory.as_ref())?;
        source.backup(rusqlite::MAIN_DB, directory.as_ref().join("store.sqlite3"), None)?;
        // 向量库备份在同目录、同名加 .vectors 后缀；有就恢复，没有就由使用方后续调 sync 重新生成。
        let vectors_snapshot = snapshot.with_file_name(format!("{}.vectors", snapshot.file_name().unwrap().to_string_lossy()));
        if vectors_snapshot.exists() {
            let vectors_source = Connection::open_with_flags(&vectors_snapshot, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            vectors_source.backup(rusqlite::MAIN_DB, directory.as_ref().join("vectors.sqlite3"), None)?;
        }
        Self::open(directory)
    }
}

pub(crate) fn now_us() -> i64 { chrono::Utc::now().timestamp_micros() }
pub(crate) fn meta(conn: &Connection, key: &str) -> Result<i64> {
    Ok(conn.query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0))?)
}

pub(crate) fn current_revision(conn: &Connection) -> Result<i64> { meta(conn, "revision") }
pub(crate) fn next_revision(conn: &Connection, _record_id: i64) -> Result<i64> {
    // 只推进全局修订号。索引与主库的一致性由「算差集」判定，不靠任何待办记录。
    conn.execute("UPDATE meta SET value=value+1 WHERE key='revision'", [])?;
    current_revision(conn)
}

/// 登记一个开放标记并返回它的整数 id；字符串只在此表出现一次。
pub(crate) fn term_id(conn: &Connection, text_value: &str) -> Result<i64> {
    let normalized = text::normalized_tag(text_value);
    conn.execute("INSERT OR IGNORE INTO strings(text) VALUES (?1)", [&normalized])?;
    Ok(conn.query_row("SELECT id FROM strings WHERE text=?1", [&normalized], |r| r.get(0))?)
}

pub(crate) fn term_text(conn: &Connection, id: i64) -> Result<String> {
    Ok(conn.query_row("SELECT text FROM strings WHERE id=?1", [id], |r| r.get(0))?)
}

/// 库里实际出现过的知识领域（记录用到的 namespace），按字典序。
/// 就绪核对按它逐个领域做：没有记录的领域没有缺口可言。
pub(crate) fn record_namespaces(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT DISTINCT s.text FROM records r JOIN strings s ON s.id=r.namespace_id ORDER BY s.text")?;
    let mut namespaces = Vec::new();
    for row in stmt.query_map([], |r| r.get::<_, String>(0))? { namespaces.push(row?); }
    Ok(namespaces)
}

pub(crate) fn validate_identity(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() || value != value.trim() || value.chars().any(char::is_control) {
        return Err(Error::Validation(format!("{label} must be nonempty, trimmed, and contain no control characters")));
    }
    Ok(())
}
pub(crate) fn validate_filter(filter: &ReadFilter) -> Result<()> {
    validate_identity("namespace", &filter.namespace)?;
    if filter.scopes.is_empty() { return Err(Error::Validation("at least one explicit read scope is required".into())); }
    for scope in &filter.scopes { validate_identity("scope", scope)?; }
    Ok(())
}
pub(crate) fn validate_limit(limit: usize) -> Result<()> {
    if !(1..=10_000).contains(&limit) { return Err(Error::Validation("limit must be between 1 and 10000".into())); }
    Ok(())
}

/// 归一化、去重后的标签文本（排序）。标签字符串统一落在 strings 表。
pub(crate) fn normalize_tags(tags: &[String]) -> Vec<String> {
    tags.iter().map(|label| text::normalized_tag(label)).filter(|tag| !tag.is_empty()).collect::<BTreeSet<_>>().into_iter().collect()
}

/// 标签集拼进哪一条的可搜正文：笔记这条自己不占索引文档，由它的第一片承载；
/// 其余记录没有切片，就是它自己。返回空格连接的标签文本，不带标签时是空串。
/// `exclude` 是已经升格成独立索引列的那些标签（笔记的目录段与文件名）——它们靠专门的列
/// 参与匹配，不能再留在正文里，否则搜目录名会命中该目录下每一篇。
pub(crate) fn tags_prefix(kind: RecordKind, tags: &[String], exclude: &[String], payload: &Value) -> String {
    let carries = match kind {
        RecordKind::Note => false,
        RecordKind::Chunk => payload.get("ordinal").and_then(Value::as_u64) == Some(0),
        _ => true,
    };
    if !carries { return String::new(); }
    tags.iter().filter(|tag| !exclude.contains(tag)).cloned().collect::<Vec<_>>().join(" ")
}

/// 笔记相对路径拆成「目录段」与「文件名（去扩展名）」。
/// 目录段原样，只有最后一段去后缀（目录名里的点不是扩展名）。写入与索引侧补文档共用同一套规则。
pub(crate) fn split_note_path(relative: &str) -> (Vec<String>, String) {
    let segments: Vec<&str> = relative.split('/').filter(|segment| !segment.is_empty()).collect();
    let Some((last, dirs)) = segments.split_last() else { return (Vec::new(), String::new()); };
    let stem = last.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(last).trim();
    (dirs.iter().map(|segment| segment.to_string()).collect(), stem.to_string())
}

/// 一篇笔记的「目录段 / 文件名」。文件名直接读写入时存下的 `notes.name`（不事后拆路径，
/// 平台的路径分隔符与扩展名边界都由 `Path` 在写入那一刻定好）；目录段取相对路径去掉末段。
/// 登记根目录后库里存的必是相对路径；迁移前留下的绝对路径没有相对目录可言，只给名字。
pub(crate) fn note_path_parts(conn: &Connection, note_id: i64) -> (Vec<String>, String) {
    let Ok((path, name)) = conn.query_row("SELECT path,name FROM notes WHERE record_id=?1", [note_id],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))) else {
        return (Vec::new(), String::new());
    };
    if Path::new(&path).is_absolute() { return (Vec::new(), name); }
    (split_note_path(&path).0, name)
}

/// 一条记录在索引里的「名字列 / 目录列 / 从可搜前缀里摘除的标签」。
/// 切片的名字列放所属笔记的文件名、目录列放它的目录段；这两样都已升格成独立列，
/// 就从可搜前缀里摘掉——否则搜目录名会命中该目录下每一篇，文件名也会被正文列弱命中重复计一次。
/// 其余记录只有名字列（实体规范名），目录列为空、无摘除。
pub(crate) fn index_columns(conn: &Connection, kind: RecordKind, payload: &Value) -> (String, String, Vec<String>) {
    if kind == RecordKind::Chunk {
        // 名字列与目录列只挂在这一篇的第一片上：否则一篇的每一片都会命中同一查询、把结果刷屏。
        if payload.get("ordinal").and_then(Value::as_u64).unwrap_or(0) != 0 { return (String::new(), String::new(), Vec::new()); }
        let note_id = payload.get("note_id").and_then(Value::as_i64).unwrap_or(0);
        let (dirs, stem) = note_path_parts(conn, note_id);
        let mut exclude = dirs.clone();
        if !stem.is_empty() { exclude.push(stem.clone()); }
        return (stem, dirs.join(" "), exclude);
    }
    (record_name(kind, payload), String::new(), Vec::new())
}

pub(crate) fn put_record(conn: &Connection, kind: RecordKind, input: &RecordInput,
    payload: &Value, text: &str) -> Result<(RecordHeader, crate::index::IndexDocument)> {
    validate_identity("namespace", &input.namespace)?;
    validate_identity("scope", &input.scope)?;
    for evidence in &input.evidence {
        if evidence.source.trim().is_empty() { return Err(Error::Validation("evidence source is required".into())); }
        match (evidence.offset, evidence.limit) {
            (None, None) => {},
            (Some(offset), Some(limit)) if offset >= 1 && limit >= 1 => {},
            _ => return Err(Error::Validation("evidence offset/limit must be a 1-based start and a positive line count".into())),
        }
    }
    let namespace_id = term_id(conn, &input.namespace)?;
    // 记录一落库，这个领域的向量分区就变了：登记它，缓存按领域精准失效。
    touch_namespace(&input.namespace);
    let scope_id = term_id(conn, &input.scope)?;
    let existing = match input.id {
        Some(id) => Some(conn.query_row("SELECT created_at_us,updated_at_us,revision,scope_id FROM records WHERE id=?1", [id],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?))).optional()?
            .ok_or_else(|| Error::NotFound(id.to_string()))?),
        None => None,
    };
    if let (Some(id), true) = (input.id, existing.as_ref().is_some_and(|v| v.3 != scope_id)) {
        // 换作用域会连带影响引用它的关系与事件，它们的端点必须与记录同域。
        // 无引用的记录（记忆、笔记、孤立实体）直接换；有引用的先清掉引用再换。
        let blocking: i64 = conn.query_row(
            "SELECT (SELECT COUNT(*) FROM relations WHERE subject_id=?1 OR object_id=?1) \
             + (SELECT COUNT(*) FROM event_participants WHERE entity_id=?1)", [id], |r| r.get(0))?;
        if blocking > 0 {
            return Err(Error::Conflict(format!("record {id} is referenced by {blocking} relation(s) or event participant row(s); remove those references before changing its scope")));
        }
    }
    if let Some(expected) = input.expected_revision {
        if existing.as_ref().map(|v| v.2) != Some(expected) { return Err(Error::StaleRevision(input.id.map(|v| v.to_string()).unwrap_or_default())); }
    }
    let now = now_us();
    let created = existing.as_ref().map(|v| v.0).unwrap_or(input.created_at_us.unwrap_or(now));
    let updated = input.updated_at_us.unwrap_or_else(|| now.max(existing.as_ref().map(|v| v.1).unwrap_or(created)));
    if updated < created { return Err(Error::Validation("updated_at_us precedes created_at_us".into())); }
    // 指纹跟着「送进搜索的文本 + 标签」走：正文或标签一变，各空间的旧向量立即失效。
    let tags = normalize_tags(&input.tags);
    let fingerprint = record_fingerprint(text, &tags);
    let metadata_json = serde_json::to_string(&input.metadata)?;
    let evidence_json = serde_json::to_string(&input.evidence)?;
    let payload_json = serde_json::to_string(payload)?;
    let (id, revision) = match input.id {
        Some(id) => {
            let revision = next_revision(conn, id)?;
            conn.execute("UPDATE records SET namespace_id=?2,kind=?3,scope_id=?4,updated_at_us=?5,revision=?6,metadata_json=?7,
                evidence_json=?8,fingerprint=?9,payload_json=?10 WHERE id=?1",
                params![id, namespace_id, kind.code(), scope_id, updated, revision, metadata_json, evidence_json,
                    fingerprint, payload_json])?;
            // Updating text invalidates every space's embedding in the same transaction.
            conn.execute("DELETE FROM vectors.embeddings WHERE record_id=?1 AND fingerprint<>?2", params![id, fingerprint])?;
            (id, revision)
        }
        None => {
            conn.execute("INSERT INTO records(namespace_id,kind,scope_id,created_at_us,updated_at_us,revision,metadata_json,evidence_json,
                fingerprint,payload_json) VALUES (?1,?2,?3,?4,?5,0,?6,?7,?8,?9)",
                params![namespace_id, kind.code(), scope_id, created, updated, metadata_json, evidence_json,
                    fingerprint, payload_json])?;
            let id = conn.last_insert_rowid();
            let revision = next_revision(conn, id)?;
            conn.execute("UPDATE records SET revision=?2 WHERE id=?1", params![id, revision])?;
            (id, revision)
        }
    };
    let tag_ids = set_record_tags(conn, id, &tags)?;
    // 索引文档就地拼好交回调用方：正文来自本次写入手上的那一份，索引阶段不再回源。
    // 标记一律带整数 id 给索引：namespace、scope、kind、tags 都不写第二份文本。
    let (name, path, exclude) = index_columns(conn, kind, payload);
    let document = crate::index::IndexDocument { id, namespace_id, scope_id, kind,
        text: text.to_string(), name, path,
        note_id: if kind == RecordKind::Chunk { payload.get("note_id").and_then(Value::as_i64).unwrap_or(0) } else { 0 },
        tags_prefix: tags_prefix(kind, &tags, &exclude, payload), tag_ids };
    Ok((RecordHeader { id, namespace: input.namespace.clone(), kind, scope: input.scope.clone(),
        created_at_us: created, updated_at_us: updated, revision, tags,
        evidence: input.evidence.clone(), metadata: input.metadata.clone() }, document))
}

/// 记录指纹：正文 + 标签一起算。两者任一变，各空间的旧向量立即失效。
pub(crate) fn record_fingerprint(text: &str, tags: &[String]) -> String {
    text::digest(&format!("text-v1\n{text}\n{}", tags.join(" ")))
}

/// 换掉一条记录的标签，返回这次的 tag id（按标签文本排序，与 `normalize_tags` 同序）。
pub(crate) fn set_record_tags(conn: &Connection, id: i64, tags: &[String]) -> Result<Vec<i64>> {
    conn.execute("DELETE FROM record_tags WHERE record_id=?1", [id])?;
    let mut tag_ids = Vec::with_capacity(tags.len());
    for tag in tags {
        let tag_id = term_id(conn, tag)?;
        conn.execute("INSERT OR IGNORE INTO record_tags(record_id,tag_id) VALUES (?1,?2)", params![id, tag_id])?;
        tag_ids.push(tag_id);
    }
    Ok(tag_ids)
}

/// 一条记录的 (tag id, 标签文本)，按文本排序。索引写 id 列与文本列都从这里取。
pub(crate) fn record_tag_pairs(conn: &Connection, id: i64) -> Result<Vec<(i64, String)>> {
    let mut stmt = conn.prepare("SELECT t.id,t.text FROM record_tags rt JOIN strings t ON t.id=rt.tag_id WHERE rt.record_id=?1 ORDER BY t.text")?;
    let mut pairs = Vec::new();
    for row in stmt.query_map([id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? { pairs.push(row?); }
    Ok(pairs)
}

/// 按库里现有状态给一条记录拼索引文档：标记（namespace / scope / tags）一律取 strings 表的 id。
pub(crate) fn index_document(conn: &Connection, id: i64, kind: RecordKind, text: String) -> Result<crate::index::IndexDocument> {
    let (namespace_id, scope_id, payload_json): (i64, i64, String) = conn.query_row("SELECT namespace_id,scope_id,payload_json FROM records WHERE id=?1",
        [id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
    let payload: Value = serde_json::from_str(&payload_json)?;
    let pairs = record_tag_pairs(conn, id)?;
    let tags: Vec<String> = pairs.iter().map(|(_, tag)| tag.clone()).collect();
    let (name, path, exclude) = index_columns(conn, kind, &payload);
    Ok(crate::index::IndexDocument { id, namespace_id, scope_id, kind, text,
        name, path,
        note_id: if kind == RecordKind::Chunk { payload.get("note_id").and_then(Value::as_i64).unwrap_or(0) } else { 0 },
        tags_prefix: tags_prefix(kind, &tags, &exclude, &payload),
        tag_ids: pairs.into_iter().map(|(tag_id, _)| tag_id).collect() })
}

/// 一批记录 id 里哪些是切片，各自属于哪一篇笔记、在原文里从第几行起。非切片不出现在结果里。
/// 「一篇笔记有多少片段命中」要按笔记精确统计，折叠后的代表命中要挂上「同一篇的其余片段」，
/// 两件事靠的都是这层「切片 → 笔记」的归属。
pub(crate) fn chunk_notes(conn: &Connection, ids: &[i64]) -> Result<BTreeMap<i64, (i64, usize)>> {
    let mut out = BTreeMap::new();
    if ids.is_empty() { return Ok(out); }
    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let mut stmt = conn.prepare(&format!("SELECT record_id,note_id,\"offset\" FROM chunks WHERE record_id IN ({placeholders})"))?;
    for row in stmt.query_map(params_from_iter(ids.iter().copied()), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))? {
        let (id, note_id, offset) = row?;
        out.insert(id, (note_id, offset.max(0) as usize));
    }
    Ok(out)
}

/// 该知识领域登记的笔记根目录；没登记就是 `None`。
pub(crate) fn namespace_root(conn: &Connection, namespace_id: i64) -> Result<Option<String>> {
    Ok(conn.query_row("SELECT root FROM namespace_roots WHERE namespace_id=?1", [namespace_id], |r| r.get(0)).optional()?)
}

/// 库里存的相对路径 → 实际文件路径：登记过根目录就拼回去，没登记就是原样那条。
pub(crate) fn absolute_note_path(conn: &Connection, namespace_id: i64, stored: &str) -> String {
    match namespace_root(conn, namespace_id) {
        Ok(Some(root)) => Path::new(&root).join(stored.replace('/', std::path::MAIN_SEPARATOR_STR)).to_string_lossy().into_owned(),
        _ => stored.to_string(),
    }
}

pub(crate) fn record_value(conn: &Connection, key: &RecordKey) -> Result<Option<Value>> {
    Ok(record_values(conn, &[key.id])?.remove(&key.id))
}

/// 批量读取记录本体：一次查 records、一次查 tags，再由同一套装配逻辑还原。
/// 语义等同于对每个 id 调用 `record_value`，只是把逐行往返压成固定两次查询。
pub(crate) fn record_values(conn: &Connection, ids: &[i64]) -> Result<BTreeMap<i64, Value>> {
    let mut out = BTreeMap::new();
    if ids.is_empty() { return Ok(out); }
    let placeholders = vec!["?"; ids.len()].join(",");
    let params = ids.iter().map(|id| SqlValue::Integer(*id)).collect::<Vec<_>>();
    let mut stmt = conn.prepare(&format!("SELECT r.id,r.namespace_id,n.text,r.kind,s.text,r.created_at_us,r.updated_at_us,r.revision,
        r.metadata_json,r.evidence_json,r.payload_json FROM records r
        JOIN strings n ON n.id=r.namespace_id JOIN strings s ON s.id=r.scope_id WHERE r.id IN ({placeholders}) ORDER BY r.id"))?;
    let mut rows: Vec<(i64, i64, String, i64, String, i64, i64, i64, String, String, String)> = Vec::new();
    for row in stmt.query_map(params_from_iter(params.iter().cloned()), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?,
        r.get::<_, String>(2)?, r.get::<_, i64>(3)?, r.get::<_, String>(4)?, r.get::<_, i64>(5)?, r.get::<_, i64>(6)?,
        r.get::<_, i64>(7)?, r.get::<_, String>(8)?, r.get::<_, String>(9)?, r.get::<_, String>(10)?)))? {
        rows.push(row?);
    }
    let mut tags_stmt = conn.prepare(&format!("SELECT rt.record_id,t.text FROM record_tags rt JOIN strings t ON t.id=rt.tag_id \
        WHERE rt.record_id IN ({placeholders}) ORDER BY rt.record_id,t.text"))?;
    let mut tags: BTreeMap<i64, Vec<String>> = BTreeMap::new();
    for row in tags_stmt.query_map(params_from_iter(params.iter().cloned()), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
        let (id, tag) = row?;
        tags.entry(id).or_default().push(tag);
    }
    // 笔记的路径与文件名都是它自己的列（原样）：读取时按 record_id 补回，
    // 路径不进标签字典，payload 里也不重复存。
    let mut note_meta: BTreeMap<i64, (String, String)> = BTreeMap::new();
    {
        let mut stmt = conn.prepare(&format!("SELECT n.record_id,n.path,n.name FROM notes n \
            WHERE n.record_id IN ({placeholders})"))?;
        for row in stmt.query_map(params_from_iter(params.iter().cloned()), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))? {
            let (id, path, name) = row?;
            note_meta.insert(id, (path, name));
        }
    }
    for (id, namespace_id, namespace, kind_code, scope, created, updated, revision, metadata, evidence, payload) in rows {
        let kind = RecordKind::from_code(kind_code).ok_or_else(|| Error::Validation("invalid stored record kind".into()))?;
        let header = RecordHeader { id, namespace, kind, scope,
            created_at_us: created, updated_at_us: updated, revision, tags: tags.remove(&id).unwrap_or_default(),
            metadata: serde_json::from_str(&metadata)?, evidence: serde_json::from_str(&evidence)? };
        let mut value = serde_json::to_value(header)?;
        let object = value.as_object_mut().ok_or_else(|| Error::Validation("invalid stored header".into()))?;
        let mut payload: Metadata = serde_json::from_str(&payload)?;
        // memory_type 也走 strings 文本表：payload 里只存 id，读取时还原文本。
        if let Some(type_id) = payload.get("memory_type_id").and_then(Value::as_i64) {
            payload.insert("memory_type".into(), Value::String(term_text(conn, type_id)?));
            payload.remove("memory_type_id");
        }
        if kind == RecordKind::Note {
            // 库里存的是相对路径（登记过领域根目录时），对外取回时拼回绝对路径。
            let (stored, name) = note_meta.remove(&id).unwrap_or_default();
            let source = absolute_note_path(conn, namespace_id, &stored);
            payload.insert("source".into(), Value::String(source));
            payload.insert("title".into(), Value::String(name));
        }
        object.extend(payload);
        out.insert(id, value);
    }
    Ok(out)
}

pub(crate) fn matches_filter(conn: &Connection, key: &RecordKey, filter: &ReadFilter) -> Result<bool> {
    validate_filter(filter)?;
    let row: Option<(i64, i64)> = conn.query_row("SELECT namespace_id,scope_id FROM records WHERE id=?1", [key.id],
        |r| Ok((r.get(0)?, r.get(1)?))).optional()?;
    let Some((namespace_id, scope_id)) = row else { return Ok(false) };
    if term_text(conn, namespace_id)? != text::normalized_tag(&filter.namespace) { return Ok(false); }
    let scope = term_text(conn, scope_id)?;
    if !filter.scopes.iter().any(|s| text::normalized_tag(s) == scope) { return Ok(false); }
    for tag in &filter.tags {
        let exists: bool = conn.query_row("SELECT EXISTS(SELECT 1 FROM record_tags rt JOIN strings t ON t.id=rt.tag_id WHERE rt.record_id=?1 AND t.text=?2)",
            params![key.id, text::normalized_tag(tag)], |r| r.get(0))?;
        if !exists { return Ok(false); }
    }
    Ok(true)
}

pub(crate) fn get<T: DeserializeOwned>(conn: &Connection, key: &RecordKey, filter: &ReadFilter) -> Result<T> {
    if !matches_filter(conn, key, filter)? { return Err(Error::NotFound(key.id.to_string())); }
    serde_json::from_value(record_value(conn, key)?.ok_or_else(|| Error::NotFound(key.id.to_string()))?).map_err(Error::from)
}

/// 批量读取一批记录，只返回满足 `filter` 的那些。
/// 语义等同于对每个 id 依次调用 `get`，但把过滤压成一条 SQL，避免逐条回库。
pub(crate) fn load_many<T: DeserializeOwned>(conn: &Connection, ids: &[i64], filter: &ReadFilter) -> Result<BTreeMap<i64, T>> {
    let mut out = BTreeMap::new();
    if ids.is_empty() { return Ok(out); }
    validate_filter(filter)?;
    let (condition, values) = filter_sql(filter, &[], true)?;
    let placeholders = vec!["?"; ids.len()].join(",");
    let mut stmt = conn.prepare(&format!("SELECT r.id FROM records r WHERE r.id IN ({placeholders}) AND {condition} ORDER BY r.id"))?;
    let params = ids.iter().map(|id| SqlValue::Integer(*id)).chain(values).collect::<Vec<_>>();
    let allowed = stmt.query_map(params_from_iter(params), |r| r.get::<_, i64>(0))?.collect::<std::result::Result<Vec<_>, _>>()?;
    for (id, value) in record_values(conn, &allowed)? {
        out.insert(id, serde_json::from_value(value)?);
    }
    Ok(out)
}

/// 生成 `records` 表上的过滤条件。
///
/// `by_ids` 用于「手里已经有一批 record_id、只在这些 id 内做属性过滤」的查询：给
/// `namespace_id` 加一元 `+`（SQLite 的 no-op，唯一作用是禁止该表达式用索引），
/// 逼优化器以 IN 列表逐行点查主键。不加的话它会拿 `records_scope` 去扫该 namespace 下的
/// 全部行——23998 行的库上取 10 条要 1167 µs，加 `+` 只要 12 µs，而且成本随库规模线性增长。
/// 反之，`select_keys` 那类「按条件翻页、手里没有 id 列表」的查询必须靠索引来缩小范围，不能禁。
pub(crate) fn filter_sql(filter: &ReadFilter, kinds: &[RecordKind], by_ids: bool) -> Result<(String, Vec<SqlValue>)> {
    validate_filter(filter)?;
    let mut query = if by_ids {
        "+r.namespace_id=(SELECT id FROM strings WHERE text=?)".to_string()
    } else {
        "r.namespace_id=(SELECT id FROM strings WHERE text=?)".to_string()
    };
    let mut values = vec![SqlValue::Text(text::normalized_tag(&filter.namespace))];
    query.push_str(" AND r.scope_id IN (SELECT id FROM strings WHERE text IN (");
    query.push_str(&vec!["?"; filter.scopes.len()].join(",")); query.push_str("))");
    values.extend(filter.scopes.iter().map(|s| SqlValue::Text(text::normalized_tag(s))));
    if !kinds.is_empty() {
        query.push_str(" AND r.kind IN ("); query.push_str(&vec!["?"; kinds.len()].join(",")); query.push(')');
        values.extend(kinds.iter().map(|k| SqlValue::Integer(k.code())));
    }
    for tag in &filter.tags {
        query.push_str(" AND EXISTS(SELECT 1 FROM record_tags rt JOIN strings t ON t.id=rt.tag_id WHERE rt.record_id=r.id AND t.text=?)");
        values.push(SqlValue::Text(text::normalized_tag(tag)));
    }
    Ok((query, values))
}

/// 按过滤条件选出记录 id（升序），供批量写入路径使用。
/// 与翻页查询不同，这里要的是全量命中，所以不带 limit。
pub(crate) fn select_ids(conn: &Connection, filter: &ReadFilter, kinds: &[RecordKind]) -> Result<Vec<i64>> {
    let (condition, values) = filter_sql(filter, kinds, false)?;
    let mut stmt = conn.prepare(&format!("SELECT r.id FROM records r WHERE {condition} ORDER BY r.id"))?;
    let ids = stmt.query_map(params_from_iter(values), |r| r.get::<_, i64>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(ids)
}

/// 过滤条件下的匹配总数（截断之前）。`with_total` 打开时才调用。
pub(crate) fn count_matches(conn: &Connection, filter: &ReadFilter, kinds: &[RecordKind]) -> Result<usize> {
    let (condition, values) = filter_sql(filter, kinds, false)?;
    let count: i64 = conn.query_row(&format!("SELECT COUNT(*) FROM records r WHERE {condition}"), params_from_iter(values), |r| r.get(0))?;
    Ok(count as usize)
}

/// 记录的正文列内容：这条记录自己的文本。
/// 切片正文来自写入时切好的那一段，不在这里算，所以这条纯函数只覆盖其余四种记录。
/// 实体的正文不含规范名——规范名单独走 `record_name` 的 name 列，不在正文里占位。
pub(crate) fn record_text(kind: RecordKind, payload: &Value) -> String {
    let field = |key: &str| payload.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    match kind {
        RecordKind::Memory => field("judgment"),
        RecordKind::Entity => entity_body(payload),
        RecordKind::Relation => format!("{} {} {} {}", field("subject_name"), field("predicate"), field("object_name"), field("reason")),
        RecordKind::Event => format!("{} {} {} {}", field("name"), field("summary"), name_list(payload), field("reason")),
        // 笔记与切片的正文另有来源：笔记的检索面交给切片，切片正文由写入流程就地提供。
        RecordKind::Note | RecordKind::Chunk => String::new(),
    }
}

/// 记录的名字列内容：只有实体有规范名，其它记录为空串（检索时该列不参与）。
pub(crate) fn record_name(kind: RecordKind, payload: &Value) -> String {
    match kind {
        RecordKind::Entity => payload.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
        _ => String::new(),
    }
}

/// 一批事件记录的正文长度（字符数），只含库里存在的 id。
///
/// 与 `record_text(RecordKind::Event, ..)` 是同一套拼法，只是走 SQL 算：事件那一路常常
/// 一次取出比字符预算大一到两个数量级的一批 id，先知道长度就能只装配真正要留下的几条。
/// 非字符串字段与缺失字段都按空串计，与 Rust 侧 `Value::as_str` 的口径一致；
/// `participant_names` 不是数组时同样按空串计。
pub(crate) fn event_text_lengths(conn: &Connection, ids: &[i64]) -> Result<BTreeMap<i64, usize>> {
    let mut out = BTreeMap::new();
    if ids.is_empty() { return Ok(out); }
    let placeholders = vec!["?"; ids.len()].join(",");
    let params = ids.iter().map(|id| SqlValue::Integer(*id)).collect::<Vec<_>>();
    let field = |name: &str| format!(
        "CASE WHEN json_type(r.payload_json,'$.{name}')='text' THEN json_extract(r.payload_json,'$.{name}') ELSE '' END");
    // 参与者名：不是数组时按空串计，是数组时只收文本元素（`j.type` 是元素的 JSON 类型名，
    // 用 `json_type(j.value)` 会把已解包的文本再当 JSON 解析一次，直接报 malformed JSON）。
    let names = "COALESCE(CASE WHEN json_type(r.payload_json,'$.participant_names')='array' \
        THEN (SELECT group_concat(j.value,' ') FROM json_each(r.payload_json,'$.participant_names') j \
            WHERE j.type='text') ELSE '' END,'')";
    let mut stmt = conn.prepare(&format!(
        "SELECT r.id, LENGTH({} || ' ' || {} || ' ' || {names} || ' ' || {}) \
         FROM records r WHERE r.id IN ({placeholders}) ORDER BY r.id",
        field("name"), field("summary"), field("reason")))?;
    for row in stmt.query_map(params_from_iter(params.iter().cloned()), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))? {
        let (id, length) = row?;
        out.insert(id, length.max(0) as usize);
    }
    Ok(out)
}

/// 一批记录里属于实体的那些的规范名（按 record_id）。重排取文档时用：正文列已不含规范名，
/// 纯名实体（别名、摘要、属性全空）的正文是空串，得把规范名拼回去才能让重排看到名字。
pub(crate) fn entity_names(conn: &Connection, ids: &[i64]) -> Result<BTreeMap<i64, String>> {
    let mut out = BTreeMap::new();
    if ids.is_empty() { return Ok(out); }
    let placeholders = vec!["?"; ids.len()].join(",");
    let mut stmt = conn.prepare(&format!("SELECT record_id,name FROM entities WHERE record_id IN ({placeholders})"))?;
    let params = ids.iter().map(|id| SqlValue::Integer(*id)).collect::<Vec<_>>();
    for row in stmt.query_map(params_from_iter(params), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
        let (id, name) = row?;
        out.insert(id, name);
    }
    Ok(out)
}

fn name_list(payload: &Value) -> String {
    payload.get("participant_names").and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

fn entity_body(payload: &Value) -> String {
    let aliases = payload.get("aliases").and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" ")).unwrap_or_default();
    let summary = payload.get("summary").and_then(Value::as_str).unwrap_or("");
    let attr_text = payload.get("attributes").and_then(Value::as_object).map(|attrs| {
        attrs.iter().map(|(key, values)| {
            let joined = values.as_array().map(|v| v.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" ")).unwrap_or_default();
            format!("{key} {joined}")
        }).collect::<Vec<_>>().join(" ")
    }).unwrap_or_default();
    format!("{aliases} {summary} {attr_text}")
}

pub(crate) fn select_keys(conn: &Connection, filter: &ReadFilter, kinds: &[RecordKind], limit: usize, after: Option<&str>) -> Result<Vec<RecordKey>> {    let (mut condition, mut values) = filter_sql(filter, kinds, false)?;
    if let Some(cursor) = after {
        let id: i64 = cursor.parse().map_err(|_| Error::Validation("invalid page cursor".into()))?;
        condition.push_str(" AND r.id>?");
        values.push(SqlValue::Integer(id));
    }
    values.push(SqlValue::Integer(limit.min(i64::MAX as usize) as i64));
    let mut stmt = conn.prepare(&format!("SELECT r.id FROM records r WHERE {condition} ORDER BY r.id LIMIT ?"))?;
    let rows = stmt.query_map(params_from_iter(values), |r| r.get::<_, i64>(0))?;
    let mut keys = Vec::new();
    for row in rows { keys.push(RecordKey { id: row? }); }
    Ok(keys)
}

pub(crate) fn list<T: DeserializeOwned>(conn: &Connection, kind: RecordKind, request: &PageRequest) -> Result<Page<T>> {
    validate_limit(request.limit)?;
    let mut keys = select_keys(conn, &request.filter, &[kind], request.limit + 1, request.after.as_deref())?;
    let has_more = keys.len() > request.limit;
    keys.truncate(request.limit);
    let next_cursor = if has_more { keys.last().map(RecordKey::index_key) } else { None };
    // 一次批量取回：逐条 `get` 会为每条记录各跑一遍过滤与装配，limit=100 就是几百次回库。
    let ids: Vec<i64> = keys.iter().map(|key| key.id).collect();
    let mut loaded: BTreeMap<i64, T> = load_many(conn, &ids, &request.filter)?;
    let items = keys.into_iter().filter_map(|key| loaded.remove(&key.id)).collect::<Vec<_>>();
    Ok(Page { items, next_cursor })
}

pub(crate) fn delete_record(conn: &Connection, key: &RecordKey) -> Result<bool> {
    // Graph references are RESTRICT, so callers explicitly remove edges first.
    // 领域名要在删之前取：删完这条记录就查不到它属于哪个领域了。
    let namespace = namespace_of(conn, key.id)?;
    let changed = conn.execute("DELETE FROM records WHERE id=?1", [key.id]);
    let changed = match changed {
        Err(rusqlite::Error::SqliteFailure(err, _)) if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            return Err(Error::Conflict(format!("record {} is still referenced", key.id))),
        other => other?,
    };
    if changed > 0 {
        // 向量外挂库没有外键级联，删记录时显式清掉它的向量。
        conn.execute("DELETE FROM vectors.embeddings WHERE record_id=?1", [key.id])?;
        next_revision(conn, key.id)?;
        if let Some(namespace) = namespace { touch_namespace(&namespace); }
    }
    Ok(changed > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 按 id 批量取回时优化器选中的计划。
    fn batched_load_plan(conn: &Connection, ids: &[i64], by_ids: bool) -> String {
        let (condition, values) = filter_sql(&ReadFilter::default(), &[], by_ids).unwrap();
        let placeholders = vec!["?"; ids.len()].join(",");
        let sql = format!("EXPLAIN QUERY PLAN SELECT r.id FROM records r WHERE r.id IN ({placeholders}) AND {condition} ORDER BY r.id");
        let params: Vec<SqlValue> = ids.iter().map(|id| SqlValue::Integer(*id)).chain(values).collect();
        let mut stmt = conn.prepare(&sql).unwrap();
        let plans: Vec<String> = stmt.query_map(params_from_iter(params), |row| row.get::<_, String>(3))
            .unwrap().map(|row| row.unwrap()).collect();
        plans.join(" | ")
    }

    fn seed(kb: &KnowledgeBase, rows: i64) {
        let inputs: Vec<crate::MemoryInput> = (1..=rows).map(|i| crate::MemoryInput::new(format!("记录 {i}"))).collect();
        kb.memories().upsert_many(&inputs).unwrap();
    }

    fn sample_ids() -> Vec<i64> { (1..=10).collect() }

    /// 手里已有 id 列表时，取回必须按主键点查。少了 `+`，优化器会去扫 records_scope 索引，
    /// 成本随库规模线性增长，而每一行都是额外的读放大。
    #[test]
    fn batched_load_stays_on_the_primary_key() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        seed(&kb, 100);
        let guard = kb.read().unwrap();
        let plan = batched_load_plan(guard.conn(), &sample_ids(), true);
        assert!(plan.contains("INTEGER PRIMARY KEY"), "批量取回退化为扫索引：{plan}");
    }

    /// 事件正文长度走 SQL 算，必须与 Rust 侧 `record_text` 逐条一致：
    /// 两者一旦漂移，事件那一路的字符预算就截在别的地方，返回的事件跟着变。
    #[test]
    fn event_text_lengths_match_record_text() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let entity = |name: &str| crate::EntityInput { record: Default::default(), name: name.into(),
            entity_type: "person".into(), aliases: vec![], attributes: BTreeMap::new(), summary: String::new() };
        let created = kb.graph().apply_batch(&crate::GraphBatch {
            entities: vec![entity("甲"), entity("乙")], ..Default::default()
        }).unwrap().value;
        let (first, second) = (created.entities[0].header.id, created.entities[1].header.id);
        let created = kb.graph().apply_batch(&crate::GraphBatch {
            events: vec![
                crate::EventInput { record: Default::default(), name: "别鹤典仪".into(), summary: "两人同去".into(),
                    participants: vec![first, second], confidence: 1.0, reason: "有人证".into() },
                crate::EventInput { record: Default::default(), name: "堂中自语".into(), summary: String::new(),
                    participants: vec![first], confidence: 1.0, reason: String::new() },
            ], ..Default::default()
        }).unwrap().value;
        let ids: Vec<i64> = created.events.iter().map(|event| event.header.id).collect();

        // 再把两条改成边角形态：非字符串字段、缺字段、participant_names 不是数组。
        {
            let raw = Connection::open(dir.path().join("store.sqlite3")).unwrap();
            let payloads = [
                r#"{"name":7,"summary":"只剩数字名","participant_names":"甲 乙","reason":null}"#,
                r#"{"name":"正常","summary":null,"participant_names":["甲",7,"乙"],"reason":"理由"}"#,
            ];
            for (id, payload) in ids.iter().zip(payloads) {
                raw.execute("UPDATE records SET payload_json=?1 WHERE id=?2", params![payload, id]).unwrap();
            }
        }

        let guard = kb.read().unwrap();
        let conn = guard.conn();
        let lengths = event_text_lengths(conn, &ids).unwrap();
        assert_eq!(lengths.len(), ids.len(), "每条事件都该有长度");
        for id in ids {
            let payload = record_values(conn, &[id]).unwrap().remove(&id).unwrap();
            assert_eq!(lengths[&id], record_text(RecordKind::Event, &payload).chars().count(),
                "事件 {id} 的 SQL 长度与 record_text 不一致");
        }
    }

    // ── 向量分区缓存的按领域失效 ──────────────────────────────────────
    fn fixture_space() -> crate::embeddings::EmbeddingSpace {
        crate::embeddings::EmbeddingSpace { id: "v".into(), model: "fixture/v1".into(),
            dimension: 2, text_version: 1, encoding: "f32".into() }
    }

    /// 让某个领域的向量分区进缓存。空领域也会被缓存成空分区，所以不必真造向量。
    fn cache_partition(kb: &KnowledgeBase, space: &crate::embeddings::EmbeddingSpace, namespace: &str) {
        kb.partition(space, namespace, "public").unwrap();
    }

    /// 当前缓存着哪些领域的向量分区。
    fn cached_namespaces(kb: &KnowledgeBase) -> BTreeSet<String> {
        kb.engine.vectors.entries.lock().keys().map(|(_, namespace, _)| namespace.clone()).collect()
    }

    fn namespace_filter(namespace: &str) -> ReadFilter {
        ReadFilter { namespace: namespace.into(), scopes: vec!["public".into()], tags: vec![], note_ids: vec![] }
    }

    /// 缓存失效按领域分开做：动过的领域清条目并推进版本号，没动过的条目与版本号都不动。
    #[test]
    fn invalidating_one_namespace_leaves_the_others_alone() {
        let cache = VectorCache::new();
        let key = |namespace: &str| ("v".to_string(), namespace.to_string(), "public".to_string());
        cache.entries.lock().insert(key("a"), None);
        cache.entries.lock().insert(key("b"), None);
        let epoch_b = cache.epoch_of("b");

        cache.invalidate_namespaces(&HashSet::from(["a".to_string()]));

        assert!(cache.entries.lock().get(&key("a")).is_none(), "写过的领域要清掉条目");
        assert!(cache.entries.lock().get(&key("b")).is_some(), "没写过的领域不该被牵连");
        assert_eq!(cache.epoch_of("b"), epoch_b, "没写过的领域版本号不动");
        assert_ne!(cache.epoch_of("a"), epoch_b, "写过的领域版本号要前进，在途载入才会作废");

        // 整体失效：所有领域一起作废，版本号只增不减。
        let epoch_a = cache.epoch_of("a");
        cache.invalidate();
        assert!(cache.entries.lock().is_empty());
        assert!(cache.epoch_of("a") > epoch_a && cache.epoch_of("b") > epoch_b);
    }

    /// 写一个领域，只该清掉那个领域已经载入的分区。
    #[test]
    fn writing_one_namespace_keeps_other_vector_partitions_cached() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let space = fixture_space();
        for namespace in ["a", "b"] { cache_partition(&kb, &space, namespace); }
        assert_eq!(cached_namespaces(&kb), BTreeSet::from(["a".to_string(), "b".to_string()]));

        let mut input = crate::MemoryInput::new("写在 a 领域的一条");
        input.record.namespace = "a".into();
        kb.memories().upsert(input).unwrap();

        assert_eq!(cached_namespaces(&kb), BTreeSet::from(["b".to_string()]), "只该清掉被写的那个领域");
    }

    /// 删记录同样精准失效。领域名必须在删之前取出来——删完这条记录就查不到它属于哪个领域了。
    #[test]
    fn deleting_a_record_evicts_only_its_own_namespace() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let mut input = crate::MemoryInput::new("要被删掉的一条");
        input.record.namespace = "a".into();
        let id = kb.memories().upsert(input).unwrap().value.header.id;

        let space = fixture_space();
        for namespace in ["a", "b"] { cache_partition(&kb, &space, namespace); }
        kb.memories().delete(id, &namespace_filter("a")).unwrap();

        assert_eq!(cached_namespaces(&kb), BTreeSet::from(["b".to_string()]), "删掉的领域要清，别的领域留着");
    }

    /// 补齐向量跑一轮，只该作废它写了向量的领域。
    /// 补齐收尾会落就绪标记，那是只写不喂向量分区的写入，不能顺手把整个缓存清掉。
    #[test]
    fn filling_vectors_only_evicts_the_namespaces_it_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let space = fixture_space();
        kb.embeddings().register_space(space.clone()).unwrap();
        kb.embeddings().register_embedder("v", |texts: &[String]| -> std::result::Result<Vec<Vec<f32>>, crate::EmbedCallbackError> {
            Ok(texts.iter().map(|_| vec![1.0f32, 0.0]).collect())
        }).unwrap();
        for namespace in ["a", "b"] { cache_partition(&kb, &space, namespace); }
        kb.memories().upsert(crate::MemoryInput::new("补齐用的一条")).unwrap();
        kb.embeddings().sync("v", 32).unwrap();

        let cached = cached_namespaces(&kb);
        assert!(cached.contains("a") && cached.contains("b"),
            "补齐只写了 default 领域，a 与 b 的分区缓存不该被牵连：{cached:?}");
    }

    /// 有写路径没登记领域时（按 id 删掉记录、或将来新增的写路径），宁可整体失效，
    /// 也不留下一个来源说不清的陈旧分区。
    #[test]
    fn an_unregistered_write_falls_back_to_invalidating_everything() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let space = fixture_space();
        for namespace in ["a", "b"] { cache_partition(&kb, &space, namespace); }

        // 直接改一行、不登记领域：模拟一条没接上登记的写路径。
        kb.mutate(|tx| Ok(tx.execute("INSERT INTO meta(key,value) VALUES ('cache_probe',1)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value", [])?)).unwrap();

        assert!(cached_namespaces(&kb).is_empty(), "登记为空却改过行时必须整体失效");
    }
}
