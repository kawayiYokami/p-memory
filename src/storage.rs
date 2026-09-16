use crate::index::TextIndex;
use crate::{schema, text, types::*, Error, Result};
use fs4::fs_std::FileExt;
use parking_lot::{Mutex, RwLock};
use rusqlite::{params, params_from_iter, types::Value as SqlValue, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::{collections::{BTreeMap, BTreeSet, HashMap}, fs::{File, OpenOptions}, path::{Path, PathBuf}, sync::{atomic::{AtomicU64, Ordering}, Arc}};

/// 写者：独占的写连接 + 跨进程文件锁。只挡其他写者，不挡读。
pub(crate) struct Writer { pub conn: Connection, _file_lock: File }

/// 只读连接池。`rusqlite::Connection` 不是 `Sync`，并发读必须各持一条独立连接；
/// 池锁只在取出与归还时短暂持有，读的整个过程不占任何全局锁。
pub(crate) struct Readers { pub idle: Vec<Connection> }

/// 向量分区缓存。按 epoch 整体失效，条目按需重建。
/// epoch 必须独立于全局 revision：向量写入不推进 revision，只盯 revision 会漏失效。
pub(crate) struct VectorCache {
    pub epoch: AtomicU64,
    pub entries: Mutex<HashMap<(String, String, String), (u64, Option<Arc<crate::embeddings::Partition>>)>>,
}

impl VectorCache {
    fn new() -> Self {
        Self { epoch: AtomicU64::new(1), entries: Mutex::new(HashMap::new()) }
    }
    /// 逻辑失效：递增 epoch 并清空条目。清空是为了让内存有界——重新载入本来就是按需的。
    /// epoch 与清空两者都要：清空负责释放内存，epoch 负责挡住「加载中撞上写入」后写入的过期条目。
    pub fn invalidate(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        self.entries.lock().clear();
    }
}

pub(crate) struct Engine {
    pub writer: Mutex<Option<Writer>>,
    pub readers: Mutex<Option<Readers>>,
    /// `TextIndex` 自带内部写锁、`IndexReader` 可并发检索，用 `Arc` 共享给所有读线程。
    /// 放进 `Option` 是为了 `close` 时能真正销毁它——Tantivy 的写锁由 `IndexWriter` 持有，
    /// 不销毁就无法释放，目录也重开不了。
    pub index: RwLock<Option<Arc<TextIndex>>>,
    pub vectors: VectorCache,
    /// 宿主注册的模型能力。它们是运行时状态（闭包 / Python 函数无法序列化），不落盘。
    pub embedders: crate::embeddings::EmbedderRegistry,
    pub rerankers: crate::search::RerankerRegistry,
    /// 最近观察到的降级档位，供健康检查读出「结果为什么变差」。
    pub degraded: Mutex<Vec<Degrade>>,
    pub root: PathBuf,
}

/// 只读连接的新建：`journal_mode` 是库文件上的持久属性，无需在每条连接上重设。
fn open_reader(root: &Path) -> Result<Connection> {
    let conn = Connection::open(root.join("store.sqlite3"))?;
    conn.execute_batch("PRAGMA busy_timeout=5000; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;")?;
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
        let index = Arc::new(TextIndex::open(&root)?);
        index.recover(&write_conn)?;
        // 只读连接在 schema 建好之后再开，保证它看到的是完整结构。
        let reader = open_reader(&root)?;
        Ok(Self { engine: Arc::new(Engine {
            writer: Mutex::new(Some(Writer { conn: write_conn, _file_lock: file_lock })),
            readers: Mutex::new(Some(Readers { idle: vec![reader] })),
            index: RwLock::new(Some(index)), vectors: VectorCache::new(),
            embedders: crate::embeddings::EmbedderRegistry::new(),
            rerankers: crate::search::RerankerRegistry::new(),
            degraded: Mutex::new(Vec::new()), root,
        }) })
    }

    pub fn directory(&self) -> &Path { &self.engine.root }

    /// 取文本索引的共享句柄。只在这一瞬间持有索引锁，拿到 `Arc` 后即可并发使用。
    pub(crate) fn index(&self) -> Result<Arc<TextIndex>> {
        self.engine.index.read().clone().ok_or(Error::Closed)
    }

    pub fn close(&self) -> Result<()> {
        let mut guard = self.engine.writer.lock();
        let result = match guard.as_ref() {
            Some(writer) => self.index()?.sync(&writer.conn),
            None => Ok(()),
        };
        *guard = None;
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

    /// 取一个向量分区：命中当前 epoch 的缓存就直接复用，否则按需载入后写入缓存。
    /// 载入过程不持缓存锁——否则一个慢分区会挡住所有其他分区的查询。
    pub(crate) fn partition(&self, conn: &Connection, space: &crate::embeddings::EmbeddingSpace,
        namespace: &str, scope: &str) -> Result<Option<Arc<crate::embeddings::Partition>>> {
        let key = (space.id.clone(), namespace.to_string(), scope.to_string());
        let epoch = self.engine.vectors.epoch.load(Ordering::SeqCst);
        let cached = {
            let entries = self.engine.vectors.entries.lock();
            entries.get(&key).filter(|(cached, _)| *cached == epoch).map(|(_, partition)| partition.clone())
        };
        if let Some(partition) = cached { return Ok(partition); }
        let loaded = crate::embeddings::Partition::load(conn, space, namespace, scope)?.map(Arc::new);
        // 载入期间可能发生了写入：epoch 变了就说明这份数据已过期，索性不写缓存。
        if self.engine.vectors.epoch.load(Ordering::SeqCst) == epoch {
            self.engine.vectors.entries.lock().insert(key, (epoch, loaded.clone()));
        }
        Ok(loaded)
    }

    /// 只在索引确实落后、且写者空闲时追平。
    /// 稳态下待办队列是空的（写路径提交后已就地清空），这一次预检就让读完全不碰写锁。
    /// 待办非空说明写入正在进行或上一次索引提交失败过：此时**只尝试、不等待**。
    /// 抢不到写锁就说明写者正在提交索引，读路径绝不能为此排队——一次提交是二十毫秒量级，
    /// 排队会把所有读都堵在门外。等待空闲时再补平，兼顾「上次提交失败后自愈」。
    pub(crate) fn sync_index_if_behind(&self, conn: &Connection) -> Result<()> {
        let pending: i64 = conn.query_row("SELECT COUNT(*) FROM index_updates", [], |r| r.get(0))?;
        if pending == 0 { return Ok(()); }
        // 抢不到写锁就说明写者正在提交索引，读路径绝不能为此排队。
        match self.engine.writer.try_lock() {
            Some(mut guard) => match guard.as_mut() {
                Some(writer) => self.index()?.sync(&writer.conn),
                None => Ok(()),
            },
            None => Ok(()),
        }
    }

    pub(crate) fn mutate<T>(&self, f: impl FnOnce(&Transaction<'_>) -> Result<T>) -> Result<WriteReceipt<T>> {
        let mut guard = self.engine.writer.lock();
        let writer = guard.as_mut().ok_or(Error::Closed)?;
        let tx = writer.conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let value = f(&tx)?;
        let revision = current_revision(&tx)?;
        tx.commit()?;
        // 失效与索引提交都放在写锁内、数据库提交之后：读路径不经过这把锁，
        // 所以索引再慢也只让其他写者排队，不会挡住任何一次读。
        self.engine.vectors.invalidate();
        let index_error = self.index()?.sync(&writer.conn).err().map(|e| e.to_string());
        Ok(WriteReceipt { value, revision, index_ready: index_error.is_none(), index_error })
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

    /// 写入后的内部向量化：库自己取文本 → 调宿主回调 → 短事务写回，宿主全程不碰向量。
    ///
    /// 这条路径**绝不阻塞写入**：拿不到该空间的回调（另一个线程正在同步）就跳过，
    /// 记录留在待补状态，交给后续写入或 `sync` 补。失败只记降级档位，不向上抛错。
    pub(crate) fn vectorize(&self, ids: &[i64]) {
        if ids.is_empty() || self.engine.embedders.is_empty() { return; }
        for degrade in crate::embeddings::vectorize_records(self, ids) {
            self.note_degrade(degrade);
        }
    }

    /// 笔记写入后连同它在本次写入中更新的切片一起向量化（切片是独立记录）。
    pub(crate) fn vectorize_note(&self, note_id: i64) {
        if self.engine.embedders.is_empty() { return; }
        let ids = match self.read() {
            Ok(state) => {
                let mut ids = vec![note_id];
                match state.conn().prepare("SELECT record_id FROM chunks WHERE note_id=?1").and_then(|mut stmt|
                    stmt.query_map([note_id], |r| r.get::<_, i64>(0)).map(|rows| rows.filter_map(|row| row.ok()).collect::<Vec<_>>())) {
                    Ok(chunks) => ids.extend(chunks),
                    Err(_) => return,
                }
                ids
            }
            Err(_) => return,
        };
        self.vectorize(&ids);
    }

    pub fn rebuild_indexes(&self) -> Result<HealthReport> {
        {
            let mut guard = self.engine.writer.lock();
            let writer = guard.as_mut().ok_or(Error::Closed)?;
            self.index()?.rebuild(&writer.conn)?;
            self.engine.vectors.invalidate();
        }
        self.health()
    }

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
            indexed_revision: meta(conn, "indexed_revision")?, record_count,
            index_document_count: self.index()?.document_count(),
            pending_index_updates: conn.query_row("SELECT COUNT(*) FROM index_updates", [], |r| r.get::<_, i64>(0))? as usize,
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
        Ok(())
    }

    /// Restores to a new directory; search indexes are rebuilt from the snapshot.
    pub fn restore(snapshot: impl AsRef<Path>, directory: impl AsRef<Path>) -> Result<Self> {
        let source = Connection::open_with_flags(snapshot.as_ref(), rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let app: i64 = source.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let version: i64 = source.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if app != schema::APPLICATION_ID { return Err(Error::Validation("snapshot is not a p-memory database".into())); }
        if version != schema::SCHEMA_VERSION { return Err(Error::SchemaVersion { found: version, supported: schema::SCHEMA_VERSION }); }
        std::fs::create_dir(directory.as_ref())?;
        source.backup(rusqlite::MAIN_DB, directory.as_ref().join("store.sqlite3"), None)?;
        Self::open(directory)
    }
}

pub(crate) fn now_us() -> i64 { chrono::Utc::now().timestamp_micros() }
pub(crate) fn meta(conn: &Connection, key: &str) -> Result<i64> {
    Ok(conn.query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0))?)
}
pub(crate) fn current_revision(conn: &Connection) -> Result<i64> { meta(conn, "revision") }
pub(crate) fn next_revision(conn: &Connection, record_id: i64) -> Result<i64> {
    conn.execute("UPDATE meta SET value=value+1 WHERE key='revision'", [])?;
    let revision = current_revision(conn)?;
    conn.execute("INSERT INTO index_updates(revision,record_id) VALUES (?1,?2)", params![revision, record_id])?;
    Ok(revision)
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

pub(crate) fn put_record(conn: &Connection, kind: RecordKind, input: &RecordInput,
    payload: &Value, embedding_text: &str) -> Result<RecordHeader> {
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
    let scope_id = term_id(conn, &input.scope)?;
    let existing = match input.id {
        Some(id) => Some(conn.query_row("SELECT created_at_us,updated_at_us,revision,scope_id FROM records WHERE id=?1", [id],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?))).optional()?
            .ok_or_else(|| Error::NotFound(id.to_string()))?),
        None => None,
    };
    if existing.as_ref().is_some_and(|v| v.3 != scope_id) {
        return Err(Error::Conflict("an existing record cannot change scope; copy it to a new ID explicitly".into()));
    }
    if let Some(expected) = input.expected_revision {
        if existing.as_ref().map(|v| v.2) != Some(expected) { return Err(Error::StaleRevision(input.id.map(|v| v.to_string()).unwrap_or_default())); }
    }
    let now = now_us();
    let created = existing.as_ref().map(|v| v.0).unwrap_or(input.created_at_us.unwrap_or(now));
    let updated = input.updated_at_us.unwrap_or_else(|| now.max(existing.as_ref().map(|v| v.1).unwrap_or(created)));
    if updated < created { return Err(Error::Validation("updated_at_us precedes created_at_us".into())); }
    let fingerprint = text::digest(&format!("text-v1\n{embedding_text}"));
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
            conn.execute("DELETE FROM embeddings WHERE record_id=?1 AND fingerprint<>?2", params![id, fingerprint])?;
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
    conn.execute("DELETE FROM record_tags WHERE record_id=?1", [id])?;
    let tags = normalize_tags(&input.tags);
    for tag in &tags {
        let tag_id = term_id(conn, tag)?;
        conn.execute("INSERT OR IGNORE INTO record_tags(record_id,tag_id) VALUES (?1,?2)", params![id, tag_id])?;
    }
    Ok(RecordHeader { id, namespace: input.namespace.clone(), kind, scope: input.scope.clone(),
        created_at_us: created, updated_at_us: updated, revision, tags,
        evidence: input.evidence.clone(), metadata: input.metadata.clone() })
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
    let mut stmt = conn.prepare(&format!("SELECT r.id,n.text,r.kind,s.text,r.created_at_us,r.updated_at_us,r.revision,
        r.metadata_json,r.evidence_json,r.payload_json FROM records r
        JOIN strings n ON n.id=r.namespace_id JOIN strings s ON s.id=r.scope_id WHERE r.id IN ({placeholders}) ORDER BY r.id"))?;
    let mut rows: Vec<(i64, String, i64, String, i64, i64, i64, String, String, String)> = Vec::new();
    for row in stmt.query_map(params_from_iter(params.iter().cloned()), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?,
        r.get::<_, i64>(2)?, r.get::<_, String>(3)?, r.get::<_, i64>(4)?, r.get::<_, i64>(5)?, r.get::<_, i64>(6)?,
        r.get::<_, String>(7)?, r.get::<_, String>(8)?, r.get::<_, String>(9)?)))? {
        rows.push(row?);
    }
    let mut tags_stmt = conn.prepare(&format!("SELECT rt.record_id,t.text FROM record_tags rt JOIN strings t ON t.id=rt.tag_id \
        WHERE rt.record_id IN ({placeholders}) ORDER BY rt.record_id,t.text"))?;
    let mut tags: BTreeMap<i64, Vec<String>> = BTreeMap::new();
    for row in tags_stmt.query_map(params_from_iter(params.iter().cloned()), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
        let (id, tag) = row?;
        tags.entry(id).or_default().push(tag);
    }
    // 笔记只把路径落在 notes 表：读取时按 record_id 补回 source（及其派生 title），
    // payload 里不再重复存路径，正文更不落库。
    let mut note_sources: BTreeMap<i64, String> = BTreeMap::new();
    {
        let mut stmt = conn.prepare(&format!("SELECT n.record_id,s.text FROM notes n JOIN strings s ON s.id=n.source_id \
            WHERE n.record_id IN ({placeholders})"))?;
        for row in stmt.query_map(params_from_iter(params.iter().cloned()), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
            let (id, source) = row?;
            note_sources.insert(id, source);
        }
    }
    for (id, namespace, kind_code, scope, created, updated, revision, metadata, evidence, payload) in rows {
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
            let source = note_sources.remove(&id).unwrap_or_default();
            let title = std::path::Path::new(&source).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            payload.insert("source".into(), Value::String(source));
            payload.insert("title".into(), Value::String(title));
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

/// 批量取回一批记录的可检索正文（按 kind 从 payload 现算）。
/// 正常情况下重排候选正文取自 Tantivy 的 stored 字段；索引不可用时退到这里，仍然拿得到正文。
pub(crate) fn search_texts(conn: &Connection, ids: &[i64]) -> Result<BTreeMap<i64, String>> {
    let mut out = BTreeMap::new();
    if ids.is_empty() { return Ok(out); }
    let placeholders = vec!["?"; ids.len()].join(",");
    let params = ids.iter().map(|id| SqlValue::Integer(*id)).collect::<Vec<_>>();
    let mut stmt = conn.prepare(&format!("SELECT id,kind,payload_json FROM records WHERE id IN ({placeholders})"))?;
    for row in stmt.query_map(params_from_iter(params), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, String>(2)?)))? {
        let (id, kind_code, payload_json) = row?;
        let kind = RecordKind::from_code(kind_code).ok_or_else(|| Error::Validation("invalid stored record kind".into()))?;
        let payload: Value = serde_json::from_str(&payload_json)?;
        out.insert(id, search_text(conn, id, kind, &payload)?);
    }
    Ok(out)
}

/// 过滤条件下的匹配总数（截断之前）。`with_total` 打开时才调用。
pub(crate) fn count_matches(conn: &Connection, filter: &ReadFilter, kinds: &[RecordKind]) -> Result<usize> {
    let (condition, values) = filter_sql(filter, kinds, false)?;
    let count: i64 = conn.query_row(&format!("SELECT COUNT(*) FROM records r WHERE {condition}"), params_from_iter(values), |r| r.get(0))?;
    Ok(count as usize)
}

/// 记录的可检索正文（写入时现算，不落 SQLite）。切片正文由其笔记原文按字符区间取出。
pub(crate) fn search_text(conn: &Connection, id: i64, kind: RecordKind, payload: &Value) -> Result<String> {
    let field = |key: &str| payload.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    Ok(match kind {
        RecordKind::Memory => field("judgment"),
        // 笔记正文不进全文索引，检索面交给切片；笔记记录只留标题（由宿主路径文件名派生）。
        RecordKind::Note => note_title(conn, id)?,
        RecordKind::Chunk => {
            let (title, content) = chunk_origin(conn, payload)?;
            match content {
                Some(content) => format!("{title}\n{}", char_slice(&content, payload)),
                None => title,
            }
        }
        RecordKind::Entity => entity_body(payload),
        RecordKind::Relation => format!("{} {} {} {}", field("subject_name"), field("predicate"), field("object_name"), field("reason")),
        RecordKind::Event => format!("{} {} {} {}", field("name"), field("summary"), name_list(payload), field("reason")),
    })
}

/// 向量化的输入文本。记忆额外拼上标签，笔记带上正文（读自宿主文件），其余与可检索正文一致。
pub(crate) fn embedding_text(conn: &Connection, id: i64, kind: RecordKind, payload: &Value, tags: &[String]) -> Result<String> {
    let field = |key: &str| payload.get(key).and_then(Value::as_str).unwrap_or("");
    Ok(match kind {
        RecordKind::Memory => format!("{}\n{}", field("judgment"), tags.join(" ")),
        RecordKind::Note => {
            let source = note_source(conn, id)?;
            // 派生路径容错：文件缺失时退回只用标题，向量化不因单个文件消失而失败。
            format!("{}\n{}", note_title(conn, id)?, std::fs::read_to_string(&source).unwrap_or_default())
        }
        _ => search_text(conn, id, kind, payload)?,
    })
}

/// 笔记的宿主路径：只落在 `notes` 表（→strings），payload 里不再重复。
fn note_source(conn: &Connection, note_id: i64) -> Result<String> {
    Ok(conn.query_row("SELECT s.text FROM notes n JOIN strings s ON s.id=n.source_id WHERE n.record_id=?1",
        [note_id], |r| r.get::<_, String>(0)).optional()?.unwrap_or_default())
}

/// 笔记标题由宿主路径的文件名派生，不落库。
fn note_title(conn: &Connection, note_id: i64) -> Result<String> {
    Ok(std::path::Path::new(&note_source(conn, note_id)?).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default())
}

fn name_list(payload: &Value) -> String {
    payload.get("participant_names").and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

fn entity_body(payload: &Value) -> String {
    let name = payload.get("name").and_then(Value::as_str).unwrap_or("");
    let aliases = payload.get("aliases").and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" ")).unwrap_or_default();
    let summary = payload.get("summary").and_then(Value::as_str).unwrap_or("");
    let attr_text = payload.get("attributes").and_then(Value::as_object).map(|attrs| {
        attrs.iter().map(|(key, values)| {
            let joined = values.as_array().map(|v| v.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" ")).unwrap_or_default();
            format!("{key} {joined}")
        }).collect::<Vec<_>>().join(" ")
    }).unwrap_or_default();
    format!("{name} {aliases} {summary} {attr_text}")
}

/// 取切片所属笔记的（标题，正文）。切片 payload 里只存 note_id 与字符区间；
/// 正文读时从笔记的宿主文件取，库里不留副本。派生路径（索引/向量/检索降级）对
/// 文件缺失容错：取不到正文时返回 `None`，让该记录退回只索引标题，而不是整个库打不开。
fn chunk_origin(conn: &Connection, payload: &Value) -> Result<(String, Option<String>)> {
    let note_id = payload.get("note_id").and_then(Value::as_i64).ok_or_else(|| Error::Validation("chunk payload is missing note_id".into()))?;
    let source = note_source(conn, note_id)?;
    Ok((note_title(conn, note_id)?, std::fs::read_to_string(&source).ok()))
}

/// 按 payload 里的 `char_start` / `char_end` 从原文取切片正文。
fn char_slice(content: &str, payload: &Value) -> String {
    let start = payload.get("char_start").and_then(Value::as_u64).unwrap_or(0) as usize;
    let end = payload.get("char_end").and_then(Value::as_u64).unwrap_or(0) as usize;
    content.chars().skip(start).take(end.saturating_sub(start)).collect()
}

/// 取某条切片记录的正文（由所属笔记原文按字符区间切出）。
pub(crate) fn chunk_content(conn: &Connection, chunk_id: i64) -> Result<String> {
    let raw: String = conn.query_row("SELECT payload_json FROM records WHERE id=?1", [chunk_id], |r| r.get(0))?;
    let payload: Value = serde_json::from_str(&raw)?;
    let (_, content) = chunk_origin(conn, &payload)?;
    let content = content.ok_or_else(|| Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "note source file is missing or unreadable")))?;
    Ok(char_slice(&content, &payload))
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
    let items = keys.iter().map(|key| get(conn, key, &request.filter)).collect::<Result<Vec<_>>>()?;
    Ok(Page { items, next_cursor })
}

pub(crate) fn delete_record(conn: &Connection, key: &RecordKey) -> Result<bool> {
    // Graph references are RESTRICT, so callers explicitly remove edges first.
    let changed = conn.execute("DELETE FROM records WHERE id=?1", [key.id]);
    let changed = match changed {
        Err(rusqlite::Error::SqliteFailure(err, _)) if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            return Err(Error::Conflict(format!("record {} is still referenced", key.id))),
        other => other?,
    };
    if changed > 0 { next_revision(conn, key.id)?; }
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
}
