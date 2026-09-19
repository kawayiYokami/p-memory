use crate::{storage, text, types::*, Error, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use tantivy::{collector::{Count, DocSetCollector, TopDocs, sort_key::{SortBySimilarityScore, SortByString}}, directory::MmapDirectory, doc, Order,
    query::{BoostQuery, BooleanQuery, ConstScoreQuery, Occur, Query, TermQuery},
    schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value as TantivyValue, INDEXED, STRING, STORED},
    tokenizer::WhitespaceTokenizer, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

const FORMAT: &str = "p-memory-text-v11";

struct Fields { key: Field, namespace: Field, scope: Field, kind: Field, tags: Field, text: Field, name: Field, path: Field, note: Field, body: Field }

/// 名字字段命中时的固定加权：规范名单独成列后，名字整段命中是最强的相关信号，给它固定倍数抬高。
const NAME_FIELD_BOOST: f32 = 3.0;

/// 索引 writer 的内存预算，只作单线程写器的内存天花板。
/// 本重建每批（`REBUILD_BATCH`）都 commit，段在每次提交即刷盘，预算不会被逼近；
/// 取值只需留足余量，不必随语料规模变化。
const WRITER_MEMORY_BUDGET: usize = 1_000_000_000;

/// 重建分页大小：每页处理这么多条记录，页间释放本页的文档与切分缓存，并提交一次。
/// 取值只需「够小以保证内存有界、够大以保证提交不过于频繁」，不承担性能调优职责。
const REBUILD_BATCH: usize = 2_000;

/// 重建进行中写在索引 payload 上的标记；与稳态的 `FORMAT:<indexed_revision>` 区分，
/// 让 `recover` 能认出「上次重建没跑完，该从游标续跑」。
fn rebuild_marker() -> String { format!("{FORMAT}:rebuild") }

/// 索引重建进度。只有重建线程写，其余线程读，因此用原子量而非锁。
pub(crate) struct RebuildProgress {
    active: AtomicBool,
    processed: AtomicU64,
    total: AtomicU64,
}

impl RebuildProgress {
    fn new() -> Self { Self { active: AtomicBool::new(false), processed: AtomicU64::new(0), total: AtomicU64::new(0) } }
}

/// 一条要写进索引的记录：正文、标签全部由写入流程就地提供。
/// 写入时读一次源文件、切一次，切片正文一路带到这里，索引阶段不再回头读文件。
pub(crate) struct IndexDocument {
    pub id: i64,
    pub namespace_id: i64,
    pub scope_id: i64,
    pub kind: RecordKind,
    /// 正文列：这条记录自己的文本。记忆是 judgment，切片是它那一段。笔记不进索引。
    pub text: String,
    /// 实体规范名：单独一列，检索时按 `NAME_FIELD_BOOST` 加权。其它记录为空串、不写这一列。
    pub name: String,
    /// 笔记所在目录：相对路径里除文件名之外的各段，空格连接。只有切片记录写这一列，
    /// 常规检索不查它——它是「书名块不够时」才动用的兜底。
    pub path: String,
    /// 切片所属笔记的记录 id（只挂切片，其它记录为 0）：用来按笔记精确统计「还有多少片段命中」。
    pub note_id: i64,
    /// 拼在可搜正文前面的标签集（空格连接，空串表示不带）。索引里没有单独的标签文本列：
    /// 标签只有承载它的那一条带——切片是第一片，其余记录是它自己；取值规则见 `storage::tags_prefix`。
    pub tags_prefix: String,
    /// 标签 id 列：只用来按标签过滤，不参与打分。
    pub tag_ids: Vec<i64>,
}

/// 已在库侧折算成 id 的检索范围。索引里的 namespace / scope / kind / tags 一律是整数，
/// 标记文本换成 id 这件事在进索引之前做完。
pub(crate) struct IndexFilter { pub namespace: i64, pub scopes: Vec<i64>, pub kinds: Vec<i64>, pub tags: Vec<i64> }

/// 文本索引。`IndexReader` 可并发检索，`IndexWriter` 收进内部互斥锁：
/// 整个结构可以直接共享给多个读线程，提交只在写者之间串行。
pub(crate) struct TextIndex {
    index: Index, reader: IndexReader, writer: Mutex<IndexWriter>, fields: Fields,
    /// 测试专用：注入查询故障，验证「索引查询失败即重建并重试」的恢复路径。
    #[cfg(test)] pub(crate) fail_search: AtomicBool,
    /// 测试专用：统计重建次数。
    #[cfg(test)] pub(crate) rebuilds: AtomicUsize,
    /// 测试专用：>0 时表示本次重建最多处理这么多页就注入中断，用来验证断点续存。
    #[cfg(test)] pub(crate) abort_rebuild_after: AtomicUsize,
    /// 重建进度快照，供其它线程轮询。
    progress: RebuildProgress,
}

impl TextIndex {
    pub fn open(root: &Path) -> Result<Self> {
        let directory = root.join("text-v2");
        std::fs::create_dir_all(&directory)?;
        let mut builder = Schema::builder();
        // 分词字段共用一套预分词 + 空格分词器：写入侧切成词元后按空格连接，
        // 查询侧用同一套规则产出词元，两边严格同源。
        let tokenized = || TextOptions::default().set_indexing_options(TextFieldIndexing::default()
            .set_tokenizer("pretokenized").set_index_option(IndexRecordOption::WithFreqsAndPositions));
        let fields = Fields {
            key: builder.add_text_field("key", (STRING | STORED).set_fast(None)),
            // 标记列一律存 strings 表的整数 id：索引里不留第二份标记文本。
            namespace: builder.add_u64_field("namespace", INDEXED),
            scope: builder.add_u64_field("scope", INDEXED),
            kind: builder.add_u64_field("kind", INDEXED),
            // 标签 id 列（多值）：按标签过滤只在这一列上做精确匹配，不参与打分。
            tags: builder.add_u64_field("tags", INDEXED),
            // 正文列：唯一承载「这条记录讲了什么」的列，也是相关性打分的主力。
            // 标签集拼在承载它的那一条的正文前面，与正文同列、同长度、共同参与打分。
            // 实体规范名不进这一列。
            text: builder.add_text_field("text", tokenized()),
            // 实体规范名单独一列：正文列不含规范名，名字只活在这一列，检索时按固定倍数加权，
            // 并作为独立的命中条件——名字命中也算这条记录被搜到。笔记的文件名也落在这一列。
            name: builder.add_text_field("name", tokenized()),
            // 笔记所在目录单独一列：目录段不进正文列（否则搜「璃月」会命中该目录下每一篇），
            // 只在「书名块不够」的兜底查询里被查。
            path: builder.add_text_field("path", tokenized()),
            // 切片所属笔记的记录 id：只挂在切片上，用来数「这一篇里有多少切片命中」。
            // 这条计数与结果窗口无关，所以只能在索引里按笔记精确统计，不能靠截断后的结果集去数。
            note: builder.add_u64_field("note", INDEXED),
            // 正文原值随命中取回：库里不留正文副本，取正文只走这一列。
            body: builder.add_text_field("body", STORED),
        };
        let schema = builder.build();
        let open = || -> Result<Index> {
            let dir = MmapDirectory::open(&directory).map_err(|e| Error::Index(e.to_string()))?;
            Ok(Index::open_or_create(dir, schema.clone())?)
        };
        let index = match open() {
            Ok(index) => index,
            Err(_) => {
                // Quarantine only this derived index, never the authoritative database.
                std::fs::rename(&directory, root.join(format!("text-v2.corrupt-{}", uuid::Uuid::new_v4())))?;
                std::fs::create_dir(&directory)?;
                open()?
            }
        };
        index.tokenizers().register("pretokenized", WhitespaceTokenizer::default());
        // 索引 writer 固定单线程：本机实测（8 核 16 线程，20 万条）1 线程最快 9.3s，
        // 2/4/8/16 线程依次退化到 11.0/14.1/19.1/32.8s——多 worker 只增加协调与合并开销，
        // 所以不交给 tantivy 按核数拆线程。
        let writer = index.writer_with_num_threads(1, WRITER_MEMORY_BUDGET)?;
        let reader = index.reader_builder().reload_policy(ReloadPolicy::Manual).try_into()?;
        Ok(Self { index, reader, writer: Mutex::new(writer), fields,
            #[cfg(test)] fail_search: AtomicBool::new(false),
            #[cfg(test)] rebuilds: AtomicUsize::new(0),
            #[cfg(test)] abort_rebuild_after: AtomicUsize::new(0),
            progress: RebuildProgress::new() })
    }

    pub fn document_count(&self) -> usize { self.reader.searcher().num_docs() as usize }

    /// 索引自检：格式过期、或上一次写入没走完提交（待办队列还压着东西）时重建。
    /// 重建是唯一允许重新读源文件的路径——正文没有第二份副本，异常恢复只能回源。
    pub fn recover(&self, conn: &Connection) -> Result<()> {
        let expected = format!("{FORMAT}:{}", storage::meta(conn, "indexed_revision")?);
        let pending: i64 = conn.query_row("SELECT COUNT(*) FROM index_updates", [], |r| r.get(0))?;
        if pending == 0 && self.index.load_metas()?.payload.as_deref() == Some(&expected) {
            // 稳态：顺手清掉可能残留的重建状态（正常收尾已清，这里是崩溃在收尾前的兜底）。
            if storage::meta_opt(conn, "rebuild_cursor")?.is_some() { storage::clear_meta(conn, "rebuild_cursor")?; }
            if storage::meta_opt(conn, "rebuild_processed")?.is_some() { storage::clear_meta(conn, "rebuild_processed")?; }
            return Ok(());
        }
        self.rebuild(conn)
    }

    /// 把写入流程就地交过来的文档写进索引：按记录 ID 覆盖，尚未提交所以对搜索不可见。
    /// commit 由 `update_index`（或关闭时的收尾）一次做完。
    pub fn stage(&self, docs: &[IndexDocument]) -> Result<()> {
        if docs.is_empty() { return Ok(()); }
        let mut writer = self.writer.lock();
        for item in docs { self.add(&mut writer, item)?; }
        Ok(())
    }

    fn add(&self, writer: &mut IndexWriter, item: &IndexDocument) -> Result<()> {
        let encoded = item.id.to_string();
        writer.delete_term(Term::from_field_text(self.fields.key, &encoded));
        let mut document = doc!(self.fields.key => encoded, self.fields.body => item.text.clone());
        document.add_u64(self.fields.namespace, item.namespace_id as u64);
        document.add_u64(self.fields.scope, item.scope_id as u64);
        document.add_u64(self.fields.kind, item.kind.code() as u64);
        for tag_id in &item.tag_ids { document.add_u64(self.fields.tags, *tag_id as u64); }
        // 标签集拼在可搜正文前面，进的是同一条正文列：它跟着这一条的文档长度一起被归一化，
        // 命中标签的分会被这一条的正文稀释。`body` 仍是纯正文——取回给人看的、送进模型的都不带标签。
        let searchable = if item.tags_prefix.is_empty() { item.text.clone() }
            else { format!("{}\n{}", item.tags_prefix, item.text) };
        let cleaned = text::clean_markdown(&searchable);
        if !cleaned.is_empty() { document.add_text(self.fields.text, text::tokenize(&cleaned).join(" ")); }
        if !item.name.is_empty() { document.add_text(self.fields.name, text::tokenize(&item.name).join(" ")); }
        if !item.path.is_empty() { document.add_text(self.fields.path, text::tokenize(&item.path).join(" ")); }
        if item.note_id != 0 { document.add_u64(self.fields.note, item.note_id as u64); }
        writer.add_document(document)?;
        Ok(())
    }

    /// 提交并记账。`revision` 是本次**实际覆盖到**的最大 revision：
    /// 并发写入时可能有更新的 revision 在读取之后才提交，记账必须只认自己真索引过的那一段。
    fn finish(&self, writer: &mut IndexWriter, conn: &Connection, revision: i64) -> Result<()> {
        let mut prepared = writer.prepare_commit()?;
        prepared.set_payload(&format!("{FORMAT}:{revision}"));
        prepared.commit()?;
        self.reader.reload()?;
        // A crash before this acknowledgement simply replays idempotent replacements.
        // 记账只增不减：并发下后到的旧提交不允许把进度回退。
        conn.execute("UPDATE meta SET value=?1 WHERE key='indexed_revision' AND value<?1", [revision])?;
        conn.execute("DELETE FROM index_updates WHERE revision<=?1", [revision])?;
        Ok(())
    }

    /// 提交本进程已经写进 writer 的文档。写入路径不再回库重建文档，所以这里只提交与记账；
    /// 队列为空是稳态（无待办时连写锁都不碰），非空说明有尚未提交的写入。
    pub fn sync(&self, conn: &Connection) -> Result<()> {
        let pending: i64 = conn.query_row("SELECT COUNT(*) FROM index_updates", [], |r| r.get(0))?;
        if pending == 0 { return Ok(()); }
        let covered: i64 = conn.query_row("SELECT COALESCE(MAX(revision),0) FROM index_updates", [], |r| r.get(0))?;
        let mut writer = self.writer.lock();
        self.finish(&mut writer, conn, covered)
    }

    /// 全量重建：清空索引，按库里现有记录重新读源文件、重跑同一套切分。
    /// 只在索引格式过期、索引损坏、或上次写入未收尾时走这里，属于异常恢复而非日常路径。
    ///
    /// 重建按 record id 分页流式进行：每页写完就提交、并把「已处理到的 id」记进 `meta`。
    /// 内存只驻留单页文档；进程中途被杀，重开时 `recover` 能从游标续跑而非从零重来。
    pub fn rebuild(&self, conn: &Connection) -> Result<()> {
        #[cfg(test)]
        self.rebuilds.fetch_add(1, Ordering::SeqCst);

        // 磁盘 payload 是「重建中」标记 → 上次没跑完，从游标接着来；否则从头清空重来。
        let resumable = self.index.load_metas()?.payload.as_deref() == Some(&rebuild_marker());
        let (start_cursor, already) = if resumable {
            // 已处理量也从 meta 读，不依赖 reader 的新鲜度：同进程重试时 reader 还停在旧快照上。
            (storage::meta_opt(conn, "rebuild_cursor")?.unwrap_or(0),
             storage::meta_opt(conn, "rebuild_processed")?.unwrap_or(0).max(0) as u64)
        } else { (0, 0) };
        // 进度里的总量只算真正进索引的记录：笔记不占索引文档。
        let total = conn.query_row("SELECT COUNT(*) FROM records WHERE kind<>?1", [RecordKind::Note.code()],
            |r| r.get::<_, i64>(0))? as u64;

        self.progress.total.store(total, Ordering::SeqCst);
        self.progress.processed.store(already, Ordering::SeqCst);
        self.progress.active.store(true, Ordering::SeqCst);
        let result = self.rebuild_pages(conn, resumable, start_cursor, already);
        // 无论成功失败都退出「进行中」态，宿主不会读到永远 active 的幽灵进度。
        self.progress.active.store(false, Ordering::SeqCst);
        result
    }

    /// 重建的分页主循环：逐页读、逐页写、逐页提交并推进持久游标。
    fn rebuild_pages(&self, conn: &Connection, resumable: bool, start_cursor: i64, mut processed: u64) -> Result<()> {
        let mut writer = self.writer.lock();
        if !resumable {
            // 从头来：旧索引整体作废，写下重建标记，游标与已处理量都归零。
            writer.delete_all_documents()?;
            storage::set_meta(conn, "rebuild_cursor", 0)?;
            storage::set_meta(conn, "rebuild_processed", 0)?;
            let mut prepared = writer.prepare_commit()?;
            prepared.set_payload(&rebuild_marker());
            prepared.commit()?;
        }
        let mut cursor = start_cursor;
        #[cfg(test)]
        let mut pages = 0usize;
        loop {
            let (documents, last_scanned) = self.rebuild_page(conn, cursor)?;
            // 本页没有推进（后面已无记录）即结束。
            if last_scanned <= cursor { break; }
            for item in &documents { self.add(&mut writer, item)?; }
            cursor = last_scanned;
            // 先提交、再写游标：崩溃只可能让游标落后于已落盘数据，重放是幂等覆盖。
            let mut prepared = writer.prepare_commit()?;
            prepared.set_payload(&rebuild_marker());
            prepared.commit()?;
            storage::set_meta(conn, "rebuild_cursor", cursor)?;
            processed += documents.len() as u64;
            storage::set_meta(conn, "rebuild_processed", processed as i64)?;
            self.progress.processed.store(processed, Ordering::SeqCst);
            #[cfg(test)]
            {
                pages += 1;
                let cap = self.abort_rebuild_after.load(Ordering::SeqCst);
                if cap != 0 && pages >= cap { return Err(Error::Index("injected rebuild abort".into())); }
            }
        }
        // 收尾：payload 落到稳态、记账、清掉重建游标与已处理量。
        self.finish(&mut writer, conn, storage::current_revision(conn)?)?;
        storage::clear_meta(conn, "rebuild_cursor")?;
        storage::clear_meta(conn, "rebuild_processed")?;
        Ok(())
    }

    /// 读一页记录折成索引文档。返回本页文档与本页扫到的最大 record id。
    /// `files`（笔记切分缓存）只在本页内存活，随函数返回释放，不再整程驻留。
    fn rebuild_page(&self, conn: &Connection, after: i64) -> Result<(Vec<IndexDocument>, i64)> {
        let mut documents = Vec::new();
        let mut files: HashMap<i64, Vec<String>> = HashMap::new();
        let mut paths: HashMap<i64, (Vec<String>, String)> = HashMap::new();
        let mut last = after;
        let mut stmt = conn.prepare("SELECT r.id,r.namespace_id,r.kind,r.scope_id,r.payload_json FROM records r WHERE r.id>?1 ORDER BY r.id LIMIT ?2")?;
        let rows = stmt.query_map(params![after, REBUILD_BATCH as i64], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?,
            r.get::<_, i64>(2)?, r.get::<_, i64>(3)?, r.get::<_, String>(4)?)))?;
        for row in rows {
            let (id, namespace_id, kind_code, scope_id, payload_json) = row?;
            last = id;
            let Some(kind) = RecordKind::from_code(kind_code) else { continue };
            // 笔记不占索引文档：路径信息以标签形态挂在它的每个切片上，文件列表按库里的标签翻。
            if kind == RecordKind::Note { continue; }
            let Ok(payload) = serde_json::from_str::<serde_json::Value>(&payload_json) else { continue };
            let text = match kind {
                RecordKind::Chunk => {
                    let note_id = payload.get("note_id").and_then(|v| v.as_i64()).unwrap_or(0);
                    let ordinal = payload.get("ordinal").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    if !files.contains_key(&note_id) { files.insert(note_id, self.note_file_chunks(conn, note_id)); }
                    files.get(&note_id).and_then(|chunks| chunks.get(ordinal)).cloned().unwrap_or_default()
                }
                _ => storage::record_text(kind, &payload),
            };
            let pairs = storage::record_tag_pairs(conn, id).unwrap_or_default();
            let tags: Vec<String> = pairs.iter().map(|(_, tag)| tag.clone()).collect();
            // 切片的名字列放文件名、目录列放所在目录，可搜前缀里要把这两样摘掉：
            // 它们已经从「标签」升格（或降格）成独立列，不能再当成正文的一部分被搜到。
            // 名字与目录只挂在这一篇的第一片上——否则每一片都会命中同一查询，把结果刷屏。
            let (name, path, exclude) = match kind {
                RecordKind::Chunk if payload.get("ordinal").and_then(|v| v.as_u64()).unwrap_or(0) == 0 => {
                    let note_id = payload.get("note_id").and_then(|v| v.as_i64()).unwrap_or(0);
                    let (dirs, stem) = paths.entry(note_id).or_insert_with(|| storage::note_path_parts(conn, note_id)).clone();
                    let mut exclude = dirs.clone();
                    if !stem.is_empty() { exclude.push(stem.clone()); }
                    (stem, dirs.join(" "), exclude)
                }
                RecordKind::Chunk => (String::new(), String::new(), Vec::new()),
                _ => (storage::record_name(kind, &payload), String::new(), Vec::new()),
            };
            documents.push(IndexDocument { id, namespace_id, scope_id, kind, text, name, path,
                note_id: if kind == RecordKind::Chunk { payload.get("note_id").and_then(|v| v.as_i64()).unwrap_or(0) } else { 0 },
                tags_prefix: storage::tags_prefix(kind, &tags, &exclude, &payload),
                tag_ids: pairs.into_iter().map(|(tag_id, _)| tag_id).collect() });
        }
        Ok((documents, last))
    }

    /// 读一次重建进度快照。纯原子读，不碰写锁，可在重建进行时从另一线程安全调用。
    pub(crate) fn rebuild_progress(&self) -> RebuildProgressReport {
        RebuildProgressReport {
            active: self.progress.active.load(Ordering::SeqCst),
            processed: self.progress.processed.load(Ordering::SeqCst),
            total: self.progress.total.load(Ordering::SeqCst),
        }
    }

    /// 读一篇笔记的源文件并重跑切分，返回每片正文（按 ordinal 顺序）。
    fn note_file_chunks(&self, conn: &Connection, note_id: i64) -> Vec<String> {
        let Ok((path, namespace_id)) = conn.query_row("SELECT n.path,n.namespace_id FROM notes n WHERE n.record_id=?1",
            [note_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))) else { return Vec::new() };
        let Ok(content) = std::fs::read_to_string(storage::absolute_note_path(conn, namespace_id, &path)) else { return Vec::new() };
        let chunk_chars = conn.query_row("SELECT payload_json FROM records WHERE id=?1", [note_id], |r| r.get::<_, String>(0))
            .ok().and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|value| value.get("chunk_chars").and_then(|v| v.as_u64()))
            .unwrap_or(220) as usize;
        crate::notes::chunk_text(&content, chunk_chars).map(|chunks| chunks.into_iter().map(|chunk| chunk.content).collect()).unwrap_or_default()
    }

    fn exact_text(field: Field, value: &str) -> Box<dyn Query> {
        Box::new(TermQuery::new(Term::from_field_text(field, value), IndexRecordOption::Basic))
    }

    fn exact_u64(field: Field, value: i64) -> Box<dyn Query> {
        Box::new(TermQuery::new(Term::from_field_u64(field, value as u64), IndexRecordOption::Basic))
    }

    fn term(field: Field, value: &str, occurrence: Occur) -> (Occur, Box<dyn Query>) {
        (occurrence, Box::new(TermQuery::new(Term::from_field_text(field, value), IndexRecordOption::WithFreqs)) as Box<dyn Query>)
    }

    fn filtered_query(&self, tokens: &[String], strict: bool, filter: &IndexFilter, field: MatchField) -> Box<dyn Query> {
        let occurrence = if strict { Occur::Must } else { Occur::Should };
        // 命中层：每个词元在指定列里命中即算这个词元命中。
        // `All` 看「正文列 + 名字列」——规范名已从正文列移出，所以名字命中也必须能独立把这条
        // 记录带进结果，不再只是加分项。目录列只在 `Path` 兜底查询里被查。
        let hit = BooleanQuery::new(tokens.iter().map(|token| {
            let column: Box<dyn Query> = match field {
                MatchField::All => Box::new(BooleanQuery::new(vec![
                    Self::term(self.fields.text, token, Occur::Should),
                    Self::term(self.fields.name, token, Occur::Should),
                ])),
                MatchField::Text => Box::new(TermQuery::new(Term::from_field_text(self.fields.text, token), IndexRecordOption::WithFreqs)),
                MatchField::Name => Box::new(TermQuery::new(Term::from_field_text(self.fields.name, token), IndexRecordOption::WithFreqs)),
                MatchField::Path => Box::new(TermQuery::new(Term::from_field_text(self.fields.path, token), IndexRecordOption::WithFreqs)),
            };
            (occurrence, column)
        }).collect());
        // 名字层只负责抬分：名字整段等于查询词的记录，不该被它那几百字的别名与属性摊薄；
        // 其它记录的 name 列为空，天然不参与。单列查询已经只认那一列，不再叠加权。
        let relevance: Box<dyn Query> = if field == MatchField::All {
            let name_relevance = BooleanQuery::new(tokens.iter().map(|token| Self::term(self.fields.name, token, Occur::Should)).collect());
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, Box::new(hit) as Box<dyn Query>),
                (Occur::Should, Box::new(BoostQuery::new(Box::new(name_relevance), NAME_FIELD_BOOST)) as Box<dyn Query>),
            ]))
        } else {
            Box::new(hit)
        };
        // 相关性这层只放 Should，所以至少要命中一个词元才算相关，过滤维度另起一层放 Must。
        // 两层不能合并：同一层里一旦存在 Must，全部 Should 都会降级为「有则加分、无也无妨」，
        // 正文条件就形同虚设，检索退化成「只按过滤条件取记录」——
        // 查一个正文里根本不存在的词，也会返回该过滤域下的任意记录。
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = vec![(Occur::Must, Box::new(relevance))];
        let scopes = BooleanQuery::new(filter.scopes.iter().map(|scope| (Occur::Should, Self::exact_u64(self.fields.scope, *scope))).collect());
        clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Self::exact_u64(self.fields.namespace, filter.namespace), 0.0))));
        clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Box::new(scopes), 0.0))));
        if !filter.kinds.is_empty() {
            let kinds = BooleanQuery::new(filter.kinds.iter().map(|kind| (Occur::Should, Self::exact_u64(self.fields.kind, *kind))).collect());
            clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Box::new(kinds), 0.0))));
        }
        for tag_id in &filter.tags {
            clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Self::exact_u64(self.fields.tags, *tag_id), 0.0))));
        }
        Box::new(BooleanQuery::new(clauses))
    }

    /// 在指定列上检索。`field` 决定命中限定在哪一列：`All` 是正文+名字（默认），
    /// `Text` / `Name` / `Path` 各自单独成路，供预设把「书名」「内容」「目录兜底」分开取。
    pub fn search_in(&self, query: &str, filter: &IndexFilter, limit: usize, field: MatchField) -> Result<Vec<(RecordKey, f64)>> {
        #[cfg(test)]
        if self.fail_search.load(Ordering::SeqCst) { return Err(Error::Index("injected index failure".into())); }
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        let searcher = self.reader.searcher();
        let mut strict_rank = HashMap::new();
        for strict in [true, false] {
            if !strict && result.len() >= limit { break; }
            let tokens = text::query_terms(query, strict);
            if tokens.is_empty() { continue; }
            let query = self.filtered_query(&tokens, strict, filter, field);
            let collector = TopDocs::with_limit(limit).order_by(((SortBySimilarityScore, Order::Desc), (SortByString::for_field("key"), Order::Asc)));
            let hits = searcher.search(&*query, &collector)?;
            for ((score, _), address) in hits {
                let document: tantivy::TantivyDocument = searcher.doc(address)?;
                let encoded = document.get_first(self.fields.key).and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Index("missing document key".into()))?;
                if !seen.insert(encoded.to_string()) { continue; }
                let id: i64 = encoded.parse().map_err(|_| Error::Index("invalid document key".into()))?;
                let key = RecordKey { id };
                strict_rank.insert(key, strict);
                result.push((key, score as f64));
            }
        }
        result.sort_by(|a, b| strict_rank[&b.0].cmp(&strict_rank[&a.0]).then_with(|| b.1.total_cmp(&a.1)).then_with(|| a.0.cmp(&b.0)));
        result.truncate(limit);
        Ok(result)
    }

    /// 数一数「某一篇笔记里，有多少条切片命中这个查询」。
    /// 与 `search_in` 同源：同一套词元、同一套过滤条件，另加一条「属于该笔记」的约束。
    /// 用来告诉调用方「这篇文档还有多少片段相关」——它是查询本身的属性，与结果窗口、翻页无关，
    /// 所以必须在这里精确统计，不能拿截断后的结果集去数。
    /// 宽松词元集是严格词元集的超集，两者的命中集因此是包含关系，只数宽松那一遍就是并集。
    pub fn count_in(&self, query: &str, filter: &IndexFilter, field: MatchField, note_id: i64) -> Result<usize> {
        let tokens = text::query_terms(query, false);
        if tokens.is_empty() { return Ok(0); }
        let base = self.filtered_query(&tokens, false, filter, field);
        let with_note = BooleanQuery::new(vec![
            (Occur::Must, base),
            (Occur::Must, Self::exact_u64(self.fields.note, note_id)),
        ]);
        Ok(self.reader.searcher().search(&with_note, &Count)?)
    }

    /// 按 id 取回索引里存储的正文：重排候选取正文、以及「切片正文从索引读」都走这里，不再回源文件。
    pub fn bodies(&self, ids: &[i64]) -> Result<BTreeMap<i64, String>> {
        let mut out = BTreeMap::new();
        if ids.is_empty() { return Ok(out); }
        let searcher = self.reader.searcher();
        for id in ids {
            let query = Self::exact_text(self.fields.key, &id.to_string());
            if let Some(address) = searcher.search(&query, &DocSetCollector)?.into_iter().next() {
                let document: tantivy::TantivyDocument = searcher.doc(address)?;
                if let Some(body) = document.get_first(self.fields.body).and_then(|v| v.as_str()) {
                    out.insert(*id, body.to_string());
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KnowledgeBase, MemoryInput, SearchRequest};

    /// 造一个比两页还多的库，确保重建会跨多个分页边界。
    fn seed(kb: &KnowledgeBase, n: usize) {
        for i in 0..n { kb.memories().upsert(MemoryInput::new(&format!("重建分页测试条目{i}"))).unwrap(); }
        kb.update_index().unwrap();
    }

    /// 读当前落盘的重建游标（不存在返回 None）。
    fn cursor_of(kb: &KnowledgeBase) -> Option<i64> {
        let guard = kb.engine.writer.lock();
        storage::meta_opt(&guard.as_ref().unwrap().conn, "rebuild_cursor").unwrap()
    }

    /// 断点续存：中断后从持久游标接着跑，而不是从零重来。
    #[test]
    fn rebuild_resumes_from_persisted_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let n = REBUILD_BATCH * 2 + 37;
        seed(&kb, n);
        let total = kb.health().unwrap().index_document_count;
        assert_eq!(total, n);

        // 让第一页提交后立刻中断，模拟进程被杀。
        let index = kb.index().unwrap();
        index.abort_rebuild_after.store(1, Ordering::SeqCst);
        assert!(kb.rebuild_indexes().is_err(), "注入的中断必须冒泡成错误");

        // 中断后的落盘状态：游标停在第一页末尾、已处理量记下已提交页、退出进行中态。
        assert_eq!(cursor_of(&kb), Some(REBUILD_BATCH as i64), "中断后必须留下停在该页末尾的持久游标");
        let progress = kb.rebuild_progress().unwrap();
        assert!(!progress.active, "中断后不应仍处于进行中态");
        assert_eq!(progress.total as usize, n);
        assert_eq!(progress.processed as usize, REBUILD_BATCH);

        // 再续跑一页后中断：游标应从 4000 而不是从 2000 重来，这才能证明「续跑而非重来」。
        assert!(kb.rebuild_indexes().is_err());
        assert_eq!(cursor_of(&kb), Some((REBUILD_BATCH * 2) as i64), "续跑必须从持久游标继续推进");
        assert_eq!(kb.rebuild_progress().unwrap().processed as usize, REBUILD_BATCH * 2);

        // 关掉钩子跑完：补齐到完整并清掉重建状态。
        index.abort_rebuild_after.store(0, Ordering::SeqCst);
        kb.rebuild_indexes().unwrap();
        assert_eq!(kb.health().unwrap().index_document_count, total);
        assert_eq!(cursor_of(&kb), None, "收尾后应清掉重建游标");
        let done = kb.rebuild_progress().unwrap();
        assert!(!done.active);
        assert_eq!(done.processed, done.total);

        // 跨进程：重开库走 recover，稳态下不再重建、索引完整。
        drop(index);
        drop(kb);
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        assert_eq!(kb.health().unwrap().index_document_count, total);
    }

    /// 跨进程续跑：中断后重开库，recover 应自动从游标补齐，无需显式重建。
    #[test]
    fn open_resumes_interrupted_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let n = REBUILD_BATCH + 11;
        seed(&kb, n);
        let total = kb.health().unwrap().index_document_count;

        let index = kb.index().unwrap();
        index.abort_rebuild_after.store(1, Ordering::SeqCst);
        assert!(kb.rebuild_indexes().is_err());
        assert!(cursor_of(&kb).is_some());
        drop(index);
        drop(kb);

        // 重开：open 内的 recover 看到「重建中」标记即从游标续跑。
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        assert_eq!(kb.health().unwrap().index_document_count, total, "recover 应自动补齐中断的重建");
        assert_eq!(cursor_of(&kb), None);
    }

    /// 跨多页流式重建后，记录不能丢也不能查不到：按名字建一批跨页记录，
    /// 重建后逐条检索与按 id 取正文都要能命中。
    #[test]
    fn streamed_rebuild_keeps_every_record_searchable() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let n = REBUILD_BATCH * 2 + 17;
        seed(&kb, n);
        kb.rebuild_indexes().unwrap();

        let index = kb.index().unwrap();
        // 逐条确认「分页边界两侧」的记录都还在索引里且可检索。
        for i in [0usize, REBUILD_BATCH - 1, REBUILD_BATCH, REBUILD_BATCH * 2, n - 1] {
            let query = format!("重建分页测试条目{i}");
            let request = SearchRequest { query: query.clone(), kinds: vec![RecordKind::Memory],
                vector: false, rerank: false, ..Default::default() };
            let result = kb.search(&request).unwrap();
            assert!(!result.hits.is_empty(), "重建后第 {i} 条应仍可检索到");
            let hit = &result.hits[0];
            let body = index.bodies(&[hit.key.id]).unwrap();
            assert!(body.get(&hit.key.id).map(|t| t.contains(&query)).unwrap_or(false),
                "重建后第 {i} 条的正文应能按 id 取回");
        }
    }

    /// 进度上报：完成后 processed 追平 total，且不再处于进行中态。
    #[test]
    fn rebuild_progress_reports_completion() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let n = REBUILD_BATCH + 5;
        seed(&kb, n);
        assert!(!kb.rebuild_progress().unwrap().active, "尚未重建时不应处于进行中态");
        kb.rebuild_indexes().unwrap();
        let p = kb.rebuild_progress().unwrap();
        assert!(!p.active);
        assert_eq!(p.total, n as u64);
        assert_eq!(p.processed, n as u64, "完成后 processed 应等于总量");
    }
}
