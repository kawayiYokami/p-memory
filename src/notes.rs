use crate::{storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn default_chunk_chars() -> usize { 220 }

/// 按文件路径同步一篇笔记的入参：库自己读文件，标题取文件名（去扩展名）。
/// 路径存为笔记自己的一列：领域登记过根目录时存减掉根目录的相对路径，否则逐字符原样存。
/// 它既用来读回正文，也用来定位同一文件。
/// 正文真相源是文件：清洗只作用于送进索引的文本，库内不留任何正文副本。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteFileInput {
    #[serde(flatten)] pub record: RecordInput,
    /// 要读取的文件路径，同时作为 `(namespace, scope, path)` 的定位键。
    pub path: PathBuf,
    #[serde(default = "default_chunk_chars")] pub chunk_chars: usize,
}
impl NoteFileInput {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { record: RecordInput::default(), path: path.into(), chunk_chars: default_chunk_chars() }
    }
}

/// 笔记的对外形态：库内只留路径，正文读时按 `source` 读文件，标题由文件名派生。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    #[serde(flatten)] pub header: RecordHeader,
    /// 文件路径：写入时给进来的那条；库内一行存的是相对根目录的形态，读回时拼成绝对路径。
    pub source: String,
    /// 由 `source` 的文件名派生，不落库。
    #[serde(default)] pub title: String,
    pub chunk_chars: usize,
}
/// 切片不再重复携带 source/title；路径与标题经 `note_id` 关联 notes 取回。
/// 切片正文只在索引里存一份：写入时切好、就地写进索引，`content` 读取时从索引取回。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    #[serde(flatten)] pub header: RecordHeader,
    pub note_id: i64, pub ordinal: usize, pub offset: usize, pub limit: usize,
    #[serde(default)] pub content: String,
}
/// `offset` 为 1 起始的起始行，`limit` 为行数。切片正文由 `content` 自带，
/// 超长段落被多个切片共享同一行号也不影响取回。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextChunk { pub ordinal: usize, pub offset: usize, pub limit: usize, pub content: String }

/// Paragraph-aware chunking. Fenced code and tables are atomic, even above the
/// target size. A long prose line may span chunks sharing that same line number.
pub fn chunk_text(content: &str, target: usize) -> Result<Vec<TextChunk>> {
    if !(16..=100_000).contains(&target) { return Err(Error::Validation("chunk_chars must be between 16 and 100000".into())); }
    let lines: Vec<&str> = content.lines().collect();
    let mut blocks: Vec<(usize, usize, String, bool)> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim().is_empty() { i += 1; continue; }
        let start = i;
        let trimmed = lines[i].trim_start();
        let fence_char = trimmed.chars().next().filter(|c| *c == '`' || *c == '~');
        let fence_len = fence_char.map(|c| trimmed.chars().take_while(|x| *x == c).count()).unwrap_or(0);
        let is_fence = fence_len >= 3;
        let is_table = lines[i].contains('|') && i + 1 < lines.len() && {
            let next = lines[i + 1].trim();
            next.contains('-') && next.contains('|') && next.chars().all(|c| matches!(c, '-' | ':' | '|' | ' ' | '\t'))
        };
        i += 1;
        if is_fence {
            while i < lines.len() {
                let line = lines[i].trim();
                i += 1;
                if line.chars().take_while(|c| Some(*c) == fence_char).count() >= fence_len
                    && line.chars().all(|c| Some(c) == fence_char || c.is_whitespace()) { break; }
            }
        } else if is_table {
            while i < lines.len() && lines[i].contains('|') && !lines[i].trim().is_empty() { i += 1; }
        } else {
            while i < lines.len() && !lines[i].trim().is_empty() {
                let line = lines[i].trim_start();
                if line.starts_with("```") || line.starts_with("~~~") || line.starts_with('#') { break; }
                if i + 1 < lines.len() && lines[i].contains('|') && lines[i + 1].contains("---") { break; }
                i += 1;
            }
        }
        blocks.push((start, i, lines[start..i].join("\n"), is_fence || is_table));
    }
    let mut chunks = Vec::new();
    for (start, end, body, atomic) in blocks {
        if atomic || body.chars().count() <= target {
            chunks.push(TextChunk { ordinal: 0, offset: start + 1, limit: end - start, content: body });
            continue;
        }
        let mut current = String::new();
        let mut count = 0;
        let mut first_line = start + 1;
        let mut last_line = first_line;
        for (line_no, line) in lines.iter().enumerate().take(end).skip(start) {
            if !current.is_empty() {
                if count + 1 >= target {
                    chunks.push(TextChunk { ordinal: 0, offset: first_line, limit: last_line - first_line + 1, content: std::mem::take(&mut current) });
                    count = 0;
                } else { current.push('\n'); count += 1; }
            }
            for ch in line.chars() {
                if count == target {
                    chunks.push(TextChunk { ordinal: 0, offset: first_line, limit: last_line - first_line + 1, content: std::mem::take(&mut current) });
                    count = 0;
                }
                if current.is_empty() { first_line = line_no + 1; }
                current.push(ch); count += 1; last_line = line_no + 1;
            }
        }
        if !current.is_empty() { chunks.push(TextChunk { ordinal: 0, offset: first_line, limit: last_line - first_line + 1, content: current }); }
    }
    for (ordinal, chunk) in chunks.iter_mut().enumerate() { chunk.ordinal = ordinal; }
    Ok(chunks)
}

/// 一篇笔记写入前的就绪形态：正文已读、切片已切、路径与标签已按库内规则解析。
/// 组合写流程（先删后加）在动任何派生数据之前把它备齐，删除落地之后的新增不再有失败窗口。
pub(crate) struct PreparedNote {
    pub record: RecordInput,
    pub namespace_id: i64, pub scope_id: i64, pub source: String,
    pub given: String, pub title: String, pub split: Vec<TextChunk>,
    pub chunk_chars: usize,
}

/// 读文件、切切片、解析领域与路径。term_id 的字典登记随调用方的事务提交。
pub(crate) fn prepare_note(conn: &Connection, input: &NoteFileInput, split: Vec<TextChunk>) -> Result<PreparedNote> {
    let given = input.path.to_string_lossy().into_owned();
    storage::validate_identity("source", &given)?;
    let namespace_id = storage::term_id(conn, &input.record.namespace)?;
    let scope_id = storage::term_id(conn, &input.record.scope)?;
    // 领域必须登记根目录：写入路径与它比对前缀，不重合直接拒绝、重合的部分裁掉。
    // 库里存的永远是相对路径——项目从头到尾不知道前面的绝对路径是什么。
    let root = storage::namespace_root(conn, namespace_id)?.ok_or_else(|| Error::Validation(
        format!("namespace {} has no registered domain root; call notes.set_root before upserting notes", input.record.namespace)))?;
    let source = relative_note_path(&root, &given)?;
    let title = input.path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default();
    let (dirs, split_stem) = storage::split_note_path(&source);
    // 文件名以写入时取好的 `title`（`file_stem`）为准，与落库的 `notes.name` 同源；
    // `split_note_path` 只用来取目录段。
    let stem = if title.is_empty() { split_stem } else { title.clone() };
    // 这一份标签挂到这篇的每一条切片上：调用方给的 + 路径拆出来的。
    let mut record = input.record.clone();
    // 笔记的身份就是 `(namespace, scope, path)`，下游给的 id 一律忽略：
    // 记录 id 由库自己分配，下游不能凭一个 id 凭空插一条笔记（那会让「笔记属于哪条路径」失守）。
    record.id = None;
    let mut tags = record.tags.clone();
    tags.extend(dirs.iter().cloned());
    if !stem.is_empty() { tags.push(stem.clone()); }
    record.tags = tags.clone();
    Ok(PreparedNote { record, namespace_id, scope_id, source, given, title, split, chunk_chars: input.chunk_chars })
}

/// 新增路径（内部方法）：笔记与切片连同「正在写入」标记一次落主库，索引文档就地折好交回。
/// 身份是 `(namespace, scope, path)`；调用方保证同一位要么是空的、要么刚被删除流程腾空。
/// 切片记录全部新开——旧切片的向量已随删除流程作废，由向量对账照主库重算。
pub(crate) fn add_note(conn: &Connection, note: &PreparedNote) -> Result<(Note, Vec<crate::index::IndexDocument>)> {
    let (header, _) = storage::put_record(conn, RecordKind::Note, &note.record,
        &json!({"chunk_chars": note.chunk_chars}), "")?;
    // 文件名在写入这一刻就从路径取好（`file_stem` 认平台分隔符），随笔记落库：
    // 索引那一列直接读它，索引侧补文档时也不必再拆一次路径。
    conn.execute("INSERT INTO notes(record_id,namespace_id,scope_id,path,name) VALUES (?1,?2,?3,?4,?5)
        ON CONFLICT(record_id) DO UPDATE SET namespace_id=excluded.namespace_id,scope_id=excluded.scope_id,path=excluded.path,name=excluded.name",
        params![header.id, note.namespace_id, note.scope_id, note.source, note.title])?;
    let mut documents = Vec::new();
    for chunk in &note.split {
        let content_digest = text::digest(&chunk.content);
        let payload = json!({"note_id":header.id,"ordinal":chunk.ordinal,"offset":chunk.offset,"limit":chunk.limit});
        let chunk_input = RecordInput { id: None, namespace: header.namespace.clone(), scope: header.scope.clone(),
            tags: header.tags.clone(), evidence: vec![], metadata: header.metadata.clone(),
            created_at_us: Some(header.created_at_us), updated_at_us: Some(header.updated_at_us), expected_revision: None };
        let (chunk_header, document) = storage::put_record(conn, RecordKind::Chunk, &chunk_input, &payload, &chunk.content)?;
        conn.execute("INSERT INTO chunks(record_id,note_id,ordinal,\"offset\",\"limit\",fingerprint) VALUES (?1,?2,?3,?4,?5,?6)",
            params![chunk_header.id, header.id, chunk.ordinal as i64, chunk.offset as i64, chunk.limit as i64, content_digest])?;
        documents.push(document);
    }
    Ok((Note { header, source: note.given.clone(), title: note.title.clone(), chunk_chars: note.chunk_chars }, documents))
}

/// 减掉领域根目录得到库里存的那条相对路径；不在根目录之内直接报错，不做猜测。
/// 只统一分隔符，不改大小写——存的是什么名字，读回文件时就找什么名字。
fn relative_note_path(root: &str, given: &str) -> Result<String> {
    let root = root.replace('\\', "/");
    let root = root.trim_end_matches('/');
    let relative = given.replace('\\', "/");
    let Some(relative) = relative.strip_prefix(root).and_then(|rest| rest.strip_prefix('/')) else {
        return Err(Error::Validation(format!("note path {given} is outside the domain root {root}")));
    };
    if relative.is_empty() { return Err(Error::Validation("note path must name a file below the domain root".into())); }
    Ok(relative.to_string())
}

#[derive(Clone)]
pub struct NoteStore(pub(crate) KnowledgeBase);

impl NoteStore {
    /// 按文件路径同步一篇笔记。更新就是「先删后加」：同路径已有笔记，先把旧笔记连同
    /// 它的切片整条删掉（写锁内落主库标记、删向量行与主库行，出锁后摘索引词条），
    /// 再当新笔记写入；没有就直接新增。切片记录全部新开，向量由向量对账照主库重算。
    pub fn upsert_file(&self, input: NoteFileInput) -> Result<WriteReceipt<Note>> {
        let mut receipts = self.upsert_files(&[input])?;
        let value = receipts.value.pop().ok_or_else(|| Error::Validation("empty upsert batch".into()))?;
        Ok(WriteReceipt { value, revision: receipts.revision })
    }

    /// 批量版：一次写锁包住整批的 SQL，每篇各自完整地先删后加。
    /// 索引操作（摘词条、交文档）在写锁外，由 Tantivy 自己的锁管。
    /// 文件在动库之前全部读好切好——任何一处读取失败，整批原样拒绝，库一个字节都没动。
    pub fn upsert_files(&self, inputs: &[NoteFileInput]) -> Result<WriteReceipt<Vec<Note>>> {
        let split = inputs.iter().map(|input| -> Result<Vec<TextChunk>> {
            let content = std::fs::read_to_string(&input.path)?;
            chunk_text(&content, input.chunk_chars)
        }).collect::<Result<Vec<_>>>()?;
        let mut notes: Vec<Note> = Vec::new();
        let (stale, documents) = self.0.with_writer_lock(|writer| {
            let mut documents: Vec<crate::index::IndexDocument> = Vec::new();
            let mut stale: Vec<i64> = Vec::new();
            for (input, split) in inputs.iter().zip(split) {
                // 解析与定位在标记事务里做：term_id 的字典登记随事务提交。
                let tx = writer.conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let prepared = prepare_note(&tx, input, split)?;
                let by_path: Option<i64> = tx.query_row("SELECT record_id FROM notes WHERE namespace_id=?1 AND scope_id=?2 AND path=?3",
                    params![prepared.namespace_id, prepared.scope_id, prepared.source], |r| r.get(0)).optional()?;
                // 笔记只按 (namespace, scope, path) 定位：同路径就是同一篇，先删后加。
                let mut removed: Vec<i64> = Vec::new();
                if let Some(id) = by_path {
                    // 切片排在笔记前面：删除按这个次序落库，RESTRICT 引用不会拦。
                    let mut stmt = tx.prepare("SELECT record_id FROM chunks WHERE note_id=?1")?;
                    for row in stmt.query_map([id], |r| r.get::<_, i64>(0))? { removed.push(row?); }
                    removed.push(id);
                    // 标记先行：主库上先把「正在删除」落定，派生动作排在它后面。
                    storage::mark_records(&tx, &removed, storage::MARK_DELETING)?;
                }
                tx.commit()?;
                if !removed.is_empty() {
                    // 权威落：删向量行 + 删主库行（切片先于笔记）。
                    let tx = writer.conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                    for id in &removed { storage::delete_record(&tx, &RecordKey { id: *id })?; }
                    tx.commit()?;
                    stale.extend(removed);
                }
                // 新增：记录连同「正在写入」标记一次落主库，索引文档就地折好带回。
                let tx = writer.conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let (note, docs) = add_note(&tx, &prepared)?;
                tx.commit()?;
                documents.extend(docs);
                notes.push(note);
            }
            Ok((stale, documents))
        })?;
        // 索引操作在写锁外：先摘旧词条、再交新文档，各自走 Tantivy 的内部锁。
        if !stale.is_empty() { self.0.index()?.stage_deletions(&stale)?; }
        self.0.index_documents(&documents)?;
        let revision = storage::current_revision(self.0.read()?.conn())?;
        Ok(WriteReceipt { value: notes, revision })
    }
    /// 登记该知识领域的笔记根目录。登记之后写入的笔记路径必须是它的子路径：
    /// 库里存相对路径，相对路径按段拆出的标签挂到这篇的每一条切片上。
    pub fn set_root(&self, namespace: &str, root: &str) -> Result<()> {
        storage::validate_identity("namespace", namespace)?;
        if !std::path::Path::new(root).is_dir() {
            return Err(Error::Validation(format!("domain root {root} is not an existing directory")));
        }
        let root = root.replace('\\', "/");
        let root = root.trim_end_matches('/').to_string();
        self.0.write(|writer| {
            let namespace_id = storage::term_id(&writer.conn, namespace)?;
            writer.conn.execute("INSERT INTO namespace_roots(namespace_id,root) VALUES (?1,?2)
                ON CONFLICT(namespace_id) DO UPDATE SET root=excluded.root", params![namespace_id, root])?;
            Ok(())
        })
    }
    /// 注销该领域的笔记根目录登记。只影响之后写入时的路径计算，已入库的笔记不动。
    /// 返回是否命中；没登记过的领域不报错。
    pub fn unset_root(&self, namespace: &str) -> Result<bool> {
        storage::validate_identity("namespace", namespace)?;
        self.0.write(|writer| {
            let namespace_id = storage::term_id(&writer.conn, namespace)?;
            Ok(writer.conn.execute("DELETE FROM namespace_roots WHERE namespace_id=?1", [namespace_id])? > 0)
        })
    }
    /// 该领域登记的笔记根目录；没登记就是 `None`。
    pub fn root(&self, namespace: &str) -> Result<Option<String>> {
        let state = self.0.read()?;
        let conn = state.conn();
        let namespace_id: Option<i64> = conn.query_row("SELECT id FROM strings WHERE text=?1",
            [text::normalized_tag(namespace)], |r| r.get(0)).optional()?;
        let Some(namespace_id) = namespace_id else { return Ok(None) };
        storage::namespace_root(conn, namespace_id)
    }
    pub fn get(&self, id: i64, filter: &ReadFilter) -> Result<Note> {
        storage::get(self.0.read()?.conn(), &RecordKey { id }, filter)
    }
    /// 批量读取一批笔记，只返回满足 `filter` 的那些。
    ///
    /// 语义等同于对每个 id 依次调用 `get`，但把过滤压成一条 SQL、一次取回，
    /// 避免宿主逐条回库的往返开销。不满足过滤条件的 id 被静默跳过（不报错）。
    pub fn get_many(&self, ids: &[i64], filter: &ReadFilter) -> Result<BTreeMap<i64, Note>> {
        storage::load_many(self.0.read()?.conn(), ids, filter)
    }
    pub fn list(&self, page: &PageRequest) -> Result<Page<Note>> { storage::list(self.0.read()?.conn(), RecordKind::Note, page) }
    pub fn get_chunk(&self, id: i64, filter: &ReadFilter) -> Result<Chunk> {
        let state = self.0.read()?;
        let conn = state.conn();
        let mut chunk: Chunk = storage::get(conn, &RecordKey { id }, filter)?;
        chunk.content = self.0.index()?.bodies(&[id])?.remove(&id).unwrap_or_default();
        Ok(chunk)
    }
    pub fn chunks(&self, note_id: i64, filter: &ReadFilter) -> Result<Vec<Chunk>> {
        let state = self.0.read()?;
        let conn = state.conn();
        let _: Note = storage::get(conn, &RecordKey { id: note_id }, filter)?;
        let ids: Vec<i64> = {
            let mut stmt = conn.prepare("SELECT record_id FROM chunks WHERE note_id=?1 ORDER BY ordinal")?;
            let rows = stmt.query_map([note_id], |r| r.get::<_, i64>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        // 一次批量取回：逐条 `get` 会为每片各跑一遍过滤与装配，一篇几百片就是几百次回库。
        let mut loaded: BTreeMap<i64, Chunk> = storage::load_many(conn, &ids, filter)?;
        let mut chunks: Vec<Chunk> = ids.iter().filter_map(|id| loaded.remove(id)).collect();
        // 一批切片一次取回正文：正文只存在索引里，逐条回库或回源文件都没有意义。
        let ids: Vec<i64> = chunks.iter().map(|chunk| chunk.header.id).collect();
        let mut bodies = self.0.index()?.bodies(&ids)?;
        for chunk in &mut chunks { chunk.content = bodies.remove(&chunk.header.id).unwrap_or_default(); }
        Ok(chunks)
    }
    /// 批量删除笔记：主库查 id，命中才继续；单条也是批量的一种。
    /// 每篇连同它的切片一起删（`chunks.note_id` 是 RESTRICT 引用）：
    /// 写锁内落主库标记、删向量行与主库行，出锁后摘索引词条。
    /// 断电留下「正在删除」标记的，下次开机把这条删除做完。
    pub fn delete(&self, ids: &[i64], filter: &ReadFilter) -> Result<WriteReceipt<usize>> {
        let removed = self.0.delete_flow(ids, filter, RecordKind::Note, |tx, id| {
            let mut out = Vec::new();
            let mut stmt = tx.prepare("SELECT record_id FROM chunks WHERE note_id=?1")?;
            for row in stmt.query_map([id], |r| r.get::<_, i64>(0))? { out.push(row?); }
            Ok(out)
        })?;
        let revision = storage::current_revision(self.0.read()?.conn())?;
        Ok(WriteReceipt { value: removed, revision })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_note_path_separates_directories_from_the_file_name() {
        assert_eq!(storage::split_note_path("notes/characters/overview.md"), (vec!["notes".to_string(), "characters".to_string()], "overview".to_string()));
        // 目录段里的点不是扩展名：只有最后一段去后缀。
        assert_eq!(storage::split_note_path("v1.2/角色.设定.md"), (vec!["v1.2".to_string()], "角色.设定".to_string()));
        assert_eq!(storage::split_note_path("overview"), (Vec::new(), "overview".to_string()));
    }

    #[test]
    fn note_paths_must_stay_inside_the_domain_root() {
        assert_eq!(relative_note_path("E:/data/demo", r"E:\data\demo\notes\overview.md").unwrap(), "notes/overview.md");
        // 前缀只是像，不是子路径；也不许正好等于根目录本身。
        assert!(relative_note_path("E:/data/demo", "E:/data/demox/overview.md").is_err());
        assert!(relative_note_path("E:/data/demo", "E:/data/other/overview.md").is_err());
        assert!(relative_note_path("E:/data/demo", "E:/data/demo").is_err());
    }
}
