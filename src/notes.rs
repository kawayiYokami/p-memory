use crate::{storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap};
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

pub(crate) fn sync_file(conn: &Connection, input: &NoteFileInput) -> Result<(Note, Vec<crate::index::IndexDocument>)> {
    let content = std::fs::read_to_string(&input.path)?;
    let given = input.path.to_string_lossy().into_owned();
    let title = input.path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default();
    storage::validate_identity("source", &given)?;
    let split = chunk_text(&content, input.chunk_chars)?;
    let mut record = input.record.clone();
    let namespace_id = storage::term_id(conn, &record.namespace)?;
    let scope_id = storage::term_id(conn, &record.scope)?;
    // 领域必须登记根目录：写入路径与它比对前缀，不重合直接拒绝、重合的部分裁掉。
    // 库里存的永远是相对路径——项目从头到尾不知道前面的绝对路径是什么。
    let root = storage::namespace_root(conn, namespace_id)?.ok_or_else(|| Error::Validation(
        format!("namespace {} has no registered domain root; call notes.set_root before upserting notes", record.namespace)))?;
    let source = relative_note_path(&root, &given)?;
    let (dirs, split_stem) = storage::split_note_path(&source);
    // 文件名以写入时取好的 `title`（`file_stem`）为准，与落库的 `notes.name` 同源；
    // `split_note_path` 只用来取目录段。
    let stem = if title.is_empty() { split_stem } else { title.clone() };
    // 这一份标签挂到这篇的每一条切片上：调用方给的 + 路径拆出来的。
    let mut merged = record.tags.clone();
    merged.extend(dirs.iter().cloned());
    if !stem.is_empty() { merged.push(stem.clone()); }
    record.tags = merged;
    let existing: Option<i64> = conn.query_row("SELECT record_id FROM notes WHERE namespace_id=?1 AND scope_id=?2 AND path=?3",
        params![namespace_id, scope_id, source], |r| r.get(0)).optional()?;
    if let Some(id) = existing {
        if record.id.is_some_and(|given| given != id) { return Err(Error::Conflict("source already belongs to another note ID".into())); }
        record.id = Some(id);
    }
    // 笔记这条记录自己不承载正文，也不占索引文档：路径信息以标签形态挂在它的每个切片上，
    // 要文件列表就按库里的标签翻笔记。
    let (header, _) = storage::put_record(conn, RecordKind::Note, &record,
        &json!({"chunk_chars":input.chunk_chars}), "")?;
    let mut documents = Vec::new();
    // 文件名在写入这一刻就从路径取好（`file_stem` 认平台分隔符），随笔记落库：
    // 索引那一列直接读它，重建时也不必再拆一次路径。
    conn.execute("INSERT INTO notes(record_id,namespace_id,scope_id,path,name) VALUES (?1,?2,?3,?4,?5)
        ON CONFLICT(record_id) DO UPDATE SET namespace_id=excluded.namespace_id,scope_id=excluded.scope_id,path=excluded.path,name=excluded.name",
        params![header.id, namespace_id, scope_id, source, title])?;
    // Reuse the record ID of any slice whose ordinal和内容都未变，让它的向量继续有效。
    let mut old = Vec::new();
    {
        let mut stmt = conn.prepare("SELECT record_id,ordinal,fingerprint FROM chunks WHERE note_id=?1")?;
        for row in stmt.query_map([header.id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, String>(2)?)))? {
            old.push(row?);
        }
    }
    let reuse: HashMap<(i64, String), i64> = old.iter().map(|(id, ordinal, fp)| ((*ordinal, fp.clone()), *id)).collect();
    // Ordinals may be reshuffled between revisions, so clear the projection first
    // and rebuild it; slices whose ordinal and content are unchanged keep their
    // record ID (and therefore their embedding).
    conn.execute("DELETE FROM chunks WHERE note_id=?1", [header.id])?;
    let mut used = BTreeSet::new();
    for chunk in &split {
        let content_digest = text::digest(&chunk.content);
        let payload = json!({"note_id":header.id,"ordinal":chunk.ordinal,"offset":chunk.offset,"limit":chunk.limit});
        let (id, document) = match reuse.get(&(chunk.ordinal as i64, content_digest.clone())) {
            Some(&id) => {
                used.insert(id);
                conn.execute("UPDATE records SET payload_json=?2 WHERE id=?1", params![id, serde_json::to_string(&payload)?])?;
                // 内容没变，但标签是这篇当前这一份：标签换了指纹跟着换，旧向量就地作废。
                let tag_ids = storage::set_record_tags(conn, id, &header.tags)?;
                let fingerprint = storage::record_fingerprint(&chunk.content, &header.tags);
                conn.execute("UPDATE records SET fingerprint=?2,updated_at_us=MAX(updated_at_us,?3) WHERE id=?1",
                    params![id, fingerprint, header.updated_at_us])?;
                conn.execute("DELETE FROM embeddings WHERE record_id=?1 AND fingerprint<>?2", params![id, fingerprint])?;
                // 标签与指纹都动过，这个领域的向量分区跟着变。
                storage::touch_namespace(&header.namespace);
                let (name, path, exclude) = storage::index_columns(conn, RecordKind::Chunk, &payload);
                (id, crate::index::IndexDocument { id, namespace_id, scope_id, kind: RecordKind::Chunk,
                    text: chunk.content.clone(),
                    name, path,
                    note_id: header.id,
                    tags_prefix: storage::tags_prefix(RecordKind::Chunk, &header.tags, &exclude, &payload),
                    tag_ids })
            }
            None => {
                let chunk_input = RecordInput { id: None, namespace: header.namespace.clone(), scope: header.scope.clone(), tags: header.tags.clone(),
                    evidence: vec![], metadata: header.metadata.clone(), created_at_us: Some(header.created_at_us), updated_at_us: Some(header.updated_at_us), expected_revision: None };
                let (chunk_header, document) = storage::put_record(conn, RecordKind::Chunk, &chunk_input, &payload, &chunk.content)?;
                (chunk_header.id, document)
            }
        };
        conn.execute("INSERT INTO chunks(record_id,note_id,ordinal,\"offset\",\"limit\",fingerprint) VALUES (?1,?2,?3,?4,?5,?6)",
            params![id, header.id, chunk.ordinal as i64, chunk.offset as i64, chunk.limit as i64, content_digest])?;
        documents.push(document);
    }
    // Delete obsolete records (and their embeddings), not just projection rows.
    for (id, _, _) in old { if !used.contains(&id) { storage::delete_record(conn, &RecordKey { id })?; } }
    Ok((Note { header, source: given, title, chunk_chars: input.chunk_chars }, documents))
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
    /// 从索引取一批记录的正文。有取不到的记录说明索引还没追上这批写入，就提交一次再取；
    /// 索引已追平时这条读路径一次都不碰写锁。
    fn bodies(&self, conn: &Connection, ids: &[i64]) -> Result<BTreeMap<i64, String>> {
        if ids.is_empty() { return Ok(BTreeMap::new()); }
        let mut bodies = self.0.index()?.bodies(ids)?;
        if ids.iter().any(|id| !bodies.contains_key(id)) {
            self.0.sync_index_if_behind(conn)?;
            bodies = self.0.index()?.bodies(ids)?;
        }
        Ok(bodies)
    }
    /// 读文件后同步一篇笔记：正文取文件原文，标题取文件名（去扩展名），路径即身份。
    /// 监听与对账在使用方；库只按给定路径处理这一个文件。
    pub fn upsert_file(&self, input: NoteFileInput) -> Result<WriteReceipt<Note>> {
        let receipt = self.0.mutate(|tx| sync_file(tx, &input))?;
        let WriteReceipt { value: (note, documents), revision } = receipt;
        // 切片正文在写入时就地切好、一路带到索引，索引阶段不再回读文件。
        self.0.index_documents(&documents)?;
        Ok(WriteReceipt { value: note, revision })
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
        chunk.content = self.bodies(conn, &[id])?.remove(&id).unwrap_or_default();
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
        let mut bodies = self.bodies(conn, &ids)?;
        for chunk in &mut chunks { chunk.content = bodies.remove(&chunk.header.id).unwrap_or_default(); }
        Ok(chunks)
    }
    pub fn delete(&self, id: i64, filter: &ReadFilter) -> Result<WriteReceipt<bool>> {
        self.0.mutate(|tx| delete_note(tx, id, filter))
    }
    /// 删除过滤条件命中的全部笔记，返回删除条数。空命中返回 0。
    /// 每篇都连同它的切片一起删：`chunks.note_id` 是 RESTRICT 引用，先切片后笔记。
    pub fn delete_by_filter(&self, filter: &ReadFilter) -> Result<WriteReceipt<usize>> {
        self.0.mutate(|tx| {
            let mut removed = 0;
            for id in storage::select_ids(tx, filter, &[RecordKind::Note])? {
                if delete_note(tx, id, filter)? { removed += 1; }
            }
            Ok(removed)
        })
    }
}

/// 删一篇笔记：先删它的切片，再删笔记本身。
fn delete_note(conn: &Connection, id: i64, filter: &ReadFilter) -> Result<bool> {
    let key = RecordKey { id };
    let _: Note = storage::get(conn, &key, filter)?;
    let mut stmt = conn.prepare("SELECT record_id FROM chunks WHERE note_id=?1")?;
    let ids = stmt.query_map([id], |r| r.get::<_, i64>(0))?.collect::<std::result::Result<Vec<_>, _>>()?;
    for child in ids { storage::delete_record(conn, &RecordKey { id: child })?; }
    storage::delete_record(conn, &key)
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
