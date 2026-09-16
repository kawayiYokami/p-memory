use crate::{storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

fn default_chunk_chars() -> usize { 220 }

/// 按文件路径同步一篇笔记的入参：库自己读文件，标题取文件名（去扩展名）。
/// 路径即身份——`(namespace, scope)` 下的定位键就是文件路径，库按它读回正文。
/// 正文真相源是文件：清洗只作用于送进索引的文本，库内不留任何正文副本。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteFileInput {
    #[serde(flatten)] pub record: RecordInput,
    /// 要读取的文件路径，同时作为 `(namespace, scope, source)` 的定位键。
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
    /// 文件路径，读时由 `notes` 表补回（库内只此一处）。
    pub source: String,
    /// 由 `source` 的文件名派生，不落库。
    #[serde(default)] pub title: String,
    pub chunk_chars: usize,
}
/// 切片不再重复携带 source/title；路径与标题经 `note_id` 关联 notes 取回。
/// `content` 不落 SQLite，读取时由笔记原文按字符区间派生。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    #[serde(flatten)] pub header: RecordHeader,
    pub note_id: i64, pub ordinal: usize, pub offset: usize, pub limit: usize,
    #[serde(default)] pub content: String,
}
/// `offset` 为 1 起始的起始行，`limit` 为行数；`char_start` / `char_end` 为正文中的字符区间（含起、不含止）。
/// 超长段落会被多个切片共享同一行号，行号无法唯一定位切片正文，字符区间才是权威。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextChunk { pub ordinal: usize, pub offset: usize, pub limit: usize, pub char_start: usize, pub char_end: usize, pub content: String }

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
            chunks.push(TextChunk { ordinal: 0, offset: start + 1, limit: end - start, char_start: 0, char_end: 0, content: body });
            continue;
        }
        let mut current = String::new();
        let mut count = 0;
        let mut first_line = start + 1;
        let mut last_line = first_line;
        for (line_no, line) in lines.iter().enumerate().take(end).skip(start) {
            if !current.is_empty() {
                if count + 1 >= target {
                    chunks.push(TextChunk { ordinal: 0, offset: first_line, limit: last_line - first_line + 1, char_start: 0, char_end: 0, content: std::mem::take(&mut current) });
                    count = 0;
                } else { current.push('\n'); count += 1; }
            }
            for ch in line.chars() {
                if count == target {
                    chunks.push(TextChunk { ordinal: 0, offset: first_line, limit: last_line - first_line + 1, char_start: 0, char_end: 0, content: std::mem::take(&mut current) });
                    count = 0;
                }
                if current.is_empty() { first_line = line_no + 1; }
                current.push(ch); count += 1; last_line = line_no + 1;
            }
        }
        if !current.is_empty() { chunks.push(TextChunk { ordinal: 0, offset: first_line, limit: last_line - first_line + 1, char_start: 0, char_end: 0, content: current }); }
    }
    for (ordinal, chunk) in chunks.iter_mut().enumerate() { chunk.ordinal = ordinal; }
    fill_char_ranges(content, &mut chunks);
    Ok(chunks)
}

/// 给切片补上正文中的字符区间。切片正文是原文的连续子串，按文档顺序用游标定位即可，
/// 无需重算分词规则。定位不到的切片（理论上不会发生）保留 0..0，读取侧会退化为空。
fn fill_char_ranges(content: &str, chunks: &mut [TextChunk]) {
    let chars: Vec<char> = content.chars().collect();
    let mut cursor = 0usize;
    for chunk in chunks.iter_mut() {
        if let Some(start) = find_chars(&chars, &chunk.content, cursor) {
            chunk.char_start = start;
            chunk.char_end = start + chunk.content.chars().count();
            cursor = chunk.char_end;
        }
    }
}

/// 在 `chars` 的 `from` 之后寻找 `needle`，返回起始字符下标。
fn find_chars(chars: &[char], needle: &str, from: usize) -> Option<usize> {
    let needle: Vec<char> = needle.chars().collect();
    if needle.is_empty() || needle.len() > chars.len() { return None; }
    let last = chars.len() - needle.len();
    for start in from..=last {
        if (0..needle.len()).all(|i| chars[start + i] == needle[i]) { return Some(start); }
    }
    None
}

pub(crate) fn sync_file(conn: &Connection, input: &NoteFileInput) -> Result<Note> {
    let content = std::fs::read_to_string(&input.path)?;
    let source = input.path.to_string_lossy().into_owned();
    let title = input.path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default();
    storage::validate_identity("source", &source)?;
    let split = chunk_text(&content, input.chunk_chars)?;
    let mut record = input.record.clone();
    let namespace_id = storage::term_id(conn, &record.namespace)?;
    let scope_id = storage::term_id(conn, &record.scope)?;
    let source_id = storage::term_id(conn, &source)?;
    let existing: Option<i64> = conn.query_row("SELECT record_id FROM notes WHERE namespace_id=?1 AND scope_id=?2 AND source_id=?3",
        params![namespace_id, scope_id, source_id], |r| r.get(0)).optional()?;
    if let Some(id) = existing {
        if record.id.is_some_and(|given| given != id) { return Err(Error::Conflict("source already belongs to another note ID".into())); }
        record.id = Some(id);
    }
    // 路径与 tags 不再混入正文被切碎：正文只留标题与内容，关键字走精确整词字段。
    let body = format!("{title}\n{content}");
    let header = storage::put_record(conn, RecordKind::Note, &record,
        &json!({"chunk_chars":input.chunk_chars}), &body)?;
    conn.execute("INSERT INTO notes(record_id,namespace_id,scope_id,source_id) VALUES (?1,?2,?3,?4)
        ON CONFLICT(record_id) DO UPDATE SET namespace_id=excluded.namespace_id,scope_id=excluded.scope_id,source_id=excluded.source_id",
        params![header.id, namespace_id, scope_id, source_id])?;
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
        let fingerprint = text::digest(&chunk.content);
        let payload = json!({"note_id":header.id,"ordinal":chunk.ordinal,"offset":chunk.offset,"limit":chunk.limit,"char_start":chunk.char_start,"char_end":chunk.char_end});
        let id = match reuse.get(&(chunk.ordinal as i64, fingerprint.clone())) {
            Some(&id) => {
                used.insert(id);
                conn.execute("UPDATE records SET payload_json=?2 WHERE id=?1", params![id, serde_json::to_string(&payload)?])?;
                id
            }
            None => {
                let chunk_input = RecordInput { id: None, namespace: header.namespace.clone(), scope: header.scope.clone(), tags: header.tags.clone(),
                    evidence: vec![], metadata: header.metadata.clone(), created_at_us: Some(header.created_at_us), updated_at_us: Some(header.updated_at_us), expected_revision: None };
                let chunk_body = format!("{title}\n{}", chunk.content);
                storage::put_record(conn, RecordKind::Chunk, &chunk_input, &payload, &chunk_body)?.id
            }
        };
        conn.execute("INSERT INTO chunks(record_id,note_id,ordinal,\"offset\",\"limit\",fingerprint) VALUES (?1,?2,?3,?4,?5,?6)",
            params![id, header.id, chunk.ordinal as i64, chunk.offset as i64, chunk.limit as i64, fingerprint])?;
    }
    // Delete obsolete records (and their embeddings), not just projection rows.
    for (id, _, _) in old { if !used.contains(&id) { storage::delete_record(conn, &RecordKey { id })?; } }
    Ok(Note { header, source, title, chunk_chars: input.chunk_chars })
}

#[derive(Clone)]
pub struct NoteStore(pub(crate) KnowledgeBase);

/// 切片正文不落 SQLite，读取时补齐。
fn with_content(conn: &Connection, mut chunk: Chunk) -> Result<Chunk> {
    if chunk.content.is_empty() { chunk.content = storage::chunk_content(conn, chunk.header.id)?; }
    Ok(chunk)
}

impl NoteStore {
    /// 读文件后同步一篇笔记：正文取文件原文，标题取文件名（去扩展名），路径即身份。
    /// 监听与对账在使用方；库只按给定路径处理这一个文件。
    pub fn upsert_file(&self, input: NoteFileInput) -> Result<WriteReceipt<Note>> {
        let receipt = self.0.mutate(|tx| sync_file(tx, &input))?;
        // 笔记与其切片一同交给内部向量化；默认关闭时这一步直接跳过。
        self.0.vectorize_note(receipt.value.header.id);
        Ok(receipt)
    }
    pub fn get(&self, id: i64, filter: &ReadFilter) -> Result<Note> {
        storage::get(self.0.read()?.conn(), &RecordKey { id }, filter)
    }
    pub fn list(&self, page: &PageRequest) -> Result<Page<Note>> { storage::list(self.0.read()?.conn(), RecordKind::Note, page) }
    pub fn get_chunk(&self, id: i64, filter: &ReadFilter) -> Result<Chunk> {
        let conn = self.0.read()?;
        let conn = conn.conn();
        with_content(conn, storage::get(conn, &RecordKey { id }, filter)?)
    }
    pub fn chunks(&self, note_id: i64, filter: &ReadFilter) -> Result<Vec<Chunk>> {
        let state = self.0.read()?;
        let conn = state.conn();
        let _: Note = storage::get(conn, &RecordKey { id: note_id }, filter)?;
        let mut stmt = conn.prepare("SELECT record_id FROM chunks WHERE note_id=?1 ORDER BY ordinal")?;
        let rows = stmt.query_map([note_id], |r| r.get::<_, i64>(0))?;
        let mut chunks = Vec::new();
        for row in rows { chunks.push(with_content(conn, storage::get(conn, &RecordKey { id: row? }, filter)?)?); }
        Ok(chunks)
    }
    pub fn delete(&self, id: i64, filter: &ReadFilter) -> Result<WriteReceipt<bool>> {
        self.0.mutate(|tx| {
            let key = RecordKey { id };
            let _: Note = storage::get(tx, &key, filter)?;
            let mut stmt = tx.prepare("SELECT record_id FROM chunks WHERE note_id=?1")?;
            let ids = stmt.query_map([id], |r| r.get::<_, i64>(0))?.collect::<std::result::Result<Vec<_>, _>>()?;
            for child in ids { storage::delete_record(tx, &RecordKey { id: child })?; }
            storage::delete_record(tx, &key)
        })
    }
}
