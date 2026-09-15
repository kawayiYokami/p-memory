use crate::{storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeSet, HashMap};

fn default_chunk_chars() -> usize { 220 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteInput {
    #[serde(flatten)] pub record: RecordInput,
    /// A host-owned path or URI. The library never reads or writes this path.
    pub source: String,
    #[serde(default)] pub title: String,
    pub content: String,
    #[serde(default = "default_chunk_chars")] pub chunk_chars: usize,
}
impl NoteInput {
    pub fn new(source: impl Into<String>, content: impl Into<String>) -> Self {
        Self { record: RecordInput::default(), source: source.into(), title: String::new(), content: content.into(), chunk_chars: default_chunk_chars() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    #[serde(flatten)] pub header: RecordHeader,
    pub source: String, pub title: String, pub content: String,
    pub source_revision: String, pub chunk_chars: usize, pub chunk_count: usize,
}
/// 切片不再重复携带 source/title；路径与标题经 `note_id` 关联 notes 取回。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    #[serde(flatten)] pub header: RecordHeader,
    pub note_id: i64, pub ordinal: usize, pub offset: usize, pub limit: usize, pub content: String,
}
/// `offset` 为 1 起始的起始行，`limit` 为行数。
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

pub(crate) fn upsert(conn: &Connection, input: &NoteInput) -> Result<Note> {
    storage::validate_identity("source", &input.source)?;
    let split = chunk_text(&input.content, input.chunk_chars)?;
    let mut record = input.record.clone();
    let namespace_id = storage::term_id(conn, &record.namespace)?;
    let scope_id = storage::term_id(conn, &record.scope)?;
    let source_id = storage::term_id(conn, &input.source)?;
    let existing: Option<i64> = conn.query_row("SELECT record_id FROM notes WHERE namespace_id=?1 AND scope_id=?2 AND source_id=?3",
        params![namespace_id, scope_id, source_id], |r| r.get(0)).optional()?;
    if let Some(id) = existing {
        if record.id.is_some_and(|given| given != id) { return Err(Error::Conflict("source already belongs to another note ID".into())); }
        record.id = Some(id);
    }
    let source_revision = text::digest(&input.content);
    // 路径与 tags 不再混入正文被切碎：正文只留标题与内容，关键字走精确整词字段。
    let body = format!("{}\n{}", input.title, input.content);
    let header = storage::put_record(conn, RecordKind::Note, &record,
        &json!({"source":input.source,"title":input.title,"content":input.content,"source_revision":source_revision,"chunk_chars":input.chunk_chars,"chunk_count":split.len()}), &body, &body)?;
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
        let payload = json!({"note_id":header.id,"ordinal":chunk.ordinal,"offset":chunk.offset,"limit":chunk.limit,"content":chunk.content});
        let id = match reuse.get(&(chunk.ordinal as i64, fingerprint.clone())) {
            Some(&id) => {
                used.insert(id);
                conn.execute("UPDATE records SET payload_json=?2 WHERE id=?1", params![id, serde_json::to_string(&payload)?])?;
                id
            }
            None => {
                let chunk_input = RecordInput { id: None, namespace: header.namespace.clone(), scope: header.scope.clone(), tags: header.tags.clone(),
                    evidence: vec![], metadata: header.metadata.clone(), created_at_us: Some(header.created_at_us), updated_at_us: Some(header.updated_at_us), expected_revision: None };
                let chunk_body = format!("{}\n{}", input.title, chunk.content);
                storage::put_record(conn, RecordKind::Chunk, &chunk_input, &payload, &chunk_body, &chunk_body)?.id
            }
        };
        conn.execute("INSERT INTO chunks(record_id,note_id,ordinal,\"offset\",\"limit\",fingerprint) VALUES (?1,?2,?3,?4,?5,?6)",
            params![id, header.id, chunk.ordinal as i64, chunk.offset as i64, chunk.limit as i64, fingerprint])?;
    }
    // Delete obsolete records (and their embeddings), not just projection rows.
    for (id, _, _) in old { if !used.contains(&id) { storage::delete_record(conn, &RecordKey { id })?; } }
    Ok(Note { header, source: input.source.clone(), title: input.title.clone(), content: input.content.clone(), source_revision,
        chunk_chars: input.chunk_chars, chunk_count: split.len() })
}

#[derive(Clone)]
pub struct NoteStore(pub(crate) KnowledgeBase);
impl NoteStore {
    pub fn upsert(&self, input: NoteInput) -> Result<WriteReceipt<Note>> { self.0.mutate(|tx| upsert(tx, &input)) }
    pub fn get(&self, id: i64, filter: &ReadFilter) -> Result<Note> {
        storage::get(self.0.read()?.conn(), &RecordKey { id }, filter)
    }
    pub fn list(&self, page: &PageRequest) -> Result<Page<Note>> { storage::list(self.0.read()?.conn(), RecordKind::Note, page) }
    pub fn get_chunk(&self, id: i64, filter: &ReadFilter) -> Result<Chunk> {
        storage::get(self.0.read()?.conn(), &RecordKey { id }, filter)
    }
    pub fn chunks(&self, note_id: i64, filter: &ReadFilter) -> Result<Vec<Chunk>> {
        let state = self.0.read()?;
        let conn = state.conn();
        let _: Note = storage::get(conn, &RecordKey { id: note_id }, filter)?;
        let mut stmt = conn.prepare("SELECT record_id FROM chunks WHERE note_id=?1 ORDER BY ordinal")?;
        let rows = stmt.query_map([note_id], |r| r.get::<_, i64>(0))?;
        let mut chunks = Vec::new();
        for row in rows { chunks.push(storage::get(conn, &RecordKey { id: row? }, filter)?); }
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
