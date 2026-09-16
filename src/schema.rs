use crate::{Error, Result};
use rusqlite::Connection;

pub(crate) const SCHEMA_VERSION: i64 = 5;
pub(crate) const APPLICATION_ID: i64 = 0x5041494d;

/// v3 -> v4：新增谓词元规则表并内置 `sys:same_as`。与 schema.sql 中同名段落保持一致。
const MIGRATION_3_TO_4: &str = "
CREATE TABLE predicate_rules (
    predicate_id INTEGER PRIMARY KEY REFERENCES strings(id) ON DELETE CASCADE,
    inverse_predicate_id INTEGER REFERENCES strings(id) ON DELETE RESTRICT,
    is_symmetric INTEGER NOT NULL DEFAULT 0,
    CHECK (is_symmetric IN (0, 1)),
    CHECK (is_symmetric = 0 OR inverse_predicate_id IS NULL)
);
INSERT OR IGNORE INTO strings(text) VALUES ('sys:same_as');
INSERT OR IGNORE INTO predicate_rules(predicate_id, is_symmetric) SELECT id, 1 FROM strings WHERE text='sys:same_as';
";

/// v4 -> v5：删掉两条派生文本列。切片 payload 的重写在 `migrate_chunk_payloads` 里。
const MIGRATION_4_TO_5: &str = "
ALTER TABLE records DROP COLUMN search_text;
ALTER TABLE records DROP COLUMN embedding_text;
";

/// 把切片 payload 从「带 content」翻成「带字符区间」：逐笔记重跑切片，按 `ordinal` 回填。
/// 切片正文是原文的连续子串，字符区间足以无损还原。
fn migrate_chunk_payloads(tx: &rusqlite::Transaction<'_>) -> Result<()> {
    let mut notes: Vec<(i64, String, usize)> = Vec::new();
    {
        let mut stmt = tx.prepare("SELECT id,payload_json FROM records WHERE kind=?1")?;
        let rows = stmt.query_map([crate::types::RecordKind::Note.code()], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows {
            let (id, payload) = row?;
            let value: serde_json::Value = serde_json::from_str(&payload)?;
            let content = value.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let chunk_chars = value.get("chunk_chars").and_then(|v| v.as_u64()).unwrap_or(220) as usize;
            notes.push((id, content, chunk_chars));
        }
    }
    for (note_id, content, chunk_chars) in notes {
        let ranges: std::collections::HashMap<usize, (usize, usize)> = crate::notes::chunk_text(&content, chunk_chars)?
            .into_iter().map(|chunk| (chunk.ordinal, (chunk.char_start, chunk.char_end))).collect();
        let mut chunks: Vec<(i64, String)> = Vec::new();
        {
            let mut stmt = tx.prepare("SELECT id,payload_json FROM records WHERE kind=?1")?;
            let rows = stmt.query_map([crate::types::RecordKind::Chunk.code()], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
            for row in rows {
                let (id, payload) = row?;
                let value: serde_json::Value = serde_json::from_str(&payload)?;
                if value.get("note_id").and_then(|v| v.as_i64()) == Some(note_id) { chunks.push((id, payload)); }
            }
        }
        for (id, payload) in chunks {
            let mut value: serde_json::Value = serde_json::from_str(&payload)?;
            let ordinal = value.get("ordinal").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let (char_start, char_end) = ranges.get(&ordinal).copied().unwrap_or((0, 0));
            if let Some(object) = value.as_object_mut() {
                object.remove("content");
                object.insert("char_start".into(), serde_json::json!(char_start));
                object.insert("char_end".into(), serde_json::json!(char_end));
            }
            tx.execute("UPDATE records SET payload_json=?2 WHERE id=?1", rusqlite::params![id, serde_json::to_string(&value)?])?;
        }
    }
    Ok(())
}

pub(crate) fn initialize(conn: &mut Connection) -> Result<()> {
    let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
    if version != 0 && !(2..=SCHEMA_VERSION).contains(&version) {
        return Err(Error::SchemaVersion { found: version, supported: SCHEMA_VERSION });
    }
    let application: i64 = conn.pragma_query_value(None, "application_id", |r| r.get(0))?;
    let tables: i64 = conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'", [], |r| r.get(0))?;
    if (application != 0 && application != APPLICATION_ID) || (application == 0 && tables > 0) {
        return Err(Error::Conflict("this is not a p-memory database; use the legacy importer".into()));
    }
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;")?;
    if version == 0 {
        let tx = conn.transaction()?;
        tx.execute_batch(include_str!("schema.sql"))?;
        tx.pragma_update(None, "application_id", APPLICATION_ID)?;
        tx.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        tx.commit()?;
    } else {
        migrate(conn, version)?;
    }
    Ok(())
}

/// 逐版本前滚，每一步都在独立事务里：任何一步失败都不会留下半升级的库。
fn migrate(conn: &mut Connection, mut version: i64) -> Result<()> {
    while version < SCHEMA_VERSION {
        let tx = conn.transaction()?;
        match version {
            // v2 -> v3：向量的磁盘编码标识；既有行全是 f32。
            2 => { tx.execute_batch("ALTER TABLE embedding_spaces ADD COLUMN encoding TEXT NOT NULL DEFAULT 'f32'")?; }
            // v3 -> v4：谓词元规则表。
            3 => { tx.execute_batch(MIGRATION_3_TO_4)?; }
            // v4 -> v5：可检索正文交给 Tantivy（索引侧重建即可），SQLite 删掉两条派生文本列；
            // 切片 payload 去 content，改存字符区间，正文由笔记原文派生。
            4 => { tx.execute_batch(MIGRATION_4_TO_5)?; migrate_chunk_payloads(&tx)?; }
            other => return Err(Error::SchemaVersion { found: other, supported: SCHEMA_VERSION }),
        }
        version += 1;
        tx.pragma_update(None, "user_version", version)?;
        tx.commit()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个历史版本的库：先用当前 schema 建全量，再按需回退到目标版本。
    fn legacy_db(directory: &std::path::Path, version: i64, rollback: &str) {
        let conn = Connection::open(directory.join("store.sqlite3")).unwrap();
        conn.execute_batch(include_str!("schema.sql")).unwrap();
        conn.execute_batch(rollback).unwrap();
        // v5 之前 records 还带着两条派生文本列；回退到旧版本时补回，供 v4→v5 迁移验证。
        if version < 5 {
            conn.execute_batch("ALTER TABLE records ADD COLUMN search_text TEXT NOT NULL DEFAULT ''; \
                ALTER TABLE records ADD COLUMN embedding_text TEXT NOT NULL DEFAULT '';").unwrap();
        }
        conn.pragma_update(None, "application_id", APPLICATION_ID).unwrap();
        conn.pragma_update(None, "user_version", version).unwrap();
    }

    #[test]
    fn rolls_v3_forward_to_current() {
        let dir = tempfile::tempdir().unwrap();
        legacy_db(dir.path(), 3, "DROP TABLE predicate_rules; DELETE FROM strings WHERE text='sys:same_as';");
        let kb = crate::KnowledgeBase::open(dir.path()).unwrap();
        assert_eq!(kb.health().unwrap().schema_version, SCHEMA_VERSION);
        // 迁移后谓词规则表可用，且内置 sys:same_as 已就位
        kb.graph().set_predicate_rule("父亲", Some("子女"), false).unwrap();
    }

    #[test]
    fn rolls_v2_forward_adding_encoding_and_rules() {
        let dir = tempfile::tempdir().unwrap();
        legacy_db(dir.path(), 2, "DROP TABLE predicate_rules; DELETE FROM strings WHERE text='sys:same_as'; ALTER TABLE embedding_spaces DROP COLUMN encoding;");
        let kb = crate::KnowledgeBase::open(dir.path()).unwrap();
        assert_eq!(kb.health().unwrap().schema_version, SCHEMA_VERSION);
        kb.graph().set_predicate_rule("同事", None, true).unwrap();
    }

    /// v4 形态的笔记 + 切片（带 content 与两条派生列）前滚到当前版本：
    /// 两条派生列消失，切片 payload 改存字符区间，正文可由笔记原文派生。
    #[test]
    fn rolls_v4_forward_dropping_derived_columns() {
        let dir = tempfile::tempdir().unwrap();
        legacy_db(dir.path(), 4, "");
        {
            let conn = Connection::open(dir.path().join("store.sqlite3")).unwrap();
            conn.execute_batch(r#"
                INSERT INTO strings(id,text) VALUES (10,'ns'),(11,'sc'),(12,'a.md');
                INSERT INTO records(id,namespace_id,kind,scope_id,created_at_us,updated_at_us,revision,metadata_json,evidence_json,search_text,embedding_text,fingerprint,payload_json)
                VALUES (1,10,4,11,1,1,1,'{}','[]','T','T','nf','{"source":"a.md","title":"T","content":"甲乙丙","source_revision":"x","chunk_chars":220,"chunk_count":1}');
                INSERT INTO notes(record_id,namespace_id,scope_id,source_id) VALUES (1,10,11,12);
                INSERT INTO records(id,namespace_id,kind,scope_id,created_at_us,updated_at_us,revision,metadata_json,evidence_json,search_text,embedding_text,fingerprint,payload_json)
                VALUES (2,10,5,11,1,1,1,'{}','[]','T','T','cfp','{"note_id":1,"ordinal":0,"offset":1,"limit":1,"content":"甲乙丙"}');
                INSERT INTO chunks(record_id,note_id,ordinal,"offset","limit",fingerprint) VALUES (2,1,0,1,1,'cfp');
            "#).unwrap();
        }
        let kb = crate::KnowledgeBase::open(dir.path()).unwrap();
        assert_eq!(kb.health().unwrap().schema_version, SCHEMA_VERSION);
        let conn = Connection::open(dir.path().join("store.sqlite3")).unwrap();
        let leftover: i64 = conn.query_row("SELECT COUNT(*) FROM pragma_table_info('records') WHERE name IN ('search_text','embedding_text')", [], |r| r.get(0)).unwrap();
        assert_eq!(leftover, 0, "派生文本列应在迁移中删除");
        let payload: String = conn.query_row("SELECT payload_json FROM records WHERE id=2", [], |r| r.get(0)).unwrap();
        let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert!(value.get("content").is_none(), "切片 payload 不再存正文");
        assert_eq!(value.get("char_start").and_then(|v| v.as_u64()), Some(0));
        assert_eq!(value.get("char_end").and_then(|v| v.as_u64()), Some(3));
    }
}

