use crate::{Error, Result};
use rusqlite::Connection;

pub(crate) const SCHEMA_VERSION: i64 = 8;
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

/// v4 -> v5：删掉两条派生文本列。切片 payload 里那副本正文由 `strip_chunk_payload_keys` 摘除。
const MIGRATION_4_TO_5: &str = "
ALTER TABLE records DROP COLUMN search_text;
ALTER TABLE records DROP COLUMN embedding_text;
";

/// 把切片 payload 里的若干键摘掉。切片正文只在索里存一份，库内副本一律不要。
fn strip_chunk_payload_keys(tx: &rusqlite::Transaction<'_>, keys: &[&str]) -> Result<()> {
    let mut rows: Vec<(i64, String)> = Vec::new();
    {
        let mut stmt = tx.prepare("SELECT id,payload_json FROM records WHERE kind=?1")?;
        for row in stmt.query_map([crate::types::RecordKind::Chunk.code()], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
            rows.push(row?);
        }
    }
    for (id, raw) in rows {
        let mut value: serde_json::Value = serde_json::from_str(&raw)?;
        if let Some(object) = value.as_object_mut() {
            for key in keys { object.remove(*key); }
        }
        tx.execute("UPDATE records SET payload_json=?2 WHERE id=?1", rusqlite::params![id, serde_json::to_string(&value)?])?;
    }
    Ok(())
}

/// 把笔记 payload 里的正文与派生字段摘掉，只留切片粒度。
/// 路径权威在 `notes` 表（→strings），标题由路径文件名派生，正文在宿主文件里。
fn migrate_note_payloads(tx: &rusqlite::Transaction<'_>) -> Result<()> {
    let mut rows: Vec<(i64, String)> = Vec::new();
    {
        let mut stmt = tx.prepare("SELECT id,payload_json FROM records WHERE kind=?1")?;
        for row in stmt.query_map([crate::types::RecordKind::Note.code()], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
            rows.push(row?);
        }
    }
    for (id, raw) in rows {
        let mut value: serde_json::Value = serde_json::from_str(&raw)?;
        if let Some(object) = value.as_object_mut() {
            for key in ["content", "title", "chunk_count", "source_revision", "source"] { object.remove(key); }
        }
        tx.execute("UPDATE records SET payload_json=?2 WHERE id=?1", rusqlite::params![id, serde_json::to_string(&value)?])?;
    }
    Ok(())
}

/// v6 -> v7：笔记路径从 `strings` 标签字典改存本表 `path` 列（原样）。
/// 路径是笔记自己的列，不是标签：进标签字典既无复用价值，又会被归一化改写
/// （全角折半角、字母小写），改一个字符就指向另一个不存在的路径。
/// 旧库里的字符串已被改写，只能原样搬过来，由上游按真实路径重新 `upsert_file` 覆盖。
fn migrate_note_paths(tx: &rusqlite::Transaction<'_>) -> Result<()> {
    tx.execute_batch("
        CREATE TABLE notes_new (
            record_id INTEGER PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
            namespace_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
            scope_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
            path TEXT NOT NULL,
            UNIQUE(namespace_id, scope_id, path)
        );
        INSERT INTO notes_new(record_id,namespace_id,scope_id,path)
            SELECT n.record_id,n.namespace_id,n.scope_id,s.text
            FROM notes n JOIN strings s ON s.id=n.source_id;
        DROP TABLE notes;
        ALTER TABLE notes_new RENAME TO notes;
    ")?;
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
/// 个别步骤要重建表（v6→v7 的 notes），SQLite 官方流程要求重建表期间关闭外键——
/// 否则 DROP 掉被 chunks 引用的 notes 会被拦下；pragma 只能在事务外改，故包一层。
fn migrate(conn: &mut Connection, version: i64) -> Result<()> {
    conn.pragma_update(None, "foreign_keys", false)?;
    let outcome = migrate_steps(conn, version);
    conn.pragma_update(None, "foreign_keys", true)?;
    outcome
}

fn migrate_steps(conn: &mut Connection, mut version: i64) -> Result<()> {
    while version < SCHEMA_VERSION {
        let tx = conn.transaction()?;
        match version {
            // v2 -> v3：向量的磁盘编码标识；既有行全是 f32。
            2 => { tx.execute_batch("ALTER TABLE embedding_spaces ADD COLUMN encoding TEXT NOT NULL DEFAULT 'f32'")?; }
            // v3 -> v4：谓词元规则表。
            3 => { tx.execute_batch(MIGRATION_3_TO_4)?; }
            // v4 -> v5：可检索正文交给 Tantivy（索引侧重建即可），SQLite 删掉两条派生文本列，
            // 切片 payload 里那副本正文一并摘除。
            4 => { tx.execute_batch(MIGRATION_4_TO_5)?; strip_chunk_payload_keys(&tx, &["content"])?; }
            // v5 -> v6：笔记 payload 不再留正文与派生字段（路径只在 notes 表，标题由路径派生），
            // 正文改由宿主文件承载；旧库里的副本就地摘除。
            5 => { migrate_note_payloads(&tx)?; }
            // v6 -> v7：笔记路径从 strings 标签字典改存本表 path 列（原样）。
            6 => { migrate_note_paths(&tx)?; }
            // v7 -> v8：切片 payload 摘掉字符区间。写入时切好的正文直接进索引、成为唯一副本，
            // 从原文按区间二次取正文这条路径整个撤掉。
            7 => { strip_chunk_payload_keys(&tx, &["char_start", "char_end"])?; }
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
        // v7 之前 notes 用 source_id 引用标签字典；回退到旧版本时换回该形态，供 v6→v7 迁移验证。
        if version < 7 {
            conn.execute_batch("CREATE TABLE notes_legacy (
                    record_id INTEGER PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
                    namespace_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
                    scope_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
                    source_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
                    UNIQUE(namespace_id, scope_id, source_id)
                );
                DROP TABLE notes;
                ALTER TABLE notes_legacy RENAME TO notes;").unwrap();
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
    /// 两条派生列消失，切片 payload 不再留正文，也不留字符区间。
    #[test]
    fn rolls_v4_forward_dropping_derived_columns() {
        let dir = tempfile::tempdir().unwrap();
        legacy_db(dir.path(), 4, "");
        // 迁移后会按权威数据全量重建索引，笔记正文从宿主文件取：造一个真实文件。
        let note_file = dir.path().join("a.md");
        std::fs::write(&note_file, "甲乙丙").unwrap();
        let source_path = note_file.to_string_lossy().replace('\\', "/");
        {
            let conn = Connection::open(dir.path().join("store.sqlite3")).unwrap();
            conn.execute_batch(&format!(r#"
                INSERT INTO strings(id,text) VALUES (10,'ns'),(11,'sc'),(12,'{source_path}');
                INSERT INTO records(id,namespace_id,kind,scope_id,created_at_us,updated_at_us,revision,metadata_json,evidence_json,search_text,embedding_text,fingerprint,payload_json)
                VALUES (1,10,4,11,1,1,1,'{{}}','[]','T','T','nf','{{"source":"{source_path}","title":"T","content":"甲乙丙","source_revision":"x","chunk_chars":220,"chunk_count":1}}');
                INSERT INTO notes(record_id,namespace_id,scope_id,source_id) VALUES (1,10,11,12);
                INSERT INTO records(id,namespace_id,kind,scope_id,created_at_us,updated_at_us,revision,metadata_json,evidence_json,search_text,embedding_text,fingerprint,payload_json)
                VALUES (2,10,5,11,1,1,1,'{{}}','[]','T','T','cfp','{{"note_id":1,"ordinal":0,"offset":1,"limit":1,"content":"甲乙丙"}}');
                INSERT INTO chunks(record_id,note_id,ordinal,"offset","limit",fingerprint) VALUES (2,1,0,1,1,'cfp');
            "#)).unwrap();
        }
        let kb = crate::KnowledgeBase::open(dir.path()).unwrap();
        assert_eq!(kb.health().unwrap().schema_version, SCHEMA_VERSION);
        let conn = Connection::open(dir.path().join("store.sqlite3")).unwrap();
        let leftover: i64 = conn.query_row("SELECT COUNT(*) FROM pragma_table_info('records') WHERE name IN ('search_text','embedding_text')", [], |r| r.get(0)).unwrap();
        assert_eq!(leftover, 0, "派生文本列应在迁移中删除");
        let payload: String = conn.query_row("SELECT payload_json FROM records WHERE id=2", [], |r| r.get(0)).unwrap();
        let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert!(value.get("content").is_none(), "切片 payload 不再存正文");
        assert!(value.get("char_start").is_none() && value.get("char_end").is_none(), "字符区间随 v8 一并摘除");
        assert_eq!(value.get("offset").and_then(|v| v.as_u64()), Some(1), "行区间保留");
        assert_eq!(value.get("limit").and_then(|v| v.as_u64()), Some(1));
        let note_payload: String = conn.query_row("SELECT payload_json FROM records WHERE id=1", [], |r| r.get(0)).unwrap();
        let note: serde_json::Value = serde_json::from_str(&note_payload).unwrap();
        assert!(note.get("content").is_none(), "笔记 payload 不再存正文");
        assert!(note.get("title").is_none(), "笔记标题由路径派生，不落库");
        assert!(note.get("source").is_none(), "路径只落在 notes 表，payload 不重复");
        assert_eq!(note.get("chunk_chars").and_then(|v| v.as_u64()), Some(220), "只留切片粒度");
        let migrated_path: String = conn.query_row("SELECT path FROM notes WHERE record_id=1", [], |r| r.get(0)).unwrap();
        assert_eq!(migrated_path, source_path, "笔记路径应从标签字典原样迁到 path 列");
    }
}

