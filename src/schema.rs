use crate::{Error, Result};
use rusqlite::Connection;

pub(crate) const SCHEMA_VERSION: i64 = 4;
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
}

