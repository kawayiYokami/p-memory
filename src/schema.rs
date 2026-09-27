use crate::{Error, Result};
use rusqlite::Connection;

pub(crate) const SCHEMA_VERSION: i64 = 15;
pub(crate) const APPLICATION_ID: i64 = 0x5041494d;

/// 建库或识别既有库。schema.sql 是最新结构定义；已发布的旧版本只保留一次性前滚迁移。
/// 库版本非 0 且不等于 `SCHEMA_VERSION` 一律拒绝打开。
pub(crate) fn initialize(conn: &mut Connection) -> Result<()> {
    let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
    if version != 0 && version != SCHEMA_VERSION && version != 14 {
        return Err(Error::SchemaVersion { found: version, supported: SCHEMA_VERSION });
    }
    let application: i64 = conn.pragma_query_value(None, "application_id", |r| r.get(0))?;
    let tables: i64 = conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'", [], |r| r.get(0))?;
    if (application != 0 && application != APPLICATION_ID) || (application == 0 && tables > 0) {
        return Err(Error::Conflict("this is not a p-memory database; use the legacy importer".into()));
    }
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;")?;
    match version {
        0 => {
            let tx = conn.transaction()?;
            tx.execute_batch(include_str!("schema.sql"))?;
            tx.pragma_update(None, "application_id", APPLICATION_ID)?;
            tx.pragma_update(None, "user_version", SCHEMA_VERSION)?;
            tx.commit()?;
        }
        14 => migrate_v14_to_v15(conn)?,
        _ => {}
    }
    Ok(())
}

fn migrate_v14_to_v15(conn: &mut Connection) -> Result<()> {
    let tx = conn.transaction()?;
    // 这些索引/列已经没有读取方；先删索引，SQLite 才允许删除被索引引用的列。
    tx.execute_batch("DROP INDEX IF EXISTS chunks_by_fingerprint; DROP INDEX IF EXISTS record_tags_by_tag;")?;
    tx.execute_batch("ALTER TABLE records DROP COLUMN fingerprint; ALTER TABLE chunks DROP COLUMN fingerprint;")?;
    // record_tags 只保留 record_id -> tag_id 这一条主库关联，主键 B-tree 直接承载表体。
    tx.execute_batch("CREATE TABLE record_tags_new (
        record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
        tag_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
        PRIMARY KEY(record_id, tag_id)
    ) WITHOUT ROWID;
    INSERT INTO record_tags_new(record_id, tag_id) SELECT record_id, tag_id FROM record_tags;
    DROP TABLE record_tags;
    ALTER TABLE record_tags_new RENAME TO record_tags;")?;
    tx.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_v14_columns_indexes_and_record_tags() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON;
            CREATE TABLE strings(id INTEGER PRIMARY KEY, text TEXT NOT NULL UNIQUE);
            CREATE TABLE records(id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL, payload_json TEXT NOT NULL);
            CREATE TABLE chunks(record_id INTEGER PRIMARY KEY, note_id INTEGER NOT NULL, fingerprint TEXT NOT NULL);
            CREATE INDEX chunks_by_fingerprint ON chunks(note_id, fingerprint);
            CREATE TABLE record_tags(record_id INTEGER NOT NULL, tag_id INTEGER NOT NULL, PRIMARY KEY(record_id, tag_id));
            CREATE INDEX record_tags_by_tag ON record_tags(tag_id, record_id);
            INSERT INTO strings VALUES (1, 'tag');
            INSERT INTO records VALUES (10, 'old', '{}');
            INSERT INTO chunks VALUES (11, 10, 'old');
            INSERT INTO record_tags VALUES (10, 1);
            PRAGMA application_id=0x5041494d;
            PRAGMA user_version=14;").unwrap();

        initialize(&mut conn).unwrap();

        assert_eq!(conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0)).unwrap(), 15);
        assert!(!conn.prepare("SELECT fingerprint FROM records").is_ok());
        assert!(!conn.prepare("SELECT fingerprint FROM chunks").is_ok());
        assert_eq!(conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE name IN ('chunks_by_fingerprint','record_tags_by_tag')", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
        assert_eq!(conn.query_row("SELECT tag_id FROM record_tags WHERE record_id=10", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
        let table_sql: String = conn.query_row("SELECT sql FROM sqlite_master WHERE name='record_tags'", [], |r| r.get(0)).unwrap();
        assert!(table_sql.contains("WITHOUT ROWID"));
    }
}
