use crate::{Error, Result};
use rusqlite::Connection;

pub(crate) const SCHEMA_VERSION: i64 = 16;
pub(crate) const APPLICATION_ID: i64 = 0x5041494d;

/// 建库或识别既有库。schema.sql 是最新结构定义；已发布的旧版本只保留一次性前滚迁移。
/// 库版本非 0 且不等于 `SCHEMA_VERSION` 一律拒绝打开。
pub(crate) fn initialize(conn: &mut Connection) -> Result<()> {
    let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
    if version != 0 && version != SCHEMA_VERSION && version != 14 && version != 15 {
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
        14 | 15 => migrate_legacy_to_v16(conn)?,
        _ => {}
    }
    Ok(())
}

/// 一张表有没有这一列。旧库里结构可能停在更早的版本上，迁移只按现状补差。
fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = stmt.query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(names.iter().any(|name| name == column))
}

/// 旧库升到当前版本。v15 那次「删掉冗余指纹、把 record_tags 收成 WITHOUT ROWID」的迁移，
/// 存在版本号已经写成 15、结构却还停在 14 的库，所以这里不假设旧库干净：
/// 每一项都先看现状，缺什么补什么，已经做过的不再动。v14 直接升级也走同一条路径。
fn migrate_legacy_to_v16(conn: &mut Connection) -> Result<()> {
    let tx = conn.transaction()?;
    // 这两个索引在 v15 就该没了；旧库里还留着会挡住后面的删列。
    tx.execute_batch("DROP INDEX IF EXISTS chunks_by_fingerprint; DROP INDEX IF EXISTS record_tags_by_tag;")?;
    for table in ["records", "chunks"] {
        if has_column(&tx, table, "fingerprint")? {
            tx.execute_batch(&format!("ALTER TABLE {table} DROP COLUMN fingerprint;"))?;
        }
    }
    // 标签关联收敛到 WITHOUT ROWID：主键 B-tree 就是表体，不留一份并行的 rowid 表。
    let tags_sql: String = tx.query_row(
        "SELECT COALESCE((SELECT sql FROM sqlite_master WHERE type='table' AND name='record_tags'), '')",
        [], |row| row.get(0))?;
    if !tags_sql.is_empty() && !tags_sql.to_uppercase().contains("WITHOUT ROWID") {
        tx.execute_batch("DROP TABLE IF EXISTS record_tags_new;
            CREATE TABLE record_tags_new (
                record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
                tag_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
                PRIMARY KEY(record_id, tag_id)
            ) WITHOUT ROWID;
            INSERT INTO record_tags_new(record_id, tag_id) SELECT record_id, tag_id FROM record_tags;
            DROP TABLE record_tags;
            ALTER TABLE record_tags_new RENAME TO record_tags;")?;
    }
    // 切片级元数据：调用方自定义切片时随切片落库，旧切片补默认空对象。
    if !has_column(&tx, "chunks", "metadata_json")? {
        tx.execute_batch("ALTER TABLE chunks ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';")?;
    }
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

        assert_eq!(conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0)).unwrap(), SCHEMA_VERSION);
        assert!(conn.prepare("SELECT metadata_json FROM chunks").is_ok());
        assert!(!conn.prepare("SELECT fingerprint FROM records").is_ok());
        assert!(!conn.prepare("SELECT fingerprint FROM chunks").is_ok());
        assert_eq!(conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE name IN ('chunks_by_fingerprint','record_tags_by_tag')", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
        assert_eq!(conn.query_row("SELECT tag_id FROM record_tags WHERE record_id=10", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
        let table_sql: String = conn.query_row("SELECT sql FROM sqlite_master WHERE name='record_tags'", [], |r| r.get(0)).unwrap();
        assert!(table_sql.contains("WITHOUT ROWID"));
    }

    /// 版本号写着 15、结构却还停在 14 的库：那次删列迁移没有真的执行。
    /// 升级不能假设它干净，得把这些残留一并补掉。
    #[test]
    fn migrates_a_15_database_that_still_carries_the_v14_columns() {
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
            PRAGMA user_version=15;").unwrap();

        initialize(&mut conn).unwrap();

        assert_eq!(conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0)).unwrap(), SCHEMA_VERSION);
        assert!(!conn.prepare("SELECT fingerprint FROM records").is_ok());
        assert!(!conn.prepare("SELECT fingerprint FROM chunks").is_ok());
        assert!(conn.prepare("SELECT metadata_json FROM chunks").is_ok());
        assert_eq!(conn.query_row("SELECT metadata_json FROM chunks WHERE record_id=11", [], |r| r.get::<_, String>(0)).unwrap(), "{}");
        assert_eq!(conn.query_row("SELECT tag_id FROM record_tags WHERE record_id=10", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
        let table_sql: String = conn.query_row("SELECT sql FROM sqlite_master WHERE name='record_tags'", [], |r| r.get(0)).unwrap();
        assert!(table_sql.contains("WITHOUT ROWID"));
    }

    /// 真正干净的 v15 库：迁移应当什么都不改，也不留下重建用的临时表。
    #[test]
    fn a_15_database_that_is_already_clean_is_left_alone() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON;
            CREATE TABLE strings(id INTEGER PRIMARY KEY, text TEXT NOT NULL UNIQUE);
            CREATE TABLE records(id INTEGER PRIMARY KEY, payload_json TEXT NOT NULL);
            CREATE TABLE chunks(record_id INTEGER PRIMARY KEY, note_id INTEGER NOT NULL,
                                metadata_json TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE record_tags (
                record_id INTEGER NOT NULL, tag_id INTEGER NOT NULL, PRIMARY KEY(record_id, tag_id)
            ) WITHOUT ROWID;
            INSERT INTO strings VALUES (1, 'tag');
            INSERT INTO records VALUES (10, '{}');
            INSERT INTO chunks VALUES (11, 10, '{\"k\":1}');
            INSERT INTO record_tags VALUES (10, 1);
            PRAGMA application_id=0x5041494d;
            PRAGMA user_version=15;").unwrap();

        initialize(&mut conn).unwrap();

        assert_eq!(conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0)).unwrap(), SCHEMA_VERSION);
        assert_eq!(conn.query_row("SELECT metadata_json FROM chunks WHERE record_id=11", [], |r| r.get::<_, String>(0)).unwrap(), "{\"k\":1}");
        assert_eq!(conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE name='record_tags_new'", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
        let table_sql: String = conn.query_row("SELECT sql FROM sqlite_master WHERE name='record_tags'", [], |r| r.get(0)).unwrap();
        assert!(table_sql.contains("WITHOUT ROWID"));
    }
}
