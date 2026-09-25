use crate::{Error, Result};
use rusqlite::Connection;

pub(crate) const SCHEMA_VERSION: i64 = 14;
pub(crate) const APPLICATION_ID: i64 = 0x5041494d;

/// 建库或识别既有库。schema.sql 是唯一的权威结构定义：没有迁移、没有前滚、没有版本升级路径。
/// 库版本非 0 且不等于 `SCHEMA_VERSION` 一律拒绝打开，交由调用方自行处理。
pub(crate) fn initialize(conn: &mut Connection) -> Result<()> {
    let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
    if version != 0 && version != SCHEMA_VERSION {
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
    }
    Ok(())
}
