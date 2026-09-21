//! End-to-end fixtures for the offline legacy importers.
//!
//! Each fixture is a throwaway SQLite database shaped after the real source
//! systems, so the tests exercise the actual column
//! mapping without touching a live library.
use p_memory::legacy::{import_legacy, ImportRequest, LegacySource};
use rusqlite::{Connection, OpenFlags};
use std::fs;
use std::path::{Path, PathBuf};

fn source(path: &Path, sql: &str) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(sql).unwrap();
}

fn request(source: LegacySource, database: PathBuf, destination: PathBuf) -> ImportRequest {
    ImportRequest {
        source,
        source_id: "fixture".into(),
        database,
        destination,
        graph_database: None,
        lookup_database: None,
        notes_root: None,
        namespace: "default".into(),
        scope: "public".into(),
        dry_run: true,
    }
}

fn count(store: &Path, table: &str) -> i64 {
    let conn = Connection::open_with_flags(store, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0)).unwrap()
}

/// P-ai: flat memories plus tags, no graph, only a file-level note index.
fn pai_fixture(dir: &Path) -> PathBuf {
    let path = dir.join("memory_store.db");
    source(&path, r#"
        CREATE TABLE memory_record(id TEXT PRIMARY KEY, memory_type TEXT, judgment TEXT, reasoning TEXT,
            strength INTEGER, is_active INTEGER, memory_scope TEXT, useful_count INTEGER, useful_score REAL,
            last_recalled_at TEXT, last_decay_at TEXT, created_at TEXT, updated_at TEXT, owner_agent_id TEXT);
        CREATE TABLE global_tag(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE memory_tag_rel(memory_id TEXT, tag_id INTEGER);
        CREATE TABLE note_index_record(source_id TEXT, note_short_id INTEGER, file_id TEXT, source_file_path TEXT,
            heading_h1 TEXT, heading_h2 TEXT, heading_h3 TEXT, heading_h4 TEXT, heading_h5 TEXT, heading_h6 TEXT,
            total_lines INTEGER, updated_at TEXT);
        INSERT INTO memory_record VALUES('m1','knowledge','Rust 是系统编程语言','因为内存安全',5,1,'public',3,0.5,NULL,NULL,'2026-02-21T13:11:40.51375Z','2026-06-13T09:26:45Z',NULL);
        INSERT INTO memory_record VALUES('m2','event','遥酱测试了记忆工具',NULL,1,0,'agent-x',0,0,NULL,NULL,'2026-02-21T13:11:40.51375Z','2026-06-12T23:58:27Z','agent-x');
        INSERT INTO memory_record VALUES('m3','skill','会写测试',NULL,2,0,'public',1,0.2,NULL,NULL,'2026-02-21T13:11:40.51375Z','2026-02-21T13:11:40.51375Z',NULL);
        INSERT INTO global_tag VALUES(1,'rust');
        INSERT INTO global_tag VALUES(2,'测试');
        INSERT INTO memory_tag_rel VALUES('m1',1);
        INSERT INTO memory_tag_rel VALUES('m1',2);
    "#);
    path
}

#[test]
fn pai_import_memories_tags_and_repeat_guard() {
    let dir = tempfile::tempdir().unwrap();
    let src = pai_fixture(dir.path());
    let destination = dir.path().join("dest");
    let mut req = request(LegacySource::Pai, src, destination.clone());

    let preview = import_legacy(&req).unwrap();
    assert_eq!(preview.counts.get("memory"), Some(&3));
    assert!(preview.conflicts.is_empty(), "{:?}", preview.conflicts);
    assert!(preview.missing_sources.is_empty());
    assert!(!preview.applied);

    req.dry_run = false;
    let report = import_legacy(&req).unwrap();
    assert!(report.applied);
    assert!(report.index_ready, "{:?}", report.index_error);
    let store = destination.join("store.sqlite3");
    assert_eq!(count(&store, "records"), 3);
    assert_eq!(count(&store, "record_tags"), 2);

    let again = import_legacy(&req).unwrap();
    assert!(again.already_imported && !again.applied);
}

#[test]
fn import_rejects_nonempty_destination() {
    let dir = tempfile::tempdir().unwrap();
    let src = pai_fixture(dir.path());
    let destination = dir.path().join("dest");
    fs::create_dir_all(&destination).unwrap();
    fs::write(destination.join("occupied.txt"), "keep").unwrap();

    let mut req = request(LegacySource::Pai, src, destination);
    req.dry_run = false;
    let report = import_legacy(&req).unwrap();
    assert!(!report.applied);
    assert!(report.conflicts.iter().any(|c| c.contains("empty directory")), "{:?}", report.conflicts);
}

/// Flat memories plus a canonical lookup graph, with a dangling
/// relation endpoint and a timezone-less timestamp — both seen in the real data.
#[test]
fn world_tree_import_skips_dangling_and_parses_zoneless_timestamps() {
    let dir = tempfile::tempdir().unwrap();
    let mem = dir.path().join("world_tree.db");
    source(&mem, r#"
        CREATE TABLE world_tree_memory(id TEXT PRIMARY KEY, judgment TEXT, memory_type TEXT, reasoning TEXT,
            created_at TEXT, updated_at TEXT, metadata_json TEXT);
        CREATE TABLE world_tree_tag(id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE world_tree_memory_tag(memory_id TEXT, tag_id TEXT);
        INSERT INTO world_tree_memory VALUES('wt1','记忆一条','world_tree',NULL,'2024-01-15T08:00:00Z','2026-03-15T03:10:49.738281Z','{"domain":"demo"}');
        INSERT INTO world_tree_memory VALUES('wt2','无时区时间戳一条','world_tree',NULL,'2026-03-21T00:05:55.250049','2026-03-21T00:05:55.250049','{"domain":"demo"}');
    "#);
    let graph = dir.path().join("world_tree_graph.db");
    source(&graph, r#"
        CREATE TABLE world_tree_graph(id TEXT PRIMARY KEY, judgment TEXT, graph_type TEXT, reasoning TEXT,
            created_at TEXT, updated_at TEXT, metadata_json TEXT);
        CREATE TABLE world_tree_graph_tag(id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE world_tree_graph_tag_map(graph_id TEXT, tag_id TEXT);
        INSERT INTO world_tree_graph VALUES('g1','图谱抽出一条','graph',NULL,'2026-03-21T00:05:55.250049','2026-03-21T00:05:55.250049','{"domain":"demo"}');
    "#);
    let lookup = dir.path().join("world_tree_lookup.db");
    source(&lookup, r#"
        CREATE TABLE entities(entity_id TEXT, domain TEXT, canonical_name TEXT, entity_type TEXT, summary TEXT, doc_hashes_json TEXT);
        CREATE TABLE entity_aliases(domain TEXT, term TEXT, term_normalized TEXT, entity_id TEXT);
        CREATE TABLE entity_attributes(entity_id TEXT, attr_key TEXT, attr_value TEXT, display_order INTEGER);
        CREATE TABLE relations(relation_id TEXT, domain TEXT, subject_entity_id TEXT, predicate TEXT, object_entity_id TEXT, reason TEXT, confidence REAL, doc_hashes_json TEXT);
        CREATE TABLE events(event_id TEXT, domain TEXT, event_name TEXT, event_type TEXT, summary TEXT, reason TEXT, doc_hashes_json TEXT, doc_titles_json TEXT);
        CREATE TABLE event_aliases(domain TEXT, term TEXT, term_normalized TEXT, event_id TEXT);
        CREATE TABLE event_participants(event_id TEXT, entity_id TEXT);
        INSERT INTO entities VALUES('e1','demo','Alice','person','','');
        INSERT INTO entities VALUES('e2','demo','Bob','person','','');
        INSERT INTO entity_aliases VALUES('demo','A','a','e1');
        INSERT INTO entity_attributes VALUES('e1','color','red',0);
        INSERT INTO relations VALUES('r1','demo','e1','knows','e2','','0.9','');
        INSERT INTO relations VALUES('r2','demo','e1','knows','ghost','','0.9','');
        INSERT INTO events VALUES('ev1','demo','alliance','','','','','');
        INSERT INTO event_participants VALUES('ev1','e1');
        INSERT INTO event_participants VALUES('ev1','ghost');
    "#);
    let destination = dir.path().join("dest");
    let mut req = request(LegacySource::WorldTree, mem, destination.clone());
    req.graph_database = Some(graph);
    req.lookup_database = Some(lookup);
    req.namespace = "demo".into();

    let preview = import_legacy(&req).unwrap();
    assert_eq!(preview.counts.get("memory"), Some(&3), "2 world-tree + 1 graph");
    assert_eq!(preview.counts.get("entity"), Some(&2));
    assert_eq!(preview.counts.get("relation"), Some(&1), "dangling relation is skipped");
    assert_eq!(preview.counts.get("event"), Some(&1));
    assert_eq!(preview.conflicts.len(), 2, "{:?}", preview.conflicts);
    assert!(preview.conflicts.iter().any(|c| c.contains("skipped relation")));
    assert!(preview.conflicts.iter().any(|c| c.contains("participant")));

    req.dry_run = false;
    assert!(import_legacy(&req).unwrap().applied);
    let store = destination.join("store.sqlite3");
    assert_eq!(count(&store, "entities"), 2);
    assert_eq!(count(&store, "relations"), 1);
    assert_eq!(count(&store, "event_participants"), 1);
}

/// angel_memory: epoch-second memories plus a note index whose bodies live on
/// disk under `notes_root`; slices are re-chunked by the core, not copied.
#[test]
fn angel_import_notes_need_notes_root_else_reported_missing() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("simple_memory.db");
    source(&src, r#"
        CREATE TABLE memory_records(id TEXT PRIMARY KEY, memory_type TEXT, judgment TEXT, reasoning TEXT,
            strength INTEGER, is_active INTEGER, memory_scope TEXT, created_at REAL, updated_at REAL,
            useful_count INTEGER, useful_score REAL, last_recalled_at REAL, last_decay_at REAL);
        CREATE TABLE global_tags(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE memory_tag_rel(memory_id TEXT, tag_id INTEGER);
        CREATE TABLE note_index_records(source_id TEXT, note_short_id INTEGER, file_id TEXT, source_file_path TEXT,
            heading_h1 TEXT, heading_h2 TEXT, heading_h3 TEXT, heading_h4 TEXT, heading_h5 TEXT, heading_h6 TEXT,
            total_lines INTEGER, updated_at REAL);
        INSERT INTO memory_records VALUES('a1','知识记忆','angel 记忆一条','reason',3,1,'public',1780029185.0,1780029185.0,0,0,NULL,NULL);
        INSERT INTO global_tags VALUES(1,'知识');
        INSERT INTO memory_tag_rel VALUES('a1',1);
        INSERT INTO note_index_records VALUES('note_file_1',0,'1','.angel/note/n1.md',NULL,NULL,NULL,NULL,NULL,NULL,10,1780029185.0);
    "#);

    let preview = import_legacy(&request(LegacySource::AngelMemory, src.clone(), dir.path().join("dest"))).unwrap();
    assert_eq!(preview.counts.get("memory"), Some(&1));
    assert_eq!(preview.missing_sources.len(), 1, "note body is missing without notes_root");
    assert_eq!(preview.missing_sources[0], ".angel/note/n1.md");

    let root = dir.path().join("raw");
    fs::create_dir_all(root.join(".angel/note")).unwrap();
    fs::write(root.join(".angel/note/n1.md"), "# 标题\n正文第一段\n正文第二段\n").unwrap();
    let destination = dir.path().join("dest2");
    let mut req = request(LegacySource::AngelMemory, src, destination.clone());
    req.notes_root = Some(root);

    let preview = import_legacy(&req).unwrap();
    assert_eq!(preview.counts.get("note"), Some(&1));
    assert!(preview.missing_sources.is_empty(), "{:?}", preview.missing_sources);
    assert!(preview.counts.get("chunk").copied().unwrap_or(0) >= 1);

    req.dry_run = false;
    assert!(import_legacy(&req).unwrap().applied);
    assert_eq!(count(&destination.join("store.sqlite3"), "notes"), 1);
}
