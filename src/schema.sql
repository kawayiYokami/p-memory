CREATE TABLE meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT INTO meta VALUES ('revision', 0), ('indexed_revision', 0);

-- 开放标记的唯一字典：id -> 归一化文本。tag / namespace / scope / entity_type /
-- predicate / attr_key / source 全部引用此表，字符串只存一份。
CREATE TABLE strings (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL UNIQUE);

-- 记录宽表。id 为内部自增主键，与任何外部 ID 无关；kind 为固定枚举整数编码。
CREATE TABLE records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
    kind INTEGER NOT NULL,
    scope_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
    created_at_us INTEGER NOT NULL, updated_at_us INTEGER NOT NULL, revision INTEGER NOT NULL,
    metadata_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE INDEX records_kind ON records(kind, id);
CREATE INDEX records_scope ON records(namespace_id, kind, scope_id, id);

-- 标签关联只存整数 id；字符串只落在 strings。
CREATE TABLE record_tags (
    record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
    PRIMARY KEY(record_id, tag_id)
);
CREATE INDEX record_tags_by_tag ON record_tags(tag_id, record_id);

-- Domain payloads live once in records; relational projections enforce graph integrity.
CREATE TABLE entities (
    record_id INTEGER PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
    name TEXT NOT NULL, entity_type_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT
);
CREATE INDEX entities_name ON entities(name);
CREATE TABLE entity_aliases (
    entity_id INTEGER NOT NULL REFERENCES entities(record_id) ON DELETE CASCADE,
    alias_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
    PRIMARY KEY(entity_id, alias_id)
);
CREATE INDEX aliases_lookup ON entity_aliases(alias_id, entity_id);
CREATE TABLE entity_attributes (
    entity_id INTEGER NOT NULL REFERENCES entities(record_id) ON DELETE CASCADE,
    attr_key_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
    attr_value TEXT NOT NULL,
    PRIMARY KEY(entity_id, attr_key_id, attr_value)
);
CREATE TABLE relations (
    record_id INTEGER PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
    subject_id INTEGER NOT NULL REFERENCES entities(record_id) ON DELETE RESTRICT,
    predicate_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
    object_id INTEGER NOT NULL REFERENCES entities(record_id) ON DELETE RESTRICT
);
CREATE INDEX relations_subject ON relations(subject_id, predicate_id);
CREATE INDEX relations_object ON relations(object_id, predicate_id);
-- 谓词元规则：声明对称谓词、已知逆谓词。物理表只存单向真实三元组，
-- 建图时据此在内存补全反向边（见 graph_search::snapshot），消除双向落盘冗余。
CREATE TABLE predicate_rules (
    predicate_id INTEGER PRIMARY KEY REFERENCES strings(id) ON DELETE CASCADE,
    inverse_predicate_id INTEGER REFERENCES strings(id) ON DELETE RESTRICT,
    is_symmetric INTEGER NOT NULL DEFAULT 0,
    CHECK (is_symmetric IN (0, 1)),
    CHECK (is_symmetric = 0 OR inverse_predicate_id IS NULL)
);
-- 内置规则：sys:same_as 是对称的别名等价关系，建图时用于并查集缩点。
INSERT OR IGNORE INTO strings(text) VALUES ('sys:same_as');
INSERT OR IGNORE INTO predicate_rules(predicate_id, is_symmetric) SELECT id, 1 FROM strings WHERE text='sys:same_as';

-- 谓词等价词：把「alpha / beta / gamma」这类同义写法登记成一组，按知识领域各存一套。
-- 表由上游提供，库不内置任何领域数据；只用于查询期扩散，不改谓词的落盘写法。
-- 同组词共享 canonical_id（组内代表词，自己那行指向自身），据此反查整组。
CREATE TABLE predicate_equivalents (
    namespace_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE CASCADE,
    predicate_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE CASCADE,
    canonical_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE CASCADE,
    PRIMARY KEY (namespace_id, predicate_id)
);
CREATE INDEX predicate_equivalents_group ON predicate_equivalents(namespace_id, canonical_id);

CREATE TABLE event_participants (
    event_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(record_id) ON DELETE RESTRICT,
    PRIMARY KEY(event_id, entity_id)
);
CREATE INDEX events_by_entity ON event_participants(entity_id, event_id);

-- 笔记持有路径/标题；切片不再重复携带 source/title，只经 note_id 关联取回。
-- 路径是笔记自己的一列、逐字符原样：它既用来读文件、也用来定位同一条，不进标签字典。
-- name 是写入时取自文件名的标题（file_stem）：索引里那一列直接读它，不再事后拆路径。
CREATE TABLE notes (
    record_id INTEGER PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
    namespace_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
    scope_id INTEGER NOT NULL REFERENCES strings(id) ON DELETE RESTRICT,
    path TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    UNIQUE(namespace_id, scope_id, path)
);
CREATE TABLE chunks (
    record_id INTEGER PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
    note_id INTEGER NOT NULL REFERENCES notes(record_id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL, "offset" INTEGER NOT NULL, "limit" INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    UNIQUE(note_id, ordinal)
);
CREATE INDEX chunks_by_fingerprint ON chunks(note_id, fingerprint);

-- 知识领域的笔记根目录。登记之后写进来的笔记路径减掉它、存相对路径，
-- 相对路径按段拆出的标签挂到这篇的每一条切片上；没登记的领域维持原样。
CREATE TABLE namespace_roots (
    namespace_id INTEGER PRIMARY KEY REFERENCES strings(id) ON DELETE CASCADE,
    root TEXT NOT NULL
);

CREATE TABLE embedding_spaces (
    id TEXT PRIMARY KEY, model TEXT NOT NULL, dimension INTEGER NOT NULL, text_version INTEGER NOT NULL,
    encoding TEXT NOT NULL DEFAULT 'sq8'
);
-- 向量住进独立库 vectors.sqlite3，与 Tantivy 同级：派生索引、自包含路由、单向消费、零反写主库。
-- 主库只留 embedding_spaces（空间定义），向量行本身在外挂库。
CREATE TABLE import_runs (
    source_id TEXT PRIMARY KEY, source_fingerprint TEXT NOT NULL, report_json TEXT NOT NULL
);
