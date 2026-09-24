-- 向量外挂库：与主库同目录，独立文件、独立 WAL、独立写连接。
-- 它是派生检索索引，自带路由字段；加载分区时纯读本库，不回主库。
CREATE TABLE IF NOT EXISTS embeddings (
    space_id TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    namespace TEXT NOT NULL,
    scope TEXT NOT NULL,
    kind INTEGER NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    note_id INTEGER NOT NULL DEFAULT 0,
    fingerprint TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY(space_id, record_id)
);
CREATE INDEX IF NOT EXISTS embeddings_partition ON embeddings(space_id, namespace, scope);
CREATE INDEX IF NOT EXISTS embeddings_by_record ON embeddings(record_id);

-- 向量域的就绪标记与派生元数据，自闭环在向量库里。
CREATE TABLE IF NOT EXISTS vector_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
