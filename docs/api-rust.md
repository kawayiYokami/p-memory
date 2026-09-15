# Rust API

核心 crate 名 `p-memory`，公开入口是 `KnowledgeBase`。以下签名省略 `pub`，`Result<T>` 即 `p_memory::Result<T>`。

## KnowledgeBase

```rust
impl KnowledgeBase {
    fn open(directory: impl AsRef<Path>) -> Result<Self>;
    fn directory(&self) -> &Path;
    fn close(&self) -> Result<()>;

    fn memories(&self)  -> MemoryStore;
    fn graph(&self)     -> GraphStore;
    fn notes(&self)     -> NoteStore;
    fn embeddings(&self) -> EmbeddingStore;

    fn search(&self, request: &SearchRequest) -> Result<SearchResult>;
    fn search_with_context(&self, request: &SearchRequest, limit: usize) -> Result<Vec<ContextualHit>>;
    fn register_reranker<F: Reranker + 'static>(&self, reranker: F) -> Result<()>;
    fn register_reranker_with<F: Reranker + 'static>(&self, reranker: F, options: RerankerOptions) -> Result<()>;
    fn unregister_reranker(&self) -> bool;
    fn reranker_registered(&self) -> bool;
    fn health(&self) -> Result<HealthReport>;
    fn rebuild_indexes(&self) -> Result<HealthReport>;

    fn backup(&self, target: impl AsRef<Path>) -> Result<()>;
    fn restore(snapshot: impl AsRef<Path>, directory: impl AsRef<Path>) -> Result<Self>;
}
```

### 打开与关闭

- `open` 创建目录、抢占 `writer.lock` 独占锁、初始化 schema、打开并恢复全文索引。锁被占用返回 `locked`。
- `KnowledgeBase` 是 `Clone` 的：克隆体**共享同一进程内引擎**——同一个写连接、同一个索引句柄与写锁。
  读取从空闲池取一条只读连接，池空则新建、读完归还（见 [architecture](architecture.md)）。
- `close()` 先同步索引再释放；关闭后任意操作返回 `closed`。

### 并发语义

- **一个数据目录在同一时刻只允许一个进程持有写锁**，同进程内可通过 `Clone` 共享句柄。
- 所有写入走 `mutate`，事务失败即整体回滚；提交与索引同步的先后见 [architecture](architecture.md)。
- 写入提交后会清空向量缓存，保证后续检索看到最新向量。
- **写是串行的，读是并发的**：所有写入共用一把写锁、按到达顺序排队；读取不经过写锁，
  每次从空闲池取一条只读连接（池空则新建），所以同一句柄上的多个读可以真正并行执行。
  跨线程共享句柄请用 `Clone`。

### 备份与恢复

- `backup(target)`：SQLite 在线备份，包含向量。目标文件已存在时拒绝覆盖（先 `create_new` 占位）。
- `restore(snapshot, directory)`：只读打开快照，校验 `application_id` 与 `user_version` 后复制到**新目录**并打开；全文索引从数据重建。目标目录已存在会失败。

### 健康检查

- `health()` 返回 `HealthReport`：schema 版本、`revision` 与 `indexed_revision`、`pending_index_updates`、索引文档数、`PRAGMA quick_check`、外键错误数、各类型记录计数，以及 `embedder_spaces` / `reranker_registered` / `last_degraded`。
- `rebuild_indexes()` 强制重建全文索引并清空向量缓存，随后返回新的健康报告。

### 注册模型回调

- `register_reranker` / `register_reranker_with` 注册重排回调：进程内单例，不绑定向量空间。库在调用前按 `RerankerOptions` 强制截断候选与文本。注册时用样本真跑一遍校验产出条数与有限性；回调当场不可用时无从校验形状，允许绑定，可用性留到检索时降级。
- 嵌入回调在 `EmbeddingStore` 上注册（见下）。

## MemoryStore

```rust
impl MemoryStore {
    fn upsert(&self, input: MemoryInput) -> Result<WriteReceipt<Memory>>;
    fn upsert_many(&self, inputs: &[MemoryInput]) -> Result<WriteReceipt<Vec<Memory>>>;
    fn upsert_by_judgment(&self, input: MemoryInput) -> Result<WriteReceipt<Memory>>;

    fn get(&self, id: i64, filter: &ReadFilter) -> Result<Memory>;
    fn list(&self, request: &PageRequest) -> Result<Page<Memory>>;
    fn delete(&self, id: i64, filter: &ReadFilter) -> Result<WriteReceipt<bool>>;

    fn feedback(&self, request: &FeedbackRequest) -> Result<WriteReceipt<FeedbackReport>>;
    fn decay(&self, filter: &ReadFilter, policy: &DecayPolicy, at: Option<i64>) -> Result<WriteReceipt<DecayReport>>;
}
```

- `upsert`：按 `id` 更新；未给 `id` 则新建。`state = None` 时保留既有状态。
- `upsert_many`：批内**全成功或全回滚**。
- `upsert_by_judgment`：在同一 `namespace`+`scope` 内按归一化论断去重。命中多条同一论断时报 `conflict`（要求改用按 ID 更新）；命中唯一记录时合并 metadata 与 evidence。
- `feedback`：`useful_ids` 必须是 `recalled_ids` 的子集。有用项提升强度/计数/有用分；**非固定保留且处于 T1（`tier0 <= score < tier1`）的未命中项减 1 强度**。返回 `FeedbackReport { recalled, boosted, penalized }`。
- `decay`：对命中的记忆按策略衰减，返回 `DecayReport { decayed, retirement_candidates }`。固定保留的记忆不衰减。详见 [lifecycle](lifecycle.md)。

## GraphStore

```rust
impl GraphStore {
    fn apply_batch(&self, batch: &GraphBatch) -> Result<WriteReceipt<GraphBatchResult>>;

    fn get(&self, kind: RecordKind, id: i64, filter: &ReadFilter) -> Result<Value>;
    fn list(&self, kind: RecordKind, page: &PageRequest) -> Result<Page<Value>>;
    fn resolve(&self, name: &str, filter: &ReadFilter, limit: usize) -> Result<Vec<Entity>>;
    fn neighbors(&self, id: i64, filter: &ReadFilter, limit: usize) -> Result<Neighborhood>;
    fn events_for_entity(&self, id: i64, filter: &ReadFilter, limit: usize) -> Result<Vec<Event>>;

    fn delete(&self, kind: RecordKind, id: i64, filter: &ReadFilter) -> Result<WriteReceipt<bool>>;
}
```

- `apply_batch` 是唯一的图谱写入入口，**单事务**，顺序为实体 → 关系 → 事件，允许批内互相引用。
- 写入单个实体后，系统会刷新引用它的关系/事件的检索文本（因为文中含实体名）；文本变化会同步失效对应向量。
- `resolve` 按归一化别名查实体，返回候选列表（不自动合并）。
- `neighbors` 返回一跳关系与对端实体；标签过滤只约束关系，作用域同时约束端点。
- `get`/`list`/`delete` 只接受 `Entity`/`Relation`/`Event`，其他类型报 `validation`。
- 删除被引用的实体或事件参与者关系，由外键 `RESTRICT` 报 `conflict`；调用方须先解除引用。

## 图搜索

`GraphStore` 提供**内存态图搜索**（算法由 [petgraph](https://github.com/petgraph/petgraph) 提供），随库内置，无需额外开关。

```rust
impl GraphStore {
    fn build_graph(&self, filter: &ReadFilter) -> Result<GraphView>;
    fn ego(&self, root: i64, depth: usize, filter: &ReadFilter, limit: usize) -> Result<Vec<Entity>>;
    fn path(&self, from: i64, to: i64, filter: &ReadFilter) -> Result<Option<Vec<Entity>>>;
    fn strongly_connected(&self, filter: &ReadFilter) -> Result<Vec<Vec<i64>>>;
    fn set_predicate_rule(&self, predicate: &str, inverse: Option<&str>, symmetric: bool) -> Result<WriteReceipt<()>>;
}

// p_memory::graph_search::GraphView
impl GraphView {
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn ego_ids(&self, root: i64, depth: usize, limit: usize) -> Vec<i64>;
    fn path_ids(&self, from: i64, to: i64) -> Option<Vec<i64>>;
    fn component_count(&self) -> usize;
}
```

- `build_graph(filter)`：读一次快照（`filter` 范围内的实体 + 关系），建**有向图**。节点 = 实体记录（含无关系的孤立实体），边 = 关系记录；范围外记录不进图，多命名空间不串。有向边只走出边，反向查询靠 `predicate_rules` 的对称/逆谓词在内存补出虚拟边打通（物理表只存单向真实边）。互为 `sys:same_as` 的实体在查询期由并查集缩点成同一节点，`ego_ids` / `path_ids` 的入参会先折算成代表 id。每次调用重建，无缓存。
- `ego_ids` / `ego`：从 `root` 出发 `depth` 跳内的实体，`limit` 截断。`ego_ids` 返回 `record_id`，`ego` 返回 `Entity`（带 name / aliases / attributes）。
- `path_ids` / `path`：`from` → `to` 的**最短桥接**（`astar`，各边等价）。不连通返回 `None`；`path` 返回路径上的实体（含两端）。
- `component_count`：连通分量数量（`connected_components`，按无向方式算）。
- `strongly_connected`：**有向**强连通环（`tarjan_scc`），只返回大小 > 1 的分量，元素为 `record_id`。
- `set_predicate_rule`：登记谓词元规则——`symmetric` 声明对称谓词（反向即自身），`inverse` 声明逆谓词（反向补一条对偶边，如 `父亲` 的逆是 `子女`），二者互斥。规则只影响后续建图的内存补边。内置 `sys:same_as` 为对称关系。
- 图搜索只服务「要在图上走一步以上」的查询；一层邻接（`neighbors`）仍走 SQL。

> 定位、能力边界与实测提醒见 [graph-search](graph-search.md)。

## NoteStore

```rust
impl NoteStore {
    fn upsert(&self, input: NoteInput) -> Result<WriteReceipt<Note>>;
    fn get(&self, id: i64, filter: &ReadFilter) -> Result<Note>;
    fn list(&self, page: &PageRequest) -> Result<Page<Note>>;
    fn get_chunk(&self, id: i64, filter: &ReadFilter) -> Result<Chunk>;
    fn chunks(&self, note_id: i64, filter: &ReadFilter) -> Result<Vec<Chunk>>;
    fn delete(&self, id: i64, filter: &ReadFilter) -> Result<WriteReceipt<bool>>;
}
```

- `upsert` 按 `(namespace, scope, source)` 定位：来源已存在则替换正文，否则新建。
- 正文替换在**同一事务**内重建切片，删除失效切片及其向量。
- `delete` 先删切片再删笔记。
- `chunk_text(content, target)` 是公开辅助函数，可脱离数据库单独调用。

## EmbeddingStore

```rust
impl EmbeddingStore {
    fn register_space(&self, space: EmbeddingSpace) -> Result<WriteReceipt<EmbeddingSpace>>;
    fn spaces(&self) -> Result<Vec<EmbeddingSpace>>;
    fn register_embedder<F: Embedder + 'static>(&self, space_id: &str, embedder: F) -> Result<()>;
    fn register_embedder_with<F: Embedder + 'static>(&self, space_id: &str, embedder: F, options: EmbedderOptions) -> Result<()>;
    fn unregister_embedder(&self, space_id: &str) -> Result<bool>;
    fn embedder_space(&self, space_id: &str) -> Result<Option<EmbeddingSpace>>;
    fn namespace_vectorization(&self, namespace: &str) -> Result<bool>;
    fn set_namespace_vectorization(&self, namespace: &str, enabled: bool) -> Result<WriteReceipt<bool>>;
    fn sync(&self, space_id: &str, batch: usize) -> Result<WriteReceipt<SyncReport>>;
    fn delete_space(&self, id: &str) -> Result<WriteReceipt<bool>>;
}
```

- 空间定义不可变；重复注册相同定义幂等。
- `register_embedder_with` 把回调绑到一个空间，**注册即用样本真跑一遍校验**（维度、有限性、非零范数、条数），不符即拒绝绑定并报 `invalid_vector`。一个向量模型对应一个向量空间。
- `sync(space_id, batch)` 是**内部同步**：库拿该空间注册的回调，把缺失向量的记录分批补齐，宿主不参与向量计算。循环严格三段式——取文本放锁 → 调回调不持锁 → 短事务写回；失败即中断，已写回的批次保留。
- 写入路径也是「写入即向量化」：`upsert` 一条记忆或笔记时，库在写入路径里补齐向量，拿不到回调就跳过留待下次补齐，不阻塞写入、不抛错。
- `namespace_vectorization` / `set_namespace_vectorization` 控制某知识域是否启用向量化，落盘在 `meta` 表。

## 类型与常量

```rust
type Metadata = Map<String, Value>;
fn default_namespace() -> String;   // "default"
fn public_scope() -> String;        // "public"
```

`SearchRequest`、`SearchHit`、`SearchResult`、`SearchDiagnostics`、`Degrade`、`ContextualHit`、`GraphPrune`、`EmbedderOptions`、`RerankerOptions`、`SyncReport`、`DecayPolicy`、`FeedbackRequest`、`Page`/`PageRequest`/`ReadFilter` 等见 [data-model](data-model.md)；检索流程见 [search](search.md)。

## 错误处理

所有接口返回 `p_memory::Error`（`thiserror`），`Error::code()` 给出稳定字符串码（见 [data-model](data-model.md#错误码)），适合跨语言映射。
