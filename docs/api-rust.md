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
    fn search_preset(&self, request: &PresetRequest) -> Result<PresetResult>;
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
- 所有写入走 `mutate`，事务失败即整体回滚；写入不就地索引，由 `update_index()` 一趟追平（见 [architecture](architecture.md)）。
- 写入提交后会清空向量缓存，保证后续检索看到最新向量。
- **写是串行的，读是并发的**：所有写入共用一把写锁、按到达顺序排队；读取不经过写锁，
  每次从空闲池取一条只读连接（池空则新建），所以同一句柄上的多个读可以真正并行执行。
  跨线程共享句柄请用 `Clone`。

### 备份与恢复

- `backup(target)`：SQLite 在线备份，包含向量。目标文件已存在时拒绝覆盖（先 `create_new` 占位）。
- `restore(snapshot, directory)`：只读打开快照，校验 `application_id` 与 `user_version` 后复制到**新目录**并打开；全文索引从数据重建。目标目录已存在会失败。

### 健康检查

- `health()` 返回 `HealthReport`：schema 版本、`revision` 与 `indexed_revision`、`pending_index_updates`、索引文档数、`PRAGMA quick_check`、外键错误数、各类型记录计数，以及 `embedder_spaces` / `reranker_registered` / `last_degraded`。
- `update_index()` 追平写入累积的待办、只提交一次，返回新的健康报告。写入路径不调用它；批量导入后调用一次即可，期间读取走自愈兜底。
- `rebuild_indexes()` 强制重建全文索引并清空向量缓存，随后返回新的健康报告。重建按 record id 分页流式进行、逐批提交，内存不随语料规模增长；进程中途被杀后重开库会从持久游标续跑。
- `rebuild_progress()` 返回一次重建进度快照 `RebuildProgressReport { active, processed, total }`，可在重建进行时从另一线程轮询。

### 注册模型回调

- `register_reranker` / `register_reranker_with` 注册重排回调：进程内单例，不绑定向量空间。库在调用前按 `RerankerOptions` 强制截断候选与文本。注册时用样本真跑一遍校验产出条数与有限性；回调当场不可用时无从校验形状，允许绑定，可用性留到检索时降级。
- 嵌入回调在 `EmbeddingStore` 上注册（见下）。

### 预设检索

`search_preset` 是预先配好的搜索方法，调用方按名字取用：

```rust
enum SearchPreset { Memory, Graph, Notes, Rag, Broad }   // "memory"/"graph"/"notes"/"rag"/"broad"

impl SearchPreset {
    const ALL: [Self; 5];
    fn as_str(self) -> &'static str;
    fn parse(value: &str) -> Result<Self>;     // 别的名字报 validation
    fn uses_memory(self) -> bool;              // memory / rag / broad
    fn uses_graph(self) -> bool;               // graph / rag / broad
    fn uses_notes(self) -> bool;               // notes / broad
}

struct PresetRequest {
    preset: SearchPreset,        // 默认 Rag
    query: String,               // 必填，空即 validation
    filter: ReadFilter,
    embed_space: Option<String>,
    text: bool, vector: bool, rerank: bool,     // 默认全真
    budget: PresetBudget,
    candidate_limit: usize,      // 默认 64，不得为 0
}
struct PresetBudget {            // 都可被调用方覆盖
    memory_chars: usize,         // 默认 2000
    notes_chars: usize,          // 默认 3000
    seed_entities: usize,        // 默认 4，不得为 0；唯一按个数的阈值
    graph_relations_chars: usize,   // 默认 1000
    graph_context_chars: usize,     // 默认 2000
    note_titles: usize,             // 默认 5，书名块想要的条数，也是路径兜底的触发线
}
struct PresetResult {
    preset: SearchPreset,
    memories: Vec<SearchHit>,    // 记忆那一路
    graph: GraphSection,         // 图谱那一路
    notes: NoteSection,          // 笔记那一路
    revision: i64,
    indexed_revision: i64,
    diagnostics: SearchDiagnostics,
}
struct NoteSection {             // 笔记那一路：书名块 / 内容块 / 路径兜底同时返回
    titles: Vec<SearchHit>,      // 文件名命中（name 列），纯全文
    contents: Vec<SearchHit>,    // 正文命中（text 列）
    paths: Vec<SearchHit>,       // 书名块不足时用目录段（path 列）补的，排最后
}
struct GraphSection {
    entities: Vec<Entity>,              // 第一步的种子，按分排，带别名
    relations: Vec<Relation>,           // 第二步命中的关系，按相关度排
    context_relations: Vec<Relation>,   // 第三步铺开的关系，不筛，按 id 升序
    context_events: Vec<Event>,         // 第三步铺开的事件，不筛，按 id 升序
}
```

- 三个字段各自独立排序、各自按字符数封顶，互不挤占；这次没走的那一路是空数组，图谱字段全空。
- 记忆、笔记两路复用 `search`（全文 + 向量 + 可选重排），按字符数截断后返回；图谱那一路按三步流程走，见 [search](search.md#预设检索)。
- 预设带的默认阈值只是起点，实例化时按场景覆盖即可。

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
    fn set_predicate_equivalents(&self, namespace: &str, groups: &[Vec<String>]) -> Result<WriteReceipt<usize>>;
    fn predicate_equivalents(&self, namespace: &str) -> Result<Vec<Vec<String>>>;
    fn expand_query(&self, namespace: &str, text: &str) -> Result<Vec<String>>;
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
- `set_predicate_equivalents`：按知识领域登记谓词等价组（如 `[["丈夫","老公","夫君"]]`），组内第一个是规范词。表由上游提供、库不内置领域数据，登记后持久化；同一个词重复登记会改写它的归属。只管同义，与 `set_predicate_rule` 的方向规则是不同维度。
- `predicate_equivalents`：列出某领域已登记的等价组（每组按文本排序）。
- `expand_query`：查询期扩散——找出 `text` 里出现的登记词，返回它们所在等价组的全部同义词（`text` 里没有登记词就返回空）。全文路与预设检索图谱路内部用同一套扩散；单次返回词元有上限（`EXPAND_QUERY_LIMIT = 64`）。
- 图搜索只服务「要在图上走一步以上」的查询；一层邻接（`neighbors`）仍走 SQL。

> 定位、能力边界与实测提醒见 [graph-search](graph-search.md)。

## NoteStore

```rust
impl NoteStore {
    fn upsert_file(&self, input: NoteFileInput) -> Result<WriteReceipt<Note>>;
    fn set_root(&self, namespace: &str, root: &str) -> Result<()>;
    fn root(&self, namespace: &str) -> Result<Option<String>>;
    fn get(&self, id: i64, filter: &ReadFilter) -> Result<Note>;
    fn list(&self, page: &PageRequest) -> Result<Page<Note>>;
    fn get_chunk(&self, id: i64, filter: &ReadFilter) -> Result<Chunk>;
    fn chunks(&self, note_id: i64, filter: &ReadFilter) -> Result<Vec<Chunk>>;
    fn delete(&self, id: i64, filter: &ReadFilter) -> Result<WriteReceipt<bool>>;
}
```

- `upsert_file` 按给定文件路径同步一篇笔记：库读文件，正文取文件原文，标题取文件名（去扩展名）；路径即身份，同一路径定位到同一笔记。监听与对账在使用方，库只处理给到的这一个文件。
- `set_root` 登记该领域的笔记根目录（必须是一个已存在的目录），`root` 读回来；落了 `namespace_roots` 表。**根目录是写入笔记的前提**：没登记就写入直接报 `validation`；登记之后写入的路径必须是它的子路径，否则同样报 `validation`，库里存减掉根目录的相对路径，`Note.source` 取回时再拼回绝对路径。
- **相对路径拆成标签**：目录段原样、文件名去扩展名，与调用方给的标签合并去重，写到这一篇的每一条切片上（库里 `record_tags`，索引里标签文本与标签 id 各一列）。
- **正文不进库**：笔记 payload 只留切片粒度，切片正文在写入时切好、随文档进全文索引。**笔记记录不进索引**，要文件列表按库里的标签翻笔记。
- 正文替换在**同一事务**内重建切片，删除失效切片及其向量。
- 切片 payload 只存 `note_id` 与行区间；`get_chunk` / `chunks` 返回的 `Chunk.content` 按切片 ID 从索引取回（索引还没提交就先提交一次）。写入时文件缺失直接报错；索引重建时文件缺失只让那批切片正文为空。
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
    fn vectorization(&self, namespace: &str, target: &str) -> Result<bool>;
    fn set_vectorization(&self, namespace: &str, target: &str, enabled: bool) -> Result<WriteReceipt<bool>>;
    fn vector_ready(&self, namespace: &str, space_id: &str, target: &str) -> Result<bool>;
    fn sync(&self, space_id: &str, batch: usize) -> Result<WriteReceipt<SyncReport>>;
    fn delete_space(&self, id: &str) -> Result<WriteReceipt<bool>>;
}
```

- 空间定义不可变；重复注册相同定义幂等。
- `register_embedder_with` 把回调绑到一个空间，**注册即用样本真跑一遍校验**（维度、有限性、非零范数、条数），不符即拒绝绑定并报 `invalid_vector`。一个向量模型对应一个向量空间。
- `sync(space_id, batch)` 是**批次结束后的补齐**，宿主不参与向量计算。它按固定顺序走完三件事：先追平索引（切片正文只存在索引里，不追平就取不到文本）→ 再按缺口分批补齐 → 最后逐档核对缺口，把缺口为 0 的档标成**就绪**。补齐循环严格三段式——取文本放锁 → 调回调不持锁 → 短事务写回；失败即中断，已写回的批次保留、未跑的批次不写。中断即未就绪。
- `vector_ready(namespace, space_id, target)` 读的是「领域 × 向量空间 × 档位」的就绪标记（落盘在 `meta`），`target` 取 `memory` / `graph` / `notes`，三档各自独立记、各自放行：记忆补完了不表示图谱也补完了。未就绪的档在检索里被剔出向量路，其余已就绪的档照常走向量；三档全被剔才记 `vector_not_ready`。写入、删除记录、改总闸或改档位都会让标记当场作废，要重新走一次 `sync` 核对过才会再标。
- 写入路径**不产生向量**：`upsert` 一条记忆或笔记只做两件事——入库与把文档写进索引 writer。一行向量都不算、一次模型都不调，所以写入耗时与模型无关；向量统一留到批次结束调 `sync`。落库后没有人调用 `sync` 的情形由库内后台线程兜底。
- 后台线程在开库时启动、关库时停下并等它收尾。它是**纯事件触发**，不轮询、不设定时：开库、注册或切换向量模型、改档位、写入提交后各叫它一次；一轮补齐跑完若还写出过向量，就接着再跑一轮收掉漏网的。线程没被叫时一直睡在条件变量上。
- `namespace_vectorization` / `set_namespace_vectorization` 是某知识域的**总闸**；`vectorization(ns, target)` / `set_vectorization(ns, target, enabled)` 是域内的**档位开关**，`target` 取 `memory` / `graph` / `notes`，三者互不牵连。读数时没设置过的档位落到内置默认（记忆开、图谱开、笔记关），非法档位名报 `validation`。四者都落盘在 `meta` 表。
- 开关只决定是否生成向量：已有向量保留，检索时按启用档位过滤；总闸关闭时该域既不生成向量、也不走向量路。

## 类型与常量

```rust
type Metadata = Map<String, Value>;
fn default_namespace() -> String;   // "default"
fn public_scope() -> String;        // "public"
```

`SearchRequest`、`SearchHit`、`SearchResult`、`SearchDiagnostics`、`MatchField`、`Degrade`、`ContextualHit`、`GraphPrune`、`NoteSection`、`EmbedderOptions`、`RerankerOptions`、`SyncReport`、`DecayPolicy`、`FeedbackRequest`、`Page`/`PageRequest`/`ReadFilter` 等见 [data-model](data-model.md)；检索流程见 [search](search.md)。

## 错误处理

所有接口返回 `p_memory::Error`（`thiserror`），`Error::code()` 给出稳定字符串码（见 [data-model](data-model.md#错误码)），适合跨语言映射。
