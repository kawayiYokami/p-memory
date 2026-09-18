# 检索

检索由 `KnowledgeBase::search` 统一入口提供，支持关键词、向量和两者混合，并可选重排。全部过滤条件在截取 Top-K **之前**应用。

向量化由库内部完成：宿主只给搜索词与目标空间，库用它注册的回调嵌入查询词——宿主不接触向量。

## 请求

```rust
struct SearchRequest {
    query: String,                    // 关键词查询，必填
    filter: ReadFilter,               // 命名空间、作用域、标签
    kinds: Vec<RecordKind>,           // 默认 memory/entity/relation/event/chunk
    limit: usize,                     // 默认 10，范围 1..=10000
    candidate_limit: Option<usize>,   // 各路召回上限
    embed_space: Option<String>,      // 走向量路时用哪条向量空间
    text_weight: f64,                 // 默认 1.0，须为正有限值
    prune: Option<GraphPrune>,        // Graph-First 剪枝，None 为全量检索
    text: bool,                       // 默认 true，本次是否走全文路
    vector: bool,                     // 默认 true，本次是否走向量路（需要 embed_space）
    rerank: bool,                     // 默认 true，本次是否重排（未注册回调时被忽略）
    with_total: bool,                 // 默认 false，是否返回过滤后的匹配总量
}

struct GraphPrune { root: i64, depth: usize, limit: usize }
```

约束：

- `query` 不能为空，否则 `validation`。
- 向量能力由库承担，宿主始终提供关键词，不传向量。
- 向量路要同时满足「`vector` 为真」与「给出 `embed_space`」；缺任一条件就不走向量路，**不是错误**。两条路都关（`text=false` 且 `vector` 无效）才报 `validation`。
- `candidate_limit` 缺省为 `(limit * 5).max(100).min(10000)`，且不得小于 `limit`。
- `prune.depth` 至少为 1、`prune.limit` 至少为 1，否则 `validation`。

## 返回

```rust
struct SearchHit {
    key: RecordKey,
    score: f64,                        // RRF 融合分，非概率
    text_score: Option<f64>,          // 全文原始分
    vector_scores: Map<String, f64>,  // 各向量空间的余弦分
    rerank_score: Option<f64>,        // 本条目在重排回调那里的分数；未重排为 None
    record: Value,                    // 完整记录
}
struct SearchResult {
    hits: Vec<SearchHit>,
    revision: i64,
    indexed_revision: i64,
    total: Option<usize>,             // 过滤之后、截断之前的匹配数；with_total=false 时为 None
    diagnostics: SearchDiagnostics,
}
struct SearchDiagnostics {
    text_used: bool, vector_used: bool, reranked: bool,
    rerank_candidates: usize,         // 实际送入重排回调的候选数
    rerank_truncated: usize,          // 因 max_docs 未送入重排的候选数
    degraded: Vec<Degrade>,           // 本次落在了哪几档降级
}
```

## 命中附带上下文

`search_with_context(request, limit)` 先走一遍 `search`，再给每条命中挂上它的实体邻域：

```rust
struct ContextualHit { hit: SearchHit, context: Neighborhood }   // Neighborhood { entities, relations }
```

「挂载」的判定：命中记录所带的 tag，若与**同一 namespace / scope 内**某实体的名字或别名文本相同，就认为该记录挂在这个实体上——两者都落在 `strings` 表、同一套归一化，直接按文本相等 join。领域边界只由 namespace / scope 决定，tag 只用来找实体、不反过来筛实体。`limit` 是每个实体邻域的规模上限；命中记录没挂到任何实体时，`context` 为空。

## 执行流程

```mermaid
flowchart LR
    A["校验参数"] --> B["库调嵌入回调嵌入查询词（不持库锁）"]
    B --> C["全文：严格 + 宽松两轮"]
    C --> D["向量：目标空间精确余弦"]
    D --> E["RRF 融合"]
    E --> F["重排：按 max_docs 截候选、按 token 预算截文本"]
    F --> G["按分数排序、截断 limit"]
```

1. **校验参数**，任何非法立即返回 `validation`。
2. **嵌入查询词**：走向量路时，库取目标空间注册的回调嵌入查询词。这是模型往返，在取库锁之前完成。
3. **同步索引**：`query` 非空时先补齐待处理的全文更新，绝不静默使用旧索引。
4. **全文召回**：严格与宽松两轮，详见下节。
5. **向量召回**：加载目标空间并精算余弦。若给了 `prune`，先按 `root` 在图里展开 `depth` 跳取邻域（含起点，节点上限 `limit`），向量打分只在这些 `record_id` 内进行——图外的记录不参与打分；未设 `prune` 时保持全量行为。邻域按 `predicate_rules` 补过对称/逆关系、并按 `sys:same_as` 缩点，详见 [graph-search](graph-search.md)。
6. **RRF 融合**：每一路按名次贡献 `weight / (60 + rank + 1)`，其中 `rank` 从 0 计。文本路权重为 `text_weight`，向量路权重固定为 1。
7. **重排**：注册了重排回调且 `rerank=true` 时，按 RRF 顺序把候选截到 `max_docs`，文档与查询词按 token 预算截断后交给回调；回调返回的分数替换 RRF 分参与排序。截断数量写进诊断。
8. **排序截断**：重排分（或 RRF 分）降序，同分按 `RecordKey` 升序，取前 `limit`。

## 全文检索

分词流程：

- `normalize_text`：去首尾空白、全角转半角、弯引号转直引号、统一小写。
- `tokenize`：英文与数字按词切分；CJK 序列同时产出**单字**与相邻**双字（bigram）**。
- `query_terms`：查询侧去重；严格模式移除纯 CJK 双字（仅保留单字），宽松模式保留。
- `truncate_to_tokens`：按同一套 token 计数把文本截到预算内，重排与嵌入的截断都用它。

两轮召回：

| 轮次 | 逻辑 | 作用 |
|---|---|---|
| 严格 | 全部词项 `Must` 命中 | 精确匹配优先 |
| 宽松 | 词项 `Should` 命中 | 补充召回 |

结果合并时**严格轮优先**：先取严格命中，再用宽松结果补齐未出现的记录，最后按组内分数与 `key` 排序。写入索引前，文本先经统一的 `clean_markdown` 清洗掉标记符（`body` 列照旧保留原文）。索引侧 schema 为 `key/namespace/scope/kind/tags/text/body`：

- `text`（正文列）：这条记录自己的文本——记忆是 `judgment`，切片是它那一段；承载标签的那一条（切片是第一片，其余记录是它自己）前面还拼着空格连接的标签集。用预切分的空白分词器（`pretokenized`）承担 1+2 分词召回。
- `tags`（标签 id 列）：整数多值，只用来按标签过滤，不参与打分。
- `namespace` / `scope` / `kind`：一律存 `strings` 表的整数 id，索引里不留第二份标记文本。折算在进索引之前做完；filter 里的文本换不到 id，说明库里没有这个标记，本次不可能有命中，直接给空结果，不必进索引碰。
- `body`：正文原值（stored），供重排取候选正文与命中回读；库里不留正文副本，取正文只走这一列。它不带标签——标签只拼进可搜的那份文本。
- `key`：记录 ID，用于删除与取回，不参与打分。

索引里没有单独的标签文本列：标签集拼在承载它的那条正文前面，进的是同一条 `text` 列，跟着这条的文档长度一起被 BM25 的长度归一化。查「澜川」时，文件名带它的那一条照样能命中，但这一份分会被它自己那两百多字的正文稀释，不再因为文档短而虚高。RRF 只用在全文路与向量路之间，与这里的一切无关。

笔记记录不进索引：它自己没有正文，路径信息以标签形态写在它每一条切片记录上（库里 `record_tags`），可搜的那一份拼在第一片的正文前面。要文件列表就按库里的标签翻笔记（`notes().list` 带 `filter.tags`）。

## 向量检索

- 每个向量空间在内存中维护一份 `VectorMatrix`（按空间缓存，写入提交后清空）。
- 精确计算余弦相似度；查询向量先归一化，存储向量写入时已归一化。
- 用**有界堆**保留 Top-K。
- 检索为**精确检索**，不做近似索引。

## 预设检索

预设是库预先配好的搜索方法，调用方按名字取用，不必自己拼参数：

```rust
fn search_preset(&self, request: &PresetRequest) -> Result<PresetResult>;
```

| 预设 | 走哪几路 | 填哪些字段 |
|---|---|---|
| `memory` | 记忆 | 记忆 |
| `graph` | 图谱 | 图谱 |
| `notes` | 切片 | 笔记 |
| `rag` | 记忆 + 图谱 | 记忆 + 图谱 |
| `broad` | 记忆 + 图谱 + 切片 | 记忆 + 图谱 + 笔记 |

返回恒为三个字段，**各自独立排序，不混在一起**；这次没走的那一路是空数组：

```rust
struct PresetResult {
    preset: SearchPreset,
    memories: Vec<SearchHit>,     // 只含记忆记录
    graph: GraphSection,          // 实体 / 关系 / 事件，见下
    notes: Vec<SearchHit>,        // 只含切片
    revision: i64,
    indexed_revision: i64,
    diagnostics: SearchDiagnostics,   // 各路的开关取或、候选数累加、降级去重
}
struct PresetRequest {
    preset: SearchPreset,             // 默认 rag
    query: String,                    // 必填，空即 validation
    filter: ReadFilter,
    embed_space: Option<String>,      // 走向量路时用哪条空间
    text: bool, vector: bool, rerank: bool,   // 默认全真
    budget: PresetBudget,             // 阈值，逐项可覆盖
    candidate_limit: usize,           // 按字符数封顶前先取多少条候选，默认 64
}
```

阈值按**字符数**封顶（重排模型按字符数算，不看条数），缺口等实测再调：

```rust
struct PresetBudget {                 // 默认值
    memory_chars: usize,              // 2000
    notes_chars: usize,               // 3000
    seed_entities: usize,             // 4，中间量，仍按个数
    graph_relations_chars: usize,     // 1000
    graph_context_chars: usize,       // 2000，铺开的关系与事件共用
}
```

单路（记忆、笔记）的候选直接复用 `search`，也就是全文 + 向量 + 可选重排一次走完，再按字符数截断：长度取索引里的纯正文，取不到正文的条目按 0 计；第一条无论多长都留下，避免预算略小就整条路空掉；`chars` 为 0 表示这一路不出。

图谱那一路按三步走，产物按阶段分块：

```rust
struct GraphSection {
    entities: Vec<Entity>,          // 第一步的种子，按分排，带别名
    relations: Vec<Relation>,       // 第二步命中的关系，按相关度排
    context_relations: Vec<Relation>,   // 第三步铺开的关系，不筛，按 id 升序
    context_events: Vec<Event>,         // 第三步铺开的事件，不筛，按 id 升序
}
```

1. **搜种子**：查询词搜实体，按分取前 `seed_entities` 个当种子。种子为空则整段为空。
2. **敲关系**：每个种子各自到「以它为端点」的关系里敲查询词，命中的留下。打分是查询词元在这条关系正文里的命中权重——二字及以上词元算 2 分、单字算 1 分，降序，同分按记录 id 升序。命中的关系算结果、进重排，按 `graph_relations_chars` 封顶。
3. **铺开**：实体集合 = 种子 + 命中关系的另一端；两端都落在集合内的关系、参与者至少两个落在集合内的**事件**全部保留、不筛，且排除第二步已命中的关系。关系按 `graph_context_chars` 吃掉一份，剩下多少给事件，两者共用这一份上限。

分块而不合榜是有意的：第三步那批本来就没跑过相关性，硬按分排等于把「不筛」又变成「按分排」；前几块的分也不在同一根尺上——种子的分来自实体正文，命中关系的分来自关系正文。下游因此一眼看得出哪块是答、哪块是背景。

准入与降级沿用 `search` 的口径：预设里的路走全文 + 向量，向量按**各档是否补齐**逐档判（记忆档 = 记忆，图谱档 = 实体 / 关系 / 事件，笔记档 = 切片），某档未就绪只让这一档不走向量，别的档照常。

## 重排

- 重排回调是进程内单例，不绑定向量空间：`rerank(query, documents) -> [score]`，与输入文档等长、按序给出。
- 库在调用前按 `RerankerOptions` 强制截断：候选截到 `max_docs`，文档按 `max_tokens_per_doc`、查询词按 `max_tokens_query` 截断。重排模型普遍有硬上限，超出的候选不是变慢就是直接报错。
- 回调失败或返回条数/有限性不符时，按融合分排序并记 `Degrade::RerankFailed`，不抛错。

## 降级

多档是多个失效点各自有退路，且每一档都可观测：

| 档位 | 触发 | Degrade |
|---|---|---|
| 纯全文 | 该 namespace 的总闸关了向量化 | `namespace_disabled` |
| 纯全文 | 目标空间没注册嵌入回调 | `no_embedder` |
| 纯全文 | 嵌入回调调用失败 | `embed_failed` |
| 纯全文 | 该领域这一档向量未补齐（该档的 `vector_ready` 标记缺失） | `vector_not_ready` |
| 按融合分排序 | 重排回调失败或产出不符 | `rerank_failed` |
| 文本路不可用 | 索引查询失败，且当场重建仍未能恢复 | `text_index_unavailable` |

任一档都返回结果、不抛错、不返回空；当前档位可从 `SearchResult.diagnostics.degraded` 与 `HealthReport.last_degraded` 读到。索引查询失败不是直接降级：检索会先按 `FORMAT` 与权威数据当场重建索引并重试，只有重建仍失败时才隔离文本路。

## 过滤

- 命名空间、作用域、记录类型、标签（AND、归一化、大小写不敏感）在构造查询或打分前统一应用。索引侧的这四类标记都是整数 id：进索引之前先把 filter 里的文本到 `strings` 表换成 id，换不到的标记直接给空结果。
- 图谱展开（`neighbors`/`events_for_entity`）遵守同样的作用域约束：标签只约束返回的关系，作用域同时约束端点实体。
- 向量路径按 `row.key.namespace`、`row.scope`、`kinds`、`row.tags` 在打分前过滤。这里的 `kinds` 是三层取交集的结果：请求要的记录类型 ∩ 该领域启用了向量化的档位 ∩ 该档已补齐（就绪）的记录类型。某档关掉或某档没补齐，它的存量向量都不参与打分，别的已就绪档照常走向量；交集为空时整条向量路不走（总闸关闭时同理，并记 `namespace_disabled`；只是没补齐则记 `vector_not_ready`）。就绪标记逐档落在 `meta` 里，键是 `vector_ready:<领域>:<空间>:<档位>`——避免拿「只补了一半」的向量去和全文融合，让先补完的那部分凭空占位。

## 索引一致性

- `index_updates` 是待提交信号：写入事务里登记 `(revision, record_id)`，同一个调用把文档写进索引 writer。`update_index` 一趟提交、把 `indexed_revision` 记到本次实际覆盖的最大 revision，并删掉已覆盖的队列行。
- 索引目录损坏时**隔离并重建**：把 `text-v2` 重命名为 `text-v2.corrupt-<uuid>`，再新建空索引，权威数据库不受影响。
- 检索前若还有待处理更新，会先提交一次，保证结果与已提交数据一致。
- 索引提交带 payload `p-memory-text-v5:<indexed_revision>`；`open` 时若 payload 与 `indexed_revision` 不符、或队列里还压着未提交的待办，即整体重建。
- 重建是唯一回读源文件的路径：切片正文没有第二份副本，按同一套切分规则重新读文件切一遍，文件缺失的那批切片正文退化为空。
