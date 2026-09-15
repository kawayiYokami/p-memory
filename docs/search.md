# 检索

检索由 `KnowledgeBase::search` 统一入口提供，支持关键词、向量和两者混合。全部过滤条件在截取 Top-K **之前**应用。

## 请求

```rust
struct SearchRequest {
    query: String,                    // 关键词查询，可为空
    filter: ReadFilter,               // 命名空间、作用域、标签
    kinds: Vec<RecordKind>,           // 默认 memory/entity/relation/event/chunk
    limit: usize,                     // 默认 10，范围 1..=10000
    candidate_limit: Option<usize>,   // 各路召回上限
    vectors: Vec<QueryVector>,        // 查询向量，可多路
    text_weight: f64,                 // 默认 1.0，须为正有限值
    prune: Option<GraphPrune>,        // Graph-First 剪枝，None 为全量检索
}

struct QueryVector { space_id: String, values: Vec<f32>, weight: f64, min_score: Option<f64> }
struct GraphPrune { root: i64, depth: usize, limit: usize }
```

约束：

- `query` 与 `vectors` 不能同时为空，否则 `validation`。
- `vectors` 中 `space_id` 不得重复，否则 `validation`。
- `min_score` 若给出须在 `[-1, 1]`。
- `candidate_limit` 缺省为 `(limit * 5).max(100).min(10000)`，且不得小于 `limit`。
- `prune.depth` 至少为 1、`prune.limit` 至少为 1，否则 `validation`。

## 返回

```rust
struct SearchHit {
    key: RecordKey,
    score: f64,                        // RRF 融合分，非概率
    text_score: Option<f64>,          // 全文原始分
    vector_scores: Map<String, f64>,  // 各向量空间的余弦分
    record: Value,                    // 完整记录
}
struct SearchResult { hits: Vec<SearchHit>, revision: i64, indexed_revision: i64 }
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
    A["校验参数"] --> B["sync 全文索引"]
    B --> C["全文：严格 + 宽松两轮"]
    C --> D["向量：各空间精确余弦"]
    D --> E["RRF 融合"]
    E --> F["按分数排序、截断 limit"]
```

1. **校验参数**，任何非法立即返回 `validation`。
2. **同步索引**：`query` 非空时先补齐待处理的全文更新，绝不静默使用旧索引。
3. **全文召回**：严格与宽松两轮，详见下节。
4. **向量召回**：对每个查询向量加载对应空间并精算余弦。若给了 `prune`，先按 `root` 在图里展开 `depth` 跳取邻域（含起点，节点上限 `limit`），向量打分只在这些 `record_id` 内进行——图外的记录不参与打分；未设 `prune` 时保持全量行为。邻域按 `predicate_rules` 补过对称/逆关系、并按 `sys:same_as` 缩点，详见 [graph-search](graph-search.md)。
5. **RRF 融合**：每一路按名次贡献 `weight / (60 + rank + 1)`，其中 `rank` 从 0 计。文本路权重为 `text_weight`，向量路权重为 `vector.weight`。
6. **排序截断**：RRF 分降序，同分按 `RecordKey` 升序，取前 `limit`。

## 全文检索

分词流程：

- `normalize_text`：去首尾空白、全角转半角、弯引号转直引号、统一小写。
- `tokenize`：英文与数字按词切分；CJK 序列同时产出**单字**与相邻**双字（bigram）**。
- `query_terms`：查询侧去重；严格模式移除纯 CJK 双字（仅保留单字），宽松模式保留。

两轮召回：

| 轮次 | 逻辑 | 作用 |
|---|---|---|
| 严格 | 全部词项 `Must` 命中 | 精确匹配优先 |
| 宽松 | 词项 `Should` 命中 | 补充召回 |

结果合并时**严格轮优先**：先取严格命中，再用宽松结果补齐未出现的记录，最后按组内分数与 `key` 排序。索引侧 schema 为 `key/namespace/scope/kind/tags/keywords/text`：`text` 使用预切分的空白分词器（`pretokenized`）承担正文 1+2 分词召回；`tags` 与 `keywords` 是精确整词字段（`STRING`），`keywords` 收纳各记录的 tags 与笔记 `source` 的各级父目录名，查询时按整词与正文并行 `Should` 召回——打关键字即能命中，且专有名词不被切碎。

## 向量检索

- 每个向量空间在内存中维护一份 `VectorMatrix`（按空间缓存，写入提交后清空）。
- 精确计算余弦相似度；查询向量先归一化，存储向量写入时已归一化。
- 用**有界堆**保留 Top-K；`min_score` 在打分时过滤。
- 检索为**精确检索**，不做近似索引。

## 过滤

- 命名空间、作用域、记录类型、标签（AND、归一化、大小写不敏感）在构造查询或打分前统一应用。
- 图谱展开（`neighbors`/`events_for_entity`）遵守同样的作用域约束：标签只约束返回的关系，作用域同时约束端点实体。
- 向量路径按 `row.key.namespace`、`row.scope`、`kinds`、`row.tags` 在打分前过滤。

## 索引一致性

- SQLite 与 Tantivy 之间用可重放的 `index_updates` 日志衔接，每条日志记录 `(revision, record_id)`：
  `revision` 既是主键也是重放游标，`record_id` 指向待同步的那条记录。
- 索引提交带 payload `p-memory-text-v2:<indexed_revision>`；`open` 时若 payload 与 `indexed_revision` 不符即整体重建。
- 索引目录损坏时**隔离并重建**：把 `text-v2` 重命名为 `text-v2.corrupt-<uuid>`，再新建空索引，权威数据库不受影响。
- 检索前若有待处理更新会先重放，保证结果与已提交数据一致。
