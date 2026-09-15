# 图谱搜索

图谱搜索是内置能力，随库一起编译，无需额外开关。

## 定位

存储仍是 SQLite；图搜索是**内存态**的——按需把 `filter` 范围内的实体与关系读进内存、建图、跑算法、用完即弃。算法一律用 [petgraph](https://github.com/petgraph/petgraph)，不自实现。

```
SQLite（权威）→ build_graph(filter) 读快照 → petgraph 图 → 跑算法 → 丢弃
```

一层邻接（A 的邻居、A↔B 直连）仍走 SQL；**要在图上走一步以上**的查询才用它。

## 接口

| 方法 | 用途 | petgraph 算法 |
|---|---|---|
| `kb.graph().build_graph(&filter)` | 建内存图（节点 = 实体，边 = 关系） | — |
| `GraphView::ego_ids(root, depth, limit)` | N 跳邻域（实体 id） | 分层 BFS |
| `GraphView::path_ids(from, to)` | 两实体最短桥接 | `astar` |
| `GraphView::component_count()` | 连通分量数 | `connected_components` |
| `kb.graph().ego(root, depth, &filter, limit)` | N 跳邻域（Entity 级，带名/别名/属性） | 同上 |
| `kb.graph().path(from, to, &filter)` | 桥接路径（Entity 级） | 同上 |
| `kb.graph().strongly_connected(&filter)` | 强连通环（只返回大小 > 1） | `tarjan_scc` |
| `kb.graph().set_predicate_rule(pred, inverse, symmetric)` | 登记对称/逆谓词元规则，供建图时内存补边 | — |

图范围由 `ReadFilter` 决定（namespace / scope / tag），跨命名空间不会串图。`build_graph` 每次调用重建——图变了无需失效逻辑；小图重建毫秒级。

## 怎么用

输入输出里的 id 都是 `record_id`（`i64`，实体记录主键）。名字先解析成 id，再查询。下面用一组人物关系做例子。

### 关系圈 ——「围绕某人的一片网」

```rust
use p_memory::types::ReadFilter;

let filter = ReadFilter { namespace: "default".into(), scopes: vec!["public".into()], tags: vec![] };
let zhang = kb.graph().resolve("张三", &filter, 1)?[0].header.id;

// 两跳内的实体（Vec<Entity>，含 header.id / name / entity_type / aliases / attributes）
for e in kb.graph().ego(zhang, 2, &filter, 50)? {
    println!("{} [{}]", e.name, e.entity_type);
}
```

- `depth`：跳数。1 = 直接相关；2 = 二度；3 起迅速膨胀（实测：一个枢纽实体 1 / 2 / 3 跳可达 100 / 1135 / 3558 个实体）。
- `limit`：返回上限，超出截断。

### 桥接 ——「这两人怎么扯上关系」

```rust
let li = kb.graph().resolve("李四", &filter, 1)?[0].header.id;

match kb.graph().path(zhang, li, &filter)? {
    Some(chain) => {
        let names: Vec<_> = chain.iter().map(|e| e.name.as_str()).collect();
        println!("{}", names.join(" → "));
    }
    None => println!("不连通"),
}
```

返回 `Option<Vec<Entity>>`，含路径两端；不连通是 `None`。

### 体检 ——「图碎不碎、有没有环」

```rust
let view = kb.graph().build_graph(&filter)?;
println!("{} 节点 / {} 边 / {} 块", view.node_count(), view.edge_count(), view.component_count());

for ring in kb.graph().strongly_connected(&filter)? {
    println!("环：{} 个实体", ring.len());
}
```

### 何时自己 `build_graph`

`ego` / `path` / `strongly_connected` **内部都会自己建图**，直接调即可。只有当你要在**同一张图**上反复查询时，才 `build_graph` 一次拿 `GraphView` 复用——它只吃、吐 `record_id`：

```rust
let view = kb.graph().build_graph(&filter)?;
let ring = view.ego_ids(zhang, 2, 50);   // Vec<i64>，纯图操作，最省
let _ = view.path_ids(zhang, li);
println!("{} 块", view.component_count());
```

要 `Entity`（名字、属性）就用 `GraphStore` 的同名方法；只认 id、想压开销就用 `GraphView`。

### Python

与 Rust 同名同参，`record_id` 为 `int`：

```python
with p_memory.KnowledgeBase("./data", namespace="default") as kb:
    zhang = kb.graph.resolve("张三", limit=1)[0]["id"]
    li    = kb.graph.resolve("李四", limit=1)[0]["id"]
    ring  = kb.graph.ego(zhang, 2, limit=50)   # list[dict]
    chain = kb.graph.path(zhang, li)           # list[dict] | None
    rings = kb.graph.strongly_connected()      # list[list[int]]
    print(kb.graph.component_count())          # int
```

## 反向关系与别名

图是**有向**的，物理只存单向真实三元组：反向查询不会自动连通。要让「由子女反查父母」这类反向走通，登记谓词元规则，建图时在内存补出虚拟边：

```rust
kb.graph().set_predicate_rule("父亲", Some("子女"), false)?;  // 逆谓词：反向补一条「子女」边
kb.graph().set_predicate_rule("同事", None, true)?;           // 对称谓词：反向即自身
```

规则只影响建图，`relations` 表始终只有一条真实边。

同一人物的不同称谓用 `sys:same_as` 关系表达，**绝不物理合并实体**（同名不等于同一实体）。建图时这些节点被并查集缩成同一超节点，别名不再打断邻域与路径：

```rust
// 林昭 --sys:same_as--> 林先生 --导师--> 导师
let view = kb.graph().build_graph(&filter)?;
view.path_ids(lin_zhao, mentor);   // 折叠后不再夹着「林先生」中间节点
view.ego_ids(lin_zhao, 1, 50);     // 一跳直接到导师
```

`sys:same_as` 是内置的对称谓词，无需登记。

## petgraph 0.6.5 能力边界

- **有**：连通分量、强连通（scc / kosaraju / tarjan）、拓扑、环检测、缩点、最短路（Dijkstra / Bellman-Ford / A* / Floyd）、k 最短路、简单路径、PageRank、最大匹配、最小生成树、子图同构。
- **没有**：社群发现（Louvain / 标签传播）、极大团、中心性（betweenness / closeness）。

**实测提醒**：PageRank 在稀疏图上退化成度数；SCC 在方向不可靠的图上会把主块闭成巨环（实测最大 SCC 2352 实体，块内仅 22% 是双向边，靠单向链绕回），一条错向边就能缝合两片网。结构类算法（连通、桥接）比排名类、方向类可靠。
