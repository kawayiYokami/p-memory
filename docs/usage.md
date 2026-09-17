# 使用指南

p-memory 是给多个宿主共用的**离线记忆底层**：它只负责存储与检索，模型调用、重排、业务规则都在宿主侧。本页回答一件事——**拿到这个库，代码里怎么用**。

## 打开一个库

```rust
use p_memory::KnowledgeBase;

let kb = KnowledgeBase::open("./data")?;   // 目录不存在会自动创建，并独占 writer.lock
```

`kb` 可 `Clone`（共享同一进程内引擎）；写入串行、读取并发（各取自空闲池的只读连接），跨线程用 `Clone`。用完 `kb.close()` 释放锁。

## 五个域

| 想要什么 | 走哪 | 典型方法 |
|---|---|---|
| 存 / 取论断式记忆 | `kb.memories()` | `upsert` / `list` / `feedback` / `decay` |
| 存 / 取知识图谱 | `kb.graph()` | `apply_batch` / `resolve` / `neighbors` / `ego` / `path` |
| 存 / 取长文档 | `kb.notes()` | `upsert` / `chunks` |
| 写入向量 / 重排（宿主提供模型回调） | `kb.embeddings()` / `kb` | `register_space` / `register_embedder` / `sync` / `register_reranker` |
| 统一检索 | `kb.search(&req)` | 关键词 / 向量 / 混合 / 重排 |

五个域共享一个数据库与事务基础：写入返回 `WriteReceipt`，读取带 `ReadFilter`。

## 记忆

```rust
use p_memory::MemoryInput;

let mut m = MemoryInput::new("用户偏好简短直接的回答，反感术语堆砌");
m.record.namespace = "chat".into();
m.record.scope = "private".into();
m.record.tags = vec!["偏好".into()];

kb.memories().upsert(m)?;                                   // 新建（带 record.id 则更新）
kb.memories().upsert_by_judgment(MemoryInput::new("…"))?;   // 按论断去重，同 namespace+scope 内合并
```

- 每条记忆有强度 / 档位 / 生命周期；`feedback` 提升有用项、`decay` 按策略衰减——详见 [lifecycle](lifecycle.md)。
- `list` 用 `PageRequest` 游标翻页，稳定。

## 检索

```rust
use p_memory::{ReadFilter, SearchRequest};

let req = SearchRequest {
    query: "偏好".into(),
    filter: ReadFilter { namespace: "chat".into(), scopes: vec!["private".into()], tags: vec![] },
    ..Default::default()
};
for hit in kb.search(&req)?.hits {
    println!("{:.3}\t{}", hit.score, hit.key.id);
}
```

- `query` 走关键词（严格 + 宽松两轮）；给了 `embed_space` 走向量（库自己嵌入查询词）；两者都开就是混合，用 RRF 融合，可选重排。详见 [search](search.md)。
- 过滤在截取 Top-K **之前**生效。

## 图谱

写入（单事务，顺序实体 → 关系 → 事件，允许批内互相引用）：

```rust
use p_memory::graph::GraphBatch;

// entities 为 Vec<EntityInput>，relations 为 Vec<RelationInput>，引用实体的 record_id
let batch = GraphBatch { entities, relations, ..Default::default() };
let got = kb.graph().apply_batch(&batch)?.value;
let alice_id = got.entities[0].header.id;   // i64
```

查询：一层邻接走 `neighbors`；要在图上走一步以上（关系圈、桥接、环）走图搜索：

```rust
kb.graph().resolve("张三", &filter, 1)?;   // 名字 → 实体
kb.graph().neighbors(id, &filter, 50)?;    // 一跳
kb.graph().ego(id, 2, &filter, 50)?;       // 两跳关系圈
kb.graph().path(a, b, &filter)?;           // 桥接
```

图谱抽取格式见 [graph-generation](graph-generation.md)，图搜索见 [graph-search](graph-search.md)。

## 笔记

```rust
use p_memory::notes::NoteFileInput;

// 只给路径：库自己读文件，标题取文件名（去扩展名），路径即身份
kb.notes().upsert_file(NoteFileInput::new("docs/readme.md"))?;
```

写入时，库在**同一事务**内重切切片；`chunks(note_id, &filter)` 取回带行号的片段。正文不落库——权威是文件，笔记 payload 只留切片粒度；切片正文由文件原文按字符区间取出；可检索正文交给全文索引承载。

## 向量与重排（宿主提供模型接口，库内部执行）

```rust
use p_memory::{EmbeddingSpace, EmbedderOptions};

kb.embeddings().register_space(EmbeddingSpace {
    id: "e5".into(), model: "e5-base".into(), dimension: 768, text_version: 1, encoding: "sq8".into(),
})?;

// 一个向量模型 = 一个向量空间；注册即用样本校验，不符即拒绝绑定。
kb.embeddings().register_embedder_with("e5", my_embed_fn, EmbedderOptions {
    max_batch: 50, max_tokens_per_text: Some(512),
})?;

// 库拿该回调把缺失向量分批补齐；写入路径也会「写入即向量化」。
kb.embeddings().sync("e5", 50)?;

// 检索：宿主只给搜索词与目标空间，库自己嵌入查询词。
```

宿主只提供「内容 + 模型接口」，向量化与重排全程在库内发生：`register_embedder` 回调算向量、`register_reranker` 回调做重排，库负责分批、截断、重试与多档降级。回调是运行时状态、不进数据库，宿主启动时注册一次即可。

## 多宿主共存

- 一个数据目录同一时刻只允许**一个进程**持写锁。
- 各宿主用不同 `namespace` 隔离；`scope`（如 `private` / `public`）区分可见性；`tags` 进一步约束。
- 读取一律经 `ReadFilter { namespace, scopes, tags }`；检索、列表、图搜索都遵守它——**跨命名空间不会串**。

## 写入回执与错误

```rust
let receipt = kb.memories().upsert(m)?;
receipt.value;         // 写入结果
receipt.revision;      // 提交后的库版本
kb.update_index()?;    // 追平待办（写入不就地索引；批量导入后调用一次即可）
```

所有错误是 `p_memory::Error`，`err.code()` 给稳定字符串码，便于跨进程 / 跨语言处理。

## 存储与备份

一个目录一个库：`store.sqlite3`（权威）+ `text-v2/`（可重建的全文索引）+ `writer.lock`。

```rust
kb.backup("./backup.sqlite3")?;                                  // 在线备份
let restored = KnowledgeBase::restore("./backup.sqlite3", "./data2")?;
```

字段级细节见 [data-model](data-model.md)，完整签名见 [api-rust](api-rust.md)。
