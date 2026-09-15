# 数据模型

所有领域记录共享同一个公共记录头，领域正文以 JSON payload 存放在同一个 `records` 表里，图谱完整性由关系投影表额外约束。

## 记录类型

```rust
enum RecordKind { Memory, Entity, Relation, Event, Note, Chunk }
```

对应字符串：`memory`、`entity`、`relation`、`event`、`note`、`chunk`。

## 公共记录头

### 输入（RecordInput）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `id` | `i64` | 库分配 | **内部主键一律自增整数**，由库统一分配；调用方不传入。外部 ID 与内部无关，仅放 `metadata` 作溯源 |
| `namespace` | `String` | `"default"` | 独立知识域 |
| `scope` | `String` | `"public"` | 作用域 |
| `tags` | `Vec<String>` | `[]` | 标签，写入时归一化去重 |
| `evidence` | `Vec<Evidence>` | `[]` | 来源引用快照 |
| `metadata` | `Map<String,Value>` | `{}` | 扩展字段 |
| `created_at_us` | `i64?` | 当前时间 | UTC 微秒 |
| `updated_at_us` | `i64?` | 当前时间 | UTC 微秒 |
| `expected_revision` | `i64?` | `None` | 乐观并发；不匹配返回 `stale_revision` |

### 输出（RecordHeader）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `i64` | 内部自增整数主键 |
| `namespace` `kind` `scope` | `String` | 归属 |
| `tags` | `Vec<String>` | 标签字符串（内部按 `tag_id` 引用，读取时从 `strings` 表取回） |
| `evidence` | `Vec<Evidence>` | 来源引用 |
| `metadata` | `Map` | 扩展字段 |
| `created_at_us` `updated_at_us` | `i64` | UTC 微秒 |
| `revision` | `i64` | 每次写入全局递增的修订号 |

领域结构（`Memory`/`Entity`/…）以 `#[serde(flatten)]` 方式内嵌 `RecordHeader`，因此序列化后是「记录头 + 领域字段」的单一对象。

### 不变式

- `namespace`/`scope` 必须非空、无首尾空白、无控制字符；`id` 由库分配，不接受调用方指定。
- 记录一旦写入，**不能改变 `scope`**；需要换作用域时显式复制成新 `id`（否则 `conflict`）。
- `updated_at_us` 不能早于 `created_at_us`。
- 指定 `expected_revision` 且与库中不符时拒绝写入。

## 来源引用（Evidence）

> 来源存的是**快照**；笔记切片被替换时，它不会随之改变。

| 字段 | 类型 | 说明 |
|---|---|---|
| `source` | `String` | 宿主拥有的路径或 URI（必填，核心从不读写该路径） |
| `source_revision` | `String?` | 来源版本 |
| `chunk_id` | `i64?` | 指向内部切片的整数 id 快照（不是外键） |
| `offset` `limit` | `usize?` | `offset` 为 1 起始的**起始行**，`limit` 为**行数**；结束行 = `offset + limit - 1` |
| `quote` | `String` | 引用原文 |
| `metadata` | `Map` | 扩展 |

## 标记与字典表

**原则：会重复出现的标记信息一律抽表、用整数 id 引用，字符串只存一份**（不只为 tag，凡重复的分类字段都适用）。按「封闭 / 开放」分两类处理。

### 固定枚举 → 整数编码

编译期可穷举的枚举直接存整数，不建表：

- `kind`：`memory / entity / relation / event / note / chunk`

### 其余标记 → 单张字典表 `strings`

不可穷举的标记统一登记到**一张**字典表，用 id 引用；不按类别拆表，类别由「谁引用它」决定：

```
strings(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL UNIQUE)
```

涵盖：`tag`、`memory_type`、`entity_type`、`predicate`、`attr_key`、`namespace`、`scope`、路径 `source`。`memory_type` 的值存于记忆 payload 的 `memory_type_id`，读取时经 `strings` 还原文本。

- `text` 存归一化后的字符串，全表唯一。
- 记录通过 id 列表引用，例如 tags `[1, 2, 3]` 对应 `我`、`身份`、`猪`。
- 归一化规则：去首尾空白、全角转半角、弯引号转直引号、小写。
- 过滤按归一化值匹配，大小写不敏感；多 tag 之间为 **AND**。

### 接口边界

对外接口（如 `ReadFilter`、写入入参）**一律收、发字符串**，`strings` 的整数 id 只在存储层内部使用：

- 调用方报名字（`namespace: "notes"`、`scopes: ["public", "private"]`），由核心查表映射成 id 再落库/检索。
- 写入遇到 `strings` 中不存在的新词时，核心自动插入一行再取 id；调用方无需预注册。
- 取舍：字典表只为压缩体积，多一次「名字↔id」映射是存储层内部成本，不外泄到接口。

## 作用域与命名空间

- `namespace` 对应独立知识域，例如按主题划分的 `domain`；存 `strings` 的 id。
- `scope` 表达公共、Agent 私有或群组范围，默认 `public`；取值动态（千变万化），存 `strings` 的 id。
- `ReadFilter` 默认只读 `["public"]`，且必须显式给出至少一个 scope（空数组报 `validation`）：

```rust
struct ReadFilter { namespace: String, scopes: Vec<String>, tags: Vec<String> }
```

- 所有读取、检索、图谱展开都受 `ReadFilter` 约束；过滤在截取 Top-K **之前**生效。

## 分页

```rust
struct PageRequest { filter: ReadFilter, limit: usize /* 默认 50 */, after: Option<String> }
struct Page<T> { items: Vec<T>, next_cursor: Option<String> }
```

- 游标是内部自增主键 `id` 的字符串形式，按 `id` 升序翻页。
- `limit` 允许 1..=10000。

## 写入回执

```rust
struct WriteReceipt<T> { value: T, revision: i64, index_ready: bool, index_error: Option<String> }
```

`index_ready == false` 表示数据已提交但全文索引未同步，`index_error` 给出原因；这**不是**写入失败。

## 统一类型词表

`memory_type` 统一为受控英文枚举：

```
knowledge | skill | emotion | event | task | graph
```

默认值 `knowledge`。字段本身是字符串，允许宿主扩展；导入来源的归一映射由导入器实现。

## 领域一：记忆（Memory）

```rust
struct Memory { header: RecordHeader, memory_type: String, judgment: String, reasoning: String, state: MemoryState }

struct MemoryState {
    pinned: bool,               // 固定保留，不参与衰减
    strength: i64,              // 强度，默认 1
    useful_count: i64,         // 有效召回次数
    useful_score: f64,          // 分档有用分
    last_recalled_at_us: i64?,
    last_decay_at_us: i64?,
}
```

- `judgment` 必填且非空白。
- `reasoning` 可空。
- `MemoryInput.state` 为 `None` 时保留既有记忆的生命周期状态。
- 生命周期字段必须有限且非负。

生命周期参数默认值（`DecayPolicy`）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `tier0_threshold` | 3.0 | 低于此值进入 T0 |
| `tier1_threshold` | 10.0 | 达到此值进入 T2 |
| `useful_score_boost` | 2.5 | 有用反馈增加的有用分 |
| `strength_boost` | 1 | 有用反馈增加的强度 |
| `tier0_cycle_days` | 3 | T0 衰减周期（天） |

强度归零只会进入 `retirement_candidates`，**最终删除由宿主决定**。详见 [lifecycle](lifecycle.md)。

## 领域二：知识图谱（Entity / Relation / Event）

### 实体

```rust
struct EntityInput {
    name: String,                                  // 必填
    entity_type: String,                           // 默认 "concept"
    aliases: Vec<String>,
    attributes: BTreeMap<String, Vec<String>>,
    summary: String,
}
```

- 别名去空白、去重，实体名本身也作为别名参与解析。

### 关系

```rust
struct RelationInput {
    subject_id: i64, predicate: String, object_id: i64,
    confidence: f64,   // 默认 0.8，必须在 [0,1]
    reason: String,
}
```

- `subject_id`/`object_id` 必须指向**同 namespace 且同 scope** 的已存在实体，否则报 `not_found`；引用在事务内校验。
- 删除仍被引用（关系端点或事件参与者）的实体报 `conflict`。
- 谓词是开放字符串，物理表只存单向真实三元组。反向查询（由「父亲」反查「子女」）、对称关系、别名等价，都靠 `predicate_rules`（谓词元规则）在建图时内存补边，不落双向物理边。规则由 `kb.graph().set_predicate_rule(...)` 受控登记；内置 `sys:same_as` 是对称的别名等价关系，建图时用于并查集缩点。详见 [graph-search](graph-search.md)。

### 事件

```rust
struct EventInput {
    name: String, summary: String,
    participants: Vec<i64>,   // 实体整数 ID，去重
    confidence: f64,             // 默认 0.8，[0,1]
    reason: String,
}
```

### 批量与解析

```rust
struct GraphBatch { entities: Vec<EntityInput>, relations: Vec<RelationInput>, events: Vec<EventInput> }
struct GraphBatchResult { entities: Vec<Entity>, relations: Vec<Relation>, events: Vec<Event> }
struct Neighborhood { entities: Vec<Entity>, relations: Vec<Relation> }
```

- 批量写入在**同一事务**内先实体、再关系、再事件，允许引用本批次内新建的实体。
- 同名实体**返回候选，不自动合并**；归并规则留在宿主。

## 领域三：笔记与切片（Note / Chunk）

```rust
struct NoteInput { source: String, title: String, content: String, chunk_chars: usize /* 默认 220 */ }
struct Note   { header, source, title, content, source_revision, chunk_chars, chunk_count }
struct Chunk  { header, note_id, ordinal, offset, limit, content }
struct TextChunk { ordinal, offset, limit, content }
```

- `source` 必填；`(namespace, scope, source)` 唯一，重复写入按来源定位到同一笔记。
- **核心只保存用于检索的正文快照，从不读写 `source` 指向的文件。**
- **路径与标题只存在笔记这一层**；切片不重复存 `source`/`title`，需要时经 `note_id` 关联取回。
- 切片规则见 [笔记切片](#笔记切片)。
- `source_revision = sha256(content)`。
- 正文替换时在同一事务内原子重建切片投影：`chunks` 存内容指纹 `fingerprint`，`ordinal` 与内容都未变的切片保留其整数 `record_id`，向量继续有效；失效的旧切片连同向量一并删除。

### 笔记切片

- 按段落切分，目标 `chunk_chars`（范围 16..=100000，默认 220）。
- **围栏代码块与表格整体保留**，即使超过目标大小也不拆分。
- `offset` 从 **1** 开始（起始行），`limit` 为行数；两者的计数单位都是**行**。
- 超长段落按字符边界切分，同一物理行可能被多个切片共享行号。
- 标题来自笔记层，仍进正文；`source` 路径作为精确整词关键字进 `keywords` 字段（含各级父目录名），不进正文参与切分。

## 领域四：向量（EmbeddingSpace / Embedding）

```rust
struct EmbeddingSpace { id: String, model: String, dimension: usize, text_version: u32 /* 恒为 1 */, encoding: String /* "f32" 或 "sq8" */ }
struct EmbeddingInput { key: RecordKey, text: String, fingerprint: String, revision: i64 }
struct EmbeddingWrite { key: RecordKey, fingerprint: String, values: Vec<f32> }
```

- 向量空间**不可变**：重复注册相同定义是幂等的，改动 model/dimension 必须换新 `id`，否则 `conflict`。
- `encoding` 决定向量落盘格式：`"f32"`（4 字节/维，无损）或 `"sq8"`（1 字节/维，逐条对称量化，体积约 -75%、召回约 99%）。新空间默认 `"sq8"`。
- `sq8` 空间在内存里同样保留 i8 码与逐条 scale，不展开回 f32，因此常驻内存也随体积一起下降。打分时查询向量做一次相同的量化，两侧都用 i8 做整数乘累加；x86_64 上按运行时检测走 AVX2，没有该指令集时自动退回标量路径。
- 维度范围 1..=65536，`text_version` 只接受 1。
- 记录内容变化时，旧指纹的向量会在**同一事务**内失效删除。
- 写回时校验维度、数值有限性、与记录的 `fingerprint` 一致；过期结果报 `stale_revision`，不写入。
- 向量在库内以归一化 `f32` 小端字节存储。

两步流程：

1. `pending(space_id, page, kinds)` → 分页返回待向量化文本、记录键与指纹。
2. 宿主机外生成向量后 `put(space_id, writes)` → 原子批量写回。

## 错误码

| code | 触发 |
|---|---|
| `validation` | 输入非法 |
| `not_found` | 记录/空间不存在 |
| `conflict` | 唯一性、引用约束、不可变空间、改 scope |
| `locked` | 数据目录已被其他进程打开 |
| `closed` | 知识库已关闭 |
| `schema_version` | 未知的更高 schema 版本 |
| `invalid_vector` | 维度不符、非有限值、零范数 |
| `stale_revision` | 乐观并发或过期向量 |
| `index` | 全文索引不可用 |
| `storage` | SQLite 错误 |
| `io` | I/O 错误 |
