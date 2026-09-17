# 数据模型

所有领域记录共享同一个公共记录头，领域正文以 JSON payload 存放在同一个 `records` 表里，图谱完整性由关系投影表额外约束。记忆的正文（`judgment`）权威在 payload，可由 payload 现算；笔记切片的正文唯一副本在全文索引的 stored 列（写入时从源文件读一次、切一次就带过去），库内只留路径与行区间。可检索文本不落 SQLite。

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

涵盖：`tag`、`memory_type`、`entity_type`、`predicate`、`attr_key`、`namespace`、`scope`。`memory_type` 的值存于记忆 payload 的 `memory_type_id`，读取时经 `strings` 还原文本。

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
struct WriteReceipt<T> { value: T, revision: i64 }
```

写入提交数据，并在同一个调用里把文档就地写进索引 writer（不 commit）；提交由使用方调用 `update_index` 一次完成，索引落后**不是**写入失败。

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
struct NoteFileInput { record, path, chunk_chars }   // 库自己读 path
struct Note   { header, source, title, chunk_chars }
struct Chunk  { header, note_id, ordinal, offset, limit, content }
struct TextChunk { ordinal, offset, limit, content }
```

- `path` 必填。领域在 `namespace_roots` 里登记过根目录时，写入路径必须是它的子路径：库里存的是**减掉根目录的相对路径**（`bwiki/沧州/澜川.md`），路径不在根目录之内直接报校验错，不做猜测；没登记的领域维持原样、逐字符存。`UNIQUE(namespace_id, scope_id, path)` 用来「同一文件重复同步定位到同一条」——路径一变身份就变，登记根目录前后同一个文件会被当成两篇，旧记录不自动改写。
- **相对路径按段拆成标签**：`bwiki/沧州/澜川.md` → `bwiki`、`沧州`、`澜川`（目录段原样，文件名去扩展名），与调用方给的标签合并去重，挂到这一篇的**每一条切片**上（库里 `record_tags` 写 `tag_id`，索引里 `tags` 与 `tags_text` 两列各一份）。要按目录筛切片就直接筛；要文件列表按库里的标签翻笔记。
- `upsert_file` 按 `path` 读一次文件、切一次片：正文取文件原文，标题取文件名（去扩展名）。
- **正文读一次、切一次，一路带到索引**：笔记 payload 只存切片粒度 `chunk_chars`，`Note` 没有 `content` 字段；切片正文在写入时就地写好、随文档进全文索引，`chunks()` 返回的 `content` 按记录 ID 从索引取回（索引还没提交就先提交一次），不回源文件、也不再按字符区间二次取正文。
- **路径只在 `notes` 表存一次**：它既用来读文件、也用来定位同一条，`payload` 不重复存 `source`；标题由路径派生，也不落库。
- **切片 payload 只存 `note_id` 与行区间**（`offset`/`limit`），不含正文、不含字符区间。
- 送进全文索引的文本先经统一 `clean_markdown` 清洗（去 HTML 标签、标题符、强调标记、链接与图片、代码块与行内代码、列表与引用符号等）；索引里存的正文原值照旧是切片原文。
- 写入时文件缺失或非 UTF-8 直接报错，不落库；索引全量重建时单个文件读不到只让那批切片正文退化为空，不让整次恢复失败。
- 切片规则见 [笔记切片](#笔记切片)。
- 正文替换时在同一事务内原子重建切片投影：`chunks` 存内容指纹 `fingerprint`，`ordinal` 与内容都未变的切片保留其整数 `record_id`，向量继续有效；失效的旧切片连同向量一并删除。记录指纹是「正文 + 标签」一起算的：标签换了（比如刚登记根目录、拆出了路径段）复用的切片也会换指纹，它的旧向量随之作废要重算。

### 笔记切片

- 按段落切分，目标 `chunk_chars`（范围 16..=100000，默认 220）。
- **围栏代码块与表格整体保留**，即使超过目标大小也不拆分。
- `offset` 从 **1** 开始（起始行），`limit` 为行数；两者的计数单位都是**行**。
- 超长段落按字符边界切分，同一物理行可能被多个切片共享行号；切片正文自己带在索引里，行号只用于定位，不需要额外记字符区间。
- 标题由路径文件名派生。文件名与目录名以**标签**的形态存在（见上），每个切片都带这份标签：搜「澜川」时，`澜川.md` 这一篇的每一片都从标签文本列拿到一份分，正文里也写着它的再加一份。

## 领域四：向量与重排（EmbeddingSpace / Embedder / Reranker）

```rust
struct EmbeddingSpace { id: String, model: String, dimension: usize, text_version: u32 /* 恒为 1 */, encoding: String /* "f32" 或 "sq8" */ }

// 宿主注册的模型接口；一个向量模型对应一个向量空间。
trait Embedder  { fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedCallbackError>; }
trait Reranker  { fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>, String>; }

struct EmbedderOptions  { max_batch: usize, max_tokens_per_text: Option<usize> }
struct RerankerOptions  { max_docs: usize, max_tokens_per_doc: usize, max_tokens_query: Option<usize> }
```

- **一个向量模型 = 一个向量空间**，一个空间只接受它自己模型的回调。向量化全程在库内发生：宿主只提供内容与模型接口，不自己算向量，也不自己重排。
- 向量空间**不可变**：重复注册相同定义是幂等的，改动 model/dimension 必须换新 `id`，否则 `conflict`。
- `encoding` 决定向量落盘格式：`"f32"`（4 字节/维，无损）或 `"sq8"`（1 字节/维，逐条对称量化，体积约 -75%、召回约 99%）。新空间默认 `"sq8"`。
- `sq8` 空间在内存里同样保留 i8 码与逐条 scale，不展开回 f32，因此常驻内存也随体积一起下降。打分时查询向量做一次相同的量化，两侧都用 i8 做整数乘累加；x86_64 上按运行时检测走 AVX2，没有该指令集时自动退回标量路径。
- 维度范围 1..=65536，`text_version` 只接受 1。
- 记录内容变化时，旧指纹的向量会在**同一事务**内失效删除。
- 写回时校验维度、数值有限性、与记录的 `fingerprint` 一致；过期结果报 `stale_revision`，不写入。
- 向量在库内以归一化 `f32` 小端字节存储。

### 注册即校验

注册嵌入回调时，库用样本真跑一遍完整链路，按该空间的 `dimension` 与写回同一套规则（维度、有限性、非零范数、条数与输入一致）逐项检查，任一不符即拒绝绑定并报 `invalid_vector`，空间定义不落盘。这样能挡住「回调绑错空间」「换模型忘改 dimension」——它们若不在这里拒掉，要等第一次写回时才炸，那时空间已建好、离原因很远。

### 回调错误的类别

回调失败必须**可分类**，类别由宿主显式给出，库不解析错误文案：

- `too_large`：单次请求条数超上限。库把 batch 减半（最小 1）后重试，减半值在进程内持久生效。
- `rate_limited`：限流 / 配额。库退避后重试有限次。
- `other`：其它。不重试，直接降级。

### 约束由宿主声明、库执行

- 嵌入：`EmbedderOptions.max_batch` 是单批条数上限；`max_tokens_per_text` 是单条文本 token 预算（`None` 不截断）。库永远不把超出声明的文本送出去。
- 重排：`RerankerOptions.max_docs` 是送入重排的候选数上限，`max_tokens_per_doc` / `max_tokens_query` 是文档与查询词的 token 预算。重排模型普遍有硬上限，库在调用前按声明强制截断。
- token 计数与全文分词同源（CJK 字 + 相邻二元组），宿主不必自己数 token。

### 默认值与三档向量化开关

- 每个 namespace 下有三个独立开关：**记忆**（`memory`）、**图谱**（`graph`）、**笔记**（`notes`）。
- 三档覆盖的记录：记忆档是记忆记录；图谱档是实体、关系、事件三类（同进同出）；笔记档是切片——笔记记录自己没有正文，给它算向量等于算空文本，所以这一档作用在切片上。
- 没被显式设置过的档位取内置默认：记忆开、图谱开、笔记关。
- 开关只决定「是否生成」：已有向量保留在库里；记录被删除、或内容变更导致旧向量失效时照常清理，与开关无关。
- 切片正文只存在索引里，所以要给切片补向量得先 `update_index`（正文进索引），再由使用方显式调用 `embeddings().sync` 补齐；写入路径本身不为切片生成向量。
- 三档与领域总闸都落盘在 `meta` 表（档位键 `vectorize:<namespace>:<档位>`，总闸键 `vectorize:<namespace>`），配置一次即生效、重启后仍有效。
- 总闸关闭时该域不生成任何向量，检索也不走向量路；档位关闭只影响该档，全文路照常命中。

### 多档降级

- 写入侧：模型可用则正常生成；模型不可用或该 namespace 关了向量化时，**记录照常写入**、向量留待补齐，不抛错。切片是例外：它的正文只存在索引里，写入时不生成向量，等 `update_index` 与 `embeddings().sync` 之后才补。
- 检索侧：全档是文本 + 向量融合；往下依次是「关向量化的 namespace → 纯全文」「回调挂了 → 纯全文」。任一档都返回结果、不抛错、不返回空，当前落在哪一档见 `SearchDiagnostics.degraded` 与 `HealthReport.last_degraded`。
- 检索的降级只到纯全文为止，不设比 BM25 更弱的检索档；索引查询失败先当场从权威数据重建并重试，重建后仍失败才隔离文本路（向量路照常，记 `text_index_unavailable`），索引目录损坏则在打开时隔离重建。


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
