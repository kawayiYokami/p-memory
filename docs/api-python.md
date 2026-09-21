# Python API

包名 `p_memory`，用 PyO3 + maturin 构建，原生模块 `p_memory._native`，支持 Python 3.10+（`abi3-py310`），命令行入口 `p-memory`。绑定只做三件事：类型转换、异常映射、调用；不复制核心逻辑。

## 公开形状

```python
import p_memory

kb = p_memory.KnowledgeBase("./data")        # 构造器直接传目录
kb.memories        # MemoryStore
kb.graph           # GraphStore
kb.notes           # NoteStore
kb.embeddings      # EmbeddingStore
kb.search("关键词") # -> SearchResult
kb.search_preset("rag", "关键词", embed_space="e5")   # 预设检索：memory / graph / notes / rag / broad
kb.register_reranker(callback, max_tokens_total=8192)
kb.health()        # -> dict
kb.update_index()  # 追平写入累积的索引待办（批量导入后调用一次）
kb.rebuild_indexes()
kb.rebuild_progress()  # -> {"active": bool, "processed": int, "total": int}，可轮询重建进度
kb.backup(target)
kb.close()

p_memory.KnowledgeBase.restore(snapshot, directory)  # 类方法
```

构造器签名：`KnowledgeBase(path, *, namespace="default", scopes=("public",), write_scope="public")`。
`namespace`、`scopes`、`write_scope` 在实例上固定，读写时自动带入，不必每次传 `filter`。
`filter` 支持 `namespace` / `scopes` / `tags` / `note_ids` 四个维度，彼此取交集。`note_ids` 给出一批笔记的记录 id 后，候选只在这批笔记的切片内产生，其余记录不参与命中，空表示不限定——这是把宿主的「书内搜索」迁进 p-memory 的入口。
异步封装 `AsyncKnowledgeBase` 才提供 `open(path, **options)` 类方法（`async def`）。

删除有两种粒度：`delete(id)` 删单条，`delete_by_filter(filter)` 按过滤条件整批删。三个存储域都有 `delete_by_filter`，返回 `{"value": 删除条数, "revision": ...}`：

```python
kb.memories.delete_by_filter(filter={"namespace": "demo"})                  # 整个域
kb.graph.delete_by_filter(filter={"namespace": "demo"})                     # 实体/关系/事件连边一起清
kb.notes.delete_by_filter(filter={"namespace": "demo", "tags": ["draft"]})  # 只删带这个标签的笔记
```

过滤条件里的 `namespace` 决定删哪个域，`scopes` / `tags` / `note_ids` 收窄范围；空命中返回 0，不是错误。图谱域按「关系 → 事件 → 实体」的顺序连边一起删，笔记域连带切片，级联顺序由库内部保证，调用方不必先解除引用。返回值只计主记录（图谱域是实体/关系/事件之和，笔记域只计笔记本身，随笔记删掉的切片不单独计数）。若某个待删实体仍被过滤条件之外的关系引用，抛 `ConflictError`，整个事务回滚，不做部分删除。

各 Store 的方法与 Rust 侧同名同参，参数与返回值使用下列映射。图搜索暴露 `ego` / `path` / `strongly_connected` / `component_count`（`record_id` 为 `int`，不暴露 `GraphView` 对象）。
谓词元规则与等价词在 Python 侧都可用：`kb.graph.set_predicate_rule(predicate, inverse=None, symmetric=False)` 登记对称/逆谓词、`kb.graph.delete_predicate_rule(predicate)` 撤销（内置 `sys:same_as` 同样可撤）；`kb.graph.set_predicate_equivalents(namespace, groups)` 按领域登记等价组（持久化，上游提供，库不内置）、`kb.graph.predicate_equivalents(namespace)` 列出、`kb.graph.delete_predicate_equivalents(namespace, predicates=None)` 撤销（不给词就清掉整域）、`kb.graph.expand_query(namespace, text)` 单独调用扩散。扩散也已在全文路与预设检索图谱路内部自动生效。

笔记多两个方法，用来登记领域根目录：

```python
kb.notes.set_root("demo", "./data/demo")   # 必须是已存在的目录
kb.notes.root("demo")                                  # -> "./data/demo"，没登记为 None
kb.notes.unset_root("demo")                            # 注销登记，返回是否命中
kb.notes.upsert_file(path="./data/demo/notes/characters/overview.md")
```

登记之后库里存的是相对路径 `notes/characters/overview.md`，拆出的 `notes` / `characters` / `overview` 与调用方标签合并、挂到这一篇的每条切片上；根目录是写入前提，没登记就写 `upsert_file` 直接报 `ValidationError`，`path` 不在根目录之内同样报 `ValidationError`。

预设检索：

```python
kb.search_preset("rag", "朱樱和白露的同学是谁", embed_space="e5")
# -> {"preset": "rag", "memories": [...], "graph": {...},
#     "notes": {"titles": [...], "contents": [...], "paths": [...]}, ...}
```

签名是 `search_preset(preset="rag", query="", *, filter=None, embed_space=None, text=True, vector=True, rerank=True, budget=None, candidate_limit=64)`；`preset` 取 `memory` / `graph` / `notes` / `rag` / `broad`。返回的 `memories` / `graph` / `notes` 三个字段各自独立排序、各自按字符数封顶，不混在一起，没走的那一路是空的。`graph` 里分 `entities` / `relations` / `context_relations` / `context_events` 四块；`notes` 里分 `titles`（文件名命中）/ `contents`（正文命中，同一篇只留最高的一片、每条带 `note_chunks` 报出这一篇的片段总数，`top_chunks` 给出这一篇在候选窗口内的几片）/ `paths`（书名块不够时用目录段兜底）三块，同时返回、互不重复。阈值按 `budget` 逐项覆盖（`dict[str, int]`，键名与 `PresetBudget` 字段一致，含 `note_titles`），默认值见 [search](search.md#预设检索)。异步封装：`await kb.search_preset("关键词", preset="broad")`。

## 向量与重排回调

向量化与重排都在库内部完成。宿主只注册模型接口并提供内容，不自己算向量、不自己重排：

```python
kb.embeddings.register_space({"id": "e5", "model": "e5-base", "dimension": 768})
kb.embeddings.register_embedder("e5", my_embed_fn, max_batch=50)   # 注册即用样本校验
kb.embeddings.sync("e5", batch=50)                                # 追平索引 → 补齐缺口 → 逐档核对并标记就绪
kb.embeddings.vector_ready("demo", "e5", "memory")           # 记忆档补完了没有；未就绪的档不走向量
kb.memories.upsert_by_judgment(judgment="……")                      # 写入不碰向量，向量留给批次结束的 sync
kb.search("偏好", embed_space="e5")                                # 库嵌入查询词，宿主只给词
kb.register_reranker(my_rerank_fn, max_tokens_total=8192, max_candidates=50)  # 进程内单例重排回调
```

- `register_embedder(space_id, callback, *, max_batch=32, max_tokens_per_text=None)`：`callback(texts: list[str]) -> list[list[float]]`。一个向量模型对应一个向量空间，注册即校验，产出不符即拒绝绑定并报 `InvalidVectorError`。
- `register_reranker(callback, *, max_tokens_total=8192, max_candidates=50, max_tokens_per_doc=1024, max_tokens_query=None)`：`callback(query: str, documents: list[str]) -> list[float]`。候选在折叠后从前往后取，条数先到 `max_candidates`、或累加 token 先超 `max_tokens_total`，就停——两者是与门。
- `EmbeddingStore` 另有 `vector_ready` / `unregister_embedder` / `embedder_space` / `spaces` / `namespace_vectorization` / `set_namespace_vectorization` / `vectorization` / `set_vectorization` / `delete_space`。
- `vectorization(namespace, target)` / `set_vectorization(namespace, target, enabled)`：`target` 取 `"memory"` / `"graph"` / `"notes"`，是该领域下三个独立开关；`namespace_vectorization` / `set_namespace_vectorization` 则是整个领域的总闸。没设置过的档位返回内置默认（记忆与图谱为 `True`，笔记为 `False`），档位名不在这三个之一时报 `ValidationError`。
- 回调是运行时状态，不进数据库：宿主启动时注册一次即可。
- 检索时宿主只给 `embed_space`，库用它注册的回调嵌入查询词；`search` 的 `text` / `vector` / `rerank` / `with_total` 各自独立开关，`match_field` 限定全文路在哪一列命中（`all` / `text` / `name` / `path`），未给 `embed_space` 时向量路自动让位（不是错误）。切片命中会折叠：同一篇笔记只留排名最高的一片，每条切片带 `note_chunks`——这一篇命中本次查询的片段总数（与结果窗口、翻页无关），以及 `top_chunks`——这一篇在候选窗口内的几片（`top_chunks_per_note` 控制条数，默认 3，0 为不聚合；第 0 条就是本条自身），按 `id` 调 `notes.get_chunk` 取回正文。

### 事件回调

库在关键执行点产出结构化事件，交给宿主自己管理——落盘、轮转、保留多久都由宿主负责，库不打开日志文件、不持有内存缓冲：

```python
def sink(event: dict) -> None:
    logging.getLogger("p-memory").info("%s", event)

kb.register_event_sink(sink)
kb.unregister_event_sink()      # 返回此前是否有注册
kb.event_sink_registered()
```

- `register_event_sink(callback)`：`callback(event: dict) -> None`。一次检索收到一条 `search` 事件，字段见 [search](search.md#事件流)：`ts` / `kind` / `ms` / `stages`（各阶段耗时）/ `candidates` / `folded` / `rerank_docs` / `rerank_tokens` / `hits` / `degraded`。`stages` 里的 `rerank` 与 `embed` 是等宿主回调返回的时间，也就是模型推理时间，不是库的开销。
- **回调必须非阻塞**：库在检索线程里同步调用它，在里面做同步 IO 或网络上报会把检索拖住，和慢的重排回调一样。
- **回调抛异常只丢这一条事件**，检索照常返回；不注册就完全不产出事件。

### 回调错误分类

嵌入回调在失败时抛出 `EmbedCallbackError`，用 `kind` 显式声明类别，库不解析错误文案：

```python
from p_memory.errors import EmbedCallbackError

def my_embed_fn(texts):
    try:
        return provider.embed(texts)
    except ProviderTooLarge as exc:
        raise EmbedCallbackError.too_large(str(exc)) from exc   # 库把 batch 减半后重试
```

`too_large` → 减半重试；`rate_limited` → 退避重试；`other`（缺省）→ 不重试、直接降级。异常上的 `kind` 属性由绑定读取，普通异常按 `other` 处理。

## 类型映射

| Rust | Python |
|---|---|
| `String` / `&str` | `str` |
| `i64`（微秒、revision） | `int` |
| `usize` | `int` |
| `f64` | `float` |
| `bool` | `bool` |
| `Option<T>` | `T \| None` |
| `Vec<T>` | `list[T]` |
| `Metadata`（`Map<String,Value>`） | `dict[str, Any]` |
| `RecordKind` | `str`（`"memory"`/`"entity"`/…） |
| 领域结构（`Memory`/`Entity`/…） | `dict`（`models.py` 用 `TypedDict` 标注，运行时就是普通字典） |
| `Page<T>` | `dict`，含 `items: list[T]`、`next_cursor: str \| None` |
| `WriteReceipt<T>` | `dict`，含 `value`、`revision` |

- 输入统一接受**同构 dict**（`TypedDict` 只作静态标注，运行时不校验、不做字段转换）。
- 输出统一为 `dict`，内嵌的 `RecordHeader` 字段平铺在同一层（与 Rust `serde(flatten)` 一致）。
- 时间统一为 UTC 微秒整数，绑定不做时区换算。

## 异常映射

以 `Error::code()` 为准，映射到 `p_memory` 的异常层级：

```text
PMemoryError                 # 基类，code = "unknown"
├── ValidationError            # validation（同时继承 ValueError）
├── NotFoundError              # not_found（同时继承 LookupError）
├── ConflictError              # conflict
├── LockedError                # locked
├── ClosedError                # closed
├── SchemaVersionError         # schema_version
├── InvalidVectorError         # invalid_vector（同时继承 ValueError）
├── StaleRevisionError         # stale_revision
├── IndexError                 # index
├── StorageError               # storage
└── IOError                    # io
```

- 绑定必须把 Rust 错误转成上述类型，**不得退化为通用 `RuntimeError`**。
- 异常携带稳定 `code` 属性，便于宿主分支处理。
- 宿主嵌入回调抛出的 `EmbedCallbackError`（`code = "embed_callback"`）不由原生层产生，它带 `kind`（`too_large` / `rate_limited` / `other`），供库判定重试与降级；见「回调错误分类」。
- `__init__.py` 的 `__all__` 导出 `PMemoryError` 与上面除 `IndexError`、`IOError` 之外的全部子类；
  这两个类仍可从 `p_memory.errors` 导入并按 `code` 捕获，但未进顶层导出列表。

## 并发与 GIL

- 同步方法在进入核心操作时**释放 GIL**，避免阻塞其他 Python 线程。
- 核心是同步的、线程安全的；绑定**不**自行创建事件循环或运行时。
- 同一 `KnowledgeBase` 句柄可跨线程共享；数据目录仍由**单进程**持写锁。

## 异步封装

核心不提供 async API；绑定在同一套同步方法上提供 `asyncio` 便捷封装，实现方式为 `asyncio.to_thread`：

```python
result = await asyncio.to_thread(kb.search, "关键词")
```

约定：

- `to_thread` 只是把同步调用丢到线程池，\*\*不引入新的并发语义\*\*；调用方须自行控制并发度。
- 不提供 `async def search(...)` 这类会隐藏线程池行为的自动包装，除非宿主明确要求。
- 该方式保证 Python 事件循环不被核心操作阻塞。

## 类型提示

- 随包提供 `py.typed` 标记与 `python/p_memory/models.py` 的 `TypedDict` 定义，覆盖公开的输入与返回形状。
- 没有 `.pyi` 存根，类型信息直接来自 `api.py` / `models.py` 里的源码标注。
- `TypedDict` 字段名与 Rust 结构逐一对齐，便于跨语言对照。

## 构建与分发

- 构建：`maturin build`，workspace 成员 `bindings/python`（crate `p-memory-python`，产物 `_native`）。
- wheel 按平台构建：`abi3-py310` 让单个 wheel 覆盖 Python 3.10 以上，但 Windows、Linux、macOS 各需在对应平台或 CI 上出一份。
- 版本与核心 crate 对齐发布，宿主通过固定版本或 Git revision 使用。
