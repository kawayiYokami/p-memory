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
kb.health()        # -> dict
kb.rebuild_indexes()
kb.backup(target)
kb.close()

p_memory.KnowledgeBase.restore(snapshot, directory)  # 类方法
```

构造器签名：`KnowledgeBase(path, *, namespace="default", scopes=("public",), write_scope="public")`。
`namespace`、`scopes`、`write_scope` 在实例上固定，读写时自动带入，不必每次传 `filter`。
异步封装 `AsyncKnowledgeBase` 才提供 `open(path, **options)` 类方法（`async def`）。

各 Store 的方法与 Rust 侧同名同参，参数与返回值使用下列映射。图搜索暴露 `ego` / `path` / `strongly_connected` / `component_count`（`record_id` 为 `int`，不暴露 `GraphView` 对象）。
**谓词元规则 `set_predicate_rule` 仅在 Rust 侧提供，Python 未暴露**；用法示例见 [graph-search](graph-search.md#python)。

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
| `WriteReceipt<T>` | `dict`，含 `value`、`revision`、`index_ready`、`index_error` |

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
