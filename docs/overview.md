# 总览

p-memory 是给多个宿主共用的**离线记忆底层**：统一记忆、知识图谱、笔记的数据模型、存储与检索实现。Rust 直接依赖核心 crate，Python 通过 PyO3 绑定调用同一套代码，各宿主使用自己的数据目录。

它只负责存储与检索。模型调用、重排、文件监控、聊天流程、业务规则和 UI 都由宿主负责。

## 架构

分层、模块结构与数据表关系见 [architecture](architecture.md)；拿到手怎么用见 [usage](usage.md)；从文本生成图谱的格式标准见 [graph-generation](graph-generation.md)；图谱搜索能力见 [graph-search](graph-search.md)。

## 三层数据职责

| 层 | 载体 | 是否权威 | 说明 |
|---|---|---|---|
| 结构化数据 | SQLite `store.sqlite3` | 权威 | 记录、标签、图谱引用、笔记路径、索引更新日志 |
| 全文索引 | Tantivy `text-v2/` | 投影，可重建 | 服务检索，并承载正文的唯一副本（stored 正文列）；重建要回源宿主文件 |
| 向量 | SQLite `vectors.sqlite3`（外挂派生库） | 派生 | 宿主注册模型回调，库内部生成并写回，按向量空间隔离 |

**SQLite 是记录与关系的唯一权威**，笔记正文是例外：它只在全文索引的 stored 列里存一份，索引重建时必须回源宿主文件。写入在事务提交后把文档就地写进索引 writer（不 commit），由使用方择时调用 `update_index` 提交一次。索引落后或重建都不会让已提交的数据失效。

## 数据目录布局

一个 `KnowledgeBase` 对应一个目录：

```
<data-dir>/
├── writer.lock          # 进程级独占写锁
├── store.sqlite3        # 权威数据
├── vectors.sqlite3      # 向量外挂派生库（独立 WAL、独立连接）
└── text-v2/             # Tantivy 全文索引（后端自管）
```

- 打开时若目录不存在会自动创建，并对 `writer.lock` 尝试独占锁；已被其他进程持有时返回 `locked`。
- 首次初始化会写入 `application_id = 0x5041494d`（"PAIM"）与 schema 版本号 `SCHEMA_VERSION`，用于识别库归属与 schema 版本；旧库打开时按版本逐级前滚迁移。

## 统一入口

`KnowledgeBase` 是唯一入口，下设五个域：

| 域 | 获取方式 | 职责 |
|---|---|---|
| 记忆 | `kb.memories()` | 论断、标签、生命周期、衰减、反馈 |
| 图谱 | `kb.graph()` | 实体与别名、属性、关系、事件 |
| 笔记 | `kb.notes()` | 文件路径与带行号的切片（正文随写入进索引，库内不留副本） |
| 向量 | `kb.embeddings()` | 向量空间注册、模型回调注册、内部同步补齐、领域总闸与记忆/图谱/笔记三档开关 |
| 检索 | `kb.search(&req)` | 关键词、向量、混合检索、重排 |

五者共享同一数据库与事务基础。写入操作走统一的事务封装，返回 `WriteReceipt`；读取操作走统一的 `ReadFilter`。

## 与宿主的关系

| 宿主 | 语言 | 接入的内容 | 备注 |
|---|---|---|---|
| P-ai | Rust | 记忆 + 笔记 | 无知识图谱 |
| angel_memory | Python | 记忆 + 笔记 | 中文类型枚举、三档衰减 |

每个宿主使用自己的数据目录；把 p-memory 接入某个宿主，是在那个宿主自己的仓库里完成的。

命令行入口 `p-memory import` 从其他记忆系统导入数据（`src/legacy.rs`）：

- 来源由 `--source` 指定，可用值见 `p-memory import --help`；`--graph-database` 与 `--lookup-database` 只对支持图谱的来源有效（其他来源传入会报 `validation`），两者均可选。
- `--notes-root` 对所有来源有效：只有落在该目录下的文件才会被当作笔记来源读入。
- 默认是 **dry run**（只预览、不写入），加 `--apply` 才落库；来源数据库全程只读。
- 导入按 `source_id` 记录批次指纹到 `import_runs`，重复导入同一快照会被识别（`already_imported`）。
- 台账可以撤：`delete_import_run(destination=..., source_id=...)` 删掉那条记录，同一目录就能重新导入同一来源；只删台账，不动已导入的数据。

接口契约见 [data-model](data-model.md)、[api-rust](api-rust.md)、[api-python](api-python.md)。
