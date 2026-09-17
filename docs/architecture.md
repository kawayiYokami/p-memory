# 架构

本文件说明 p-memory 的架构：分层、模块、数据表关系与关键时序。

## 1. 系统分层

```mermaid
flowchart TD
    subgraph H["宿主"]
        A["P-ai（Rust）"]
        C["angel_memory（Python）"]
    end
    A -->|"crate 依赖"| K["p-memory 核心"]
    C --> P
    P -->|"protocol.rs：版本化 JSON"| K
    K --> API["KnowledgeBase"]
    API --> D["memories / graph / notes / embeddings / search"]
    D --> ST["SQLite store.sqlite3：权威"]
    D --> IX["Tantivy text-v2：可重建投影，承载切片正文唯一副本"]
```

- Rust 宿主直接依赖核心 crate；Python 宿主经 `p_memory` 绑定，绑定只做类型转换并把调用交给 `protocol.rs` 的 JSON 分发。
- 存储分层与权威关系见 [overview](overview.md)。

## 2. 核心模块

```mermaid
flowchart TD
    lib["lib.rs — KnowledgeBase 入口"]
    lib --> memory["memory.rs — 记忆与生命周期"]
    lib --> gstore["graph.rs — 实体 / 关系 / 事件"]
    lib --> notes["notes.rs — 笔记与切片"]
    lib --> embed["embeddings.rs — 向量空间"]
    lib --> search["search.rs — 关键词 / 向量 / RRF"]

    memory --> storage
    gstore --> storage
    notes --> storage
    embed --> storage
    search --> storage
    search --> index["index.rs — Tantivy 全文投影"]
    gstore --> gsearch["graph_search.rs — 内存态图搜索（petgraph）"]

    storage["storage.rs — 事务 / 读写 / strings 映射 / 锁 / 备份"]
    storage --> schema["schema.rs + schema.sql — 建表与版本"]
    storage --> types["types.rs — 公共类型"]
    storage --> text["text.rs — 归一化 / 分词 / 摘要"]
    storage --> err["error.rs — 错误码"]

    legacy["legacy.rs — 离线导入"] --> storage
    protocol["protocol.rs — 版本化 JSON 分发"] --> lib
```

| 模块 | 职责 |
|---|---|
| `lib.rs` | 唯一入口 `KnowledgeBase`，聚合五个域 |
| `storage.rs` | 事务封装、记录读写、读连接池（空闲复用 / 池空新建）、向量分区缓存、`strings` 字典映射、写锁、备份恢复 |
| `types.rs` | `RecordInput`/`RecordHeader`/`ReadFilter`/`PageRequest`/`WriteReceipt`/`Evidence` |
| `text.rs` | 归一化、CJK 分词、内容摘要 |
| `error.rs` | 错误类型与错误码 |
| `schema.rs` / `schema.sql` | 建表语句、schema 版本、初始化 |
| `memory.rs` / `graph.rs` / `notes.rs` / `embeddings.rs` / `search.rs` | 五个领域 |
| `index.rs` | Tantivy 全文投影，可重放 |
| `graph_search.rs` | 内存态图搜索：按 `ReadFilter` 读一次快照建 petgraph 图，补虚拟边与 `sys:same_as` 缩点，用完即弃 |
| `legacy.rs` | 三来源离线导入 |
| `protocol.rs` | 给薄绑定用的版本化 JSON 边界，只解码与分发 |

## 3. 数据表关系

`strings` 是唯一的字典表：所有会重复出现的标记（tag、namespace、scope、entity_type、predicate、attr_key、source，以及 payload 内的 `memory_type`）都只把文本存这一份，别处只存整数 id。`kind` 是编译期可穷举的固定枚举，直接存整数、不建表。

```mermaid
erDiagram
    strings }o--o{ records : "namespace_id / scope_id"
    strings }o--o{ record_tags : "tag_id"
    records ||--o{ record_tags : "record_id"

    records ||--|| entities : "record_id"
    strings }o--o{ entities : "entity_type_id"
    entities ||--o{ entity_aliases : "entity_id"
    strings }o--o{ entity_aliases : "alias_id"
    entities ||--o{ entity_attributes : "entity_id"
    strings }o--o{ entity_attributes : "attr_key_id"

    records ||--|| relations : "record_id"
    entities ||--o{ relations : "subject_id / object_id"
    strings }o--o{ relations : "predicate_id"

    records ||--o{ event_participants : "event_id"
    entities ||--o{ event_participants : "entity_id"

    records ||--|| notes : "record_id"
    strings }o--o{ notes : "namespace_id / scope_id"
    notes ||--o{ chunks : "note_id"
    records ||--|| chunks : "record_id"

    records ||--o{ embeddings : "record_id"
    embedding_spaces ||--o{ embeddings : "space_id"

    strings {
        INTEGER id PK
        TEXT text
    }
    records {
        INTEGER id PK
        INTEGER namespace_id FK
        INTEGER kind
        INTEGER scope_id FK
        INTEGER revision
        TEXT payload_json
        TEXT fingerprint
    }
    record_tags {
        INTEGER record_id FK
        INTEGER tag_id FK
    }
    entities {
        INTEGER record_id PK
        TEXT name
        INTEGER entity_type_id FK
    }
    entity_aliases {
        INTEGER entity_id FK
        INTEGER alias_id FK
    }
    entity_attributes {
        INTEGER entity_id FK
        INTEGER attr_key_id FK
        TEXT attr_value
    }
    relations {
        INTEGER record_id PK
        INTEGER subject_id FK
        INTEGER predicate_id FK
        INTEGER object_id FK
    }
    event_participants {
        INTEGER event_id FK
        INTEGER entity_id FK
    }
    notes {
        INTEGER record_id PK
        INTEGER namespace_id FK
        INTEGER scope_id FK
        TEXT path
    }
    chunks {
        INTEGER record_id PK
        INTEGER note_id FK
        INTEGER ordinal
        INTEGER offset
        INTEGER limit
        TEXT fingerprint
    }
    embedding_spaces {
        TEXT id PK
        TEXT model
        INTEGER dimension
        INTEGER text_version
    }
    embeddings {
        TEXT space_id FK
        INTEGER record_id FK
        TEXT fingerprint
        BLOB vector
    }
```

关键点：

- `records` 是所有领域的宽表，主键 `id` 自增整数，与任何外部 ID 无关；`kind` 区分领域。领域正文的权威：记忆在 `payload_json` 的 `judgment`；笔记切片的正文唯一副本在全文索引的 stored 列（写入时从源文件读一次、切好带过去），`payload_json` 只留切片粒度，路径在 `notes` 表。记忆的 `memory_type` 也以 `memory_type_id` 存在 payload 内，指向 `strings`，读取时还原文本。
- 可检索正文与向量输入都是**派生文本**，不落 SQLite 列：写入时按 `kind` 从 payload（笔记则读文件）现算，可检索正文交给 Tantivy 的 stored 字段。
- 各领域投影表（`entities`/`relations`/`notes`/`chunks` 等）的 `record_id` 直接复用 `records.id`，靠外键与级联删除维持一致性。
- 笔记的路径只在 `notes` 表存一次；领域登记过根目录（`namespace_roots`）时存的是减掉根目录的相对路径，标题由路径文件名派生。路径**不占索引列**：它按 `/` 拆段变成标签，挂在这一篇的每条切片上，检索面留在切片。
- `chunks.fingerprint` 保留「内容未变则复用向量」的语义：`ordinal` 与内容都没变的切片复用原 `record_id`，向量继续有效。记录指纹按「正文 + 标签」算，标签换了（例如刚登记根目录拆出路径段）指纹就换，旧向量作废。
- `embedding_spaces` 的 `id` 是文本（模型标识），`embeddings` 是「空间 × 记录」的复合主键。
- `index_updates`、`import_runs`、`meta` 是辅助表：索引重放日志、导入批次指纹、修订号计数器。

## 4. 一次写入的时序

调用方报字符串，核心在存储层内部换算成 id。

```mermaid
sequenceDiagram
    participant H as 宿主
    participant K as KnowledgeBase
    participant S as storage（SQLite 事务）
    participant I as Tantivy

    H->>K: upsert(input)（namespace / scope / tag 均为字符串；笔记按 path 读文件并切好正文）
    K->>S: mutate() 开启 IMMEDIATE 事务
    S->>S: 字符串 → strings id（不存在则插入新行）
    S->>S: 写 records 与投影表，revision += 1，登记 index_updates
    S->>S: 该领域各档的就绪标记就地作废
    S-->>K: 提交事务
    K->>I: 就地写入索引文档（正文一路带过来，未提交、对搜索不可见）
    K-->>H: WriteReceipt{ revision }
    H->>K: update_index()（批量导入后调用一次）
    K->>I: 一趟提交并记账（只提交，不回读文件）
    H->>K: embeddings().sync(space, batch)（批次结束后触发一次）
    K->>K: 先追平索引，再调模型补齐缺口，最后逐档核对缺口标成就绪
```

- 数据提交与索引更新分开：写入只登记待办（`index_updates`）并在同一个调用里把文档写进索引 writer，不 commit；索引由使用方在合适时机调用 `update_index()` 一趟提交、只提交一次。读取若发现待办会尽力自愈（抢不到写锁就跳过，绝不排队），所以写入后照样能查到，只是索引可能滞后到那一刻。
- **写入一行向量都不产生、一次模型都不调**。向量化是独立的一步：批次结束后由调用方调 `sync` 补齐，库内线程兜底，且线程是纯事件触发（开库、注册或切换模型、改档位、写入提交后各叫一次），不轮询、不设定时。库内线程只写 `embeddings` 表，不写业务记录、不改索引。
- 正文只读一次、只切一次：写入时就地切好，随文档一路带到索引；索引阶段不再回读源文件。唯一需要回源的是重建（格式升级、索引损坏、上次写入未收尾）与向量化取切片正文——后者的正文只存在索引里，所以补齐前先把索引追平。
