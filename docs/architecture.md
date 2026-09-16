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
    D --> IX["Tantivy text-v3：可重建投影"]
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

- `records` 是所有领域的宽表，主键 `id` 自增整数，与任何外部 ID 无关；`kind` 区分领域。领域正文的权威：记忆在 `payload_json` 的 `judgment`，笔记在宿主文件（`payload_json` 只留切片粒度，路径在 `notes` 表）。记忆的 `memory_type` 也以 `memory_type_id` 存在 payload 内，指向 `strings`，读取时还原文本。
- 可检索正文与向量输入都是**派生文本**，不落 SQLite 列：写入时按 `kind` 从 payload（笔记则读文件）现算，可检索正文交给 Tantivy 的 stored 字段。
- 各领域投影表（`entities`/`relations`/`notes`/`chunks` 等）的 `record_id` 直接复用 `records.id`，靠外键与级联删除维持一致性。
- 笔记的路径只在 `notes` 表存一次（→strings）；标题由路径派生。`chunks` 不重复携带正文，只存 `note_id` 与行/字符区间，切片正文按区间从文件原文取出。
- `chunks.fingerprint` 保留「内容未变则复用向量」的语义：`ordinal` 与内容都没变的切片复用原 `record_id`，向量继续有效。
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

    H->>K: upsert(input)（namespace / scope / tag 均为字符串）
    K->>S: mutate() 开启 IMMEDIATE 事务
    S->>S: 字符串 → strings id（不存在则插入新行）
    S->>S: 写 records 与投影表，revision += 1，登记 index_updates
    S-->>K: 提交事务
    K->>I: 同步全文索引（可检索正文按 payload / 文件现算，写入 stored 字段）
    K-->>H: WriteReceipt{ revision, index_ready }
```

- 数据提交与索引更新分离：事务先提交，再写索引。`index_ready=false` 只表示索引未同步，数据仍然已提交。
