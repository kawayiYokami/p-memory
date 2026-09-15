# p-memory

> [English](README.md) | 简体中文

**一个可嵌入的记忆库：持久记忆、知识图谱、笔记与检索，共用一个本地目录，Rust 与 Python 皆可用。**

p-memory 由宿主应用直接嵌入，没有要单独运行的服务。你在 Rust 或 Python 里打开一个数据目录，就得到写入、索引、过滤、召回这一整条链路；不依赖外部数据库，也没有网络请求。

## 快速开始

```rust
use p_memory::{KnowledgeBase, MemoryInput, SearchRequest};

fn main() -> p_memory::Result<()> {
    let kb = KnowledgeBase::open("./data/app")?;
    kb.memories().upsert(MemoryInput::new("User prefers unsweetened tea"))?;
    let hits = kb.search(&SearchRequest { query: "unsweetened tea".into(), ..Default::default() })?;
    println!("{:?}", hits);
    kb.close()
}
```

```python
from p_memory import KnowledgeBase

with KnowledgeBase("./data/app") as kb:
    kb.memories.upsert_by_judgment(judgment="User prefers unsweetened tea")
    print(kb.search("unsweetened tea")["hits"])
```

## 特性

- **换嵌入模型不毁库。** 换模型时新登记一个嵌入空间，旧数据原地保留，不必重建整库。
- **体积尽可能小。** 文本用规范化的表存：字符串共用一张字典，标签只落整数 id，重复的不存第二遍。对个人规模而言，把索引体积压到最小是我们刻意的追求。
- **记忆、图谱、笔记一次搜全。** 三者共用同一套索引，一次检索同时返回三处的命中，用 RRF 融成一个列表。
- **快。** LLM 的记忆里，响应速度最重要——本地单文件，没有网络往返。

## 记忆、图谱、笔记

**记忆** —— 既能记零碎的片段，也能记成体系的用户画像，两类可以并存。还能撑住千人级聊天室：每个人的记忆各归各的，不会混到一起。

**图谱** —— 实体、关系与事件。可以按名字找到某个实体、看它周围一圈的邻域，或者找出两个实体之间最短的关联路径。

**笔记** —— 文档按段落切片，每一片都留着自己来自哪个文件、原文件里的哪几行。

**隔离** —— 记忆、图谱、笔记都支持按领域、按 agent 隔离。同一个库可以同时服务多个领域、多个 agent，各自只看得到自己的内容。

**如何结合** —— 三者彼此打通：给一条记忆打上标签，这个标签可以是图谱里的某个实体，于是记忆和实体指着同一个人或物，顺着就能互相找到，不用另外建关联。检索也不分工种，一次查询就能同时拿到记忆、图谱和笔记的结果。

每个宿主应用使用**各自的数据目录**。p-memory 提供库与导入工具；把它接入 **P-ai**、**astrbot_plugin_angel_memory** 或其他宿主，是在那个宿主自己的仓库里完成的。

## Rust

```toml
[dependencies]
p-memory = { path = "../p-memory" }
# Pin a version once the repository is pushed to your own Git remote.
```

图搜索开箱即用，无需 feature 开关：

```rust
use p_memory::{KnowledgeBase, ReadFilter};

let kb = KnowledgeBase::open("./data/rust-app")?;
let filter = ReadFilter { namespace: "default".into(), scopes: vec!["public".into()], tags: vec![] };

let alice = kb.graph().resolve("Alice", &filter, 1)?[0].header.id;
let bob   = kb.graph().resolve("Bob",   &filter, 1)?[0].header.id;

let ring  = kb.graph().ego(alice, 2, &filter, 50)?;   // entities within two hops
let chain = kb.graph().path(alice, bob, &filter)?;    // shortest bridge between two entities
```

> Python 绑定暴露同样的能力：`ego` / `path` / `strongly_connected` / `component_count`。

## Python

CPython 3.10+。Windows 与 Linux 的预编译 wheel 无需 Rust；从源码构建需要 Rust 1.88+ 和一套 C/C++ 工具链。

```powershell
# from the project root
python -m pip install .
# or install a prebuilt wheel
python -m pip install <path-to-wheel>
```

```python
from p_memory import KnowledgeBase

with KnowledgeBase("./data/python-app") as kb:
    receipt = kb.memories.upsert_by_judgment(
        judgment="User prefers unsweetened tea", tags=["preference"]
    )
    print(receipt["value"]["id"])
    print(kb.search("unsweetened tea")["hits"])
```

异步宿主用 `async with AsyncKnowledgeBase(...) as kb`，再 `await kb.memories.upsert(...)` / `await kb.search(...)`。操作在工作线程上执行并释放 GIL；取消一次等待无法回滚已经开始进行的写入。

## 简易 agent

```powershell
# Model name and endpoint come from your provider. The API key is read from the environment only.
$env:OPENAI_BASE_URL = "http://localhost:8000/v1"
$env:P_MEMORY_MODEL = "your-tool-calling-model"
# Set OPENAI_API_KEY if the endpoint requires authentication.

p-memory chat --data ./data/agent --prompt "Remember: I like unsweetened tea" --trace
p-memory chat --data ./data/agent --prompt "What do I like to drink?" --trace
p-memory chat --data ./data/agent
```

使用 Chat Completions 的 `tools` / `tool_calls`，支持自定义端点、模型、超时、轮次上限、只读工具模式与多工具回复。默认 BM25；`--embedding-model` 与 `--embedding-dimension` 可启用语义检索。实现见 [agent.py](python/p_memory/agent.py)。

## 导入既有数据

```powershell
p-memory import --source p_ai --source-id my-host `
  --database ./backup/memory_store.db --destination ./data/imported
# Review the report, then add --apply. The default is a dry run that creates no destination.
```

支持从其他记忆系统导入快照，可用来源见 `p-memory import --help`。原始字段、生命周期、作用域与稳定的 ID 映射都会被保留；来源数据库全程只读。图谱查找与笔记根目录可显式指定。旧的向量 / FAISS / Tantivy 缓存会被重建，报告中会列出尚待嵌入的文本。

## 文档与测试

- [概览](docs/overview.md) · [用法](docs/usage.md)
- [架构](docs/architecture.md) · [数据模型](docs/data-model.md)
- [Rust API](docs/api-rust.md) · [Python API](docs/api-python.md)
- [检索](docs/search.md) · [生命周期](docs/lifecycle.md)
- [图谱生成](docs/graph-generation.md) · [图搜索](docs/graph-search.md)
- [示例](examples/)

```powershell
cargo test
python -m pytest
```

Apache-2.0。见 [LICENSE](LICENSE)。
