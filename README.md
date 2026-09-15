# p-memory

> English | [简体中文](README.zh-CN.md)

**An embedded memory library: durable memories, a knowledge graph, notes, and retrieval in one local directory, usable from Rust and Python.**

p-memory embeds into your application; there is no separate service to run. Open a data directory from Rust or Python and it gives you the whole path — write, index, filter, recall — with no external database and nothing leaving your machine.

## Quick start

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

## Features

- **Changing the embedding model doesn't break the database.** Register a new embedding space for the new model; the old data stays where it is, with no full rebuild.
- **As small as possible.** Text lives in normalized tables — one shared string dictionary, tag links as plain integer ids, nothing stored twice. For personal scale, keeping the store minimal is a deliberate goal.
- **Memory, graph, and notes in a single search.** They share one index, so one query returns all three, fused with RRF.
- **Fast.** For LLM memory, responsiveness comes first: local, single-file, no network round trips.

## Memory, the graph, and notes

**Memory** — holds both loose fragments and structured user profiles, and the two coexist. It also scales to a chat room of a thousand people: each person's memory stays its own, with no bleed between them.

**The graph** — entities, relations, and events. Find an entity by name, look at its neighborhood, or trace the shortest way two entities are connected.

**Notes** — documents chunked by paragraph, each chunk keeping its source file and line range.

**Isolation** — memories, the graph, and notes all support separation by domain and by agent. One store can serve several domains and several agents at once, each seeing only its own.

**How they combine** — the three are wired together: tag a memory and that tag can be an entity in the graph, so the memory and the entity point at the same person or thing and can be walked between without building a link by hand. Retrieval is undivided too — one query returns memories, the graph, and notes together.

Every host application uses its **own data directory**. p-memory ships the library and the import tooling; wiring it into **P-ai**, **astrbot_plugin_angel_memory**, or another host happens in that host's repository.

## Rust

```toml
[dependencies]
p-memory = { path = "../p-memory" }
# Pin a version once the repository is pushed to your own Git remote.
```

Graph search is built in and needs no feature flags:

```rust
use p_memory::{KnowledgeBase, ReadFilter};

let kb = KnowledgeBase::open("./data/rust-app")?;
let filter = ReadFilter { namespace: "default".into(), scopes: vec!["public".into()], tags: vec![] };

let alice = kb.graph().resolve("Alice", &filter, 1)?[0].header.id;
let bob   = kb.graph().resolve("Bob",   &filter, 1)?[0].header.id;

let ring  = kb.graph().ego(alice, 2, &filter, 50)?;   // entities within two hops
let chain = kb.graph().path(alice, bob, &filter)?;    // shortest bridge between two entities
```

> The Python binding exposes the same capabilities: `ego` / `path` / `strongly_connected` / `component_count`.

## Python

CPython 3.10+. Windows and Linux wheels require no Rust; building from source needs Rust 1.88+ and a C/C++ toolchain.

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

Async hosts use `async with AsyncKnowledgeBase(...) as kb`, then `await kb.memories.upsert(...)` / `await kb.search(...)`. Operations run on a worker thread and release the GIL; cancelling a wait cannot roll back a write that has already started.

## Simple agent

```powershell
# Model name and endpoint come from your provider. The API key is read from the environment only.
$env:OPENAI_BASE_URL = "http://localhost:8000/v1"
$env:P_MEMORY_MODEL = "your-tool-calling-model"
# Set OPENAI_API_KEY if the endpoint requires authentication.

p-memory chat --data ./data/agent --prompt "Remember: I like unsweetened tea" --trace
p-memory chat --data ./data/agent --prompt "What do I like to drink?" --trace
p-memory chat --data ./data/agent
```

Uses Chat Completions `tools` / `tool_calls`, with custom endpoint, model, timeout, round cap, read-only tool mode, and multi-tool replies. BM25 is the default; `--embedding-model` and `--embedding-dimension` enable semantic retrieval. Implementation: [agent.py](python/p_memory/agent.py).

## Importing existing data

```powershell
p-memory import --source p_ai --source-id my-host `
  --database ./backup/memory_store.db --destination ./data/imported
# Review the report, then add --apply. The default is a dry run that creates no destination.
```

Supports importing snapshots from other memory systems; run `p-memory import --help` for the available sources. Original fields, lifecycle, scopes, and stable ID mappings are preserved; the source database is read-only. Graph lookup and note roots can be set explicitly. Old vector / FAISS / Tantivy caches are rebuilt, and the report lists the text still pending embedding.

## Documentation and testing

- [Overview](docs/overview.md) · [Usage](docs/usage.md)
- [Architecture](docs/architecture.md) · [Data model](docs/data-model.md)
- [Rust API](docs/api-rust.md) · [Python API](docs/api-python.md)
- [Search](docs/search.md) · [Lifecycle](docs/lifecycle.md)
- [Graph generation](docs/graph-generation.md) · [Graph search](docs/graph-search.md)
- [Examples](examples/)

```powershell
cargo test
python -m pytest
```

Apache-2.0. See [LICENSE](LICENSE).
