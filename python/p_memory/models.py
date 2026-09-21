"""JSON-shaped public types. All inputs are validated by the shared Rust core."""
from typing import Any, Literal, TypedDict

RecordKind = Literal["memory", "entity", "relation", "event", "note", "chunk"]


class ReadFilter(TypedDict, total=False):
    namespace: str
    scopes: list[str]
    tags: list[str]
    note_ids: list[int]


class RecordKey(TypedDict):
    id: int


class Evidence(TypedDict, total=False):
    source: str
    source_revision: str | None
    chunk_id: int | None
    offset: int | None
    limit: int | None
    quote: str
    metadata: dict[str, Any]


class RecordInput(TypedDict, total=False):
    id: int
    namespace: str
    scope: str
    tags: list[str]
    evidence: list[Evidence]
    metadata: dict[str, Any]
    created_at_us: int
    updated_at_us: int
    expected_revision: int


class MemoryState(TypedDict, total=False):
    pinned: bool
    strength: int
    useful_count: int
    useful_score: float
    last_recalled_at_us: int | None
    last_decay_at_us: int | None


class MemoryInput(RecordInput, total=False):
    judgment: str
    memory_type: str
    reasoning: str
    state: MemoryState | None


class EntityInput(RecordInput, total=False):
    name: str
    entity_type: str
    aliases: list[str]
    attributes: dict[str, list[str]]
    summary: str


class RelationInput(RecordInput, total=False):
    subject_id: int
    predicate: str
    object_id: int
    confidence: float
    reason: str


class EventInput(RecordInput, total=False):
    name: str
    summary: str
    participants: list[int]
    confidence: float
    reason: str


class NoteFileInput(RecordInput, total=False):
    """按文件路径同步一篇笔记：库自己读文件，路径即身份，标题取文件名（去扩展名）。"""

    path: str
    chunk_chars: int


class EmbeddingSpace(TypedDict):
    id: str
    model: str
    dimension: int
    text_version: int
    encoding: str


class EmbedderOptions(TypedDict, total=False):
    max_batch: int
    max_tokens_per_text: int | None


class RerankerOptions(TypedDict, total=False):
    max_tokens_total: int
    max_candidates: int
    max_tokens_per_doc: int
    max_tokens_query: int | None


class SyncReport(TypedDict):
    scanned: int
    written: int
    batches: int
    interrupted: str | None


class SearchDiagnostics(TypedDict, total=False):
    text_used: bool
    vector_used: bool
    reranked: bool
    rerank_candidates: int
    rerank_truncated: int
    degraded: list[str]


class LogEvent(TypedDict, total=False):
    """库产出的一条事件。`kind` 决定哪些字段有值，无值的字段不出现。

    `stages` 是各阶段耗时（毫秒）；`rerank` 是「等宿主重排回调返回」的时间，
    也就是模型推理时间，不是库的开销。事件里不含查询原文与正文。"""

    ts: str
    kind: str
    ms: int
    stages: dict[str, int]
    candidates: int
    folded: int
    rerank_docs: int
    rerank_tokens: int
    hits: int
    documents: int
    format: str
    degraded: list[str]


class WriteReceipt(TypedDict):
    value: Any
    revision: int


class Page(TypedDict):
    items: list[Any]
    next_cursor: str | None


class ChunkRef(TypedDict):
    """折叠后挂在代表命中上的一个片段：只带定位，正文按 `id` 另取。"""

    id: int
    offset: int


class SearchHit(TypedDict):
    key: RecordKey
    score: float
    text_score: float | None
    vector_scores: dict[str, float]
    rerank_score: float | None
    note_chunks: int | None
    top_chunks: list[ChunkRef]
    record: dict[str, Any]


class SearchResult(TypedDict):
    hits: list[SearchHit]
    revision: int
    indexed_revision: int
    total: int | None
    diagnostics: SearchDiagnostics
