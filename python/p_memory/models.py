"""JSON-shaped public types. All inputs are validated by the shared Rust core."""
from typing import Any, Literal, TypedDict

RecordKind = Literal["memory", "entity", "relation", "event", "note", "chunk"]


class ReadFilter(TypedDict, total=False):
    namespace: str
    scopes: list[str]
    tags: list[str]


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


class NoteInput(RecordInput, total=False):
    source: str
    title: str
    content: str
    chunk_chars: int


class NoteFileInput(RecordInput, total=False):
    """按文件路径同步一篇笔记：库自己读文件，标题取文件名（去扩展名）。"""

    source: str
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
    max_docs: int
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


class WriteReceipt(TypedDict):
    value: Any
    revision: int
    index_ready: bool
    index_error: str | None


class Page(TypedDict):
    items: list[Any]
    next_cursor: str | None


class SearchHit(TypedDict):
    key: RecordKey
    score: float
    text_score: float | None
    vector_scores: dict[str, float]
    rerank_score: float | None
    record: dict[str, Any]


class SearchResult(TypedDict):
    hits: list[SearchHit]
    revision: int
    indexed_revision: int
    total: int | None
    diagnostics: SearchDiagnostics
