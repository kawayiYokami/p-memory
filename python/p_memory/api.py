"""Thin Python API. Persistence, tokenization and ranking all run in Rust."""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from . import _native
from .errors import ClosedError, ValidationError, from_native
from .models import (EmbeddingSpace, EntityInput, EventInput, MemoryInput,
                     NoteFileInput, Page, ReadFilter, RecordKind, RelationInput,
                     SearchResult, WriteReceipt)


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc


def _unwrap(encoded: str) -> Any:
    response = json.loads(encoded)
    if not response["ok"]:
        raise from_native(response["error"])
    return response["result"]


def _open_native(factory: Callable, *args: str):
    return _native_call(factory, *args)


def _native_call(factory: Callable, *args: Any) -> Any:
    try:
        return factory(*args)
    except RuntimeError as exc:
        try:
            error = json.loads(str(exc))
        except (ValueError, TypeError):
            raise
        raise from_native(error) from exc


class KnowledgeBase:
    """An embedded database with an exclusive process lock on its directory.

    Scopes are explicit. Public records are not implicitly included when reading
    a private scope. A host must enforce its own authorization before passing
    filters. Instances in different applications should use separate paths.
    """

    def __init__(self, path: str | os.PathLike, *, namespace: str = "default",
                 scopes: Sequence[str] = ("public",), write_scope: str = "public"):
        self._configure(namespace, scopes, write_scope)
        self._native = _open_native(_native.NativeKnowledgeBase, os.fspath(path))

    def _configure(self, namespace: str, scopes: Sequence[str], write_scope: str) -> None:
        if isinstance(scopes, str) or not scopes:
            raise ValidationError("scopes must be a nonempty sequence of scope names")
        if any(not isinstance(s, str) or not s.strip() or s != s.strip() or any(ord(c) < 32 for c in s)
               for s in (namespace, write_scope, *scopes)):
            raise ValidationError("namespace and scopes must be nonempty trimmed strings")
        self.namespace, self.scopes, self.write_scope = namespace, tuple(scopes), write_scope
        self.memories = MemoryStore(self)
        self.graph = GraphStore(self)
        self.notes = NoteStore(self)
        self.embeddings = EmbeddingStore(self)

    @classmethod
    def restore(cls, snapshot: str | os.PathLike, directory: str | os.PathLike,
                **options: Any) -> KnowledgeBase:
        value = cls.__new__(cls)
        value._configure(options.pop("namespace", "default"), options.pop("scopes", ("public",)), options.pop("write_scope", "public"))
        if options:
            raise TypeError(f"unexpected options: {', '.join(options)}")
        value._native = _open_native(_native.NativeKnowledgeBase.restore, os.fspath(snapshot), os.fspath(directory))
        return value

    def invoke(self, operation: str, payload: Any = None) -> Any:
        """Call the version 1 JSON protocol directly. Normal users use stores."""
        return _unwrap(self._native.call(operation, _json({} if payload is None else payload)))

    def _filter(self, filter: ReadFilter | None = None) -> dict:
        return {"namespace": self.namespace, "scopes": list(self.scopes), **(filter or {})}

    def _input(self, data: Mapping[str, Any] | None, kwargs: dict | None = None) -> dict:
        return {"namespace": self.namespace, "scope": self.write_scope, **(data or {}), **(kwargs or {})}

    def search(self, query: str = "", *, filter: ReadFilter | None = None,
               kinds: Sequence[RecordKind] | None = None, limit: int = 10,
               candidate_limit: int | None = None, embed_space: str | None = None,
               text: bool = True, vector: bool = True, rerank: bool = True,
               with_total: bool = False, text_weight: float = 1.0,
               match_field: str = "all", top_chunks_per_note: int = 3) -> SearchResult:
        """检索。向量能力由库内部完成：给出 `embed_space`，库用该空间注册的
        回调嵌入查询词；宿主只给搜索词，不给向量。`text` / `vector` / `rerank`
        / `with_total` 各自独立开关。`match_field` 限定全文路在哪一列命中：
        `all`（默认，正文+名字）/ `text` / `name` / `path`。切片按笔记折叠后，
        `top_chunks_per_note`（默认 3，0 为不聚合）决定每条命中最多带回这一篇
        在候选窗口内的几片，落在命中的 `top_chunks` 里。"""
        payload = {"query": query, "filter": self._filter(filter), "limit": limit,
                   "candidate_limit": candidate_limit, "text_weight": text_weight,
                   "text": text, "vector": vector, "rerank": rerank, "with_total": with_total,
                   "match_field": match_field, "top_chunks_per_note": top_chunks_per_note}
        if embed_space is not None:
            payload["embed_space"] = embed_space
        if kinds is not None:
            payload["kinds"] = list(kinds)
        return self.invoke("search", payload)

    def search_preset(self, preset: str = "rag", query: str = "", *,
                      filter: ReadFilter | None = None, embed_space: str | None = None,
                      text: bool = True, vector: bool = True, rerank: bool = True,
                      budget: Mapping[str, int] | None = None,
                      candidate_limit: int = 64) -> dict[str, Any]:
        """按预设检索：库预先配好的搜索方法，`preset` 取 memory / graph / notes
        / rag / broad。返回的记忆、图谱、笔记三个字段各自独立排序，不混在一起；
        字符数等阈值可在 `budget` 里逐项覆盖。"""
        payload = {"preset": preset, "query": query, "filter": self._filter(filter),
                   "text": text, "vector": vector, "rerank": rerank,
                   "candidate_limit": candidate_limit}
        if embed_space is not None:
            payload["embed_space"] = embed_space
        if budget is not None:
            payload["budget"] = dict(budget)
        return self.invoke("preset", payload)

    def register_reranker(self, callback: Callable, *, max_tokens_total: int = 8192,
                          max_candidates: int = 50,
                          max_tokens_per_doc: int = 1024,
                          max_tokens_query: int | None = None) -> None:
        """注册重排回调：`callback(query: str, documents: list[str]) -> list[float]`。
        库在调用前按声明的条数上限与 token 预算强制截断，两者是与门：候选从前往后取，
        条数先到 `max_candidates`、或累加 token 先超 `max_tokens_total`，就停下不再送。"""
        _native_call(self._native.register_reranker, callback, max_tokens_total,
                     max_candidates, max_tokens_per_doc, max_tokens_query)

    def register_event_sink(self, callback: Callable) -> None:
        """注册事件回调：`callback(event: dict) -> None`。

        库在关键执行点产出事件（`search`），交给这个回调；
        落盘、轮转、保留多久都由宿主自己负责，库不碰文件。

        库在检索线程里同步调用回调，所以它必须非阻塞——在里面做同步 IO 或
        网络上报会把检索拖住，和慢的重排回调一样。回调抛错只丢这一条事件，
        不影响检索。不注册就完全不产出事件。"""
        _native_call(self._native.register_event_sink, callback)

    def unregister_event_sink(self) -> bool:
        """注销事件回调，返回此前是否有注册。"""
        return self._native.unregister_event_sink()

    def event_sink_registered(self) -> bool:
        """当前是否注册了事件回调。"""
        return self._native.event_sink_registered()

    def health(self) -> dict[str, Any]:
        return self.invoke("health")

    def update_index(self) -> dict[str, Any]:
        """提交写入攒下的索引增删（批量写入后调用一次即可）。"""
        return self.invoke("update_index")

    def reconcile_index(self) -> dict[str, Any]:
        """显式对账：把全文索引与主库求差集对齐。停电恢复覆盖不了的残余
        （例如绕过 API 直改主库）由这里一次收敛；稳态下调用是空操作。"""
        return self.invoke("reconcile")

    def backup(self, path: str | os.PathLike) -> None:
        self.invoke("backup", {"path": os.fspath(path)})

    def close(self) -> None:
        self.invoke("close")

    def __enter__(self) -> KnowledgeBase:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class _Store:
    prefix: str

    def __init__(self, kb: KnowledgeBase):
        self._kb = kb

    def _call(self, method: str, payload: Any) -> Any:
        return self._kb.invoke(f"{self.prefix}.{method}", payload)

    def get(self, id: str, *, filter: ReadFilter | None = None) -> dict:
        return self._call("get", {"id": id, "filter": self._kb._filter(filter)})

    def get_many(self, ids: Sequence[int], *, filter: ReadFilter | None = None) -> dict[int, dict]:
        """批量读取：等价于逐条 `get`，但一次往返。

        只返回满足 `filter` 的记录，返回 `{id: record}`；未命中过滤条件的 id
        不会出现在结果里（与逐条 `get` 报 NotFound 不同，请按缺失处理）。
        """
        result = self._call("get_many", {"ids": [int(i) for i in ids], "filter": self._kb._filter(filter)})
        return {int(key): value for key, value in result.items()}

    def list(self, *, filter: ReadFilter | None = None, limit: int = 50, after: str | None = None) -> Page:
        return self._call("list", {"filter": self._kb._filter(filter), "limit": limit, "after": after})

    def delete(self, ids: int | str | Sequence[int | str], *, filter: ReadFilter | None = None) -> WriteReceipt:
        """批量删除：`ids` 是一批记录 id（单个 id 也是批量的一种）。
        主库查 id，命中才继续；未命中的 id 静默跳过，返回实际删除条数（`value`）。
        主库标记先行、索引词条派生先行退、向量与主库行最后落；断电留下的
        「正在删除」标记会在下次开机把没删完的做完。"""
        if isinstance(ids, (int, str)):
            ids = [ids]
        return self._call("delete", {"ids": [int(i) for i in ids], "filter": self._kb._filter(filter)})


class MemoryStore(_Store):
    prefix = "memories"

    def upsert(self, data: MemoryInput | None = None, **fields: Any) -> WriteReceipt:
        return self._call("upsert", self._kb._input(data, fields))

    def upsert_many(self, inputs: Sequence[MemoryInput]) -> WriteReceipt:
        return self._call("upsert_many", [self._kb._input(item) for item in inputs])

    def upsert_by_judgment(self, data: MemoryInput | None = None, **fields: Any) -> WriteReceipt:
        return self._call("upsert_by_judgment", self._kb._input(data, fields))

    def feedback(self, recalled_ids: Sequence[str], useful_ids: Sequence[str] = (), *,
                 filter: ReadFilter | None = None, now_us: int | None = None,
                 policy: Mapping[str, Any] | None = None) -> WriteReceipt:
        return self._call("feedback", {"recalled_ids": list(recalled_ids), "useful_ids": list(useful_ids),
                                      "filter": self._kb._filter(filter), "now_us": now_us, "policy": dict(policy or {})})

    def decay(self, *, filter: ReadFilter | None = None, now_us: int | None = None,
              policy: Mapping[str, Any] | None = None) -> WriteReceipt:
        return self._call("decay", {"filter": self._kb._filter(filter), "now_us": now_us, "policy": dict(policy or {})})


class GraphStore(_Store):
    prefix = "graph"

    def apply_batch(self, *, entities: Sequence[EntityInput] = (), relations: Sequence[RelationInput] = (),
                    events: Sequence[EventInput] = ()) -> WriteReceipt:
        return self._call("apply_batch", {"entities": [self._kb._input(v) for v in entities],
                                         "relations": [self._kb._input(v) for v in relations],
                                         "events": [self._kb._input(v) for v in events]})

    def get(self, kind: RecordKind, id: str, *, filter: ReadFilter | None = None) -> dict:
        return self._call("get", {"kind": kind, "id": id, "filter": self._kb._filter(filter)})

    def list(self, kind: RecordKind, *, filter: ReadFilter | None = None, limit: int = 50, after: str | None = None) -> Page:
        return self._call("list", {"kind": kind, "page": {"filter": self._kb._filter(filter), "limit": limit, "after": after}})

    def delete(self, kind: RecordKind, ids: int | str | Sequence[int | str], *,
               filter: ReadFilter | None = None) -> WriteReceipt:
        """批量删除图记录：`ids` 是一批记录 id（单个 id 也是批量的一种）。"""
        if isinstance(ids, (int, str)):
            ids = [ids]
        return self._call("delete", {"kind": kind, "ids": [int(i) for i in ids], "filter": self._kb._filter(filter)})

    def resolve(self, name: str, *, filter: ReadFilter | None = None, limit: int = 10) -> list[dict]:
        return self._call("resolve", {"name": name, "filter": self._kb._filter(filter), "limit": limit})

    def set_predicate_equivalents(self, namespace: str, groups: Sequence[Sequence[str]]) -> WriteReceipt:
        """Register predicate equivalence groups for a knowledge domain.

        Each group is a list of interchangeable predicate spellings (e.g.
        ``["alpha", "beta", "gamma"]``); the first entry is the canonical one. The
        table is supplied by the caller per domain and persisted; the library
        ships no built-in domain data. Re-registering a word moves it to the
        new group. Only affects query-time expansion, never stored predicates.
        """
        return self._call("set_predicate_equivalents",
                          {"namespace": namespace, "groups": [list(group) for group in groups]})

    def delete_predicate_equivalents(self, namespace: str, predicates: Sequence[str] | None = None) -> WriteReceipt:
        """Drop registered predicate equivalence entries for a domain.

        Pass ``predicates`` to remove just those words, or leave it out to clear
        the whole domain's table. Words that were never registered are ignored.
        """
        return self._call("delete_predicate_equivalents",
                          {"namespace": namespace, "predicates": None if predicates is None else list(predicates)})

    def predicate_equivalents(self, namespace: str) -> list[list[str]]:
        """List the predicate equivalence groups registered for a domain."""
        return self._call("predicate_equivalents", {"namespace": namespace})

    def set_predicate_rule(self, predicate: str, inverse: str | None = None, symmetric: bool = False) -> WriteReceipt:
        """Declare a predicate rule: symmetric, or carrying a known inverse.

        The two are mutually exclusive, and rules only affect the in-memory edges
        built at graph-search time.
        """
        return self._call("set_predicate_rule",
                          {"predicate": predicate, "inverse": inverse, "symmetric": symmetric})

    def delete_predicate_rule(self, predicate: str) -> WriteReceipt:
        """Remove a predicate rule. The built-in ``sys:same_as`` can be removed too."""
        return self._call("delete_predicate_rule", {"predicate": predicate})

    def expand_query(self, namespace: str, text: str) -> list[str]:
        """Expand ``text`` with synonyms of any registered predicate it contains.

        Returns the extra terms to append to a search (empty when nothing
        matched or the domain has no table). The same expansion is applied
        automatically inside full-text and preset search.
        """
        return self._call("expand_query", {"namespace": namespace, "text": text})

    def neighbors(self, id: str, *, filter: ReadFilter | None = None, limit: int = 50) -> dict:
        return self._call("neighbors", {"id": id, "filter": self._kb._filter(filter), "limit": limit})

    def events_for_entity(self, id: str, *, filter: ReadFilter | None = None, limit: int = 50) -> list[dict]:
        return self._call("events_for_entity", {"id": id, "filter": self._kb._filter(filter), "limit": limit})

    def ego(self, id: int, depth: int = 1, *, filter: ReadFilter | None = None, limit: int = 50) -> list[dict]:
        """Entities within `depth` hops of `id` (excludes `id` itself)."""
        return self._call("ego", {"id": id, "depth": depth, "filter": self._kb._filter(filter), "limit": limit})

    def path(self, source: int, target: int, *, filter: ReadFilter | None = None) -> list[dict] | None:
        """Shortest bridge from `source` to `target` (entities, both ends included); None if disconnected."""
        return self._call("path", {"from": source, "to": target, "filter": self._kb._filter(filter)})

    def strongly_connected(self, *, filter: ReadFilter | None = None) -> list[list[int]]:
        """Strongly connected rings (size > 1), as lists of record ids."""
        return self._call("strongly_connected", {"filter": self._kb._filter(filter)})

    def component_count(self, *, filter: ReadFilter | None = None) -> int:
        """Number of connected components in the graph."""
        return self._call("component_count", {"filter": self._kb._filter(filter)})


class NoteStore(_Store):
    prefix = "notes"

    def upsert_file(self, data: NoteFileInput | Sequence[NoteFileInput] | None = None,
                    **fields: Any) -> WriteReceipt:
        """按文件路径同步笔记：正文取文件原文，路径即身份，标题取文件名（去扩展名）。
        更新就是「先删后加」：同路径已有笔记，先把旧笔记连同切片整条删掉再当新笔记写入，
        切片记录全部新开，向量由向量对账照主库重算。传列表即批量，一次写锁整批处理。"""
        if isinstance(data, (list, tuple)):
            payload = [self._kb._input(item) for item in data]
        else:
            payload = self._kb._input(data, fields)
        return self._call("upsert_file", payload)

    def set_root(self, namespace: str, root: str | os.PathLike) -> None:
        """登记该知识领域的笔记根目录：之后写入的路径必须是它的子路径，库里存相对路径。"""
        return self._call("set_root", {"namespace": namespace, "root": os.fspath(root)})

    def unset_root(self, namespace: str) -> WriteReceipt:
        """Deregister a domain's notes root directory.

        Only affects how later writes resolve paths; notes already stored keep
        the relative paths they were written with.
        """
        return self._kb.invoke("notes.unset_root", {"namespace": namespace})

    def root(self, namespace: str) -> str | None:
        """该领域登记的笔记根目录；没登记就是 None。"""
        return self._call("root", {"namespace": namespace})

    def chunks(self, note_id: str, *, filter: ReadFilter | None = None) -> list[dict]:
        return self._call("chunks", {"id": note_id, "filter": self._kb._filter(filter)})

    def get_chunk(self, id: str, *, filter: ReadFilter | None = None) -> dict:
        return self._call("get_chunk", {"id": id, "filter": self._kb._filter(filter)})


class EmbeddingStore:
    def __init__(self, kb: KnowledgeBase):
        self._kb = kb

    def register_space(self, data: EmbeddingSpace | None = None, **fields: Any) -> WriteReceipt:
        return self._kb.invoke("embeddings.register_space", {"text_version": 1, "encoding": "sq8", **(data or {}), **fields})

    def register_embedder(self, space_id: str, callback: Callable, *, max_batch: int = 32,
                          max_tokens_per_text: int | None = None) -> None:
        """注册嵌入回调：`callback(texts: list[str]) -> list[list[float]]`。
        一个向量模型对应一个向量空间，注册即用样本校验，产出不符该空间契约就拒绝绑定。"""
        _native_call(self._kb._native.register_embedder, space_id, callback, max_batch, max_tokens_per_text)

    def unregister_embedder(self, space_id: str) -> bool:
        return self._kb.invoke("embeddings.unregister_embedder", {"space_id": space_id})

    def embedder_space(self, space_id: str) -> EmbeddingSpace | None:
        return self._kb.invoke("embeddings.embedder_space", {"space_id": space_id})

    def spaces(self) -> list[EmbeddingSpace]:
        return self._kb.invoke("embeddings.spaces")

    def namespace_vectorization(self, namespace: str) -> bool:
        return self._kb.invoke("embeddings.namespace_vectorization", {"namespace": namespace})

    def set_namespace_vectorization(self, namespace: str, enabled: bool) -> WriteReceipt:
        return self._kb.invoke("embeddings.set_namespace_vectorization", {"namespace": namespace, "enabled": enabled})

    def vectorization(self, namespace: str, target: str) -> bool:
        """某档开关在该领域下的取值；`target` 取 `memory` / `graph` / `notes`。"""
        return self._kb.invoke("embeddings.vectorization", {"namespace": namespace, "target": target})

    def set_vectorization(self, namespace: str, target: str, enabled: bool) -> WriteReceipt:
        """设置某档开关；只决定以后是否生成向量，已有向量保留。"""
        return self._kb.invoke("embeddings.set_vectorization", {"namespace": namespace, "target": target, "enabled": enabled})

    def sync(self, space_id: str, *, batch: int = 32) -> WriteReceipt:
        """批次处理完之后由使用方调一次补齐：把缺失向量的记录分批补齐，
        最后核对缺口、把补齐的领域标成就绪。库内没有后台线程，没人调就一直缺着。"""
        return self._kb.invoke("embeddings.sync", {"space_id": space_id, "batch": batch})

    def vector_ready(self, namespace: str, space_id: str, target: str) -> bool:
        """该领域的某一档在该空间下是否已补齐，`target` 取 memory / graph / notes。
        未就绪的那一档，检索不让它的向量参与打分。"""
        return self._kb.invoke("embeddings.vector_ready", {"namespace": namespace, "space_id": space_id, "target": target})

    def delete_space(self, id: str) -> WriteReceipt:
        return self._kb.invoke("embeddings.delete_space", {"id": id})


def import_legacy(*, source: str, source_id: str, database: str | os.PathLike, destination: str | os.PathLike,
                  graph_database: str | os.PathLike | None = None, lookup_database: str | os.PathLike | None = None,
                  notes_root: str | os.PathLike | None = None, namespace: str = "default", scope: str = "public",
                  dry_run: bool = True) -> dict:
    return _unwrap(_native.import_legacy(_json({"source": source, "source_id": source_id,
        "database": os.fspath(database), "destination": os.fspath(destination),
        "graph_database": os.fspath(graph_database) if graph_database is not None else None,
        "lookup_database": os.fspath(lookup_database) if lookup_database is not None else None,
        "notes_root": os.fspath(notes_root) if notes_root is not None else None,
        "namespace": namespace, "scope": scope, "dry_run": dry_run})))


def delete_import_run(*, destination: str | os.PathLike, source_id: str) -> bool:
    """Drop one import-ledger entry so the same source can be imported again.

    Only the ledger row is removed; data already imported from that source stays.
    """
    return _unwrap(_native.delete_import_run(
        _json({"destination": os.fspath(destination), "source_id": source_id})))


class _AsyncStore:
    def __init__(self, owner: AsyncKnowledgeBase, name: str):
        self._owner, self._name = owner, name

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        if name.startswith("_"):
            raise AttributeError(name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            kb = await self._owner._ensure_open()
            return await asyncio.to_thread(getattr(getattr(kb, self._name), name), *args, **kwargs)
        return call


class AsyncKnowledgeBase:
    """Async facade using asyncio.to_thread; construction defers opening.

    Cancellation stops waiting but cannot roll back an in-flight native write.
    Use `async with` or `await open(...)`, and always close before opening the
    same directory from another process.
    """
    def __init__(self, path: str | os.PathLike, **options: Any):
        self._path, self._options = path, options
        self._sync: KnowledgeBase | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        self.memories = _AsyncStore(self, "memories")
        self.graph = _AsyncStore(self, "graph")
        self.notes = _AsyncStore(self, "notes")
        self.embeddings = _AsyncStore(self, "embeddings")

    @classmethod
    async def open(cls, path: str | os.PathLike, **options: Any) -> AsyncKnowledgeBase:
        result = cls(path, **options)
        await result._ensure_open()
        return result

    async def _ensure_open(self) -> KnowledgeBase:
        async with self._lock:
            if self._closed:
                raise ClosedError("knowledge base is closed")
            if self._sync is None:
                self._sync = await asyncio.to_thread(KnowledgeBase, self._path, **self._options)
            return self._sync

    async def search(self, query: str = "", **options: Any) -> SearchResult:
        kb = await self._ensure_open()
        return await asyncio.to_thread(kb.search, query, **options)

    async def search_preset(self, query: str = "", **options: Any) -> dict[str, Any]:
        kb = await self._ensure_open()
        return await asyncio.to_thread(kb.search_preset, options.pop("preset", "rag"), query, **options)

    async def health(self) -> dict:
        kb = await self._ensure_open()
        return await asyncio.to_thread(kb.health)

    async def register_reranker(self, callback: Callable, **options: Any) -> None:
        kb = await self._ensure_open()
        await asyncio.to_thread(kb.register_reranker, callback, **options)

    async def invoke(self, operation: str, payload: Any = None) -> Any:
        kb = await self._ensure_open()
        return await asyncio.to_thread(kb.invoke, operation, payload)

    async def backup(self, path: str | os.PathLike) -> None:
        kb = await self._ensure_open()
        await asyncio.to_thread(kb.backup, path)

    async def update_index(self) -> dict:
        kb = await self._ensure_open()
        return await asyncio.to_thread(kb.update_index)

    async def reconcile_index(self) -> dict:
        kb = await self._ensure_open()
        return await asyncio.to_thread(kb.reconcile_index)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            if self._sync is not None:
                await asyncio.to_thread(self._sync.close)

    async def __aenter__(self) -> AsyncKnowledgeBase:
        await self._ensure_open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
