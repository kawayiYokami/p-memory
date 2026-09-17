"""Python 绑定契约测试。

Rust 侧 `tests/core.rs` 覆盖核心语义；这里保护的是绑定自己的契约：
返回结构是扁平 dict（不是对象）、异常按 `code` 映射、作用域与分页在 Python 侧可用、
以及四条存储域与导入器的端到端路径。
"""
import asyncio
import sqlite3

import pytest

from p_memory import (
    AsyncKnowledgeBase,
    ClosedError,
    ConflictError,
    InvalidVectorError,
    KnowledgeBase,
    LockedError,
    NotFoundError,
    PMemoryError,
    ValidationError,
    import_legacy,
)


# ── 记忆 ──────────────────────────────────────────────────────────────

def test_memory_roundtrip_is_a_flat_dict(kb):
    """写回、取回、检索三处拿到的都是同一个扁平 dict，记录头字段已平铺。"""
    receipt = kb.memories.upsert_by_judgment(judgment="用户偏好简短回答", tags=["偏好"])
    memory = receipt["value"]

    assert isinstance(memory, dict)
    assert isinstance(memory["id"], int)
    assert memory["judgment"] == "用户偏好简短回答"
    assert memory["kind"] == "memory"
    assert memory["memory_type"] == "knowledge"
    assert memory["tags"] == ["偏好"]
    # 记录头平铺：没有嵌套的 header 对象
    for field in ("namespace", "scope", "revision", "created_at_us", "updated_at_us"):
        assert field in memory, f"记录头字段 {field} 未平铺"

    assert kb.memories.get(memory["id"]) == memory

    hits = kb.search("简短回答")["hits"]
    assert [h["record"]["judgment"] for h in hits] == ["用户偏好简短回答"]
    assert set(hits[0]) == {"key", "record", "score", "text_score", "vector_scores", "rerank_score"}


def test_upsert_by_judgment_deduplicates_within_scope(kb):
    """同一句论断重复写入合并成一条，不产生第二行。"""
    first = kb.memories.upsert_by_judgment(judgment="同一句论断")
    second = kb.memories.upsert_by_judgment(judgment="同一句论断")

    assert first["value"]["id"] == second["value"]["id"]
    assert kb.health()["record_count"] == 1


def test_scope_filter_hides_records_written_to_other_scopes(open_kb):
    """只读 public 的句柄看不到写入 private 的记录；两个 scope 都给才读得到。"""
    public_only = open_kb("a", scopes=("public",))
    public_only.memories.upsert_by_judgment(judgment="公开内容")
    public_only.memories.upsert_by_judgment(judgment="私密内容", scope="private")

    assert kb_judgments(public_only) == ["公开内容"]
    assert [m["scope"] for m in public_only.memories.list(limit=10)["items"]] == ["public"]

    both = open_kb("b", scopes=("public", "private"))
    both.memories.upsert_by_judgment(judgment="公开内容", scope="public")
    both.memories.upsert_by_judgment(judgment="私密内容", scope="private")
    assert sorted(kb_judgments(both)) == ["公开内容", "私密内容"]


def test_tags_filter_is_conjunctive(kb):
    """多标签是 AND：只带其中一个标签的记录不会被 tags=[a,b] 选中。"""
    kb.memories.upsert_by_judgment(judgment="带两个标签", tags=["a", "b"])
    kb.memories.upsert_by_judgment(judgment="只带一个标签", tags=["a"])

    hits = kb.search("标签", filter={"tags": ["a", "b"]})["hits"]
    assert [h["record"]["judgment"] for h in hits] == ["带两个标签"]


def test_pagination_cursor_walks_every_record_once(kb):
    """游标翻页走完整个集合，不重不漏且按 id 升序。"""
    for i in range(5):
        kb.memories.upsert_by_judgment(judgment=f"分页记录 {i}")

    seen, cursor, pages = [], None, 0
    while True:
        page = kb.memories.list(limit=2, after=cursor)
        seen += [m["id"] for m in page["items"]]
        pages += 1
        cursor = page["next_cursor"]
        assert pages <= 10, "游标未收敛"
        if cursor is None:
            break

    assert len(seen) == 5
    assert seen == sorted(seen)
    assert len(set(seen)) == 5


# ── 图谱 ──────────────────────────────────────────────────────────────

def test_graph_batch_traversal_aliases_and_components(kb):
    """批量写入后，别名解析、一跳邻接、N 跳邻域、桥接与连通分量都可用。"""
    batch = kb.graph.apply_batch(
        entities=[
            {"name": "张三", "entity_type": "person", "aliases": ["小张"], "attributes": {"城市": ["杭州"]}},
            {"name": "李四", "entity_type": "person"},
            {"name": "王五", "entity_type": "person"},
        ],
    )
    zhang, li, wang = [e["id"] for e in batch["value"]["entities"]]

    kb.graph.apply_batch(relations=[
        {"subject_id": zhang, "predicate": "同事", "object_id": li},
        {"subject_id": li, "predicate": "同事", "object_id": wang},
    ])

    # 别名解析回同一个实体，属性与别名都在
    resolved = kb.graph.resolve("小张")
    assert [e["id"] for e in resolved] == [zhang]
    assert resolved[0]["attributes"] == {"城市": ["杭州"]}

    # 一跳邻接
    assert {e["id"] for e in kb.graph.neighbors(zhang)["entities"]} == {li}

    # N 跳邻域不含起点
    assert [e["id"] for e in kb.graph.ego(zhang, 1)] == [li]
    assert {e["id"] for e in kb.graph.ego(zhang, 2)} == {li, wang}

    # 桥接含两端
    assert [e["id"] for e in kb.graph.path(zhang, wang)] == [zhang, li, wang]

    # 单向链既是 1 个连通分量，也没有强连通环
    assert kb.graph.component_count() == 1
    assert kb.graph.strongly_connected() == []


def test_deleting_a_referenced_entity_reports_conflict(kb):
    """被关系引用的实体不能删；解除引用后可以删。"""
    batch = kb.graph.apply_batch(entities=[{"name": "甲"}, {"name": "乙"}])
    jia, yi = [e["id"] for e in batch["value"]["entities"]]
    kb.graph.apply_batch(relations=[{"subject_id": jia, "predicate": "同事", "object_id": yi}])

    with pytest.raises(ConflictError):
        kb.graph.delete("entity", jia)

    standalone = kb.graph.apply_batch(entities=[{"name": "丙"}])["value"]["entities"][0]["id"]
    assert kb.graph.delete("entity", standalone)["value"] is True
    with pytest.raises(NotFoundError):
        kb.graph.get("entity", standalone)


# ── 笔记 ──────────────────────────────────────────────────────────────

def test_note_chunks_are_line_ranges_and_reuse_ids(kb, tmp_path):
    """切片行号从 1 起；同一路径追加内容时复用未变切片的 id，并删掉失效切片。"""
    path = tmp_path / "a.md"
    path.write_text("A段\n\nB段\n\nC段", encoding="utf-8", newline="")
    first = kb.notes.upsert_file(path=str(path))
    note_id = first["value"]["id"]
    chunks = kb.notes.chunks(note_id)

    assert len(chunks) == 3
    assert [(c["ordinal"], c["offset"], c["limit"]) for c in chunks] == [(0, 1, 1), (1, 3, 1), (2, 5, 1)]
    assert all(c["note_id"] == note_id for c in chunks)

    path.write_text("A段\n\nB段\n\nC段\n\nD段", encoding="utf-8", newline="")
    second = kb.notes.upsert_file(path=str(path))
    assert second["value"]["id"] == note_id, "同一路径应复用同一条笔记"

    grown = kb.notes.chunks(note_id)
    assert [c["ordinal"] for c in grown] == [0, 1, 2, 3]
    assert [c["id"] for c in grown][:3] == [c["id"] for c in chunks], "未变切片应复用原 id"


def test_note_rejects_out_of_range_chunk_size(kb, tmp_path):
    """切片大小超范围由核心拒绝。"""
    path = tmp_path / "x.md"
    path.write_text("正文", encoding="utf-8", newline="")
    with pytest.raises(ValidationError):
        kb.notes.upsert_file(path=str(path), chunk_chars=5)


def test_note_upsert_file_uses_file_stem_and_keeps_raw_text(kb, tmp_path):
    """按路径同步：标题取文件名，路径即身份，正文不进库、切片正文随写入进索引。"""
    path = tmp_path / "世界观.md"
    raw = "# 标题\n\n这里有 **独有措辞** 正文。"
    path.write_text(raw, encoding="utf-8", newline="")

    note = kb.notes.upsert_file(path=str(path))["value"]
    assert note["title"] == "世界观"
    assert note["source"] == str(path), "路径即身份"
    assert kb.search("独有措辞", kinds=["chunk"])["hits"], "清洗后的文本仍可检索"
    assert any("独有措辞" in c["content"] for c in kb.notes.chunks(note["id"])), "切片正文随写入进索引"

    path.write_text("改过的正文 **新词** 在这里。", encoding="utf-8", newline="")
    updated = kb.notes.upsert_file(path=str(path))["value"]
    assert updated["id"] == note["id"], "同一路径复用同一笔记"

    with pytest.raises(PMemoryError):
        kb.notes.upsert_file(path=str(tmp_path / "nope.md"))
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe\xfd")
    with pytest.raises(PMemoryError):
        kb.notes.upsert_file(path=str(bad))


def test_note_body_lives_in_the_index_and_path_tags_ride_on_every_chunk(kb, tmp_path):
    """切片正文随写入进索引：源文件删掉仍读得到；登记根目录后路径段拆成标签，挂在每个切片上。"""
    root = tmp_path / "domain"
    directory = root / "绝区零" / "角色"
    directory.mkdir(parents=True)
    path = directory / "雅.md"
    path.write_text("苹果 香蕉 橘子", encoding="utf-8", newline="")
    kb.notes.set_root("default", root)
    assert kb.notes.root("default") == str(root).replace("\\", "/")
    note = kb.notes.upsert_file(path=str(path))["value"]

    path.unlink()
    chunks = kb.notes.chunks(note["id"])
    assert [c["content"] for c in chunks] == ["苹果 香蕉 橘子"], "正文在索引里，源文件没了也读得到"
    assert [c["tags"] for c in chunks] == [["绝区零", "角色", "雅"]], "路径段标签挂在切片上"
    assert kb.search("角色", kinds=["chunk"])["hits"], "路径段标签让切片被搜到"
    assert kb.search("角色", kinds=["note"])["hits"] == [], "笔记不占索引文档"
    page = kb.notes.list(filter={"tags": ["角色"]})
    assert [item["id"] for item in page["items"]] == [note["id"]], "按标签翻笔记仍能筛出这一篇"

    outside = tmp_path / "外面.md"
    outside.write_text("根目录之外", encoding="utf-8", newline="")
    with pytest.raises(PMemoryError):
        kb.notes.upsert_file(path=str(outside))


# ── 向量与重排 ────────────────────────────────────────────────────────

def test_embedder_registers_validates_and_search_embeds_query(kb):
    """注册即校验；写入只入库不入向量，批次结束 sync 之后向量路才放行；检索只给搜索词。"""
    kb.embeddings.register_space({"id": "e5", "model": "e5-base", "dimension": 4})
    assert kb.embeddings.spaces() == [
        {"id": "e5", "model": "e5-base", "dimension": 4, "text_version": 1, "encoding": "sq8"}
    ]

    calls: list[int] = []

    def embedder(texts):
        calls.append(len(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    kb.embeddings.register_embedder("e5", embedder, max_batch=4)
    assert calls and max(calls) <= 4, "注册即用样本真跑一遍校验"

    target = kb.memories.upsert_by_judgment(judgment="需要向量化的记忆")["value"]["id"]
    assert len(calls) == 1, "写入不碰模型：注册校验那次之后回调没再被调到"
    assert kb.embeddings.vector_ready("default", "e5") is False, "还没补过，谈不上就绪"

    gated = kb.search("向量化", embed_space="e5", text=False, rerank=False)
    assert gated["hits"] == [] and "vector_not_ready" in gated["diagnostics"]["degraded"]

    assert kb.embeddings.sync("e5")["value"]["written"] == 1, "批次结束补一次"
    assert kb.embeddings.vector_ready("default", "e5") is True

    hits = kb.search("向量化", embed_space="e5", text=False, rerank=False)["hits"]
    assert hits[0]["key"]["id"] == target
    assert hits[0]["vector_scores"]["e5"] == pytest.approx(1.0)
    assert kb.embeddings.embedder_space("e5")["dimension"] == 4
    assert kb.health()["embedder_spaces"] == ["e5"]
    assert kb.embeddings.unregister_embedder("e5") is True
    assert kb.embeddings.unregister_embedder("e5") is False


def test_embedder_registration_rejects_wrong_dimension(kb):
    """回调产出维度与空间契约不符时拒绝绑定，并映射到 invalid_vector。"""
    kb.embeddings.register_space({"id": "v", "model": "m", "dimension": 3})
    with pytest.raises(InvalidVectorError):
        kb.embeddings.register_embedder("v", lambda texts: [[1.0, 2.0] for _ in texts])


def test_namespace_vectorization_switch_is_per_namespace(kb):
    """一个知识域关掉向量化不影响另一个；开关落盘可读。"""
    kb.embeddings.register_space({"id": "v", "model": "m", "dimension": 4})
    kb.embeddings.register_embedder("v", lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts])
    assert kb.embeddings.namespace_vectorization("default") is True

    kb.memories.upsert_by_judgment(judgment="另一个域的内容", namespace="other")
    kb.embeddings.set_namespace_vectorization("other", False)
    assert kb.embeddings.namespace_vectorization("other") is False

    result = kb.search("另一个域的内容", filter={"namespace": "other"}, embed_space="v")
    assert "namespace_disabled" in result["diagnostics"]["degraded"]
    assert result["diagnostics"]["vector_used"] is False
    assert result["hits"], "关掉向量化后全文路仍然给结果"


def test_vectorization_targets_are_independent(kb, tmp_path):
    """每个领域下三档独立开关：默认值、按档读写、非法档位名被拒、设置落盘。"""
    kb.embeddings.register_space({"id": "v", "model": "m", "dimension": 4})
    kb.embeddings.register_embedder("v", lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts])
    assert kb.embeddings.vectorization("default", "memory") is True
    assert kb.embeddings.vectorization("default", "graph") is True
    assert kb.embeddings.vectorization("default", "notes") is False

    kb.embeddings.set_vectorization("default", "memory", False)
    kb.embeddings.set_vectorization("default", "notes", True)
    assert kb.embeddings.vectorization("default", "memory") is False
    assert kb.embeddings.vectorization("default", "graph") is True, "关掉一档不影响另一档"
    with pytest.raises(ValidationError):
        kb.embeddings.set_vectorization("default", "knowledge", True)
    with pytest.raises(ValidationError):
        kb.embeddings.vectorization("default", "记忆")

    # 关掉的那档不生成向量，也不进向量路；全文路照常给结果。
    kb.memories.upsert_by_judgment(judgment="关掉记忆档之后写入的内容")
    muted = kb.search("关掉记忆档之后写入的内容", kinds=["memory"], embed_space="v", text=False)
    assert muted["hits"] == []
    assert muted["diagnostics"]["vector_used"] is False, "该档关闭时向量路整条不走"
    assert kb.search("关掉记忆档之后写入的内容")["hits"], "全文路不受开关影响"
    assert kb.embeddings.sync("v")["value"]["written"] == 0, "关闭的档位不会被 sync 补出向量"

    # 设置落盘：重开同一个库仍然生效。
    kb.close()
    reopened = KnowledgeBase(str(tmp_path / "data"))
    try:
        assert reopened.embeddings.vectorization("default", "memory") is False
        assert reopened.embeddings.vectorization("default", "notes") is True
    finally:
        reopened.close()


def test_reranker_reorders_and_reports_diagnostics(kb):
    """重排回调在融合之后生效，截断与是否重排都写进诊断；总量按过滤后统计。"""
    for i in range(4):
        kb.memories.upsert_by_judgment(judgment=f"重排对象 {i}")

    baseline = [h["key"]["id"] for h in kb.search("重排对象", vector=False, rerank=False)["hits"]]
    assert len(baseline) == 4

    kb.register_reranker(lambda query, documents: [float(i) for i in range(len(documents))], max_docs=2)
    result = kb.search("重排对象", vector=False, rerank=True)
    assert result["diagnostics"]["reranked"] is True
    assert result["diagnostics"]["rerank_candidates"] == 2
    assert result["diagnostics"]["rerank_truncated"] == 2
    assert result["hits"][0]["key"]["id"] == baseline[1], "截到 2 条候选中，给后者更高分者排最前"
    assert result["hits"][0]["rerank_score"] is not None

    counted = kb.search("重排对象", vector=False, rerank=False, limit=1, with_total=True)
    assert len(counted["hits"]) == 1
    assert counted["total"] == 4


# ── 生命周期 ──────────────────────────────────────────────────────────

def test_lifecycle_reports_candidates_without_deleting(kb):
    """衰减只产出候选，不删数据；固定保留项不进候选，强度也不变。"""
    ordinary = kb.memories.upsert_by_judgment(judgment="普通记忆")["value"]["id"]
    pinned = kb.memories.upsert_by_judgment(judgment="固定保留的记忆", state={"pinned": True})["value"]["id"]
    before = kb.health()["record_count"]

    far_future = 1_900_000_000_000_000  # 微秒
    report = kb.memories.decay(now_us=far_future)["value"]

    candidates = [c["id"] for c in report["retirement_candidates"]]
    assert ordinary in candidates
    assert pinned not in candidates
    assert kb.health()["record_count"] == before, "衰减不得删除记录"
    assert kb.memories.get(pinned)["state"]["strength"] == 1
    assert kb.memories.get(ordinary)["state"]["strength"] == 0


def test_feedback_boosts_useful_recall(kb):
    """被标记为有用的召回项提升强度与计数。"""
    target = kb.memories.upsert_by_judgment(judgment="被反馈的记忆")["value"]["id"]
    report = kb.memories.feedback([target], [target])["value"]

    assert report == {"recalled": 1, "boosted": 1, "penalized": 0}
    state = kb.memories.get(target)["state"]
    assert state["strength"] == 2
    assert state["useful_count"] == 1
    assert state["useful_score"] == pytest.approx(2.5)


# ── 健康、备份与错误映射 ──────────────────────────────────────────────

def test_health_exposes_core_counters(kb):
    """健康报告含库归属、修订号、索引进度与完整性检查。"""
    kb.memories.upsert_by_judgment(judgment="一条记忆")
    # 写入只登记待办、不就地索引：显式追平之前进度是落后的。
    assert kb.health()["pending_index_updates"] >= 1
    kb.update_index()
    report = kb.health()

    assert report["schema_version"] == 9
    assert report["revision"] >= 1
    assert report["indexed_revision"] == report["revision"]
    assert report["pending_index_updates"] == 0
    assert report["record_count"] == 1
    assert report["counts"]["memory"] == 1
    assert report["sqlite_integrity"] == "ok"
    assert report["foreign_key_errors"] == 0


def test_backup_and_restore_roundtrip(kb, tmp_path):
    """备份出独立快照，restore 到新目录后数据与索引都可用。"""
    kb.memories.upsert_by_judgment(judgment="备份里的记忆")
    snapshot = tmp_path / "snapshot.sqlite3"
    kb.backup(str(snapshot))
    assert snapshot.exists()

    restored = KnowledgeBase.restore(str(snapshot), str(tmp_path / "restored"))
    try:
        assert [h["record"]["judgment"] for h in restored.search("备份")["hits"]] == ["备份里的记忆"]
        assert restored.health()["record_count"] == 1
    finally:
        restored.close()


def test_error_codes_map_to_exception_classes(open_kb, tmp_path):
    """锁定、未找到、参数非法、已关闭四类错误都映射到对应异常，并带稳定 code。"""
    kb = open_kb("locked")

    with pytest.raises(LockedError) as locked:
        KnowledgeBase(str(tmp_path / "locked"))
    assert locked.value.code == "locked"

    with pytest.raises(NotFoundError) as missing:
        kb.memories.get(424242)
    assert missing.value.code == "not_found"

    with pytest.raises(ValidationError) as invalid:
        open_kb("bad", scopes=[])
    assert invalid.value.code == "validation"

    kb.close()
    with pytest.raises(ClosedError) as closed:
        kb.health()
    assert closed.value.code == "closed"


def test_search_rejects_empty_query_without_vectors(kb):
    """既无关键词也无向量时由核心拒绝。"""
    with pytest.raises(ValidationError):
        kb.search("")


# ── 异步封装与导入器 ──────────────────────────────────────────────────

def test_async_facade_mirrors_the_sync_store(tmp_path):
    """异步封装的读写走同一套核心，语义与同步一致。"""

    async def scenario() -> list[str]:
        async with await AsyncKnowledgeBase.open(str(tmp_path / "async")) as kb:
            await kb.memories.upsert_by_judgment(judgment="异步写入的记忆")
            hits = await kb.search("异步")
            assert (await kb.health())["record_count"] == 1
            return [h["record"]["judgment"] for h in hits["hits"]]

    assert asyncio.run(scenario()) == ["异步写入的记忆"]


def test_import_legacy_dry_run_then_apply(tmp_path):
    """导入器默认只预览、不落盘；--apply 后才生成目标库。"""
    source = tmp_path / "memory_store.db"
    conn = sqlite3.connect(source)
    conn.executescript("""
        CREATE TABLE memory_record(id TEXT PRIMARY KEY, memory_type TEXT, judgment TEXT, reasoning TEXT,
            strength INTEGER, is_active INTEGER, memory_scope TEXT, useful_count INTEGER, useful_score REAL,
            last_recalled_at TEXT, last_decay_at TEXT, created_at TEXT, updated_at TEXT, owner_agent_id TEXT);
        CREATE TABLE global_tag(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE memory_tag_rel(memory_id TEXT, tag_id INTEGER);
        INSERT INTO memory_record VALUES('m1','knowledge','Rust 是系统编程语言','因为内存安全',5,1,'public',3,0.5,
            NULL,NULL,'2026-02-21T13:11:40.51375Z','2026-06-13T09:26:45Z',NULL);
        INSERT INTO global_tag VALUES(1,'rust');
        INSERT INTO memory_tag_rel VALUES('m1',1);
    """)
    conn.close()

    destination = tmp_path / "imported"
    preview = import_legacy(source="p_ai", source_id="fixture", database=str(source), destination=str(destination))
    assert preview["applied"] is False
    assert preview["counts"]["memory"] == 1
    assert not preview["conflicts"]
    assert not destination.exists(), "dry run 不得创建目标目录"

    applied = import_legacy(source="p_ai", source_id="fixture", database=str(source),
                            destination=str(destination), dry_run=False)
    assert applied["applied"] is True
    assert destination.exists()

    imported = KnowledgeBase(str(destination))
    try:
        hits = imported.search("系统编程语言")["hits"]
        assert [h["record"]["judgment"] for h in hits] == ["Rust 是系统编程语言"]
        assert hits[0]["record"]["tags"] == ["rust"]
        assert hits[0]["record"]["state"]["strength"] == 5
    finally:
        imported.close()


def kb_judgments(kb) -> list[str]:
    """当前 filter 下可见的全部记忆论断，按 id 升序。"""
    return [m["judgment"] for m in kb.memories.list(limit=100)["items"]]
