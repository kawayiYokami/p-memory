"""Python 绑定契约测试。

Rust 侧 `tests/core.rs` 覆盖核心语义；这里保护的是绑定自己的契约：
返回结构是扁平 dict（不是对象）、异常按 `code` 映射、作用域与分页在 Python 侧可用、
以及四条存储域与导入器的端到端路径。
"""
import asyncio
import sqlite3
import threading
import time

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


def wait_until_ready(kb, namespace: str, space_id: str, target: str, budget_s: float = 15.0) -> bool:
    """等某一档补齐。补齐由库内线程按事件触发，这里只等结果，不假定是谁补的。"""
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        if kb.embeddings.vector_ready(namespace, space_id, target):
            return True
        time.sleep(0.025)
    return False


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
    assert set(hits[0]) == {"key", "record", "score", "text_score", "vector_scores", "rerank_score", "note_chunks", "top_chunks"}


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


def test_chunks_from_one_note_collapse_with_the_note_total(kb, tmp_path):
    """同一篇笔记命中多片时只出一条，并报出这一篇共有多少片段命中。"""
    many = tmp_path / "对话.md"
    many.write_text("派蒙" * 800, encoding="utf-8", newline="")
    many_note = kb.notes.upsert_file(path=str(many))["value"]
    many_chunks = kb.notes.chunks(many_note["id"])
    assert len(many_chunks) > 1, "这篇应当切成多片"

    once = tmp_path / "独白.md"
    once.write_text("派蒙", encoding="utf-8", newline="")
    once_note = kb.notes.upsert_file(path=str(once))["value"]

    hits = kb.search("派蒙", kinds=["chunk"], limit=50)["hits"]
    by_note = {hit["record"]["note_id"]: hit for hit in hits}
    assert set(by_note) == {many_note["id"], once_note["id"]}, "两篇各出且只出一条，多片段那篇不许刷屏"
    assert by_note[many_note["id"]]["note_chunks"] == len(many_chunks), "多片段那篇报出片段总数"
    assert by_note[once_note["id"]]["note_chunks"] == 1, "单片段那篇报 1"

    # 折叠掉的那几片不丢：代表命中带这一篇排名最高的若干片，第 0 条就是本条自身。
    many_hit = by_note[many_note["id"]]
    assert len(many_hit["top_chunks"]) == 3, "默认把这一篇排名最高的三片聚合在一条里"
    assert many_hit["top_chunks"][0]["id"] == many_hit["key"]["id"], "第 0 条就是本条自身"
    assert many_hit["top_chunks"][0]["offset"] == many_hit["record"]["offset"]
    offsets = {chunk["id"]: chunk["offset"] for chunk in many_chunks}
    assert all(chunk["id"] in offsets for chunk in many_hit["top_chunks"]), "聚合进来的都是这一篇的片段"
    assert [chunk["offset"] for chunk in many_hit["top_chunks"]] == [offsets[chunk["id"]] for chunk in many_hit["top_chunks"]], "行号与切片表一致"
    assert [chunk["id"] for chunk in by_note[once_note["id"]]["top_chunks"]] == [by_note[once_note["id"]]["key"]["id"]], "只命中一片的篇，聚合里就它自己"

    # 条数可控：0 表示不聚合；计数与聚合开关无关。
    plain = {hit["record"]["note_id"]: hit for hit in kb.search("派蒙", kinds=["chunk"], limit=50, top_chunks_per_note=0)["hits"]}
    assert plain[many_note["id"]]["top_chunks"] == [], "关掉聚合就不带片段"
    assert plain[many_note["id"]]["note_chunks"] == len(many_chunks), "计数与聚合开关无关"


def test_note_body_lives_in_the_index_and_the_first_chunk_carries_the_path_tags(kb, tmp_path):
    """切片正文随写入进索引：源文件删掉仍读得到；文件名进名字列可搜，目录段只当标签可筛不可搜。"""
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
    assert [c["tags"] for c in chunks] == [["绝区零", "角色", "雅"]], "路径段标签挂在切片记录上"
    assert kb.search("角色", kinds=["chunk"])["hits"] == [], "目录段不参与常规匹配"
    hits = kb.search("雅", kinds=["chunk"])["hits"]
    assert [h["key"]["id"] for h in hits] == [chunks[0]["id"]], "文件名进名字列，第一片被搜到"
    assert kb.search("雅", kinds=["note"])["hits"] == [], "笔记不占索引文档"
    page = kb.notes.list(filter={"tags": ["角色"]})
    assert [item["id"] for item in page["items"]] == [note["id"]], "按目录段筛笔记仍能筛出这一篇"

    outside = tmp_path / "外面.md"
    outside.write_text("根目录之外", encoding="utf-8", newline="")
    with pytest.raises(PMemoryError):
        kb.notes.upsert_file(path=str(outside))


# ── 向量与重排 ────────────────────────────────────────────────────────

def test_embedder_registers_validates_and_search_embeds_query(kb):
    """注册即校验；写入只入库不入向量，补齐之后向量路才放行；检索只给搜索词。"""
    kb.embeddings.register_space({"id": "e5", "model": "e5-base", "dimension": 4})
    assert kb.embeddings.spaces() == [
        {"id": "e5", "model": "e5-base", "dimension": 4, "text_version": 1, "encoding": "sq8"}
    ]

    calls: list[tuple[str, int]] = []

    def embedder(texts):
        calls.append((threading.current_thread().name, len(texts)))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    # 还没有模型可用：写入照样成功，一行向量都不产生。
    target = kb.memories.upsert_by_judgment(judgment="需要向量化的记忆")["value"]["id"]
    assert kb.embeddings.vector_ready("default", "e5", "memory") is False, "还没补过，谈不上就绪"

    gated = kb.search("向量化", embed_space="e5", text=False, rerank=False)
    assert gated["hits"] == [] and "vector_not_ready" in gated["diagnostics"]["degraded"]

    # 接上回调：注册这个动作自己就触发一次补齐。
    kb.embeddings.register_embedder("e5", embedder, max_batch=4)
    mine = [length for name, length in calls if name == threading.current_thread().name]
    assert mine and max(mine) <= 4, "注册即用样本真跑一遍校验"
    # 模型已就位的情况下再写一条：调用方这条线程上依然一次模型都不调。
    before = len(mine)
    kb.memories.upsert_by_judgment(judgment="第二条需要向量化的记忆")
    mine = [length for name, length in calls if name == threading.current_thread().name]
    assert len(mine) == before, "写入不碰模型"
    assert wait_until_ready(kb, "default", "e5", "memory"), "补齐之后该放行"

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
    """重排回调在两路候选合并之后生效，截断与是否重排都写进诊断；总量按过滤后统计。"""
    for i in range(4):
        kb.memories.upsert_by_judgment(judgment=f"重排对象 {i}")

    baseline = [h["key"]["id"] for h in kb.search("重排对象", vector=False, rerank=False)["hits"]]
    assert len(baseline) == 4

    kb.register_reranker(lambda query, documents: [float(i) for i in range(len(documents))], max_tokens_total=10)
    result = kb.search("重排对象", vector=False, rerank=True)
    assert result["diagnostics"]["reranked"] is True
    assert result["diagnostics"]["rerank_candidates"] == 2
    assert result["diagnostics"]["rerank_truncated"] == 2
    assert result["hits"][0]["key"]["id"] == baseline[1], "截到 2 条候选中，给后者更高分者排最前"
    assert result["hits"][0]["rerank_score"] is not None

    counted = kb.search("重排对象", vector=False, rerank=False, limit=1, with_total=True)
    assert len(counted["hits"]) == 1
    assert counted["total"] == 4

    # 条数上限：token 预算装得下全部 4 条，仍按条数封顶。
    kb.register_reranker(lambda query, documents: [0.0 for _ in documents], max_candidates=1)
    capped = kb.search("重排对象", vector=False, rerank=True)
    assert capped["diagnostics"]["rerank_candidates"] == 1
    assert capped["diagnostics"]["rerank_truncated"] == 3


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

    assert report["schema_version"] == 12
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


# ── 预设检索 ──────────────────────────────────────────────────────────

def notes_empty(section) -> bool:
    """笔记那一路是否空着：书名块、内容块、路径兜底三块都空才算空。"""
    return not section["titles"] and not section["contents"] and not section["paths"]


def test_search_preset_keeps_the_three_fields_apart(kb, tmp_path):
    """预设检索：记忆、图谱、笔记各占一个字段，互不混排；图谱走实体 → 关系 → 事件。"""
    created = kb.graph.apply_batch(entities=[{"name": "朱樱"}, {"name": "白露"},
                                             {"name": "青萍"}, {"name": "玄霜"}])["value"]["entities"]
    ids = {item["name"]: item["id"] for item in created}
    kb.graph.apply_batch(
        relations=[
            {"subject_id": ids["朱樱"], "predicate": "同学", "object_id": ids["青萍"]},
            {"subject_id": ids["白露"], "predicate": "同学", "object_id": ids["玄霜"]},
            {"subject_id": ids["青萍"], "predicate": "同门", "object_id": ids["玄霜"]},
        ],
        events=[
            {"name": "别鹤典仪", "participants": [ids["朱樱"], ids["青萍"]]},
            {"name": "堂中自语", "participants": [ids["青萍"]]},
        ],
    )
    kb.memories.upsert({"judgment": "朱樱的同学是青萍"})
    path = tmp_path / "预设笔记.md"
    path.write_text("朱樱的同学是青萍" * 20, encoding="utf-8", newline="")
    kb.notes.upsert_file(path=str(path))

    rag = kb.search_preset("rag", "朱樱和白露的同学是谁")
    assert kb_judgments(kb) and [item["record"]["judgment"] for item in rag["memories"]], "记忆那一路有结果"
    assert notes_empty(rag["notes"]), "RAG 不出笔记那一路"
    assert sorted(entity["name"] for entity in rag["graph"]["entities"]) == ["朱樱", "白露"], "命中的实体成为种子"
    predicates = [relation["predicate"] for relation in rag["graph"]["relations"]]
    assert predicates.count("同学") == 2, "种子实体各自敲出的同学关系都在结果里"
    assert [relation["predicate"] for relation in rag["graph"]["context_relations"]] == ["同门"], "两两之间的关系进第三段"
    assert [event["name"] for event in rag["graph"]["context_events"]] == ["别鹤典仪"], "参与者至少两个才算"

    broad = kb.search_preset("broad", "朱樱和白露的同学是谁")
    assert broad["memories"] and not notes_empty(broad["notes"]) and broad["graph"]["entities"], "广撒网三个字段都填"

    memory_only = kb.search_preset("memory", "朱樱和白露的同学是谁")
    assert memory_only["memories"] and notes_empty(memory_only["notes"]) and memory_only["graph"]["entities"] == []


def test_event_sink_receives_search_events(kb):
    """注册事件回调后，一次检索收到一条 `search` 事件；注销后不再产出。"""
    kb.memories.upsert_by_judgment(judgment="事件流 甲")
    events: list[dict] = []
    kb.register_event_sink(events.append)
    assert kb.event_sink_registered()

    result = kb.search("事件流", kinds=["memory"], vector=False, rerank=False)
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "search"
    assert event["hits"] == len(result["hits"])
    assert "text" in event["stages"]
    assert "vector" not in event["stages"], "没走向量路就不该有向量格"
    assert event["rerank_docs"] == 0, "没启用重排就不送候选"
    assert isinstance(event["ms"], int) and event["ts"]

    assert kb.unregister_event_sink()
    assert not kb.event_sink_registered()
    kb.search("事件流", kinds=["memory"], vector=False, rerank=False)
    assert len(events) == 1, "注销后不再产出"


def test_event_sink_swallows_callback_errors(kb):
    """回调抛异常只丢这一条事件，检索照常返回同样的结果。"""
    kb.memories.upsert_by_judgment(judgment="事件抛错 甲")
    options = {"kinds": ["memory"], "vector": False, "rerank": False}
    baseline = [hit["key"]["id"] for hit in kb.search("事件抛错", **options)["hits"]]

    def boom(_event):
        raise RuntimeError("sink 自己炸了")

    kb.register_event_sink(boom)
    after = [hit["key"]["id"] for hit in kb.search("事件抛错", **options)["hits"]]
    assert after == baseline


def test_note_ids_limit_the_search_to_the_given_notes(kb, tmp_path):
    """按笔记限定：候选在生成阶段就收窄到目标笔记，而不是取回来再筛。"""
    kb.notes.set_root("default", str(tmp_path))
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text("harbor的契约与地契", encoding="utf-8", newline="")
    second.write_text("harbor的商船与货单", encoding="utf-8", newline="")
    a = kb.notes.upsert_file(path=str(first))["value"]["id"]
    b = kb.notes.upsert_file(path=str(second))["value"]["id"]

    everything = kb.search("harbor", kinds=["chunk"], limit=50)["hits"]
    assert {hit["record"]["note_id"] for hit in everything} == {a, b}, "不限定范围时两篇都命中"

    scoped = kb.search("harbor", kinds=["chunk"], filter={"note_ids": [a]}, limit=50)["hits"]
    assert scoped, "限定后目标笔记仍有命中"
    assert {hit["record"]["note_id"] for hit in scoped} == {a}, "命中全部来自被限定的笔记"

    assert kb.search("harbor", kinds=["chunk"], filter={"note_ids": [10**9]}, limit=50)["hits"] == [], "不存在的笔记 id 给空结果"


# ── 按过滤批量删除 ────────────────────────────────────────────────────

def test_batch_delete_clears_a_memory_domain_in_one_call(kb):
    """一条命令清掉整个域，返回删除条数。"""
    kb.memories.upsert_many([{"judgment": f"待清的记忆 {i}"} for i in range(3)])
    assert kb.memories.delete_by_filter()["value"] == 3
    assert kb.memories.list()["items"] == []


def test_batch_delete_takes_graph_edges_along(kb):
    """图谱域清空时实体与它的边一起走，不留下半截状态。"""
    batch = kb.graph.apply_batch(entities=[{"name": "甲"}, {"name": "乙"}])
    jia, yi = [e["id"] for e in batch["value"]["entities"]]
    kb.graph.apply_batch(relations=[{"subject_id": jia, "predicate": "同事", "object_id": yi}])
    # 计数是实体加关系，边本身是级联产物。
    assert kb.graph.delete_by_filter()["value"] == 3
    for kind in ("entity", "relation", "event"):
        assert kb.graph.list(kind)["items"] == []


def test_batch_delete_of_notes_takes_chunks_along(kb, tmp_path):
    """删笔记连同它的切片，切片不会变成孤儿。"""
    path = tmp_path / "n.md"
    path.write_text("笔记正文里的独有措辞", encoding="utf-8", newline="")
    note_id = kb.notes.upsert_file(path=str(path))["value"]["id"]
    chunk_id = kb.notes.chunks(note_id)[0]["id"]
    assert kb.notes.delete_by_filter()["value"] == 1
    with pytest.raises(NotFoundError):
        kb.notes.get_chunk(chunk_id)


def test_batch_delete_without_matches_returns_zero(kb):
    """空命中不是错误。"""
    assert kb.memories.delete_by_filter()["value"] == 0
    assert kb.graph.delete_by_filter()["value"] == 0
    assert kb.notes.delete_by_filter()["value"] == 0
