"""最小端到端 Demo。

运行：`python -m p_memory.demo`

它只做一件事：把一小份记忆与档案写进库，注册一个向量模型回调和一个重排模型回调，
然后直接拿搜索结果。向量化与重排都在库内部完成——宿主只提供内容与模型接口，
不自己算向量，也不自己重排。它跑通，就说明整条路径没有漏。
"""
from __future__ import annotations

import tempfile

from .api import KnowledgeBase

DIMENSION = 32


def fake_embedding(text: str, dimension: int = DIMENSION) -> list[float]:
    """确定性假向量：同一文本永远得到同一向量，也不为零。零外部依赖、结果可复现。"""
    values = [0.0] * dimension
    for index, byte in enumerate(text.encode("utf-8")):
        values[(index + byte) % dimension] += 1.0 + (byte % 7) * 0.1
    if all(value == 0.0 for value in values):
        values[0] = 1.0
    return values


def fake_relevance(query: str, document: str) -> float:
    """确定性假重排：按查询词里出现在文档中的字符数给分。"""
    return float(sum(1 for ch in query if not ch.isspace() and ch in document))


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        with KnowledgeBase(directory) as kb:
            # 一个向量模型 = 一个向量空间。注册即用样本真跑一遍校验，产出不符契约会被拒绝绑定。
            kb.embeddings.register_space(
                {"id": "demo-v1", "model": "demo-embed", "dimension": DIMENSION, "encoding": "f32"}
            )
            kb.embeddings.register_embedder(
                "demo-v1", lambda texts: [fake_embedding(text) for text in texts]
            )
            # 重排：库按声明的定长约束截断候选与文档后再调用，宿主不自己重排。
            kb.register_reranker(
                lambda query, documents: [fake_relevance(query, document) for document in documents]
            )

            # 一小份记忆与档案：只写内容，向量化由库在写入路径里完成。
            for judgment, tag in (
                ("红豆喜欢简短直接的回答", "偏好"),
                ("p-memory 是嵌入式记忆库", "项目"),
                ("搜索优先返回精确结果", "原则"),
            ):
                kb.memories.upsert_by_judgment(judgment=judgment, tags=[tag])
            kb.notes.upsert(
                source="docs/demo.md",
                content="档案：最小 Demo 只用一小份样本。\n\n它直接完成搜索，不做多余处理。",
            )

            # 直接检索：宿主只给搜索词与目标空间，库用它注册的回调嵌入查询词。
            result = kb.search(
                "p-memory 搜索",
                kinds=["memory", "note", "chunk"],
                limit=5,
                embed_space="demo-v1",
                with_total=True,
            )
            diagnostics = result["diagnostics"]

            print(f"命中 {len(result['hits'])} 条（过滤后总量 {result['total']}）")
            print(
                "走了哪几条路：全文={text_used} 向量={vector_used} 重排={reranked}".format(**diagnostics)
            )
            if diagnostics["rerank_truncated"] > 0:
                print(f"重排截断：{diagnostics['rerank_truncated']} 条候选未送入重排")
            if diagnostics["degraded"]:
                print(f"降级档位：{diagnostics['degraded']}")
            for rank, hit in enumerate(result["hits"], start=1):
                record = hit["record"]
                label = record.get("judgment") or record.get("source") or "<记录>"
                print(f"{rank}. #{hit['key']['id']} {label}  score={hit['score']:.4f} rerank={hit['rerank_score']}")

            # 多档降级：注销嵌入回调后再搜同一个词。空间还注册着、向量开关也开着，但回调不在了，
            # 于是落到「无回调」档走纯全文——结果照常返回，降级档位可观测。
            kb.embeddings.unregister_embedder("demo-v1")
            degraded = kb.search("p-memory 搜索", kinds=["memory", "note", "chunk"], limit=5, embed_space="demo-v1")
            flags = degraded["diagnostics"]
            print("--- 注销嵌入回调后 ---")
            print(
                f"命中 {len(degraded['hits'])} 条，走了哪几条路："
                f"全文={flags['text_used']} 向量={flags['vector_used']} 降级={flags['degraded']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
