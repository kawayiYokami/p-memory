"""对真实嵌入 / 重排模型跑一遍库内端到端链路。

用法：

    cp .env.sample .env      # 填入真实 base_url / api_key / 模型名
    python examples/real_models.py

它走的是与最小 Demo 同一条路径，只是把确定性假回调换成真实模型：
注册真实嵌入回调与真实重排回调 -> 写入一小份记忆与档案 -> 检索 -> 注销嵌入回调再检索一次。
向量化、分批、截断、重试、降级都在库内部发生，宿主不接触向量。

接口形状（对不上就改这里）：嵌入走 OpenAI 兼容的 `POST {base}/embeddings`；
重排走 `POST {base}/rerank`（Jina / SiliconFlow / DashScope 这一族）。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from p_memory import KnowledgeBase
from p_memory.errors import EmbedCallbackError, PMemoryError

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path) -> None:
    """把 .env 读进 os.environ（不覆盖已有值）。缺文件直接跳过。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def post(url: str, payload: dict, api_key: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise ProviderHTTPError(status) from None


class ProviderHTTPError(Exception):
    def __init__(self, status: int):
        super().__init__(f"provider returned HTTP {status}")
        self.status = status


def embedding_vectors(base: str, api_key: str, model: str, texts: list[str]) -> list[list[float]]:
    """真实嵌入回调。类别在宿主这边显式给出，库不解析错误文案。"""
    try:
        data = post(f"{base}/embeddings", {"model": model, "input": texts, "encoding_format": "float"}, api_key)
    except ProviderHTTPError as exc:
        if exc.status == 413:
            raise EmbedCallbackError.too_large(str(exc)) from exc
        if exc.status == 429:
            raise EmbedCallbackError.rate_limited(str(exc)) from exc
        raise EmbedCallbackError.other(str(exc)) from exc
    items = data.get("data")
    if not isinstance(items, list) or len(items) != len(texts):
        raise EmbedCallbackError.other("embedding response count does not match input count")
    ordered: list[list[float] | None] = [None] * len(texts)
    for item in items:
        ordered[item["index"]] = item["embedding"]
    if any(vector is None for vector in ordered):
        raise EmbedCallbackError.other("embedding response has missing indices")
    return ordered  # type: ignore[return-value]


def rerank_scores(base: str, api_key: str, model: str, query: str, documents: list[str]) -> list[float]:
    """真实重排回调：返回与 documents 等长、按序对齐的分数。"""
    data = post(f"{base}/rerank", {"model": model, "query": query, "documents": documents}, api_key)
    items = data.get("results") or data.get("data")
    if not isinstance(items, list) or len(items) != len(documents):
        raise ValueError("rerank response count does not match document count")
    ordered: list[float | None] = [None] * len(documents)
    for item in items:
        ordered[item["index"]] = float(item["relevance_score"])
    if any(score is None for score in ordered):
        raise ValueError("rerank response has missing indices")
    return ordered  # type: ignore[return-value]


def main() -> int:
    load_env(REPO_ROOT / ".env")
    base = os.environ.get("P_MEMORY_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("P_MEMORY_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    embed_model = os.environ.get("P_MEMORY_EMBEDDING_MODEL", "")
    dimension = int(os.environ.get("P_MEMORY_EMBEDDING_DIMENSION", "0") or 0)
    batch = int(os.environ.get("P_MEMORY_EMBEDDING_BATCH", "50") or 50)
    rerank_model = os.environ.get("P_MEMORY_RERANK_MODEL", "").strip()
    if not (base and api_key and embed_model and dimension):
        print("缺少配置：请在 .env 里填好 P_MEMORY_BASE_URL / P_MEMORY_API_KEY / "
              "P_MEMORY_EMBEDDING_MODEL / P_MEMORY_EMBEDDING_DIMENSION", file=sys.stderr)
        return 2

    embed_batches: list[int] = []
    rerank_docs: list[int] = []

    def embedder(texts: list[str]) -> list[list[float]]:
        embed_batches.append(len(texts))
        return embedding_vectors(base, api_key, embed_model, texts)

    def reranker(query: str, documents: list[str]) -> list[float]:
        rerank_docs.append(len(documents))
        return rerank_scores(base, api_key, rerank_model, query, documents)

    with tempfile.TemporaryDirectory() as directory:
        with KnowledgeBase(directory) as kb:
            kb.embeddings.register_space(
                {"id": "real-v1", "model": embed_model, "dimension": dimension, "encoding": "f32"}
            )
            kb.embeddings.register_embedder("real-v1", embedder, max_batch=batch)
            if rerank_model:
                kb.register_reranker(reranker, max_tokens_total=8192, max_tokens_per_doc=1024)

            for judgment, tags in [
                ("红豆喜欢简短直接的回答", ["偏好"]),
                ("p-memory 把向量化收进库内部", ["项目"]),
                ("检索要优先返回精确结果", ["原则"]),
                ("笔记默认关闭向量化，但仍进全文召回", ["记忆"]),
            ]:
                kb.memories.upsert_by_judgment(judgment=judgment, tags=tags)
            note_path = Path(directory) / "real.md"
            note_path.write_text(
                "档案：真实模型验证只用一小份样本。\n\n它直接完成搜索，不做多余处理。",
                encoding="utf-8",
            )
            kb.notes.upsert_file(path=str(note_path))

            report = kb.embeddings.sync("real-v1", batch=batch)["value"]
            print(f"内部同步：scanned={report['scanned']} written={report['written']} "
                  f"batches={report['batches']} interrupted={report['interrupted']}")

            result = kb.search("真实模型 检索", kinds=["memory", "note", "chunk"], limit=5,
                               embed_space="real-v1", with_total=True)
            flags = result["diagnostics"]
            print(f"命中 {len(result['hits'])} 条（过滤后总量 {result['total']}）")
            print(f"走了哪几条路：全文={flags['text_used']} 向量={flags['vector_used']} "
                  f"重排={flags['reranked']}")
            if flags["rerank_truncated"]:
                print(f"重排截断：{flags['rerank_truncated']} 条候选未送入")
            if flags["degraded"]:
                print(f"降级档位：{flags['degraded']}")
            for rank, hit in enumerate(result["hits"], start=1):
                record = hit["record"]
                label = record.get("judgment") or record.get("source") or "<记录>"
                print(f"{rank}. #{hit['key']['id']} {label}  score={hit['score']:.6f} "
                      f"rerank={hit['rerank_score']}")

            print(f"嵌入回调实收批次：{embed_batches}")
            if rerank_model:
                print(f"重排回调实收候选数：{rerank_docs}")

            kb.embeddings.unregister_embedder("real-v1")
            degraded = kb.search("真实模型 检索", kinds=["memory", "note", "chunk"], limit=5,
                                 embed_space="real-v1")
            flags = degraded["diagnostics"]
            print("--- 注销嵌入回调后 ---")
            print(f"命中 {len(degraded['hits'])} 条，全文={flags['text_used']} "
                  f"向量={flags['vector_used']} 降级={flags['degraded']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PMemoryError as exc:
        print(f"库错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
