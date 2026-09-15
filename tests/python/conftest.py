"""Python 绑定的共享夹具。

每个用例一个独立数据目录；库实例统一由夹具关闭，避免 `writer.lock` 泄漏到下个用例。
"""
import pytest

from p_memory import ClosedError, KnowledgeBase


@pytest.fixture
def open_kb(tmp_path):
    """返回一个建库工厂：`open_kb()`、`open_kb("other", scopes=("private",))`。"""
    opened: list[KnowledgeBase] = []

    def make(name: str = "data", **options) -> KnowledgeBase:
        kb = KnowledgeBase(str(tmp_path / name), **options)
        opened.append(kb)
        return kb

    yield make
    for kb in opened:
        try:
            kb.close()
        except ClosedError:
            pass


@pytest.fixture
def kb(open_kb):
    """默认库：namespace=default、scopes=(public,)、write_scope=public。"""
    return open_kb()
