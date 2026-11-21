"""同路径共享句柄与引用计数。

目录锁是进程级的：同一目录第二次真正去打开会抛 `LockedError`。宿主应用重载
自己的模块之后会再构造一次 `KnowledgeBase`，如果每次都真的开目录，就会撞锁
——这是插件热重载后「记忆功能整轮失效」的根因。

这里锁定行为：同路径共享同一个 native 句柄，引用计数归零才真正关闭并释放锁。
"""

import pytest

from p_memory import KnowledgeBase
from p_memory.errors import ClosedError


def test_same_path_shares_one_handle(tmp_path):
    """同路径第二次构造不再撞锁，拿到的就是同一个句柄。"""
    root = str(tmp_path / "shared")
    first = KnowledgeBase(root)
    second = KnowledgeBase(root)
    try:
        assert first._native is second._native
        assert first.counts()["total"] == 0
        assert second.counts()["total"] == 0
    finally:
        first.close()
        second.close()


def test_handle_closes_only_when_last_released(tmp_path):
    """还有人在用就不关；归零之后锁才释放。"""
    root = str(tmp_path / "counted")
    first = KnowledgeBase(root)
    second = KnowledgeBase(root)

    first.close()
    # 第二个持有者还在，库必须仍可用
    assert second.counts()["total"] == 0
    # 已归还的那个不再接受访问
    with pytest.raises(ClosedError):
        first.counts()

    second.close()
    # 归零之后目录锁释放，同路径能重新打开
    third = KnowledgeBase(root)
    try:
        assert third.counts()["total"] == 0
    finally:
        third.close()


def test_close_is_idempotent(tmp_path):
    """重复关闭不报错，也不能把计数减成负数。"""
    root = str(tmp_path / "idempotent")
    kb = KnowledgeBase(root)
    kb.close()
    kb.close()

    again = KnowledgeBase(root)
    try:
        assert again.counts()["total"] == 0
    finally:
        again.close()


def test_closed_instance_rejects_access(tmp_path):
    """关闭之后的实例一律拒绝访问，而不是把 native 错误透出去。"""
    kb = KnowledgeBase(str(tmp_path / "closed"))
    kb.close()
    with pytest.raises(ClosedError):
        kb.invoke("counts")
