# 记忆生命周期

记忆的生命周期由**反馈**和**衰减**两个显式操作驱动。核心只计算强度与候选，**最终删除由宿主决定**。

## 状态字段

```rust
struct MemoryState {
    pinned: bool,               // 固定保留，永不衰减
    strength: i64,              // 强度，默认 1
    useful_count: i64,          // 有效召回次数
    useful_score: f64,          // 有用分
    last_recalled_at_us: i64?,
    last_decay_at_us: i64?,
}
```

## 三档

按 `useful_score` 与阈值分档，默认阈值 `tier0 = 3.0`、`tier1 = 10.0`：

| 档 | 条件 | 行为 |
|---|---|---|
| T0 | `score < 3.0` | 时间自然遗忘 |
| T1 | `3.0 <= score < 10.0` | 被召回且无用才遗忘 |
| T2 | `score >= 10.0` | 永不遗忘 |

## 反馈（feedback）

```rust
struct FeedbackRequest {
    filter: ReadFilter,
    recalled_ids: Vec<i64>,   // 本次被召回的记录
    useful_ids: Vec<i64>,     // 其中被判为有用的
    now_us: Option<i64>,
    policy: DecayPolicy,
}
struct FeedbackReport { recalled: usize, boosted: usize, penalized: usize }
```

规则：

- `useful_ids` 必须是 `recalled_ids` 的子集，否则 `validation`。
- 有用项：`strength += strength_boost`、`useful_count += 1`、`useful_score += useful_score_boost`、`last_recalled_at_us = now`。
- 未命中且**非固定保留**、且处于 **T1** 的召回项：`strength -= 1`（不低于 0）。
- 其余（T0、T2、固定保留）不变。

## 衰减（decay）

```rust
fn decay(filter, policy, at) -> WriteReceipt<DecayReport>
struct DecayReport { decayed: usize, retirement_candidates: Vec<RecordKey> }
```

规则：

- 只处理命中筛选、**非固定保留**、处于 **T0** 且 `strength > 0` 的记忆。
- 参考时间 `reference = max(created_at, last_recalled_at, last_decay_at)`。
- 经过的周期数 `steps = (now - reference) / (tier0_cycle_days * 1 天)`。
- `steps > 0` 时：`strength -= steps`（不低于 0），`last_decay_at = reference + steps * cycle`。
- 收尾：`strength == 0` 且 `tier < 2` 的记忆进入 `retirement_candidates`。

**衰减只产生候选，不删除数据。** 宿主拿到候选后自行决定删除、降级或永久化。

## 默认参数

```rust
struct DecayPolicy {
    tier0_threshold: f64,     // 3.0
    tier1_threshold: f64,     // 10.0
    useful_score_boost: f64,  // 2.5
    strength_boost: i64,      // 1
    tier0_cycle_days: u32,    // 3
}
```

校验：阈值/提升须为有限正数，`tier1 > tier0`，`strength_boost >= 1`，`tier0_cycle_days != 0`。

## 与其他域的关系

- 记忆内容变化会重算 `fingerprint`，使旧向量失效（写入时同事务删除）。
- 衰减与反馈都是写入操作，返回 `WriteReceipt`；索引未就绪不影响数据提交。
