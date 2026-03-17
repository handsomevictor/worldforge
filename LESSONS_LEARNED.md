# 经验教训（持续更新）

> 本文档只记录**实际编写代码时遇到的真实问题**。没有经历过的坑不会写在这里。

---

## Step 02-06：core + 全部分布（2026-03-17）

### BUG-01：`math.erfinv` 不存在于 Python 标准库

**现象：** 写了 `math.erfinv(2 * q - 1)` 用于 Normal.ppf()，跑测试时报：
```
AttributeError: module 'math' has no attribute 'erfinv'
```

**根因：** 误以为 `math.erfinv` 在 Python 3.13 中已加入标准库，实际上 Python（截至 3.14）的 `math` 模块只有 `erf` 和 `erfc`，没有逆函数。

**修复：** 添加 `_erfinv(z)` 辅助函数：优先使用 `scipy.special.erfinv`，不可用时用 Halley's 初始近似 + Newton 迭代收敛。

**教训：** 假设标准库功能前先用 `python -c "import math; help(math)"` 验证。

---

### BUG-02：`Pareto.ppf(0.0)` 不应该报错

**现象：** 测试写了 `assert d.ppf(0.0) == scale`，但实现里 `if not 0 < q < 1` 会对 q=0 抛 DistributionError。

**根因：** 照搬了 Normal.ppf 的参数校验（`0 < q < 1` 开区间），但 Pareto 的 PPF 在 q=0 有合法值（等于 scale，即分布的最小值）。

**修复：** Pareto.ppf 改为 `0 <= q < 1`，q=1 才是无效的（对应无穷大）。

**教训：** 不同分布的 ppf 边界条件不同，不要无脑复制粘贴参数校验。

---

## Step 07-09：Agent + EventQueue + SimContext（2026-03-17）

这三步**零 bug，132 个测试一次全绿**。没有遇到问题，所以没有经验教训可记。

但有两个**设计决策**值得记录（在写代码前就思考清楚，避免了潜在 bug）：

### 设计决策 01：agent 删除必须延迟

在 `_run_tick()` 中遍历所有 agent 并调用 `step()`。如果 agent 在 `step()` 内调用 `ctx.remove_agent(self)`，而 `remove_agent` 直接修改 `self._agents`，就会在迭代 dict 时修改它，导致 `RuntimeError: dictionary changed size during iteration`。

**解决方案：** `remove_agent()` 只是把 agent 放入 `_pending_removals` 列表。等 `_run_tick()` 完成所有 `step()` 调用后，再统一执行 `_flush_pending()`。

### 设计决策 02：`on_event` 广播只分发给重写了它的 agent

如果对每个 `emit()` 都遍历全部 agent 并调用 `on_event`，在 1M agent 规模下开销巨大。

**判断方式：** `type(agent).on_event is not Agent.on_event` 在 O(1) 时间内检查 agent 是否重写了 `on_event`。没有重写的 agent 完全跳过，无需额外数据结构。

---

## Step 16-18：输出、Runner 与集成测试（2026-03-17）

### BUG-03：可复现性测试失败——agent ID 跨 run 不重置

**现象：** `test_reproducibility` 调用 `sim.run()` 两次，对比两次结果发现 user_id 不同（第一次 ID 从 1 开始，第二次从 11 开始），导致购买事件 dict 不相等。

**根因：** Agent ID 由全局 `itertools.count` 生成。`_reset_id_counter()` 只在测试 fixture 里调用了一次（测试函数开始前），没有在两次 `run()` 之间再次调用。

**修复：** `SequentialRunner.run()` 在每次运行开始时调用 `_reset_id_counter(1)`，确保同一个 `seed` 无论运行多少次 ID 序列都从 1 起始。

**教训：** 可复现性不只是 RNG seed，所有有状态的全局计数器都要在运行入口处重置。

---

## Step 19-30：环境、行为、场景、CLI、Benchmark（2026-03-17）

这批步骤**零 bug，162 个测试一次全绿**。没有遇到运行时问题，所以没有 bug 经验教训可记。

**性能结果：** 1,000 agents × 1,000 steps = **65ms**（目标 < 1 秒），超额完成 15 倍。
