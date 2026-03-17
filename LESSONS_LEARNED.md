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

---

## 第十一/十二阶段：新场景 + 16 个数据逻辑 Bug 修复（2026-03-17）

### BUG-04：epidemic 同一 tick 内双重感染
**现象：** 高传播率（transmission_prob=0.99）时，同一个人在同一 tick 内被记录为被感染多次（多条 InfectionEvent 指向同一 person_id）。
**根因：** `spread()` 全局规则逐个遍历已感染者，对每个感染者的接触对象立即设置 `target.state = "I"`。若同一 susceptible person 在同一 tick 内被多个感染者接触，会被重复感染，生成多条 InfectionEvent。
**修复：** 引入 `newly_infected: dict[str, tuple]`，用 agent_id 作为 key 去重。所有感染候选先写入 dict（自动去重），遍历结束后统一应用，确保每人每 tick 最多一次 InfectionEvent。
**教训：** 在多个 agent 可以并发修改同一目标状态的场景中，必须用"先收集候选，再原子提交"模式。

### BUG-05：epidemic 模块级全局变量跨 run 污染
**现象：** BatchRunner 连续执行两个 `epidemic_world(recovery_rate=0.01)` 和 `epidemic_world(recovery_rate=0.5)`，后者的 recovery_rate 会覆盖前者，导致第一个 sim 使用了错误参数。
**根因：** `Person.step()` 引用模块级 `_recovery_rate` 全局变量。`epidemic_world()` 用 `global _recovery_rate = recovery_rate` 赋值。多个实例共享同一全局变量，顺序调用时会相互干扰。
**修复：** 将 `recovery_rate` 改为 `Person` 的 per-agent `field()`，`epidemic_world()` 用工厂函数为每个 agent 注入参数值。`_transmission_prob` 则保留为闭包局部变量（在 `spread()` rule 内引用），不再用全局变量。
**教训：** 场景工厂函数的参数绝对不能通过模块级全局变量传递给 Agent 类方法。要么用 per-agent field，要么用闭包捕获。

### BUG-06：gmv_daily 指标是累计值而非当日值
**现象：** ecommerce 场景的 `gmv_daily` 指标每天单调递增，第 30 天的值等于整个仿真期间的总 GMV。用户以为是当日营业额，实际是总营业额。
**根因：** `ctx.event_sum(PurchaseEvent, "amount")` 不带 `last=` 参数时，默认对从仿真开始到现在的所有事件求和。`AggregatorProbe` 每天触发一次，但每次都是全量累计。
**修复：** 改为 `ctx.event_sum(PurchaseEvent, "amount", last="1 day")`，同时新增 `gmv_cumulative` 保留完整累计值。fintech 场景的月度指标做了相同修复（`last=30`）。
**教训：** 所有"日指标""月指标"等周期性指标，必须明确加 `last=` 参数窗口，否则它们只是变相的累计指标。命名中带"daily/monthly"要与数据语义一致。

### BUG-07：IoT 异常计数统计的是所有读数
**现象：** `n_anomalies` 时序指标的数值等于传感器总读数（n_sensors × steps），远大于实际异常数（anomaly_rate=0.005 时应约为 0.5%）。
**根因：** `ctx.event_count(SensorReading)` 统计所有 SensorReading 事件，包括正常和异常，没有过滤 `is_anomaly==True` 的字段。
**修复：** 改为 `sum(1 for e in ctx._event_log if isinstance(e, SensorReading) and e.is_anomaly)`，并新增 `anomaly_rate`（异常数/总读数）指标。
**教训：** 事件类型过滤只是第一层过滤；事件内字段的条件过滤需要单独处理，`ctx.event_count()` 无法自动做到。

### BUG-08：org_dynamics HireEvent 从不被发出
**现象：** `EventLogProbe` 注册了 `HireEvent`，但运行后 event_log 中 `HireEvent` 记录为空，无法追踪招聘行为。
**根因：** `backfill_hiring` 全局规则调用 `ctx.spawn(Employee, count=1)` 创建新员工，但从不调用 `ctx.emit(HireEvent(...))`。
**修复：** 利用 `ctx.spawn()` 的 `init` 回调参数，在 agent 被创建后立即 emit HireEvent：`ctx.spawn(Employee, count=1, init=_emit_hire)`。
**教训：** 凡是在代码中注册了某种 Event 到 Probe 但实际找不到 emit 调用的，说明遗漏了事件发射点。应在写场景时做一次"event trace"检查。

### BUG-09：能源电网储能效率不对称
**现象：** 电池充放电的能量账算不平。充电 100 MWh 只存入 92 MWh（efficiency=0.92），但放电 92 MWh 直接交付给电网 92 MWh，没有损耗。
**根因：** 充电路径应用了效率（`charge * bat.efficiency`），放电路径没有（直接 `bat.charge_level_mwh -= discharge`），违反能量守恒。
**修复：** 放电时计算：要向电网输出 `deficit` MW，需要从电池取出 `deficit / efficiency` MWh；实际取出量 = min(charge_rate/efficiency, stored_needed, charge_level)；电网获得 = 取出量 × efficiency。
**教训：** 物理系统中有损耗的组件（电池、泵、变压器）必须在充/放两侧都施加效率系数，单侧会引入能量凭空产生或消失的 bug。

### 设计决策 03：sim.set_environment() 替代手动注入
market_microstructure 场景原来用 `sim._market_env = env`（私有属性）存储环境，并在注释里说"runner must wire this manually"。这是一个 footgun：用户忘记手动注入时，所有 `ctx.environment.mid_price()` 调用都会抛 AttributeError。
**解决方案：** 添加 `sim.set_environment(env)` 公开 API，`SequentialRunner.run()` 在启动时自动检查 `sim._environment` 并注入到 `ctx.environment`。用户无需手动处理。
**教训：** 框架内部的依赖注入不应交给用户手动完成。凡是 "you must wire X before running" 的注释，都是应该由框架自动处理的信号。
