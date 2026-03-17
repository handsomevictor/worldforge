# STRUCTURE.md — worldforge 模块参考

每个模块的职责和公开 API 说明。

---

## 顶层结构

```
worldforge/
├── src/worldforge/     # 可安装包
├── tests/              # pytest 测试套件
├── CLAUDE.md           # 项目章程与实现计划
├── README.md           # 用户文档
├── STRUCTURE.md        # 本文件
├── PROGRESS_LOG.md     # 实现进度追踪
├── TUTORIAL.md         # 分步教程
├── LESSONS_LEARNED.md  # 实际 debug 经验记录
├── pyproject.toml      # 构建与依赖配置
└── .gitignore
```

---

## `src/worldforge/`

### `__init__.py`
公开导出：`Agent`、`Simulation`、`field`、`SimContext`。
用户可以直接 `from worldforge import Agent, Simulation, field`。

---

### `core/`

#### `clock.py` ✅
抽象基类 `Clock` + `DiscreteClock`。
- `tick()` → 推进一步
- `now` → 当前时间
- `is_done` → 仿真是否结束
- `reset()` → 重置到初始状态

#### `event_queue.py`
基于 `heapq` 的优先队列事件调度器。
- `schedule(event, at)` — 在时间 at 入队
- `pop()` → 下一个事件
- `peek_time()` → 不消费地查看下一事件时间

#### `registry.py`
全局 name→class 注册表。
- `register(name, cls)`
- `lookup(name)` → class

#### `context.py`
`SimContext` — 运行时上下文，传入每个 `step()`、rule、probe。
关键 API：
- `ctx.now` — 当前仿真时间
- `ctx.rng` — 已 seed 的 `numpy.random.Generator`
- `ctx.emit(event)` — 发布事件
- `ctx.agents(AgentType, filter=None)` → list
- `ctx.agent_count(AgentType)` → int
- `ctx.agent_mean(AgentType, field)` → float
- `ctx.remove_agent(agent)`
- `ctx.spawn(AgentType, count, **kwargs)`
- `ctx.event_sum(EventType, field, last=None)` → float
- `ctx.event_count(EventType, last=None)` → int

#### `exceptions.py` ✅
异常层次：
- `WorldForgeError`（基类）
  - `ConfigurationError`
  - `SimulationError`
  - `AgentError`
  - `EventOrderError`
  - `DistributionError`

---

### `agent.py`
`Agent` 基类和 `field()` 描述符。

`field(initializer)` 接受：
- 常量值：`field(0.0)`
- `Distribution` 实例：`field(Normal(5000, 1000))`
- callable `(agent) -> value`：`field(lambda a: f"user_{a.id}")`
- `ConditionalDistribution`

Agent 生命周期钩子（可选重写）：
- `on_born(ctx)`
- `step(ctx)`
- `on_die(ctx)`
- `on_event(event, ctx)`

---

### `simulation.py`
`Simulation` — 顶层用户入口。

```python
sim = Simulation(name, seed, clock)
sim.add_agents(AgentType, count, factory=None)
sim.add_probe(probe)
sim.add_shock(shock)
sim.on(EventType)(handler)
sim.global_rule(every=...)(fn)
result = sim.run(progress=False)
sim.checkpoint(path)
Simulation.from_checkpoint(path)
```

---

### `time/`

#### `discrete.py` — `DiscreteClock`
抽象整数步长时钟，运行 N 个 tick。

#### `calendar.py` — `CalendarClock`
真实日历时间，支持时区。
- `start`、`end`、`step`（timedelta 或字符串如 `"1 day"`）

#### `event_driven.py` — `EventDrivenClock`
下一事件时间推进，适合稀疏事件系统。

---

### `distributions/` ✅

所有分布共享统一接口：
```python
dist.mean() -> float
dist.std() -> float
dist.pdf(x) -> float
dist.cdf(x) -> float
dist.ppf(q) -> float
dist.sample(rng) -> scalar
dist.sample_batch(n, rng) -> np.ndarray
```

#### `base.py` ✅ — `Distribution` 抽象基类

#### `continuous.py` ✅
- `Normal(mu, sigma, clip=None)`
- `LogNormal(mu, sigma, clip=None)`
- `Exponential(scale, clip=None)`
- `Pareto(alpha, scale=1.0)`
- `Gamma(shape, scale)`
- `Beta(alpha, beta)`
- `Uniform(low, high)`
- `Triangular(low, mode, high)`
- `Weibull(shape, scale)`

#### `discrete.py` ✅
- `Poisson(lam)`
- `Binomial(n, p)`
- `Geometric(p)`
- `Empirical(values, weights=None)` / `Empirical.from_data(data)`
- `Categorical(choices, weights)`

#### `temporal.py` ✅
- `HourOfDay(pattern: dict[int, float])` — 小时乘数
- `DayOfWeek(pattern: dict[str, float])` — 星期乘数
- `Seasonal(base, hour_multiplier=None, day_multiplier=None)`

#### `mixture.py` ✅ — `MixtureDistribution(components, weights)`

#### `conditional.py` ✅ — `ConditionalDistribution(condition, mapping, default=None)`

#### `correlated.py` ✅ — `CorrelatedDistributions(distributions, correlation)`（Gaussian copula）

---

### `behaviors/`

#### `state_machine.py` — `StateMachineBehavior`
声明式 FSM，支持概率转移和停留时间分布。

#### `decision.py` — `DecisionBehavior`
基于效用的决策。

#### `memory.py` — `MemoryBehavior`
Agent 短/长期记忆。

#### `social.py` — `SocialBehavior`
影响力传播与意见动力学。

#### `lifecycle.py` — `LifecycleBehavior`
基于年龄的生命事件。

---

### `environments/`

#### `base.py` — `Environment` 抽象基类
- `add_agent(agent)` / `remove_agent(agent)` / `agents()` / `step(ctx)`

#### `temporal.py` — `TemporalEnvironment`
无空间结构，仅时间维度。

#### `network.py` — `NetworkEnvironment`
NetworkX 图环境。
- `scale_free(n, m)` / `erdos_renyi(n, p)` / `small_world(n, k, p)`
- `neighbors(agent_id)` / `agents_within_hops(agent, hops)`

#### `grid.py` — `GridEnvironment`
2D 网格，支持 Moore/von Neumann 邻域，平面/环面拓扑。

#### `continuous.py` — `ContinuousSpace`
连续 2D/3D 空间。

#### `market.py` — `MarketEnvironment`
限价订单簿。
- `submit_order(agent_id, asset, side, price, qty)` → order_id
- `mid_price(asset)` / `spread(asset)` / `trade_history(last)`

---

### `events/`

#### `base.py` — `Event`
所有事件的冻结 dataclass 基类。`timestamp` 在 emit 时自动设置。

#### `lifecycle.py` — `AgentCreated` / `AgentRemoved`

#### `interaction.py` — `AgentInteraction`

#### `external.py` — `ExternalShock(at, effect, description)`

---

### `probes/`

#### `base.py` — `Probe` 抽象基类
- `collect(ctx)` / `finalize()` / `every`

#### `event_log.py` — `EventLogProbe(events)`
记录指定类型的所有事件。

#### `snapshot.py` — `SnapshotProbe(agent_type, fields, every, sample_rate)`
定期快照 agent 字段。

#### `aggregator.py` — `AggregatorProbe(metrics, every)`
定期聚合指标。

#### `timeseries.py` — `TimeSeriesProbe(series, every)`
高频时序指标。

---

### `output/`

#### `result.py` — `SimulationResult`
`sim.run()` 返回的容器。

- `result[probe_name]` → probe 数据
- `result.to_pandas()` / `to_polars()` / `to_dict()`
- `result.to_csv(path)` / `to_parquet(path)` / `to_json(path)` / `to_sql(engine)`
- `result.summary()` / `result.validate()`

---

### `runner/`

#### `sequential.py` — `SequentialRunner`
单线程仿真引擎，默认运行器。

#### `parallel.py` — `ParallelRunner`
多进程运行器，用于多个独立复制。

#### `batch.py` — `BatchRunner`
Monte Carlo 参数扫描。

---

### `scenarios/`

| 模块 | 函数 | 描述 |
|------|------|------|
| `ecommerce.py` | `ecommerce_world(...)` | 用户、商品、购买、欺诈、季节性 |
| `fintech.py` | `fintech_world(...)` | 账户、交易、信用评分 |
| `saas.py` | `saas_world(...)` | 注册、激活、流失漏斗 |
| `epidemic.py` | `epidemic_world(...)` | SIR/SEIR 网络传播与干预 |
| `supply_chain.py` | `supply_chain_world(...)` | 供应商、仓库、物流 |
| `market_microstructure.py` | `market_microstructure_world(...)` | 做市商、知情/噪音交易者 |
| `social_network.py` | `social_network_world(...)` | 意见动力学、影响力传播 |
| `iot_timeseries.py` | `iot_world(...)` | 传感器、异常、漂移 |

---

## `tests/`

```
tests/
├── conftest.py               # 共享 fixtures（rng 等）
├── unit/
│   ├── test_clock.py         ✅ 6 个测试
│   ├── test_distributions.py ✅ 72 个测试
│   ├── test_agent.py
│   ├── test_event_queue.py
│   ├── test_context.py
│   ├── test_behaviors.py
│   ├── test_environments.py
│   └── test_probes.py
├── integration/
│   ├── test_basic_sim.py
│   ├── test_calendar_sim.py
│   └── test_batch_runner.py
└── benchmarks/
    ├── bench_1k_agents.py
    └── bench_100k_agents.py
```

### 关键测试（来自 CLAUDE.md §十一）

| 测试 | 验证内容 |
|------|---------|
| `test_reproducibility` | 相同 seed → 完全一致的结果 |
| `test_no_future_events` | 事件时间戳永远 >= 当前仿真时间 |
| `test_state_machine_valid_transitions` | 所有 FSM 转移在合法转移表内 |
| `test_distribution_moments` | 10万次采样：均值/方差在 2% 误差内 ✅ |
| `test_conditional_distribution_respects_condition` | 条件分布采样落在正确范围 ✅ |
| `test_large_scale_performance` | 100k agents × 365 steps < 30 秒 |
| `test_agent_removal_no_dangling_refs` | 删除 Agent 后无悬空引用 |
| `test_event_ordering` | 事件按时间戳严格顺序处理 |
| `test_probe_data_integrity` | Probe 行数与仿真步数严格对应 |
