# worldforge

**工业级多范式 Python 仿真框架。**

> 用纯 Python 表达任意复杂系统的演化规则，让框架自动运行仿真并输出结构化数据。

---

## 为什么选择 worldforge？

现有工具各有所长，但没有一个能覆盖全部需求：

| 工具 | 擅长 | 缺失 |
|------|------|------|
| SimPy | 离散事件、资源队列 | 无 Agent 状态、无数据导出、无 ABM |
| Mesa | Agent-Based Modeling | 无事件驱动、无内置概率系统、数据导出弱 |
| Faker/factory_boy | 静态测试数据 | 无时间演化、无因果关系、无 Agent 行为 |
| AnyLogic | 多范式仿真 | 商业软件、Java/GUI、不可编程扩展 |

worldforge 填补这个空白：**Python 为第一公民**，统一支持三种仿真范式——**同一个仿真中可混合使用**：

1. **Agent-Based (ABM)** — 个体行为驱动涌现
2. **Discrete-Event (DES)** — 事件驱动状态转变
3. **Time-Step** — 固定步长演化（物理、经济、时序数据）

---

## 快速开始

```bash
pip install worldforge
```

```python
from worldforge import Agent, Simulation, field
from worldforge.distributions import Normal, Categorical
from worldforge.time import CalendarClock
from worldforge.probes import AggregatorProbe

class User(Agent):
    balance: float = field(Normal(mu=5000, sigma=2000, clip=(0, None)))
    tier: str = field(Categorical(
        choices=["free", "pro", "enterprise"],
        weights=[0.70, 0.20, 0.10]
    ))
    churn_risk: float = field(0.0)

    def step(self, ctx):
        self.churn_risk = 0.01 if self.tier != "free" else 0.05
        if ctx.rng.random() < self.churn_risk:
            ctx.remove_agent(self)

sim = Simulation(
    name="user_churn",
    seed=42,
    clock=CalendarClock(start="2024-01-01", end="2024-12-31", step="1 day"),
)
sim.add_agents(User, count=10_000)
sim.add_probe(AggregatorProbe(
    metrics={"dau": lambda ctx: ctx.agent_count(User)},
    every="1 day",
))

result = sim.run(progress=True)
df = result.to_pandas()["aggregator"]
print(df.head())
```

---

## 方法论（Methodology）

worldforge 的设计基于以下系统仿真理论与工程实践的结合：

### 1. 时间推进（Time Advance）

worldforge 支持三种时间推进机制，对应不同仿真场景：

| 时钟类型 | 类 | 适用场景 |
|----------|-----|---------|
| 固定步长 | `DiscreteClock(steps=N)` | 每步计时均匀，ABM、时序数据生成 |
| 日历时间 | `CalendarClock(start, end, step)` | 真实时间轴，商业/金融仿真 |
| 事件驱动 | `EventDrivenClock(max_time)` | 稀疏事件系统，网络/DES |

每个 tick 按以下顺序执行：
1. 时钟推进（`clock.tick()`）
2. 所有 Agent 执行 `step(ctx)`（确定性顺序，id 升序）
3. 延迟变更刷新（`_flush_pending`）：Agent 的创建/删除在 tick 末尾批量生效，防止迭代中修改集合
4. ExternalShock 触发（如当前时间 == 冲击时间）
5. 全局规则（`@sim.global_rule`）调用
6. 所有 Probe 的 `on_step(ctx, step)` 调用

### 2. Agent 字段系统

Agent 字段通过 `field()` 声明，支持四种初始化方式：

```python
class MyAgent(Agent):
    # 常量
    name: str = field("unknown")

    # 概率分布 → 每个 Agent 在创建时独立采样
    balance: float = field(Normal(mu=5000, sigma=1000))

    # 条件分布 → 依赖 Agent 已有字段
    bonus: float = field(ConditionalDistribution(
        condition=lambda a: a.tier,
        mapping={"free": Uniform(0, 0), "pro": Normal(500, 100)},
    ))

    # Lambda → 依赖 Agent 自身（在所有其他字段解析后调用）
    email: str = field(lambda a: f"user_{a.id}@example.com")
```

字段解析顺序遵循声明顺序（包含父类字段），确保 Lambda 字段能安全引用其他字段。

### 3. 事件系统

事件采用**发布-订阅**模型：

- `ctx.emit(event)` 立即广播事件
- 事件被追加到 `ctx._event_log`（时间戳自动设置）
- 已注册的全局处理器（`@sim.on(EventType)`）同步调用
- 只有**重写了 `on_event()` 的 Agent** 收到广播（O(1) 判断，不遍历全体）

这确保在 1M Agent 规模下 `emit()` 开销为 O(k)，其中 k 是重写了 `on_event` 的 Agent 数量，而非 O(N)。

### 4. 可复现性

给定相同 `seed`，仿真在任何机器上产生完全一致的结果。保障机制：

1. `SequentialRunner.run()` 在每次运行开始时调用 `_reset_id_counter(1)`，Agent ID 序列从 1 起始
2. RNG 通过 `np.random.default_rng(seed)` 创建，为 PCG64 算法（高质量、可跳转）
3. Agent `step()` 调用顺序固定（dict 插入顺序，即创建顺序）
4. 延迟变更（`_flush_pending`）在所有 step() 完成后批量处理，避免顺序依赖

### 5. 概率分布

所有分布实现 `Distribution` 抽象基类，提供统一接口：

```python
dist.sample(rng)            # 单次采样
dist.sample_batch(n, rng)   # 批量采样 → numpy array
dist.mean()                 # 理论均值
dist.std()                  # 理论标准差
dist.pdf(x)                 # 概率密度
dist.cdf(x)                 # 累积分布
dist.ppf(q)                 # 分位数函数（逆 CDF）
```

高级组合：
- `MixtureDistribution` — 混合分布（双峰等）
- `ConditionalDistribution` — 条件分布（依赖上下文）
- `CorrelatedDistributions` — 高斯 Copula 多变量相关

### 6. 数据采集（Probe 系统）

Probe 是数据采集的核心抽象。所有 Probe 继承 `Probe` 基类，实现：
- `collect(ctx)` — 在每个触发步执行数据收集
- `finalize()` — 仿真结束后返回 `list[dict]`

触发频率由 `every` 参数控制，支持整数步数或字符串时长（`"1 day"`、`"1 week"`）。

| Probe | 用途 |
|-------|------|
| `EventLogProbe` | 记录指定类型的全部事件（输出包含 `event_type` 字段，方便多类型合并后区分来源） |
| `SnapshotProbe` | 周期性拍摄 Agent 字段快照（支持采样率） |
| `AggregatorProbe` | 周期性计算聚合指标（DAU、GMV 等） |
| `TimeSeriesProbe` | 高频标量时序（每小时、每分钟）|
| `CustomProbe` | 通过 `@sim.probe()` 装饰器定义的自定义采集逻辑 |

### 7. 状态机行为

`StateMachineBehavior` 实现概率有限状态机（PFSM）：

- 转移规则：`(概率, 目标状态, 停留时间分布)` 三元组列表
- 停留时间：可以是 Distribution 实例、固定数值或 `float("inf")`
- 惰性初始化：首次 `step()` 时才进行初始转移采样
- 终止检测：`is_terminal` 属性，终止状态后 `step()` 为无操作

### 8. 性能设计

| 规模 | 实测（纯 Python step） |
|------|----------------------|
| 1,000 agents × 1,000 steps | **~65ms** |
| 目标：< 1 秒 | ✅ 超额完成 15 倍 |

性能策略：
- Agent 存储于 Python dict（O(1) 查找/删除）
- `on_event` 广播使用 `type(a).on_event is not Agent.on_event` 跳过未重写的 Agent
- 批量变更在 tick 末统一刷新，避免 GC 压力
- 大规模场景可使用 `ParallelRunner` 或 `BatchRunner` 多进程扩展

---

## 核心特性

### 概率分布——一等公民

```python
from worldforge.distributions import (
    Normal, LogNormal, Exponential, Pareto, Gamma, Beta,
    Uniform, Triangular, Weibull,
    Poisson, Binomial, Geometric,
    Empirical, Categorical,
    HourOfDay, DayOfWeek, Seasonal,
    MixtureDistribution, ConditionalDistribution, CorrelatedDistributions,
)

# 截断分布
income = Normal(mu=5000, sigma=2000, clip=(0, 100_000))

# 双峰消费分布
spending = MixtureDistribution(
    components=[Normal(50, 10), Normal(500, 100)],
    weights=[0.8, 0.2]
)

# 季节性到达率
arrival = Seasonal(
    base=Poisson(lam=100),
    hour_multiplier=HourOfDay(pattern={8: 0.5, 12: 1.2, 18: 1.5}),
)
```

### 状态机行为

```python
from worldforge.behaviors import StateMachineBehavior
from worldforge.distributions import Exponential

class OrderFSM(StateMachineBehavior):
    states = ["pending", "paid", "shipped", "delivered", "cancelled"]
    initial = "pending"
    terminal = ["delivered", "cancelled"]
    transitions = {
        "pending": [(0.85, "paid", Exponential(300)), (0.15, "cancelled", Exponential(3600))],
        "paid":    [(1.00, "shipped", Exponential(86400))],
        "shipped": [(0.95, "delivered", Exponential(3*86400)), (0.05, "cancelled", Exponential(86400))],
    }
```

### 丰富的环境系统

```python
from worldforge.environments import NetworkEnvironment, GridEnvironment, MarketEnvironment

# 无标度社交网络
env = NetworkEnvironment.scale_free(n=10_000, m=3)

# 2D 网格（元胞自动机、流行病等）
env = GridEnvironment(width=100, height=100, topology="torus")

# 限价订单簿
env = MarketEnvironment(assets=["BTC"], initial_prices={"BTC": 50_000})
```

### 结构化数据导出

```python
result.to_pandas()          # {probe_name: pd.DataFrame}
result.to_polars()          # {probe_name: pl.DataFrame}
result.to_json("out/")      # 每个 probe 一个 JSON 文件
result.to_csv("out/")       # 每个 probe 一个 CSV 文件
result.to_parquet("out/")   # 每个 probe 一个 Parquet 文件
result.to_sql(engine)       # SQLAlchemy，自动建表
result.summary()            # 人类可读摘要
```

### 数据完整性验证

```python
report = result.validate()
if not report.passed:
    for err in report.errors:
        print("ERROR:", err)
# ValidationReport(PASS, 0 warnings)
```

### Monte Carlo 批量参数扫描

```python
from worldforge.runner import BatchRunner

batch = BatchRunner(
    sim_factory=lambda p: build_sim(**p),
    param_grid={"churn_rate": [0.01, 0.05, 0.10], "n_users": [1000, 10000]},
    n_replications=5,
    workers=8,
)
df = batch.run().to_pandas()
```

### 内置场景

```python
from worldforge.scenarios import (
    ecommerce_world, epidemic_world, fintech_world,
    saas_world, market_microstructure_world,
    iot_world, supply_chain_world, social_network_world,
    rideshare_world, game_economy_world,
    org_dynamics_world, energy_grid_world,
)

# 流行病仿真（SIR 模型）
sim = epidemic_world(
    population=100_000,
    transmission_prob=0.3,
    recovery_rate=0.05,
    duration_days=180,
    seed=42,
)
result = sim.run()
```

---

## 安装

```bash
# 仅核心依赖（numpy + faker）
pip install worldforge

# 按需安装可选依赖
pip install "worldforge[pandas]"          # pandas 输出
pip install "worldforge[polars]"          # polars 输出
pip install "worldforge[network]"         # NetworkX 环境
pip install "worldforge[science]"         # scipy（更快的 ppf 计算）
pip install "worldforge[pandas,network]"  # 组合安装

# 完整安装
pip install "worldforge[all]"

# 开发环境（含测试、lint、benchmark 工具）
pip install "worldforge[dev]"
```

**环境要求：** Python >= 3.13，numpy >= 1.26，faker >= 24.0

---

## 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev,pandas]"

# 运行全部测试
python -m pytest tests/

# 仅运行单元测试
python -m pytest tests/unit/

# 仅运行集成测试
python -m pytest tests/integration/

# 运行 benchmark（排除慢速测试）
python -m pytest tests/benchmarks/ -k "not slow"

# 运行慢速性能测试（100k agents × 365 steps）
python -m pytest tests/benchmarks/ -m slow

# 查看详细输出
python -m pytest tests/ -v

# 生成覆盖率报告
python -m pytest tests/ --cov=worldforge --cov-report=html
```

---

## CLI 使用

```bash
# 列出所有内置场景
worldforge list

# 运行场景（快速预览）
worldforge run ecommerce --n-agents 1000 --steps 30 --seed 42

# 保存结果到 JSON
worldforge run epidemic --n-agents 5000 --steps 90 --output ./output/

# 查看环境信息
worldforge info
```

---

## 项目结构

```
worldforge/
├── src/worldforge/       # 框架源码
│   ├── core/             # 时钟、事件队列、上下文、异常
│   ├── distributions/    # 17 种概率分布
│   ├── behaviors/        # 状态机、决策、记忆、社交、生命周期行为
│   ├── environments/     # 网络、网格、连续空间、市场订单簿
│   ├── events/           # 事件基类及内置事件
│   ├── probes/           # 5 种数据采集探针
│   ├── output/           # SimulationResult + pandas/polars/SQL/流式输出
│   ├── runner/           # 顺序、并行、批量运行器
│   ├── scenarios/        # 12 个预置仿真场景
│   ├── simulation.py     # Simulation 主类
│   ├── rl.py             # Gymnasium 兼容强化学习接口
│   └── cli.py            # 命令行接口
└── tests/
    ├── unit/             # 单元测试（每个模块独立）
    ├── integration/      # 集成测试（端到端仿真）
    └── benchmarks/       # 性能基准测试
```

详细模块说明见 [STRUCTURE.md](STRUCTURE.md)。

---

## 设计原则

1. **Python 即模型语言** — 不学 DSL，不写 YAML/XML
2. **声明 What，框架处理 How** — 调度、时间、数据收集全部自动
3. **组合优于继承** — 行为自由混合，无深层继承树
4. **数据导出是一等公民** — 输出直接用于 ML、分析、数据库
5. **可复现** — 相同 seed → 任意机器结果完全一致

---

## 许可证

MIT
