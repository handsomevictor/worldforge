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
result.to_parquet("out/")   # 每个 probe 一个文件
result.to_sql(engine)       # SQLAlchemy，自动建表
result.to_json("out/")
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
from worldforge.scenarios import ecommerce_world, epidemic_world, market_microstructure_world

sim = epidemic_world(
    population=1_000_000,
    network_type="small_world",
    R0=2.5,
    interventions=[{"day": 30, "type": "lockdown", "compliance": 0.7}],
)
result = sim.run()
```

---

## 性能目标

| 规模 | 目标 |
|------|------|
| 1,000 agents × 1,000 steps | < 1 秒 |
| 100,000 agents × 365 steps | < 30 秒 |
| 1,000,000 agents × 30 steps | < 120 秒（numpy 向量化） |

---

## 安装

```bash
# 仅核心依赖
pip install worldforge

# 可选扩展
pip install "worldforge[pandas,polars,network,science,viz]"

# 全部安装
pip install "worldforge[all]"

# 开发环境
pip install "worldforge[dev]"
```

**环境要求：** Python >= 3.13，numpy >= 1.26，faker >= 24.0

---

## 项目结构

```
worldforge/
├── src/worldforge/       # 框架源码
│   ├── core/             # 时钟、事件队列、上下文、注册表
│   ├── distributions/    # 概率分布
│   ├── behaviors/        # 状态机、决策、记忆、社交行为
│   ├── environments/     # 网络、网格、市场、连续空间
│   ├── events/           # 事件基类及内置事件
│   ├── probes/           # 数据采集探针
│   ├── output/           # 结果格式化输出
│   ├── runner/           # 顺序、并行、批量运行器
│   └── scenarios/        # 预置仿真场景
└── tests/
    ├── unit/
    ├── integration/
    └── benchmarks/
```

详细模块说明见 [STRUCTURE.md](STRUCTURE.md)。

---

## 设计原则

1. **Python 即模型语言** — 不学 DSL，不写 YAML/XML
2. **声明 What，框架处理 How** — 调度、时间、数据收集全部自动
3. **组合优于继承** — 行为自由混合
4. **数据导出是一等公民** — 输出直接用于 ML、分析、数据库
5. **可复现** — 相同 seed → 任意机器结果完全一致

---

## 许可证

MIT
