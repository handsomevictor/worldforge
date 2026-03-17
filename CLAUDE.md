# CLAUDE.md — worldforge

## 一、项目定位

**worldforge** 是一个工业级 Python 仿真框架，核心价值是：

> 用纯 Python 表达任意复杂系统的演化规则，让框架自动运行仿真并输出结构化数据。

它不是以下任何一种现有工具的替代品，而是填补它们之间的空白：

| 工具 | 擅长 | 缺陷 |
|------|------|------|
| SimPy | 离散事件、资源队列 | 无 Agent 状态、无数据导出、无 ABM |
| Mesa | Agent-Based Modeling | 无事件驱动、无内置概率系统、数据导出弱 |
| Faker/factory_boy | 静态测试数据 | 无时间演化、无因果关系、无 Agent 行为 |
| AnyLogic | 多范式仿真 | 商业软件、Java/GUI、不可编程扩展 |

worldforge 的定位：以 Python 为第一公民，统一支持三种仿真范式：
1. Agent-Based (ABM) — 个体行为驱动涌现
2. Discrete-Event (DES) — 事件驱动状态转变
3. Time-Step — 固定步长演化（物理、经济、时序数据）

三者可以在同一个仿真中混合使用。

---

## 二、设计哲学

### 原则一：Python 即模型语言
用户不学 DSL，不写 YAML/XML。所有规则、行为、约束、概率分布都是普通 Python 代码。任何 Python 能表达的逻辑，框架都能支持。

### 原则二：声明 What，框架处理 How
用户声明"这个系统由什么构成、规则是什么、要测量什么"，框架负责调度、时间推进、数据收集、并行化。

### 原则三：组合优于继承
功能通过 mixin、插件、装饰器组合，而不是深层继承树。用户可以替换任何内部组件。

### 原则四：数据导出是一等公民
仿真结束后产出的数据必须能直接用于分析、训练 ML 模型、写入数据库。

### 原则五：可复现
相同 seed 必须在任何机器上产生完全相同的结果。

---

## 三、适用场景（不限于此）

- 电商/金融系统行为仿真
- 流行病/舆论传播网络仿真
- 供应链/物流系统压力测试数据
- 交通流、能源电网
- 游戏经济系统设计验证
- 市场微结构仿真（订单簿、做市商）
- 用户行为漏斗/留存/病毒传播
- IoT 传感器时序数据生成
- 服务器/队列系统负载仿真
- 任何可以用规则描述的演化系统

---

## 四、架构总览

```
worldforge/
├── src/worldforge/
│   ├── __init__.py
│   ├── core/
│   │   ├── clock.py            # 仿真时钟
│   │   ├── event_queue.py      # 优先队列事件调度器
│   │   ├── registry.py         # 注册表
│   │   ├── context.py          # SimContext 运行时上下文
│   │   └── exceptions.py       # 异常层次
│   ├── agent.py                # Agent 基类
│   ├── simulation.py           # Simulation 主类
│   ├── time/
│   │   ├── discrete.py         # 固定步长时钟
│   │   ├── event_driven.py     # 事件驱动时钟
│   │   └── calendar.py         # 日历时间
│   ├── distributions/
│   │   ├── base.py             # Distribution 抽象基类
│   │   ├── continuous.py       # Normal, LogNormal, Exponential, Pareto,
│   │   │                       # Gamma, Beta, Uniform, Triangular, Weibull
│   │   ├── discrete.py         # Poisson, Binomial, Geometric, Empirical, Categorical
│   │   ├── temporal.py         # HourOfDay, DayOfWeek, Seasonal
│   │   ├── mixture.py          # MixtureDistribution
│   │   ├── conditional.py      # ConditionalDistribution
│   │   └── correlated.py       # CorrelatedDistributions
│   ├── behaviors/
│   │   ├── state_machine.py    # StateMachineBehavior
│   │   ├── decision.py         # DecisionBehavior
│   │   ├── memory.py           # MemoryBehavior
│   │   ├── social.py           # SocialBehavior
│   │   └── lifecycle.py        # LifecycleBehavior
│   ├── environments/
│   │   ├── base.py
│   │   ├── network.py          # NetworkEnvironment（图/网络）
│   │   ├── grid.py             # GridEnvironment（2D/3D 网格）
│   │   ├── continuous.py       # ContinuousSpace
│   │   ├── market.py           # MarketEnvironment（订单簿）
│   │   └── temporal.py         # TemporalEnvironment（无空间）
│   ├── events/
│   │   ├── base.py
│   │   ├── lifecycle.py        # AgentCreated, AgentRemoved
│   │   ├── interaction.py      # AgentInteraction
│   │   └── external.py         # ExternalShock
│   ├── probes/
│   │   ├── base.py
│   │   ├── snapshot.py         # SnapshotProbe
│   │   ├── event_log.py        # EventLogProbe
│   │   ├── aggregator.py       # AggregatorProbe
│   │   └── timeseries.py       # TimeSeriesProbe
│   ├── output/
│   │   ├── result.py           # SimulationResult
│   │   ├── pandas_out.py
│   │   ├── polars_out.py
│   │   ├── sql_out.py
│   │   ├── json_out.py
│   │   └── streaming_out.py
│   ├── runner/
│   │   ├── sequential.py
│   │   ├── parallel.py
│   │   └── batch.py            # Monte Carlo 批量扫描
│   └── scenarios/
│       ├── ecommerce.py
│       ├── fintech.py
│       ├── saas.py
│       ├── epidemic.py
│       ├── supply_chain.py
│       ├── market_microstructure.py
│       ├── social_network.py
│       └── iot_timeseries.py
└── tests/
    ├── unit/
    ├── integration/
    └── benchmarks/
```

---

## 五、核心 API 设计

### 5.1 Agent

```python
from worldforge import Agent, field
from worldforge.distributions import Normal, Categorical, ConditionalDistribution
from worldforge.behaviors import StateMachineBehavior

class User(Agent):
    # 字段声明：支持固定值、分布、callable
    balance: float = field(Normal(mu=5000, sigma=2000, clip=(0, None)))
    tier: str = field(Categorical(
        choices=["free", "pro", "enterprise"],
        weights=[0.70, 0.20, 0.10]
    ))
    age_days: int = field(0)
    churn_risk: float = field(0.0)

    # 条件分布字段
    bonus: float = field(ConditionalDistribution(
        condition=lambda a: a.tier,
        mapping={
            "free":       Uniform(0, 0),
            "pro":        Normal(500, 100),
            "enterprise": Normal(5000, 1000),
        }
    ))

    # lambda 字段（依赖其他字段）
    email: str = field(lambda agent: f"user_{agent.id}@example.com")

    def step(self, ctx):
        # ctx 提供：当前时间、环境引用、随机数生成器、事件发射器
        self.age_days += 1
        self.churn_risk = self._compute_churn_risk()

        if ctx.rng.random() < self.purchase_probability():
            amount = Normal(mu=200, sigma=100, clip=(1, self.balance)).sample(ctx.rng)
            self.emit(PurchaseEvent(user_id=self.id, amount=amount, ts=ctx.now))
            self.balance -= amount

    def _compute_churn_risk(self) -> float:
        base = 0.01
        if self.tier == "free" and self.age_days > 30:
            base += 0.05
        return min(base, 1.0)

    def purchase_probability(self) -> float:
        return {"free": 0.02, "pro": 0.08, "enterprise": 0.20}[self.tier]

    # 生命周期钩子（可选重写）
    def on_born(self, ctx): ...
    def on_die(self, ctx): ...
    def on_event(self, event, ctx): ...  # 收到事件时
```

### 5.2 Simulation

```python
from worldforge import Simulation
from worldforge.time import CalendarClock
from worldforge.probes import EventLogProbe, SnapshotProbe, AggregatorProbe

sim = Simulation(
    name="user_behavior_sim",
    seed=42,
    clock=CalendarClock(
        start="2024-01-01",
        end="2024-12-31",
        step="1 day"
    ),
)

# 添加 Agent（批量）
sim.add_agents(User, count=10_000)

# 添加 Agent（工厂函数，精细控制）
sim.add_agents(
    User,
    count=1000,
    factory=lambda i, rng: User(
        tier="enterprise" if i < 50 else "pro",
        balance=rng.uniform(10000, 100000)
    )
)

# 事件处理器
@sim.on(PurchaseEvent)
def handle_purchase(event, ctx):
    ctx.emit(InvoiceEvent(
        user_id=event.user_id,
        amount=event.amount,
        due_date=ctx.now + timedelta(days=30)
    ))

# 全局规则（不属于任何 Agent）
@sim.global_rule(every="1 week")
def weekly_churn_check(ctx):
    for agent in ctx.agents(User):
        if agent.churn_risk > 0.8 and ctx.rng.random() < agent.churn_risk:
            ctx.remove_agent(agent)

# 数据采集
sim.add_probe(EventLogProbe(events=[PurchaseEvent, InvoiceEvent]))
sim.add_probe(SnapshotProbe(
    agent_type=User,
    fields=["id", "balance", "tier", "churn_risk"],
    every="1 week",
    sample_rate=0.1,
))
sim.add_probe(AggregatorProbe(
    metrics={
        "dau":       lambda ctx: ctx.agent_count(User),
        "gmv_daily": lambda ctx: ctx.event_sum(PurchaseEvent, "amount", last="1 day"),
        "avg_bal":   lambda ctx: ctx.agent_mean(User, "balance"),
    },
    every="1 day",
))

result = sim.run(progress=True)
```

### 5.3 时间系统

```python
from worldforge.time import DiscreteClock, CalendarClock, EventDrivenClock

# 固定步长（抽象时间）
clock = DiscreteClock(steps=1000)

# 真实日历时间
clock = CalendarClock(
    start="2020-01-01",
    end="2025-12-31",
    step="1 hour",
    timezone="UTC",
)

# 事件驱动（next-event advance，适合稀疏事件系统）
clock = EventDrivenClock(max_time=1e9, min_step=1e-9)
```

### 5.4 分布系统

分布是一等公民，可以作为字段初始化器、在 step() 中直接调用、互相组合。

```python
from worldforge.distributions import (
    # 连续
    Normal, LogNormal, Exponential, Pareto, Gamma, Beta,
    Uniform, Triangular, Weibull,
    # 离散
    Poisson, Binomial, Geometric, Empirical, Categorical,
    # 时序
    HourOfDay, DayOfWeek, Seasonal,
    # 组合
    MixtureDistribution, ConditionalDistribution, CorrelatedDistributions,
)

# 截断
income = Normal(mu=5000, sigma=2000, clip=(0, 100000))

# 混合分布（双峰消费分布）
spending = MixtureDistribution(
    components=[Normal(50, 10), Normal(500, 100)],
    weights=[0.8, 0.2]
)

# 多变量相关（price 与 quantity 负相关）
price, qty = CorrelatedDistributions(
    distributions=[LogNormal(4, 0.5), Poisson(10)],
    correlation=-0.7
).sample(rng)

# 周期性时序分布
arrival_rate = Seasonal(
    base=Poisson(lam=100),
    hour_multiplier=HourOfDay(
        pattern={0: 0.1, 8: 0.5, 12: 1.2, 18: 1.5, 22: 0.3}
    ),
    day_multiplier=DayOfWeek(
        pattern={"Mon": 1.0, "Sat": 1.5, "Sun": 0.8}
    ),
)

# 从数据拟合经验分布
empirical = Empirical.from_data([45, 52, 48, 61, 55])

# 所有分布的通用接口
dist.mean()
dist.std()
dist.pdf(x)
dist.cdf(x)
dist.ppf(0.95)
dist.sample(rng)
dist.sample_batch(n=1000, rng=rng)  # 返回 numpy array
```

### 5.5 状态机行为

```python
from worldforge.behaviors import StateMachineBehavior
from worldforge.distributions import Exponential

class OrderFSM(StateMachineBehavior):
    states = ["pending", "paid", "processing", "shipped", "delivered",
              "cancelled", "refunded"]
    initial = "pending"
    terminal = ["delivered", "cancelled", "refunded"]

    # 转移规则：(概率/条件, 目标状态, 停留时间分布)
    transitions = {
        "pending": [
            (0.85, "paid",      Exponential(scale=300)),
            (0.15, "cancelled", Exponential(scale=3600)),
        ],
        "paid": [
            (1.00, "processing", Exponential(scale=600)),
        ],
        "processing": [
            (0.98, "shipped",   Exponential(scale=86400)),
            (0.02, "refunded",  Exponential(scale=3600)),
        ],
        "shipped": [
            (0.95, "delivered", Exponential(scale=3*86400)),
            (0.05, "refunded",  Exponential(scale=86400)),
        ],
    }

    # 转移回调
    def on_transition(self, from_state, to_state, ctx):
        ctx.emit(OrderStatusChanged(
            order_id=self.agent.id,
            from_status=from_state,
            to_status=to_state,
            ts=ctx.now
        ))

class Order(Agent):
    fsm: OrderFSM = field(OrderFSM)
    amount: float = field(LogNormal(mu=5, sigma=1))

    def step(self, ctx):
        self.fsm.step(ctx)

    @property
    def status(self):
        return self.fsm.current_state
```

### 5.6 环境系统

```python
from worldforge.environments import (
    NetworkEnvironment, GridEnvironment,
    ContinuousSpace, MarketEnvironment, TemporalEnvironment,
)

# 网络环境
env = NetworkEnvironment.scale_free(n=10_000, m=3)
env = NetworkEnvironment.erdos_renyi(n=1000, p=0.01)
env = NetworkEnvironment.from_edgelist("edges.csv")

neighbors = env.neighbors(agent_id)
env.add_edge(a_id, b_id, weight=0.5)
nearby = env.agents_within_hops(agent, hops=2)

# 网格环境
env = GridEnvironment(width=100, height=100,
                      topology="torus", neighborhood="moore")
env.place(agent, x=10, y=20)
env.move(agent, dx=1, dy=0)

# 市场订单簿
env = MarketEnvironment(
    assets=["BTC", "ETH"],
    initial_prices={"BTC": 50000, "ETH": 3000},
    tick_size=0.01,
)
env.submit_order(agent_id, asset="BTC", side="buy", price=49900, qty=0.1)
env.mid_price("BTC")
env.trade_history(last=100)
```

### 5.7 Agent 间交互

```python
# 方式一：事件总线（解耦）
class Seller(Agent):
    def step(self, ctx):
        ctx.emit(PriceUpdate(seller_id=self.id, price=self.price))

class Buyer(Agent):
    def on_event(self, event, ctx):
        if isinstance(event, PriceUpdate):
            if event.price < self.max_price:
                ctx.emit(PurchaseIntent(buyer_id=self.id, seller_id=event.seller_id))

# 方式二：直接访问（通过 ctx）
class Predator(Agent):
    def step(self, ctx):
        nearby = ctx.agents_near(self, radius=5, type=Prey)
        if nearby:
            prey = ctx.rng.choice(nearby)
            ctx.remove_agent(prey)
            self.energy += 10

# 方式三：动态生成子 Agent
class Company(Agent):
    def step(self, ctx):
        if self.is_hiring:
            ctx.spawn(
                Employee,
                count=Poisson(lam=10).sample(ctx.rng),
                parent=self,
                init=lambda e: e.update(
                    company_id=self.id,
                    salary=Normal(80000, 20000).sample(ctx.rng)
                )
            )
```

### 5.8 数据采集

```python
from worldforge.probes import (
    EventLogProbe, SnapshotProbe,
    AggregatorProbe, TimeSeriesProbe, CustomProbe,
)

# 事件日志
sim.add_probe(EventLogProbe(events=[PurchaseEvent, ChurnEvent]))

# 状态快照
sim.add_probe(SnapshotProbe(
    agent_type=User,
    fields=["id", "balance", "tier", "churn_risk"],
    every="1 week",
    sample_rate=0.1,
))

# 聚合指标
sim.add_probe(AggregatorProbe(
    metrics={
        "n_users":   lambda ctx: ctx.agent_count(User),
        "gmv":       lambda ctx: ctx.event_sum(PurchaseEvent, "amount"),
        "new_users": lambda ctx: ctx.event_count(UserCreated),
    },
    every="1 day",
))

# 时序指标
sim.add_probe(TimeSeriesProbe(
    series={
        "avg_balance": lambda ctx: ctx.agent_mean(User, "balance"),
        "p95_balance": lambda ctx: ctx.agent_percentile(User, "balance", 0.95),
    },
    every="1 hour",
))

# 完全自定义
@sim.probe(every="1 week")
def weekly_cohort(ctx, collector):
    for tier in ["free", "pro", "enterprise"]:
        users = ctx.agents(User, filter=lambda u: u.tier == tier)
        collector.record({
            "week": ctx.now.isocalendar().week,
            "tier": tier,
            "count": len(users),
        })
```

### 5.9 仿真结果

```python
result = sim.run()

# 转换格式
result.to_pandas()          # {name: DataFrame}
result.to_polars()          # {name: pl.DataFrame}
result.to_dict()            # Python dict
result.to_json("./output/")
result.to_csv("./output/")
result.to_sql(engine)       # SQLAlchemy，自动建表
result.to_parquet("./output/")

# 流式（超大规模）
with result.stream_to_parquet("./output/") as writer:
    sim.run(stream=writer)

# 内置分析
result.summary()
result.plot_metrics()
result.validate()

# 单独访问
purchase_df = result["event_log"].to_pandas()
user_snapshots = result["user_snapshot"].to_pandas()
```

### 5.10 批量参数扫描（Monte Carlo）

```python
from worldforge.runner import BatchRunner

batch = BatchRunner(
    sim_factory=lambda params: build_sim(**params),
    param_grid={
        "churn_rate":     [0.01, 0.05, 0.10],
        "initial_users":  [1000, 5000, 10000],
        "viral_coeff":    Uniform(0.5, 2.0),
    },
    n_samples=100,
    n_replications=5,
    workers=16,
)

batch_result = batch.run()
df = batch_result.to_pandas()
batch_result.sensitivity_analysis("gmv_total")
```

---

## 六、高级特性

### 外部数据注入

```python
real_prices = pd.read_csv("btc_prices.csv")

@sim.on_step
def inject_real_prices(ctx):
    price = real_prices.loc[real_prices.date == ctx.now.date(), "close"].iloc[0]
    ctx.environment.set_price("BTC", price)
```

### 外部冲击

```python
from worldforge.events import ExternalShock

sim.add_shock(ExternalShock(
    at="2024-03-15",
    effect=lambda ctx: [
        setattr(a, "churn_risk", min(a.churn_risk * 3, 1.0))
        for a in ctx.agents(User)
    ],
    description="Competitor launched free tier"
))
```

### 强化学习接口（Gymnasium 兼容）

```python
from worldforge.rl import GymWrapper

env = GymWrapper(
    sim=sim,
    observation=lambda ctx: np.array([
        ctx.agent_mean(User, "balance"),
        ctx.agent_count(User),
        ctx.event_rate(PurchaseEvent, window="1 day"),
    ]),
    action_space="continuous",
    reward=lambda ctx: ctx.event_sum(PurchaseEvent, "amount", last="1 day"),
)
```

### 检查点与恢复

```python
sim.checkpoint("checkpoint_day_100.pkl")

sim2 = Simulation.from_checkpoint("checkpoint_day_100.pkl")
sim2.update_param("churn_base_rate", 0.02)
result2 = sim2.run()
```

### 实时仿真

```python
sim = Simulation(
    clock=CalendarClock(step="1 second", realtime=True),
)
# 用于数字孪生、实时监控系统测试
```

---

## 七、内置场景（参考实现）

```python
from worldforge.scenarios import (
    ecommerce_world,
    fintech_world,
    epidemic_world,
    market_microstructure_world,
    iot_world,
    saas_world,
    supply_chain_world,
    social_network_world,
)

# 电商
sim = ecommerce_world(
    n_users=10_000, n_products=1_000,
    duration="2 years", locale="zh_CN",
    include_fraud=True, seasonality=True,
)

# 流行病
sim = epidemic_world(
    population=1_000_000,
    network_type="small_world",
    R0=2.5,
    incubation_period=Gamma(shape=5.8, scale=0.95),
    interventions=[
        {"day": 30, "type": "lockdown", "compliance": 0.7},
        {"day": 90, "type": "vaccination", "rate_per_day": 10000},
    ],
)

# 市场微结构（订单簿）
sim = market_microstructure_world(
    n_market_makers=10,
    n_informed_traders=50,
    n_noise_traders=500,
    asset="STOCK",
    initial_price=100.0,
    duration="1 trading day",
)

# IoT 时序
sim = iot_world(
    n_sensors=1_000,
    sensor_types=["temperature", "pressure", "vibration"],
    duration="30 days",
    sample_interval="1 minute",
    anomaly_rate=0.005,
)
```

每个场景函数接收参数，返回完整配置好的 Simulation 对象，用户可以在此基础上继续添加 Agent、规则、Probe。

---

## 八、性能目标

| 规模 | 目标 |
|------|------|
| 1,000 agents × 1,000 steps | < 1 秒 |
| 100,000 agents × 365 steps | < 30 秒 |
| 1,000,000 agents × 30 steps | < 120 秒（numpy 向量化） |

**性能策略：**
- Agent 字段存储在 numpy structured array（列式存储），非 Python 对象列表
- 批量 step() 优先 numpy 向量化，回退 Python 循环
- 事件队列用 heapq，事件 dispatch 用 dict
- 大规模仿真自动流式输出，避免内存爆炸

---

## 九、依赖策略

**硬依赖：**
- `numpy >= 1.26`
- `faker >= 24.0`

**软依赖（按需）：**
- `pandas >= 2.0`
- `polars >= 0.20`
- `sqlalchemy >= 2.0`
- `networkx >= 3.0`
- `scipy >= 1.11`
- `matplotlib >= 3.7`
- `gymnasium >= 0.29`

---

## 十、pyproject.toml

```toml
[project]
name = "worldforge"
version = "0.1.0"
description = "Industrial-grade multi-paradigm simulation framework for Python"
readme = "README.md"
requires-python = ">=3.13"
license = { text = "MIT" }
keywords = ["simulation", "agent-based", "discrete-event",
            "synthetic-data", "abm", "des", "monte-carlo"]
dependencies = ["numpy>=1.26", "faker>=24.0"]

[project.optional-dependencies]
pandas  = ["pandas>=2.0"]
polars  = ["polars>=0.20"]
sql     = ["sqlalchemy>=2.0"]
network = ["networkx>=3.0"]
science = ["scipy>=1.11"]
viz     = ["matplotlib>=3.7"]
rl      = ["gymnasium>=0.29"]
all     = ["pandas>=2.0", "polars>=0.20", "sqlalchemy>=2.0",
           "networkx>=3.0", "scipy>=1.11", "matplotlib>=3.7"]
dev     = ["pytest>=8.0", "pytest-cov", "mypy>=1.10", "ruff>=0.4",
           "hypothesis>=6.0", "pytest-benchmark"]

[project.scripts]
worldforge = "worldforge.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/worldforge"]
```

---

## 十一、关键测试

```python
def test_reproducibility():
    """相同 seed，两次运行完全一致"""

def test_no_future_events():
    """事件时间戳永远 >= 当前仿真时间"""

def test_state_machine_valid_transitions():
    """所有状态转移都在合法转移表内"""

def test_distribution_moments():
    """采样 100000 次，均值/方差在 1% 误差内"""

def test_conditional_distribution_respects_condition():
    """条件分布在不同条件下采样值落在正确范围"""

def test_large_scale_performance():
    """100k agents × 365 steps 在 30 秒内完成"""

def test_agent_removal_no_dangling_refs():
    """删除 Agent 后无悬空引用"""

def test_event_ordering():
    """事件按时间戳严格顺序处理"""

def test_probe_data_integrity():
    """Probe 收集的数据行数与仿真步数严格对应"""
```

---

## 十二、实现顺序

按以下顺序实现，每步完成后运行 pytest 全部通过再继续：

```
Step 01: pyproject.toml + 项目骨架（空文件）
Step 02: core/exceptions.py + core/clock.py（DiscreteClock）
Step 03: distributions/base.py + distributions/continuous.py（Normal, Uniform, Exponential）
Step 04: distributions/discrete.py（Poisson, Categorical, Empirical）
Step 05: distributions/temporal.py + distributions/mixture.py + distributions/conditional.py
Step 06: distributions/correlated.py + 全部分布单元测试
Step 07: agent.py（字段声明系统 + field()）
Step 08: core/event_queue.py + events/base.py
Step 09: core/context.py（SimContext）
Step 10: environments/base.py + environments/temporal.py
Step 11: simulation.py（骨架 + 基本 run() 循环）
Step 12: time/discrete.py + time/calendar.py + time/event_driven.py
Step 13: behaviors/state_machine.py + behaviors/lifecycle.py
Step 14: probes/base.py + probes/event_log.py + probes/snapshot.py
Step 15: probes/aggregator.py + probes/timeseries.py
Step 16: output/result.py + output/pandas_out.py + output/dict_backend.py
Step 17: runner/sequential.py（完整可运行单线程引擎）
Step 18: 集成测试：跑通 10 Agent × 100 step 完整仿真
Step 19: environments/network.py（NetworkX 集成）
Step 20: environments/grid.py + environments/continuous.py
Step 21: environments/market.py（订单簿）
Step 22: behaviors/decision.py + behaviors/social.py + behaviors/memory.py
Step 23: runner/parallel.py + runner/batch.py
Step 24: output/polars_out.py + output/sql_out.py + output/streaming_out.py
Step 25: scenarios/ecommerce.py
Step 26: scenarios/epidemic.py + scenarios/fintech.py
Step 27: scenarios/market_microstructure.py + scenarios/iot_timeseries.py + scenarios/saas.py
Step 28: rl/ Gymnasium 接口
Step 29: cli.py
Step 30: benchmarks/ + 全量测试 + README
```