# TUTORIAL.md — worldforge 完整使用教程

本教程覆盖所有用户可用的公开 API，每个示例均标明**预期输出/行为**。

---

## 目录

1. [如何运行测试](#如何运行测试)
2. [快速入门](#快速入门)
3. [Agent 与字段系统](#agent-与字段系统)
4. [时间系统](#时间系统)
5. [概率分布](#概率分布)
6. [事件系统](#事件系统)
7. [数据采集（Probe）](#数据采集probe)
8. [Simulation 主类 API](#simulation-主类-api)
9. [仿真结果（SimulationResult）](#仿真结果simulationresult)
10. [行为系统（Behaviors）](#行为系统behaviors)
11. [环境系统（Environments）](#环境系统environments)
12. [运行器（Runner）](#运行器runner)
13. [内置场景（Scenarios）](#内置场景scenarios)
14. [CLI 使用](#cli-使用)
15. [常见模式](#常见模式)

---

## 如何运行测试

### 安装开发环境

```bash
# 克隆项目
git clone <repo-url>
cd worldforge

# 安装（带 dev 和 pandas 依赖）
pip install -e ".[dev,pandas]"
```

### 运行测试命令

```bash
# 运行全部测试（推荐）
python -m pytest tests/

# 仅运行单元测试（快，无 I/O）
python -m pytest tests/unit/

# 运行集成测试（端到端仿真）
python -m pytest tests/integration/

# 运行 benchmark（排除需要 100k agents 的慢速测试）
python -m pytest tests/benchmarks/ -k "not slow"

# 运行慢速性能测试（需要几十秒）
python -m pytest tests/benchmarks/ -m slow

# 详细输出（显示每个测试名称）
python -m pytest tests/ -v

# 只运行特定测试文件
python -m pytest tests/unit/test_behaviors.py

# 只运行特定测试类
python -m pytest tests/unit/test_behaviors.py::TestStateMachineBehavior

# 只运行特定测试方法
python -m pytest tests/unit/test_behaviors.py::TestStateMachineBehavior::test_transitions_after_steps

# 生成 HTML 覆盖率报告
python -m pytest tests/ --cov=worldforge --cov-report=html
# 之后在浏览器打开 htmlcov/index.html
```

### 预期结果

```
tests/unit/test_agent.py          20 passed
tests/unit/test_batch_runner.py    9 passed (部分 skip，需要 pandas)
tests/unit/test_behaviors.py      34 passed
tests/unit/test_clock.py           6 passed
tests/unit/test_context.py        25 passed
tests/unit/test_distributions.py  72 passed
tests/unit/test_environments.py   ~30 passed (NetworkEnvironment 需要 networkx)
tests/unit/test_event_queue.py     9 passed
tests/unit/test_output.py         ~18 passed (部分 skip，需要 pandas)
tests/unit/test_probes.py         26 passed
tests/unit/test_time.py           30 passed
tests/integration/test_basic_sim.py    14 passed
tests/integration/test_scenarios.py   13 passed
tests/benchmarks/test_performance.py   3 passed (非慢速)
```

---

## 快速入门

最简单的仿真：10 个 Counter agent 运行 100 步。

```python
from worldforge import Agent, Simulation, field
from worldforge.core.clock import DiscreteClock

class Counter(Agent):
    count: int = field(0)

    def step(self, ctx):
        self.count += 1

sim = Simulation(
    name="hello",
    seed=42,
    clock=DiscreteClock(steps=100),
)
sim.add_agents(Counter, count=10)
result = sim.run()

print(result.summary())
```

**预期输出：**
```
SimulationResult summary
----------------------------------------
  name: hello
  seed: 42
  steps: 100
  elapsed_seconds: 0.xxx
  agent_count_final: 10
  events_total: 0
  probes: []
```

---

## Agent 与字段系统

### 字段声明方式

```python
from worldforge import Agent, field
from worldforge.distributions import Normal, Categorical, Uniform
from worldforge.distributions import ConditionalDistribution

class User(Agent):
    # 1. 常量字段
    active: bool = field(True)
    age_days: int = field(0)

    # 2. 概率分布 → 每个 Agent 创建时独立采样
    balance: float = field(Normal(mu=5000, sigma=1000, clip=(0, None)))
    tier: str = field(Categorical(
        choices=["free", "pro", "enterprise"],
        weights=[0.70, 0.20, 0.10],
    ))

    # 3. 条件分布 → 依赖 Agent 当前状态
    bonus: float = field(ConditionalDistribution(
        condition=lambda a: a.tier,
        mapping={
            "free":       Uniform(0, 0),
            "pro":        Normal(500, 100, clip=(0, None)),
            "enterprise": Normal(5000, 1000, clip=(0, None)),
        },
    ))

    # 4. Lambda 字段 → 依赖其他已解析的字段
    email: str = field(lambda a: f"user_{a.id}@example.com")
```

**预期行为：**
- 每个 `User()` 实例有不同的 `balance`（从 Normal 分布采样）
- 90% 的用户 `tier` 为 "free" 或 "pro"，10% 为 "enterprise"
- `bonus` 随 `tier` 不同采样不同范围
- `email` 引用 `self.id`，格式固定

### override 覆盖默认值

```python
# 通过 kwarg 覆盖：跳过分布采样，直接使用给定值
user = User(balance=10000.0, tier="enterprise")
assert user.balance == 10000.0
assert user.tier == "enterprise"
```

### 生命周期钩子

```python
class TrackedUser(Agent):
    balance: float = field(1000.0)

    def on_born(self, ctx):
        """Agent 加入仿真时调用一次。"""
        print(f"User {self.id} born at t={ctx.now}")

    def step(self, ctx):
        """每个 tick 调用。"""
        self.balance *= 1.001

    def on_die(self, ctx):
        """Agent 被移除时调用一次。"""
        print(f"User {self.id} removed at t={ctx.now}")

    def on_event(self, event, ctx):
        """有事件广播时调用（需要重写才会被调度）。"""
        pass
```

---

## 时间系统

### DiscreteClock — 抽象步长

```python
from worldforge.core.clock import DiscreteClock

clock = DiscreteClock(steps=100)

# 属性
clock.now      # int，当前步数（从 0 开始）
clock.is_done  # True 当 now >= steps
clock.tick()   # 推进一步，now += 1
clock.reset()  # 回到 now = 0
```

**预期行为：** 运行 100 步后 `is_done == True`，`now == 100`。

### CalendarClock — 日历时间

```python
from worldforge.time.calendar import CalendarClock, parse_duration

clock = CalendarClock(
    start="2024-01-01",
    end="2024-12-31",
    step="1 day",
    timezone="UTC",
)

# 属性
clock.now    # datetime 对象
clock.step   # timedelta 对象
clock.is_done

# 字符串时长解析
parse_duration("1 day")     # timedelta(days=1)
parse_duration("2 hours")   # timedelta(hours=2)
parse_duration("30 minutes")# timedelta(minutes=30)
parse_duration("1 week")    # timedelta(weeks=1)
```

**预期行为：** `start="2024-01-01"`, `end="2024-01-08"`, `step="1 day"` → 7 次 tick 后 `is_done == True`。

### EventDrivenClock — 事件驱动

```python
from worldforge.time.event_driven import EventDrivenClock

clock = EventDrivenClock(max_time=1_000_000, min_step=1e-12)

clock.advance_to(42.5)  # 跳转到指定时间
clock.now               # 42.5
clock.is_done           # False（< max_time）

# 后退会抛出异常
clock.advance_to(10.0)  # → EventOrderError
```

---

## 概率分布

所有分布都实现相同接口：

```python
import numpy as np
rng = np.random.default_rng(42)

from worldforge.distributions import Normal

dist = Normal(mu=100, sigma=15)

# 单次采样
x = dist.sample(rng)             # float，如 112.7

# 批量采样
arr = dist.sample_batch(1000, rng)  # numpy array, shape (1000,)

# 统计量
dist.mean()    # 100.0
dist.std()     # 15.0
dist.pdf(100)  # ≈ 0.0266
dist.cdf(100)  # 0.5
dist.ppf(0.95) # ≈ 124.7（第 95 百分位）
```

### 所有分布类型

```python
from worldforge.distributions import (
    # 连续分布
    Normal(mu, sigma, clip=None)          # 正态分布
    LogNormal(mu, sigma, clip=None)       # 对数正态
    Exponential(scale, clip=None)         # 指数分布（scale=1/λ）
    Pareto(alpha, scale, clip=None)       # 帕累托分布
    Gamma(shape, scale, clip=None)        # Gamma 分布
    Beta(alpha, beta)                     # Beta 分布，值域 [0,1]
    Uniform(low, high)                    # 均匀分布
    Triangular(low, mode, high)           # 三角分布
    Weibull(shape, scale, clip=None)      # Weibull 分布

    # 离散分布
    Poisson(lam)                          # 泊松分布
    Binomial(n, p)                        # 二项分布
    Geometric(p)                          # 几何分布
    Empirical(values, weights=None)       # 经验分布（从列表采样）
    Categorical(choices, weights)         # 离散类别

    # 时序分布
    HourOfDay(pattern: dict)              # 小时模式（0-23 → 乘数）
    DayOfWeek(pattern: dict)              # 星期模式
    Seasonal(base, hour_multiplier, day_multiplier)  # 季节性

    # 高级组合
    MixtureDistribution(components, weights)          # 混合分布
    ConditionalDistribution(condition, mapping)       # 条件分布
    CorrelatedDistributions(distributions, correlation) # 相关多变量
)
```

**示例：双峰消费分布**

```python
from worldforge.distributions import MixtureDistribution, Normal

spending = MixtureDistribution(
    components=[Normal(50, 10), Normal(500, 100)],
    weights=[0.8, 0.2],
)

# 80% 的采样值在 ~50 附近，20% 在 ~500 附近
samples = spending.sample_batch(10000, rng)
# 均值 ≈ 0.8*50 + 0.2*500 = 140
assert abs(samples.mean() - 140) < 5
```

**示例：多变量相关（Gaussian Copula）**

```python
from worldforge.distributions import CorrelatedDistributions, LogNormal, Poisson

# price 与 quantity 负相关（价格高则购买量少）
corr_dist = CorrelatedDistributions(
    distributions=[LogNormal(4, 0.5), Poisson(10)],
    correlation=-0.7,
)
price, qty = corr_dist.sample(rng)
# 多次采样后，price 与 qty 的秩相关性约为 -0.7
```

---

## 事件系统

### 定义事件

```python
from dataclasses import dataclass
from worldforge.events.base import Event

@dataclass
class PurchaseEvent(Event):
    user_id: str
    product_id: str
    amount: float
    # timestamp 由框架自动设置，不需要声明
```

### 发射事件

```python
class Buyer(Agent):
    wallet: float = field(1000.0)

    def step(self, ctx):
        if ctx.rng.random() < 0.1:
            amount = 50.0
            self.wallet -= amount
            # self.emit() 等价于 ctx.emit()，但只能在 step() 内调用
            self.emit(PurchaseEvent(
                user_id=self.id,
                product_id="item_1",
                amount=amount,
            ))
```

**预期行为：** 事件 `timestamp` 会被框架自动设置为 `ctx.now`。

### 全局事件处理器

```python
sim = Simulation(...)

@sim.on(PurchaseEvent)
def handle_purchase(event, ctx):
    """在每次 PurchaseEvent 发射后立即同步调用。"""
    print(f"Purchase: user={event.user_id}, amount={event.amount}, ts={event.timestamp}")
```

### Agent 级事件处理

```python
class Seller(Agent):
    revenue: float = field(0.0)

    def on_event(self, event, ctx):
        """重写此方法即可接收所有广播事件。"""
        if isinstance(event, PurchaseEvent):
            if event.product_id.startswith("my_"):
                self.revenue += event.amount
```

**注意：** 只有**重写了 `on_event`** 的 Agent 类才会收到广播。未重写的 Agent 会被框架跳过（O(1) 判断，不影响性能）。

### ExternalShock（外部冲击）

```python
from worldforge.events.external import ExternalShock

sim.add_shock(ExternalShock(
    at="2024-06-15",          # CalendarClock：日期字符串；DiscreteClock：整数步数
    description="竞品降价",
    effect=lambda ctx: [
        setattr(u, "churn_risk", min(u.churn_risk * 3, 1.0))
        for u in ctx.agents(User)
    ],
))
```

**预期行为：** 当 `ctx.now.date() == 2024-06-15` 时，`effect` 函数被调用一次。

---

## 数据采集（Probe）

### EventLogProbe

记录所有指定类型的事件。

```python
from worldforge.probes import EventLogProbe

probe = EventLogProbe(
    events=[PurchaseEvent, ChurnEvent],
    name="event_log",
)
sim.add_probe(probe)
result = sim.run()

records = result["event_log"]
# records 是 list[dict]，每条记录对应一个事件
# 每条记录包含事件的所有非下划线开头字段 + "timestamp"
# 例：{"user_id": "1", "amount": 50.0, "timestamp": datetime(2024,1,2)}
```

### SnapshotProbe

周期性拍摄 Agent 字段快照。

```python
from worldforge.probes import SnapshotProbe

probe = SnapshotProbe(
    agent_type=User,
    fields=["id", "balance", "tier", "churn_risk"],
    every="1 week",       # 或 every=7（整数步数）
    sample_rate=0.10,     # 仅采样 10% 的 Agent（默认 1.0=全部）
    name="user_snapshot",
)
sim.add_probe(probe)
result = sim.run()

records = result["user_snapshot"]
# records 是 list[dict]
# 每条：{"timestamp": ..., "id": "3", "balance": 4500.0, "tier": "pro", "churn_risk": 0.01}
# 不存在的字段返回 None
```

### AggregatorProbe

周期性计算聚合指标。

```python
from worldforge.probes import AggregatorProbe

probe = AggregatorProbe(
    metrics={
        "dau":       lambda ctx: ctx.agent_count(User),
        "gmv_daily": lambda ctx: ctx.event_sum(PurchaseEvent, "amount"),
        "avg_bal":   lambda ctx: ctx.agent_mean(User, "balance"),
    },
    every="1 day",
    name="daily_metrics",
)
sim.add_probe(probe)
result = sim.run()

records = result["daily_metrics"]
# 每天一条：{"timestamp": ..., "dau": 9823, "gmv_daily": 450000.0, "avg_bal": 4981.2}
# 若某个 lambda 抛出异常，对应字段值为 None
```

### TimeSeriesProbe

高频标量时序采集。

```python
from worldforge.probes import TimeSeriesProbe

probe = TimeSeriesProbe(
    series={
        "avg_balance": lambda ctx: ctx.agent_mean(User, "balance"),
        "p95_balance": lambda ctx: ctx.agent_percentile(User, "balance", 0.95),
    },
    every="1 hour",     # 每小时一条
    name="timeseries",
)
sim.add_probe(probe)
result = sim.run()

records = result["timeseries"]
# 每条：{"timestamp": ..., "avg_balance": 4981.2, "p95_balance": 7200.0}
```

### CustomProbe（@sim.probe 装饰器）

完全自定义的数据采集逻辑。

```python
@sim.probe(every="1 week")
def weekly_cohort(ctx, collector):
    """
    ctx:       SimContext（可用所有查询方法）
    collector: 有 record(dict) 方法，调用它保存一行数据
    """
    for tier in ["free", "pro", "enterprise"]:
        users = ctx.agents(User, filter=lambda u: u.tier == tier)
        collector.record({
            "week":        ctx.now,
            "tier":        tier,
            "count":       len(users),
            "avg_balance": sum(u.balance for u in users) / max(len(users), 1),
        })

result = sim.run()
records = result["weekly_cohort"]
# 每周 3 条（每个 tier 一条）
# 格式：{"week": ..., "tier": "free", "count": 7000, "avg_balance": 3200.0}
```

---

## Simulation 主类 API

### 构造

```python
from worldforge import Simulation
from worldforge.core.clock import DiscreteClock
from worldforge.time.calendar import CalendarClock

sim = Simulation(
    name="my_sim",    # 字符串，出现在结果 metadata 中
    seed=42,          # int，控制所有随机性
    clock=DiscreteClock(steps=365),  # 时钟（默认 DiscreteClock(100)）
)
```

### add_agents

```python
# 基本用法：批量创建，字段从 field() 声明采样
sim.add_agents(User, count=10_000)

# 工厂函数：精细控制每个 Agent 的属性
sim.add_agents(
    User,
    count=1000,
    factory=lambda i, rng: User(
        tier="enterprise" if i < 50 else "pro",
        balance=float(rng.uniform(10_000, 100_000)),
    ),
)
```

**预期行为：** 工厂函数中 `i` 从 0 到 count-1，`rng` 是已 seed 的 Generator。

### on — 全局事件处理器

```python
@sim.on(PurchaseEvent)
def handle_purchase(event, ctx):
    # 返回值被忽略
    pass
```

### global_rule — 全局规则

```python
@sim.global_rule(every="1 week")   # 或 every=7
def weekly_churn(ctx):
    for u in ctx.agents(User):
        if ctx.rng.random() < u.churn_risk:
            ctx.remove_agent(u)
```

### run

```python
result = sim.run(
    progress=True,   # True → 打印进度条到 stdout；默认 False
)
# 返回 SimulationResult
```

### checkpoint / from_checkpoint

```python
sim.checkpoint("checkpoint.pkl")
sim2 = Simulation.from_checkpoint("checkpoint.pkl")
result2 = sim2.run()
```

---

## 仿真结果（SimulationResult）

### 访问数据

```python
result = sim.run()

# 按名称访问 probe 数据
records = result["event_log"]    # list[dict]

# 检查 probe 是否存在
"event_log" in result            # True or False

# 列出所有 probe 名称
result.keys()                    # ["event_log", "daily_metrics", ...]
```

### 转换格式

```python
# 转 dict（原始格式）
data = result.to_dict()
# 返回 {"probe_name": [{"field": val, ...}, ...]}

# 转 pandas DataFrame（需要 pandas）
dfs = result.to_pandas()
# 返回 {"probe_name": pd.DataFrame}
dfs["event_log"].head()

# 转 polars DataFrame（需要 polars）
from worldforge.output import to_polars
dfs = to_polars(result)

# 保存为 JSON 文件
result.to_json("./output/")
# 生成 ./output/event_log.json, ./output/daily_metrics.json ...

# 保存为 CSV 文件（需要 pandas）
result.to_csv("./output/")
# 生成 ./output/event_log.csv, ...

# 写入数据库（需要 sqlalchemy）
from worldforge.output import to_sql
from sqlalchemy import create_engine
engine = create_engine("sqlite:///my_sim.db")
to_sql(result, engine, if_exists="replace")
```

### 摘要

```python
print(result.summary())
# SimulationResult summary
# ----------------------------------------
#   name: my_sim
#   seed: 42
#   steps: 365
#   elapsed_seconds: 1.23
#   agent_count_final: 9821
#   events_total: 147832
#   probes: ['event_log', 'daily_metrics', 'user_snapshot']
#     'event_log': 147832 records
#     'daily_metrics': 365 records
#     'user_snapshot': 3650 records
```

### metadata

```python
result.metadata["name"]          # "my_sim"
result.metadata["seed"]          # 42
result.metadata["steps"]         # 365
result.metadata["elapsed_seconds"]  # 1.23
result.metadata["agent_count_final"]  # int
result.metadata["events_total"]   # int
```

---

## 行为系统（Behaviors）

### StateMachineBehavior — 概率有限状态机

```python
from worldforge.behaviors import StateMachineBehavior
from worldforge.distributions import Exponential

class OrderFSM(StateMachineBehavior):
    states = ["pending", "paid", "shipped", "delivered", "cancelled"]
    initial = "pending"
    terminal = ["delivered", "cancelled"]
    transitions = {
        "pending": [
            (0.85, "paid",       Exponential(scale=300)),   # (概率, 目标, 停留时间分布)
            (0.15, "cancelled",  Exponential(scale=3600)),
        ],
        "paid": [
            (1.00, "shipped", Exponential(scale=86400)),
        ],
        "shipped": [
            (0.95, "delivered", Exponential(scale=3*86400)),
            (0.05, "cancelled", Exponential(scale=86400)),
        ],
    }

    def on_transition(self, from_state, to_state, ctx):
        """状态转移时调用（可选重写）。"""
        ctx.emit(StatusChangedEvent(
            order_id=self.agent.id,
            from_status=from_state,
            to_status=to_state,
        ))

class Order(Agent):
    fsm: OrderFSM = field(OrderFSM)

    def step(self, ctx):
        self.fsm.step(ctx)
        if self.fsm.is_terminal:
            ctx.remove_agent(self)

    @property
    def status(self):
        return self.fsm.current_state
```

**预期行为：**
- `fsm.current_state` — 当前状态字符串
- `fsm.is_terminal` — 是否处于终止状态
- 首次 `step()` 时惰性初始化，采样第一个转移
- 停留时间到达后自动转移并调用 `on_transition`

### LifecycleBehavior — 生命周期管理

```python
from worldforge.behaviors import LifecycleBehavior
from worldforge.distributions import Normal

class Person(Agent):
    lifecycle: LifecycleBehavior = field(
        lambda a: LifecycleBehavior(lifespan=Normal(mu=70, sigma=10, clip=(20, None)))
    )

    def step(self, ctx):
        self.lifecycle.step(ctx)  # 年龄+1，lifespan 到达时触发 ctx.remove_agent

# 属性
# lifecycle.age       → int，已过步数
# lifecycle.is_alive  → bool，age < lifespan
```

### DecisionBehavior — 规则决策

```python
from worldforge.behaviors import DecisionBehavior

class UserDecision(DecisionBehavior):
    rules = [
        # (条件函数, 动作函数)，按顺序评估，第一个匹配的执行
        (lambda a, ctx: a.balance < 10,    lambda a, ctx: print(f"{a.id} is broke")),
        (lambda a, ctx: a.balance > 5000,  lambda a, ctx: setattr(a, "invested", True)),
        (lambda a, ctx: True,              lambda a, ctx: None),  # 默认动作
    ]

class User(Agent):
    balance: float = field(1000.0)
    invested: bool = field(False)
    decision: UserDecision = field(UserDecision)

    def step(self, ctx):
        self.decision.step(ctx)

# 动态添加规则
user.decision.add_rule(
    condition=lambda a, ctx: a.balance > 10000,
    action=lambda a, ctx: setattr(a, "tier", "vip"),
    priority=0,  # 插入到最前（None = 追加到末尾）
)
```

### SocialBehavior — 社会影响

```python
from worldforge.behaviors import SocialBehavior

class MySocial(SocialBehavior):
    opinion_field = "sentiment"   # 要更新的字段名
    influence_rate = 0.1          # 每步向邻居均值靠拢的比例
    conformity_bias = 0.0         # 额外向 0 靠拢的力度

class Person(Agent):
    sentiment: float = field(Uniform(-1.0, 1.0))
    social: MySocial = field(MySocial)

    def step(self, ctx):
        neighbors = ctx.environment.neighbors(self.id)
        self.social.influence(neighbors, ctx)
        # sentiment 向邻居均值靠拢
```

### ContagionBehavior — 传染行为

```python
from worldforge.behaviors import ContagionBehavior

class SIRC(ContagionBehavior):
    transmission_prob = 0.3   # 每个感染邻居的传染概率
    recovery_rate = 0.05      # 每步恢复概率

class Person(Agent):
    state: str = field("S")
    contagion: SIRC = field(SIRC)

    def step(self, ctx):
        neighbors = ctx.agents(Person)  # 简化：全局混合
        infected_neighbors = [n for n in neighbors if n.state == "I"]
        self.state = self.contagion.step_state(self.state, infected_neighbors, ctx)
```

**`step_state(state, infected_neighbors, ctx)` 逻辑：**
- `"S"` + n 个感染邻居 → P(感染) = 1 - (1 - p)^n
- `"I"` → P(恢复) = `recovery_rate`
- `"R"` → 不变

### MemoryBehavior — 有限记忆

```python
from worldforge.behaviors import MemoryBehavior

class ShortMemory(MemoryBehavior):
    capacity = 30  # 最多记住 30 条（循环缓冲区）

class Trader(Agent):
    memory: ShortMemory = field(ShortMemory)

    def step(self, ctx):
        # 记录一条
        self.memory.remember({"ts": ctx.now, "price": 100.0})

        # 查询最近 5 条
        recent = self.memory.recall(last=5)

        # 提取某个字段的历史值
        prices = self.memory.query("price")

        # 清除记忆
        self.memory.forget()

# len(self.memory) → 当前记忆条数
```

---

## 环境系统（Environments）

### GridEnvironment — 2D 网格

```python
from worldforge.environments import GridEnvironment

env = GridEnvironment(
    width=50,
    height=50,
    topology="torus",          # "bounded"（有边界）或 "torus"（环绕）
    neighborhood="moore",      # "moore"（8邻）或 "von_neumann"（4邻）
)

# 放置 Agent
env.place(agent, x=10, y=20)

# 相对移动
env.move(agent, dx=1, dy=-1)

# 查询位置（返回 (x, y) tuple 或 None）
pos = env.position(agent)

# 查询邻居（radius=1 → 直接相邻的格子）
neighbors = env.neighbors(agent, radius=1)  # list[Agent]

# 查询某格的全部 Agent
agents_here = env.agents_at(x=10, y=20)
```

**预期行为（torus）：** `place(agent, 51, 0)` 等价于 `place(agent, 1, 0)`。

### ContinuousSpace — 连续空间

```python
from worldforge.environments import ContinuousSpace

env = ContinuousSpace(
    width=100.0,
    height=100.0,
    topology="bounded",   # "bounded" 或 "torus"
)

env.place(agent, x=10.5, y=20.3)
env.move(agent, dx=0.5, dy=-1.0)

# 半径内的 Agent
nearby = env.agents_near(agent, radius=5.0)
nearby_users = env.agents_near(agent, radius=5.0, agent_type=User)

# 两个 Agent 之间的距离
dist = env.distance(agent_a, agent_b)   # float

pos = env.position(agent)  # (x, y) tuple 或 None
```

### NetworkEnvironment — 图网络（需要 networkx）

```python
from worldforge.environments import NetworkEnvironment

# 构造方式
env = NetworkEnvironment.scale_free(n=1000, m=3)
env = NetworkEnvironment.small_world(n=1000, k=4, p=0.1)
env = NetworkEnvironment.erdos_renyi(n=1000, p=0.01)
env = NetworkEnvironment.from_edgelist("edges.csv")

# 查询邻居（返回 Agent 对象列表，非 ID）
neighbors = env.neighbors(agent.id)

# 增删边
env.add_edge(agent_a.id, agent_b.id, weight=0.5)
env.remove_edge(agent_a.id, agent_b.id)

# 跳数内的 Agent
nearby = env.agents_within_hops(agent, hops=2)

# 节点度数
d = env.degree(agent.id)

# 直接访问 networkx Graph
g = env.graph
```

### MarketEnvironment — 限价订单簿

```python
from worldforge.environments import MarketEnvironment

env = MarketEnvironment(
    assets=["BTC", "ETH"],
    initial_prices={"BTC": 50000.0, "ETH": 3000.0},
    tick_size=0.01,
)

# 提交限价单（返回成交列表）
trades = env.submit_order(
    agent_id="trader_1",
    asset="BTC",
    side="buy",     # "buy" 或 "sell"
    price=49900.0,
    qty=0.1,
)
# trades: list[Trade]，每个 Trade 有 buyer_id, seller_id, price, qty

# 价格查询
mid = env.mid_price("BTC")        # 最优买卖均价
bid = env.best_bid("BTC")         # 最优买入价（None 若无挂单）
ask = env.best_ask("BTC")         # 最优卖出价（None 若无挂单）

# 注入外部价格（数字孪生用）
env.set_price("BTC", 51000.0)

# 成交历史
history = env.trade_history("BTC")            # 所有成交
recent = env.trade_history("BTC", last=100)   # 最近 100 笔
```

---

## 运行器（Runner）

### SequentialRunner（内置于 sim.run()）

通常不需要直接使用，`sim.run()` 内部调用它。

```python
from worldforge.runner import SequentialRunner

runner = SequentialRunner(sim)
result = runner.run(progress=True)
```

### ParallelRunner — 并行运行多个仿真

```python
from worldforge.runner import ParallelRunner

sims = [build_sim(seed=i) for i in range(20)]
runner = ParallelRunner(
    sims=sims,
    workers=4,         # 进程数（默认 4）
    use_threads=False, # True → 线程池（适合 I/O 密集），False → 进程池
)
results = runner.run()  # list[SimulationResult]，顺序与输入一致
```

### BatchRunner — Monte Carlo 参数扫描

```python
from worldforge.runner import BatchRunner
from worldforge.distributions import Uniform

batch = BatchRunner(
    sim_factory=lambda params: build_sim(**params),
    param_grid={
        "churn_rate":    [0.01, 0.05, 0.10],       # list → 枚举
        "initial_users": [1000, 5000],
        "viral_coeff":   Uniform(0.5, 2.0),         # 分布 → 随机采样 n_samples 次
    },
    n_samples=10,        # 分布型参数采样次数
    n_replications=3,    # 每组参数运行几次（不同 seed）
    workers=1,           # 1=顺序；>1=多进程
)

batch_result = batch.run()

# 转 pandas DataFrame（每行=一次运行）
df = batch_result.to_pandas()
# 列：param 名称 + replication + result.metadata 中的所有字段

# 参数敏感性分析（打印各参数对指标的影响）
batch_result.sensitivity_analysis("gmv_total")  # gmv_total 需要在 result.metadata 中

# 总运行次数
len(batch_result)  # = 3 churn_rates × 2 users × 10 samples × 3 reps = 180
```

---

## 内置场景（Scenarios）

每个场景函数返回**已配置好的 Simulation 对象**，调用 `.run()` 即可执行。

```python
from worldforge.scenarios import (
    ecommerce_world,          # 电商用户行为
    epidemic_world,           # SIR 流行病传播
    fintech_world,            # 银行/金融用户
    saas_world,               # SaaS 订阅生命周期
    market_microstructure_world,  # 限价订单簿
    iot_world,                # IoT 传感器时序
    supply_chain_world,       # 供应链库存
    social_network_world,     # 舆论动力学
)
```

### ecommerce_world

```python
sim = ecommerce_world(
    n_users=10_000,
    duration="90 days",   # 日历时间字符串
    seed=42,
)
result = sim.run()
# result 包含：event_log, daily_metrics, user_snapshot
```

### epidemic_world

```python
sim = epidemic_world(
    population=100_000,
    initial_infected=50,
    transmission_prob=0.3,
    recovery_rate=0.05,
    duration_days=180,
    seed=42,
)
result = sim.run()
# result 包含：sir_curve (S/I/R 每步), event_log
rows = result["sir_curve"]
print(f"Peak infected: {max(r['I'] for r in rows)}")
```

### saas_world

```python
sim = saas_world(n_users=5000, duration_days=365, seed=42)
result = sim.run()
# result 包含：monthly_metrics (mrr, n_users, churned), event_log
rows = result["monthly_metrics"]
print(f"Final MRR: {rows[-1]['mrr']:.2f}")
```

### iot_world

```python
sim = iot_world(n_sensors=500, duration_steps=1440, anomaly_rate=0.005, seed=42)
result = sim.run()
# result 包含：sensor_readings (sensor_id, sensor_type, value, is_anomaly), hourly_summary
n_anomalies = sum(1 for r in result["sensor_readings"] if r["is_anomaly"])
print(f"Anomalies detected: {n_anomalies}")
```

---

## CLI 使用

```bash
# 查看可用场景
worldforge list
# 输出：
# Available scenarios:
#   ecommerce                  E-commerce user behavior (purchases, churn)
#   epidemic                   SIR epidemic spreading
#   ...

# 运行场景（终端显示进度条 + 汇总）
worldforge run ecommerce --n-agents 1000 --steps 30 --seed 42

# 保存 JSON 结果
worldforge run epidemic --n-agents 5000 --steps 90 --output ./results/

# 查看环境信息（版本、已安装的可选依赖）
worldforge info
```

---

## 常见模式

### 可复现性验证

```python
r1 = Simulation(name="t", seed=42, clock=DiscreteClock(100)).run()
r2 = Simulation(name="t", seed=42, clock=DiscreteClock(100)).run()

# 两次结果必须完全相同
assert r1.to_dict() == r2.to_dict()
```

### 动态生成子 Agent（spawn）

```python
class Company(Agent):
    headcount: int = field(0)

    def step(self, ctx):
        if ctx.rng.random() < 0.01:
            n = int(ctx.rng.poisson(5))
            ctx.spawn(
                Employee,
                count=n,
                parent=self,
                init=lambda e: setattr(e, "company_id", self.id),
            )
            self.headcount += n
```

**预期行为：** 新 Agent 在 tick 末尾添加，不影响当前 tick 的迭代。

### 注入真实外部数据

```python
import pandas as pd
prices = pd.read_csv("btc_prices.csv", parse_dates=["date"])

@sim.global_rule(every=1)
def inject_real_price(ctx):
    row = prices[prices.date == ctx.now.date()]
    if not row.empty:
        ctx.environment.set_price("BTC", float(row.iloc[0]["close"]))
```

### 有序字段依赖（Lambda 依赖其他字段）

```python
class User(Agent):
    # 字段按声明顺序解析
    tier: str = field(Categorical(["free", "pro"], [0.7, 0.3]))
    # 此 lambda 在 tier 已解析后执行 —— 安全引用 a.tier
    max_spend: float = field(lambda a: 100.0 if a.tier == "free" else 1000.0)
    email: str = field(lambda a: f"user_{a.id}@example.com")
```

### SimContext 常用查询

```python
# 在 step(ctx)、global_rule、probe lambda 中使用：

ctx.now                          # 当前时间（int 或 datetime）
ctx.rng                          # np.random.Generator（已 seed）

ctx.agents(User)                 # list[User]
ctx.agents(User, filter=lambda u: u.balance > 1000)  # 带过滤
ctx.agent_count(User)            # int
ctx.agent_mean(User, "balance")  # float
ctx.agent_percentile(User, "balance", 0.95)  # float

ctx.emit(event)                  # 发射事件
ctx.remove_agent(agent)          # 标记删除（tick 末生效）
ctx.spawn(AgentType, count=3, init=lambda a: ...)  # 标记新增

ctx.event_count(PurchaseEvent)   # int
ctx.event_sum(PurchaseEvent, "amount")  # float
ctx.event_count(PurchaseEvent, last=7)  # 最近 7 步内的数量
```
