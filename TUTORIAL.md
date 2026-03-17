# TUTORIAL.md — worldforge 分步教程

---

## 教程 1：Hello Simulation（DiscreteClock）

最简单的仿真：agent 计步。

```python
from worldforge import Agent, Simulation, field
from worldforge.time import DiscreteClock

class Counter(Agent):
    count: int = field(0)

    def step(self, ctx):
        self.count += 1

sim = Simulation(
    name="hello",
    seed=42,
    clock=DiscreteClock(steps=10),
)
sim.add_agents(Counter, count=5)
result = sim.run()

# 每个 agent 的 count 应为 10
```

**核心概念：**
- `field(default)` 声明 agent 属性
- `step(ctx)` 每个 tick 对每个 agent 调用一次
- `DiscreteClock(steps=N)` 恰好运行 N 个 tick

---

## 教程 2：概率分布作为字段默认值

```python
from worldforge import Agent, Simulation, field
from worldforge.distributions import Normal, Categorical
from worldforge.time import DiscreteClock

class Customer(Agent):
    # 每个用户的余额从 Normal(5000, 1000) 独立采样
    balance: float = field(Normal(mu=5000, sigma=1000, clip=(0, None)))
    tier: str = field(Categorical(
        choices=["free", "pro", "enterprise"],
        weights=[0.70, 0.20, 0.10]
    ))

    def step(self, ctx):
        if self.tier == "pro":
            # 使用 ctx.rng，不要用 random.random()
            self.balance -= Normal(mu=50, sigma=10).sample(ctx.rng)

sim = Simulation(name="customers", seed=42, clock=DiscreteClock(steps=30))
sim.add_agents(Customer, count=1000)
result = sim.run()
```

**核心概念：**
- Distribution 实例作为 `field()` 初始化器——每个 agent 独立采样
- `ctx.rng` 是已 seed 的随机数生成器——**始终使用它，不要用 `random.random()`**
- `clip=(low, high)` 防止越界值

---

## 教程 3：日历时间与数据采集

```python
from worldforge import Agent, Simulation, field
from worldforge.distributions import Normal, Categorical
from worldforge.time import CalendarClock
from worldforge.probes import AggregatorProbe, SnapshotProbe
from worldforge.events import Event
from dataclasses import dataclass

@dataclass
class PurchaseEvent(Event):
    user_id: str
    amount: float

class User(Agent):
    balance: float = field(Normal(5000, 2000, clip=(0, None)))
    tier: str = field(Categorical(["free", "pro", "enterprise"], [0.7, 0.2, 0.1]))

    def step(self, ctx):
        buy_prob = {"free": 0.02, "pro": 0.08, "enterprise": 0.20}[self.tier]
        if ctx.rng.random() < buy_prob:
            amount = Normal(200, 50, clip=(1, self.balance)).sample(ctx.rng)
            self.balance -= amount
            ctx.emit(PurchaseEvent(user_id=self.id, amount=amount))

sim = Simulation(
    name="ecommerce_basic",
    seed=42,
    clock=CalendarClock(start="2024-01-01", end="2024-12-31", step="1 day"),
)
sim.add_agents(User, count=5_000)

# 每日聚合指标
sim.add_probe(AggregatorProbe(
    metrics={
        "dau":       lambda ctx: ctx.agent_count(User),
        "gmv_daily": lambda ctx: ctx.event_sum(PurchaseEvent, "amount", last="1 day"),
    },
    every="1 day",
))

# 每周对 10% 用户拍快照
sim.add_probe(SnapshotProbe(
    agent_type=User,
    fields=["id", "balance", "tier"],
    every="1 week",
    sample_rate=0.10,
))

result = sim.run(progress=True)
dfs = result.to_pandas()
print(dfs["aggregator"].head())
```

---

## 教程 4：事件与事件处理器

事件解耦 agent 之间的通信。一个 agent 发出，其他 agent 响应。

```python
@dataclass
class PriceUpdate(Event):
    seller_id: str
    price: float

class Seller(Agent):
    price: float = field(Normal(100, 5, clip=(1, None)))

    def step(self, ctx):
        self.price *= ctx.rng.normal(1.0, 0.01)
        ctx.emit(PriceUpdate(seller_id=self.id, price=self.price))

class Buyer(Agent):
    max_price: float = field(Normal(110, 10))

    def on_event(self, event, ctx):
        if isinstance(event, PriceUpdate) and event.price <= self.max_price:
            ctx.emit(OrderEvent(buyer_id=self.id, seller_id=event.seller_id))

# 全局事件处理器
@sim.on(OrderEvent)
def handle_order(event, ctx):
    ...
```

---

## 教程 5：状态机行为（订单生命周期）

```python
from worldforge.behaviors import StateMachineBehavior
from worldforge.distributions import Exponential, LogNormal
from worldforge.events import Event
from dataclasses import dataclass

@dataclass
class OrderStatusChanged(Event):
    order_id: str
    from_status: str
    to_status: str

class OrderFSM(StateMachineBehavior):
    states = ["pending", "paid", "processing", "shipped", "delivered", "cancelled"]
    initial = "pending"
    terminal = ["delivered", "cancelled"]

    transitions = {
        "pending": [
            (0.85, "paid",       Exponential(scale=300)),
            (0.15, "cancelled",  Exponential(scale=3600)),
        ],
        "paid": [
            (1.00, "processing", Exponential(scale=600)),
        ],
        "processing": [
            (0.98, "shipped",    Exponential(scale=86400)),
            (0.02, "cancelled",  Exponential(scale=3600)),
        ],
        "shipped": [
            (0.95, "delivered",  Exponential(scale=3*86400)),
            (0.05, "cancelled",  Exponential(scale=86400)),
        ],
    }

    def on_transition(self, from_state, to_state, ctx):
        ctx.emit(OrderStatusChanged(
            order_id=self.agent.id,
            from_status=from_state,
            to_status=to_state,
        ))

class Order(Agent):
    fsm: OrderFSM = field(OrderFSM)
    amount: float = field(LogNormal(mu=5, sigma=1))

    def step(self, ctx):
        self.fsm.step(ctx)
        if self.fsm.is_terminal:
            ctx.remove_agent(self)
```

---

## 教程 6：网络环境（流行病仿真）

```python
from worldforge import Agent, Simulation, field
from worldforge.environments import NetworkEnvironment
from worldforge.time import DiscreteClock

class Person(Agent):
    state: str = field("susceptible")
    infection_day: int = field(-1)

    def step(self, ctx):
        if self.state == "infected":
            for neighbor_id in ctx.environment.neighbors(self.id):
                neighbor = ctx.get_agent(neighbor_id)
                if neighbor.state == "susceptible" and ctx.rng.random() < 0.03:
                    neighbor.state = "infected"
                    neighbor.infection_day = ctx.now
            if ctx.now - self.infection_day > 14:
                self.state = "recovered"

sim = Simulation(name="epidemic", seed=42, clock=DiscreteClock(steps=180))
env = NetworkEnvironment.small_world(n=10_000, k=6, p=0.1)
sim.set_environment(env)
sim.add_agents(Person, count=10_000)
# 种入 10 个感染者
for i in range(10):
    sim.agents[i].state = "infected"
    sim.agents[i].infection_day = 0
result = sim.run(progress=True)
```

---

## 教程 7：Monte Carlo 批量参数扫描

```python
from worldforge.runner import BatchRunner
from worldforge.distributions import Uniform

def build_sim(churn_rate, initial_users, viral_coeff):
    sim = Simulation(name="saas_mc", seed=None, clock=DiscreteClock(steps=365))
    # ... 用参数配置仿真 ...
    return sim

batch = BatchRunner(
    sim_factory=lambda p: build_sim(**p),
    param_grid={
        "churn_rate":    [0.01, 0.05, 0.10],
        "initial_users": [1000, 5000, 10000],
        "viral_coeff":   Uniform(0.5, 2.0),
    },
    n_samples=50,
    n_replications=5,
    workers=8,
)

df = batch.run().to_pandas()
batch.run().sensitivity_analysis(target_metric="gmv_total")
```

---

## 教程 8：自定义 Probe

```python
@sim.probe(every="1 week")
def weekly_cohort(ctx, collector):
    for tier in ["free", "pro", "enterprise"]:
        users = ctx.agents(User, filter=lambda u: u.tier == tier)
        collector.record({
            "week":        ctx.now.isocalendar().week,
            "tier":        tier,
            "count":       len(users),
            "avg_balance": sum(u.balance for u in users) / max(len(users), 1),
        })
```

---

## 教程 9：外部冲击

```python
from worldforge.events import ExternalShock

sim.add_shock(ExternalShock(
    at="2024-06-15",
    description="竞品推出免费套餐",
    effect=lambda ctx: [
        setattr(u, "churn_risk", min(u.churn_risk * 3, 1.0))
        for u in ctx.agents(User)
    ],
))
```

---

## 教程 10：可复现性检查

```python
result_a = Simulation(name="test", seed=42, clock=DiscreteClock(100)).run()
result_b = Simulation(name="test", seed=42, clock=DiscreteClock(100)).run()

df_a = result_a.to_pandas()["aggregator"]
df_b = result_b.to_pandas()["aggregator"]
assert df_a.equals(df_b), "仿真不可复现！"
```

如果断言失败，检查：
- 是否用了 `random.random()` 而非 `ctx.rng`
- 是否对 set/dict 迭代顺序有依赖
- 并行代码中的浮点不确定性

---

## 常用模式

### 精细控制 agent 初始化

```python
sim.add_agents(
    User,
    count=1000,
    factory=lambda i, rng: User(
        tier="enterprise" if i < 50 else "pro",
        balance=rng.uniform(10_000, 100_000),
    )
)
```

### 动态生成子 agent

```python
class Company(Agent):
    def step(self, ctx):
        if self.is_hiring:
            ctx.spawn(
                Employee,
                count=Poisson(lam=5).sample(ctx.rng),
                parent=self,
                init=lambda e: setattr(e, "company_id", self.id),
            )
```

### 注入真实数据

```python
import pandas as pd
prices = pd.read_csv("btc_prices.csv", parse_dates=["date"])

@sim.on_step
def inject_prices(ctx):
    row = prices[prices.date == ctx.now.date()]
    if not row.empty:
        ctx.environment.set_price("BTC", row.iloc[0]["close"])
```
