# worldforge 设计笔记

> 写给自己看的技术备忘录。当有人问"你这个项目是怎么解决 XXX 问题的"时，这里有完整的思路和答案。
> 面向人群：Python 初中级开发者，有一定面向对象基础，没有仿真框架背景。

---

## 一、这个项目是做什么的？为什么要做它？

worldforge 是一个**Python 仿真框架**。它能让你用普通 Python 代码描述一个系统里有哪些"角色"（Agent）、它们的行为规则是什么，然后让框架自动运行仿真、收集数据、输出结构化结果。

**一句话解释：** 就像 SimCity 的后端——你定义城市里有哪些居民、商店、道路，框架负责让时间流逝并记录发生了什么。

### 为什么现有工具不够用？

| 现有工具 | 能做什么 | 缺什么 |
|---------|---------|-------|
| SimPy | 离散事件仿真（医院排队、工厂流水线） | 没有 Agent 状态管理，没有数据导出 |
| Mesa | Agent-Based 建模（鸟群、市场） | 没有事件系统，数据导出很弱 |
| Faker | 生成假数据（姓名、邮件、地址） | 数据之间没有因果关系，不随时间演化 |
| AnyLogic | 商业级多范式仿真 | Java/GUI，不能用 Python，不能 pip install |

worldforge 填补了这个空白：纯 Python，三种范式（Agent-Based、事件驱动、时间步长）在一个框架里，数据导出是一等公民。

---

## 二、整体架构是怎么设计的？

### 2.1 为什么选择这个分层结构？

```
用户代码
    ↓ 声明 Agent、规则、分布
Simulation（编排器）
    ↓ 委托给
SequentialRunner（执行引擎）
    ↓ 驱动
Clock（时间推进） + SimContext（运行时状态）
    ↓ 触发
Agent.step() → 产生 Events → 触发 Probes → 收集数据
    ↓
SimulationResult（结构化输出）
```

**为什么这样分层？**

- `Simulation` 类只负责"配置"——添加 agent、添加 probe、注册规则。它不知道如何运行。
- `SequentialRunner` 负责"执行"——它知道时钟怎么推进、agent 的 step() 怎么调用、probe 什么时候触发。
- 这样做的好处：未来可以换一个 `ParallelRunner` 或 `BatchRunner`，`Simulation` 代码完全不变。这是经典的**策略模式（Strategy Pattern）**。

**曾经考虑过的替代方案：**
- `Simulation` 直接包含 run() 逻辑 → 问题：并行、批量等变体就要继承 `Simulation`，继承层次会很深。
- 纯函数式设计（每步传入状态，返回新状态）→ 问题：Python 里写起来很别扭，Agent 对象的可变状态更自然。

### 2.2 为什么用 `SimContext` 作为"传话筒"？

在 Agent 的 `step(ctx)` 里，`ctx` 是连接一切的纽带。Agent 通过它：
- 查询当前时间（`ctx.now`）
- 发布事件（`ctx.emit(event)`）
- 生成随机数（`ctx.rng`）
- 查询其他 Agent（`ctx.agents(SomeType)`）
- 访问环境（`ctx.environment`）

**为什么不直接让 Agent 持有引用到 Simulation？**

因为 Simulation 在 run() 阶段之后就不该被 Agent 改变。`SimContext` 是一个精心设计的"切面"——它只暴露 Agent 在一个 tick 里合理能做的事，不暴露"重置仿真""修改参数"等不应该在 step() 里做的操作。

---

## 三、Agent 字段系统是怎么实现的？

### 3.1 问题：如何让 `field(Normal(5000, 1000))` 工作？

用户写：
```python
class User(Agent):
    balance: float = field(Normal(5000, 1000))
```

创建 1000 个 User 时，每个 User 的 balance 都从 Normal 分布独立采样，而不是共享同一个 5000。

**核心难点：** 类属性（`field(Normal(5000, 1000))`）是所有实例共享的，但我们需要每个实例独立采样。

**实现方案：**

1. `field()` 返回一个 `FieldSpec` 对象，存储"如何初始化"的描述（不是值本身）
2. `AgentMeta`（自定义 metaclass）在创建 Agent 子类时，扫描所有类属性，收集所有 `FieldSpec`
3. `Agent.__init__()` 遍历这些 `FieldSpec`，对每个字段调用 `spec.resolve(agent, rng)` 来采样真实值，然后用 `object.__setattr__(self, name, value)` 设置到实例上

**字段解析顺序：** 按照类定义中的声明顺序（包含父类字段），这样 `lambda` 字段可以安全地引用同一 Agent 的其他字段。

**曾经考虑过的替代方案：**
- Python descriptor 协议（`__get__`/`__set__`）→ 问题：每次访问都要重新采样，无法固定值
- dataclass + `field(default_factory=...)`  → 问题：无法传 `Distribution` 对象，也不支持 `lambda a: ...` 依赖其他字段

### 3.2 Lambda 字段的执行顺序问题

```python
class User(Agent):
    tier: str = field(Categorical(["free", "pro"], [0.7, 0.3]))
    bonus: float = field(lambda a: 500.0 if a.tier == "pro" else 0.0)
```

`bonus` 依赖 `tier`，所以必须在 `tier` 解析完之后才能计算 `bonus`。

**解决方案：** 两遍解析。第一遍解析所有非 lambda 字段（常量、分布），第二遍解析 lambda 字段。这确保了 lambda 字段执行时，其他字段已经有了具体值。

---

## 四、事件系统是怎么设计的？

### 4.1 发布-订阅模型

```
ctx.emit(PurchaseEvent(...))
    → 设置 event.timestamp = ctx.now
    → 追加到 ctx._event_log
    → 遍历已注册的全局 handlers，同步调用
    → 广播给所有"重写了 on_event() 的 Agent"
```

**为什么同步调用而不是异步？**

在仿真中，因果顺序必须确定。如果 A 发出事件，B 在同一 tick 处理，这要显式控制。异步会引入不确定性。

### 4.2 大规模下的 O(1) 广播

**问题：** 100 万个 Agent，只有 1000 个重写了 `on_event()`。每次 emit 遍历 100 万个 Agent 太慢。

**解决方案：**
```python
if type(agent).on_event is not Agent.on_event:
    agent.on_event(event, ctx)
```

Python 中，如果子类没有重写某个方法，`type(subclass_instance).method is BaseClass.method` 为 `True`。这是一个 O(1) 的检查，不需要额外的数据结构（如集合或标记位）。

**效果：** 即使有 100 万个 Agent，emit 的开销只与"重写了 on_event 的 Agent 数量"成比例，而不是总 Agent 数。

### 4.3 为什么选 dataclass 作为事件基类？

```python
@dataclass
class PurchaseEvent(Event):
    user_id: str
    amount: float
```

- 自动生成 `__init__`、`__repr__`、`__eq__`
- 字段类型注解即文档
- `timestamp` 字段在基类 `Event` 中定义，`emit()` 自动设置

**曾经考虑过的替代方案：**
- 纯 dict → 问题：没有类型安全，`isinstance(event, PurchaseEvent)` 无法使用
- Pydantic model → 问题：引入额外依赖，且验证开销在高频事件场景中明显

---

## 五、时间系统是怎么设计的？

### 5.1 三种时钟，一个接口

```python
class Clock(ABC):
    def tick(self) -> None: ...
    @property
    def now(self) -> Any: ...   # int 或 datetime
    @property
    def is_done(self) -> bool: ...
    def reset(self) -> None: ...
```

三种实现：

| 时钟 | `now` 类型 | 适用场景 |
|-----|-----------|---------|
| `DiscreteClock(steps=N)` | `int` (0→N) | 快速仿真，单位"步" |
| `CalendarClock(start, end, step)` | `datetime` | 日历时间，商业/金融场景 |
| `EventDrivenClock(max_time)` | `float` | 稀疏事件，时间可跳跃 |

**为什么不直接用 `datetime`？**

`DiscreteClock` 的整数步更快（整数比较 vs datetime 比较），且对于"运行 1000 步"这种需求，日历时间反而增加了不必要的概念。两种时钟使用同一个接口，用户可以自由切换。

### 5.2 CalendarClock 的 `parse_duration()`

用户写 `step="1 day"` 而不是 `step=timedelta(days=1)`，更直观。

```python
parse_duration("1 day")    → timedelta(days=1)
parse_duration("2 hours")  → timedelta(hours=2)
parse_duration("30 minutes") → timedelta(minutes=30)
```

实现很简单：`parts = dur.split()` 取数字和单位，查表映射到 `timedelta`。

### 5.3 Realtime 模式（数字孪生场景）

```python
clock = CalendarClock(start="2024-01-01", end="2024-01-02",
                      step="1 minute", realtime=True, realtime_factor=1.0)
```

每 tick 调用 `time.sleep(realtime_factor)` 来同步真实时钟。用于实时监控或数字孪生场景。

**为什么 `realtime_factor=0.0` 可以用来测试？**

不用真的 sleep 就能测试代码路径，只需把 factor 设为 0。这是个设计上的小细节，让测试不会因为 sleep 变慢。

---

## 六、可复现性是怎么保证的？

这是一个经常被问到的问题，因为仿真天然涉及随机数。

### 保证机制

**1. 固定 seed 创建 RNG：**
```python
rng = np.random.default_rng(seed)  # PCG64 算法，高质量伪随机
```

**2. 每次 run() 重置 Agent ID 计数器：**
```python
_reset_id_counter(1)  # 第一行，在创建任何 agent 之前
```

这是 BUG-03 的教训：如果不重置，第一次 run() 产生 ID 1-1000，第二次 run() 产生 ID 1001-2000，即使 seed 相同，结果也不同。

**3. Agent step() 调用顺序固定：**

Agent 存在 `dict` 中，Python 3.7+ 保证 dict 的迭代顺序等于插入顺序。Agent 按创建顺序 step()，结果确定。

**4. 延迟变更（deferred mutations）：**

所有 agent 的 step() 完成后，才统一处理新增/删除。如果 step() 中即时修改，那么 step() 顺序会影响结果（A step() 删除了 B，B 就不会 step()）。延迟处理确保每个 tick 的行为与 step() 顺序无关。

---

## 七、概率分布系统是怎么设计的？

### 7.1 统一接口的好处

所有 17 种分布实现同一个 `Distribution` 抽象基类：

```python
dist.sample(rng)              # 单次采样
dist.sample_batch(n, rng)     # 批量采样 → numpy array（快！）
dist.mean() / dist.std()      # 理论统计量
dist.pdf(x) / dist.cdf(x)    # 概率密度/累积分布
dist.ppf(q)                   # 百分位函数（逆 CDF）
```

好处：用户可以随时换分布，代码不用改。`field(Normal(...))` 和 `field(Exponential(...))` 调用方式完全一样。

### 7.2 为什么不直接用 scipy？

- 核心依赖只有 numpy（确保 `pip install worldforge` 快速且无痛）
- scipy 是软依赖（`pip install "worldforge[science]"`），安装后自动使用更快的 PPF 实现
- 这个选择让框架在没有 scipy 的环境（如容器、CI）里也能运行

### 7.3 高级组合分布

**MixtureDistribution（混合分布）：**
```python
spending = MixtureDistribution(
    components=[Normal(50, 10), Normal(500, 100)],
    weights=[0.8, 0.2]
)
```
每次采样先按权重选一个分量，再从该分量采样。用于"大部分用户小额消费，少数用户大额消费"这类双峰场景。

**CorrelatedDistributions（Gaussian Copula）：**
```python
price, qty = CorrelatedDistributions(
    distributions=[LogNormal(4, 0.5), Poisson(10)],
    correlation=-0.7   # 价格越高，购买量越少
).sample(rng)
```

实现原理：
1. 从标准正态分布 N(0,1) 采样两个相关的变量 (u, v)，相关系数由 Cholesky 分解控制
2. 用各自分布的 PPF（逆 CDF）把 N(0,1) 的样本映射到目标分布

这是 Gaussian Copula 的核心思想：在 normal space 里控制相关，再映射到任意边际分布。

---

## 八、数据采集（Probe 系统）是怎么设计的？

### 8.1 为什么需要 Probe？

仿真运行时，Agent 的状态在不断变化。如果用户想知道"每天有多少 DAU"，他不能等仿真跑完再回头看——那时 Agent 可能已经被删除了。

Probe 是一个"观察者"，在仿真运行过程中定期采集数据，存到内存里，仿真结束后汇总成结构化数据表。

### 8.2 触发频率的设计

```python
AggregatorProbe(..., every="1 day")   # CalendarClock
AggregatorProbe(..., every=7)          # DiscreteClock：每7步
```

`_resolve_every(every, clock)` 把各种形式统一转换成整数"步数间隔"，在 `probe.on_step(ctx, step)` 里用 `step % interval == 0` 判断是否触发。

### 8.3 EventLogProbe 的 `event_type` 字段

每条 EventLog 记录自动加入 `event_type` 字段（事件类名）：
```python
{"event_type": "PurchaseEvent", "user_id": "42", "amount": 150.0, "timestamp": "2024-01-15"}
```

**为什么加这个？** 用户的 event_log 里可能有多种事件混在一起。如果没有类型标记，DataFrame 里无法直接 `df[df["event_type"] == "PurchaseEvent"]` 来过滤，需要导入原始类做 isinstance 判断，对分析很不方便。

### 8.4 事件窗口过滤（last= 参数）

```python
# 统计过去 1 天的 GMV（不是累计 GMV）
ctx.event_sum(PurchaseEvent, "amount", last="1 day")

# 统计过去 7 步的事件数
ctx.event_count(ChurnEvent, last=7)

# 也支持 timedelta
from datetime import timedelta
ctx.event_sum(PurchaseEvent, "amount", last=timedelta(hours=6))
```

**这个功能是修复了一个经典 bug 后加的。** 最初的代码里 `gmv_daily` 没有 `last=` 参数，导致它实际上是从仿真开始到现在的累计 GMV，每天单调递增。命名叫"daily"但语义是"cumulative"——这是一个很容易犯的错误。现在的实现强制要求用户在设计周期指标时显式指定窗口。

---

## 九、数据逻辑一致性是怎么保证的？

这是一个很重要的问题，因为仿真产生的数据如果在逻辑上自相矛盾，后续分析就没有意义。

### 9.1 发现了哪些逻辑问题？

做了一次全面的逻辑审计，发现 16 个 bug，主要分三类：

**类型一：指标语义错误（最常见）**
- `gmv_daily` 实际是累计值 → 加 `last="1 day"` 窗口
- `n_anomalies` 统计了所有传感器读数 → 过滤 `is_anomaly == True`
- 月度指标全是累计 → 加 `last=30` 窗口

**类型二：并发修改竞态（仿真特有）**
- epidemic 场景：同一个 person 在同一 tick 被多个 infected agent 接触，被标记为感染多次
- 修复：先用 dict 收集所有"新感染候选"（key=person_id 自动去重），遍历结束后统一提交

**类型三：模块级全局状态污染**
- epidemic 场景：`_recovery_rate` 全局变量，多次调用 `epidemic_world()` 时相互覆盖
- game_economy 场景：`_market_prices` 全局 dict，第二次运行会继承第一次运行结束时的物价
- 修复：epidemic 改为 per-agent field；game_economy 在每次工厂函数调用时 reset 为默认值

### 9.2 result.validate() 运行时检查

```python
report = result.validate()
# 检查: NaN/Inf 数值字段, timestamp 倒退, 负数 agent 数
```

实现在 `output/result.py` 的 `_validate_result()` 函数中。遍历每个 probe 的每条记录，检查：
- `isinstance(val, float) and math.isnan(val)` → ERROR
- `isinstance(val, float) and math.isinf(val)` → ERROR
- `ts < prev_ts` → ERROR（时间戳倒退）
- 计数字段 < 0 → ERROR

**为什么是事后检查而不是运行时断言？**

仿真可能运行很久（数百万 step），在每一步强制校验会极大降低性能。事后检查是在仿真结束后统一扫描结果数据，对性能无影响，且能给用户一次性报告所有问题。

### 9.3 能量守恒（energy_grid 场景）

```
充电：grid → battery
  stored = input × efficiency   (e.g. 100 MW → 92 MWh stored)

放电：battery → grid
  需要从 battery 取出 = demand / efficiency  (e.g. 需要 92 MW → 取出 100 MWh)
  实际输出到 grid = 取出量 × efficiency
```

两侧都应用效率，实现能量守恒。初始版本只在充电侧应用效率（放电侧无损耗），导致每个充放电循环都凭空"增加"了能量。

---

## 十、性能优化是怎么做的？

### 10.1 基准测试结果

```
1,000 agents × 1,000 steps = 65ms（目标 < 1 秒，超额完成 15 倍）
```

### 10.2 具体的优化手段

**1. Agent 存储用 dict（O(1) 查找/删除）：**

```python
self._agents: dict[str, Agent] = {}   # id → agent
```

按类型分组：
```python
self._by_type: dict[type, list[Agent]] = {}
```

`ctx.agents(SomeType)` 直接返回已分组的列表，不用遍历所有 agent。

**2. `on_event` 广播跳过未重写的 Agent：**

```python
if type(agent).on_event is not Agent.on_event:
    agent.on_event(event, ctx)
```

Python 方法查找会检查 MRO（Method Resolution Order）。如果子类没有重写 `on_event`，`type(inst).on_event` 就是 `Agent.on_event`，用 `is` 比较是 O(1)。

**3. 延迟变更批量处理：**

```python
# tick 期间
self._pending_removals.append(agent)   # 不立即删除

# tick 结束后
for agent in self._pending_removals:
    self._unregister_agent(agent)
```

避免在遍历 `_agents` 时修改它（会引发 RuntimeError），同时减少 GC 压力（批量操作比逐个操作效率高）。

**4. 批量采样（`sample_batch`）：**

```python
dist.sample_batch(n=1000, rng=rng)  # → numpy array
# 比 [dist.sample(rng) for _ in range(1000)] 快约 10 倍
```

numpy 的向量化操作在底层调用 C 代码，避免 Python 解释器循环开销。

### 10.3 为什么没有用多进程？

单线程 SequentialRunner 已经能在 65ms 内处理 1k×1k。对于更大规模（100k agents）可以用 `ParallelRunner` 或 `BatchRunner`，但默认不用多进程，因为：
- 多进程有启动开销（fork/pickle），小规模反而更慢
- 避免 GIL 的方式是多进程，但 Agent 状态共享会很复杂
- 对大多数用户来说，单线程够用

---

## 十一、12 个内置场景的设计思路

每个场景都是一个工厂函数，返回一个配置好的 `Simulation` 对象，用户可以直接 `.run()` 或继续添加 Agent/Probe。

### 11.1 场景选择标准

选的都是有"涌现现象"的系统——系统级行为无法从单个 Agent 的规则直接预测：
- 电商：个体购买行为 → DAU、GMV、留存率
- 流行病：个体传播 → 群体 S/I/R 曲线
- 网约车：司机/乘客供需 → 动态定价（surge）
- 游戏经济：玩家交易 → 物价通胀/通缩

### 11.2 各场景的核心设计决策

**epidemic_world（SIR 模型）：**

最初把 `recovery_rate` 放在模块级全局变量，被 BatchRunner 调用时发生参数污染（BUG-05）。最终改为 per-agent field，用工厂函数注入：
```python
sim.add_agents(Person, count=N,
    factory=lambda i, rng: Person(recovery_rate=_recovery_rate, _rng=rng))
```

**rideshare_world（动态定价）：**

Surge 定价用供需比实现：
```python
ratio = waiting_riders / idle_drivers
if ratio > 2.0: surge = 2.5
elif ratio > 1.0: surge = 1.5
else: surge = 1.0
```

这比直接写公式更符合现实中 Uber/滴滴的定价逻辑（离散档位，不是连续函数）。

**energy_grid_world（能量守恒）：**

电池充放电必须两侧都施加效率：
- 充电：存入 = 输入 × efficiency
- 放电：需要取出 = 需求 / efficiency，实际输出 = 取出 × efficiency

初始版只有充电侧有效率，放电无损耗，导致每次充放循环都凭空增加能量。

**game_economy_world（物价动态）：**

物价根据最近 200 次交易的需求弹性调整：
- 需求量 > 30 次：涨价 5%
- 需求量 < 5 次：降价 3%

这模拟了真实市场的价格发现机制。初始版本 `_market_prices` 是模块级全局 dict，第二次调用时会继承上次仿真结束时的物价（BUG-03）。修复：每次工厂函数调用时 reset 为默认值。

---

## 十二、GymWrapper（强化学习接口）是怎么设计的？

### 12.1 为什么要支持 RL？

worldforge 的仿真是一个天然的 RL 环境：
- 状态（observation）：仿真某一时刻的统计量（感染率、DAU、库存水平...）
- 奖励（reward）：某个业务指标（康复人数、GMV、利润...）
- 动作（action）：在每步修改仿真参数（隔离政策、定价策略、采购量...）

### 12.2 接口设计

```python
env = GymWrapper(
    sim=sim,
    observation=lambda ctx: np.array([...]),  # 用户定义
    reward=lambda ctx: float(...),             # 用户定义
    action_fn=lambda action, ctx: ...,         # 用户定义
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

**为什么用 lambda 而不是子类重写？**

lambda 更轻量，不需要用户创建一个新的 class。对于探索性实验，3 行 lambda 比定义一个新类更快。进阶用户可以传入任意 callable（包括类实例的方法）。

### 12.3 gym 作为软依赖

```python
try:
    gym = __import__("gymnasium")
except ImportError:
    return spec  # 没有装 gymnasium 也能用，只是 action_space 是 placeholder
```

这样没有安装 `gymnasium` 的用户不会报 ImportError，只是得到一个占位的 action_space。

---

## 十三、CLI 是怎么设计的？

```
worldforge run ecommerce --n-agents 1000 --steps 30 --seed 42
worldforge list
worldforge info
```

用 `argparse` 实现，每个场景注册一个 runner 函数。

**为什么不用 Click？**

argparse 是标准库，不引入额外依赖。对于这个规模的 CLI，argparse 完全够用。如果未来命令变复杂，可以换 typer/click。

---

## 十四、测试策略

### 14.1 测试分层

```
tests/unit/         # 每个模块独立，mock 外部依赖
tests/integration/  # 端到端仿真，验证模块协作
tests/benchmarks/   # 性能基准，防止性能退化
```

### 14.2 关键测试原则

**可复现性测试：**
```python
def test_reproducibility():
    r1 = sim.run()
    r2 = sim.run()
    assert r1["event_log"] == r2["event_log"]
```
相同 seed，两次结果完全一致。这是最重要的性质之一。

**数据完整性回归测试（test_data_integrity.py）：**

每个修复的 bug 都有对应的回归测试，确保不会复现：
- `test_each_person_infected_at_most_once_per_run` → BUG-04
- `test_two_sims_independent` → BUG-05
- `test_second_run_starts_with_default_prices` → BUG-06
- 等 20 个...

**为什么回归测试很重要？**

bug 修复后如果没有测试覆盖，未来的代码改动很可能把 bug 重新引入。回归测试是防止"倒退"的安全网。

### 14.3 可选依赖的处理

```python
def test_to_pandas():
    pd = pytest.importorskip("pandas")  # 没有 pandas 就 skip，不 fail
    ...
```

`pytest.importorskip` 确保测试在没有安装可选依赖时被跳过而不是失败，CI 可以在精简环境里运行。

---

## 十五、常见问题 Q&A

**Q: 如何在多个场景之间复用 Agent 类型？**

A: 直接在多个场景的工厂函数里都引用同一个 Agent 类。Agent 类本身不包含仿真逻辑，只包含状态和 step() 行为，可以自由复用。

**Q: 如果 Agent 在 step() 里删除自己，会不会出问题？**

A: 不会。`ctx.remove_agent(self)` 是延迟操作，实际删除在当前 tick 所有 agent 的 step() 执行完毕后才发生。所以 agent 在 step() 里删除自己后，剩余的 step() 逻辑仍然正常执行。

**Q: CalendarClock 和 DiscreteClock 可以混用吗？**

A: 不能在同一个仿真里同时用两个时钟（一个仿真只有一个 clock）。但你可以把两个仿真用同一组 Agent 类型，只是一个用 CalendarClock，另一个用 DiscreteClock。

**Q: BatchRunner 的参数网格支持随机参数吗？**

A: 支持。在 `param_grid` 里可以直接传 Distribution 实例：
```python
BatchRunner(
    param_grid={"alpha": Uniform(0.0, 1.0)},  # 随机采样
    n_samples=100,   # 采样 100 个 alpha 值
)
```
列表参数（`[0.01, 0.05, 0.10]`）做笛卡尔积，Distribution 参数做随机采样，两种方式可以混用。

**Q: 仿真结果能直接用于机器学习训练吗？**

A: 可以。`result.to_pandas()` 返回 pandas DataFrame，`result.to_parquet()` 输出 Parquet 文件，两者都可以直接喂给 sklearn、PyTorch 等框架。EventLogProbe 产生的事件序列可以用于时序模型训练。

**Q: 如何调试 "为什么这个 Agent 没有被删除"？**

A: 检查 `ctx.remove_agent(self)` 是否在 step() 里被调用，以及调用条件是否满足。也可以用 `result.validate()` 检查数据是否有异常，或在 global_rule 里打印调试信息。注意：`remove_agent` 是延迟的，在当前 tick 内 agent 仍然存在。

---

*最后更新：2026-03-17*
