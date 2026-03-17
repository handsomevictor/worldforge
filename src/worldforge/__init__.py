"""
worldforge — industrial-grade multi-paradigm simulation framework.

Quick start::

    from worldforge import Simulation, Agent, field
    from worldforge.distributions import Normal, Categorical
    from worldforge.time import DiscreteClock

    class User(Agent):
        balance: float = field(Normal(5000, 1000))
        tier: str = field(Categorical(["free", "pro"], [0.7, 0.3]))

        def step(self, ctx):
            self.balance *= 1.001  # 0.1% daily interest

    sim = Simulation(seed=42, clock=DiscreteClock(steps=365))
    sim.add_agents(User, count=1000)
    result = sim.run()
"""

from worldforge.agent import Agent, field
from worldforge.simulation import Simulation

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "field",
    "Simulation",
    "__version__",
]
