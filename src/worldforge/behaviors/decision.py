"""DecisionBehavior: rule-based decision making with priority actions."""
from __future__ import annotations

from typing import Any, Callable


class DecisionBehavior:
    """
    Evaluates a prioritized list of (condition, action) rules and executes
    the first matching action each step.

    Example::

        class UserDecision(DecisionBehavior):
            rules = [
                (lambda a, ctx: a.balance < 10,   lambda a, ctx: a.request_topup(ctx)),
                (lambda a, ctx: a.balance > 5000,  lambda a, ctx: a.invest(ctx)),
                (lambda a, ctx: True,              lambda a, ctx: a.browse(ctx)),
            ]

        class User(Agent):
            decision: UserDecision = field(UserDecision)

            def step(self, ctx):
                self.decision.step(ctx)
    """

    # Override in subclass: list of (condition, action) tuples (high → low priority)
    rules: list[tuple[Callable, Callable]] = []

    def __init__(self) -> None:
        self.agent: Any = None  # set by Agent.__init__ link-back

    def step(self, ctx: Any) -> None:
        """Evaluate rules in order and execute the first matching action."""
        for condition, action in self.rules:
            try:
                if condition(self.agent, ctx):
                    action(self.agent, ctx)
                    return
            except Exception:
                continue

    def add_rule(
        self,
        condition: Callable,
        action: Callable,
        priority: int | None = None,
    ) -> None:
        """Dynamically add a rule at the given priority index (None = append)."""
        rule = (condition, action)
        if priority is None:
            self.rules = list(self.rules) + [rule]
        else:
            rules = list(self.rules)
            rules.insert(priority, rule)
            self.rules = rules
