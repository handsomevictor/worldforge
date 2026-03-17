"""SequentialRunner: single-threaded simulation engine."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from worldforge.core.context import SimContext
from worldforge.output.result import SimulationResult

if TYPE_CHECKING:
    from worldforge.simulation import Simulation


class SequentialRunner:
    """
    Runs a Simulation synchronously in a single thread.

    The execution loop:
      for each tick while not clock.is_done:
        1. clock.tick()
        2. _run_tick() — call step() on all agents, flush deferred mutations
        3. fire any ExternalShocks scheduled at this time
        4. call global rules (at their configured intervals)
        5. call probes (at their configured intervals)
    """

    def __init__(self, sim: "Simulation") -> None:
        self._sim = sim

    def run(self, progress: bool = False) -> SimulationResult:
        sim = self._sim
        clock = sim.clock
        clock.reset()

        # Reset agent ID counter so each run with the same seed is fully reproducible
        from worldforge.agent import _reset_id_counter
        _reset_id_counter(1)

        rng = np.random.default_rng(sim.seed)
        ctx = SimContext(clock=clock, rng=rng)

        # Register event handlers
        for event_type, handler in sim._event_handlers:
            ctx.register_event_handler(event_type, handler)

        # Add initial agents
        for agent_type, count, factory in sim._agent_specs:
            for i in range(count):
                if factory is not None:
                    agent = factory(i, rng)
                else:
                    agent = agent_type(_rng=rng)
                ctx._register_agent(agent)
                agent.on_born(ctx)

        # Configure probes
        for probe in sim._probes:
            probe.configure(clock)

        # Build global-rule step intervals
        rule_intervals: list[tuple[Any, int]] = []
        for fn, every in sim._global_rules:
            from worldforge.probes.base import _resolve_every
            interval = _resolve_every(every, clock)
            rule_intervals.append((fn, interval))

        t0 = time.time()
        step = 0
        total_steps: int | None = _estimate_steps(clock)

        while not clock.is_done:
            clock.tick()
            step += 1

            # Run all agent steps
            ctx._run_tick()

            # Fire external shocks
            for shock in sim._shocks:
                shock_time = shock.at
                if _times_match(clock.now, shock_time):
                    shock.fire(ctx)

            # Invoke global rules
            for fn, interval in rule_intervals:
                if step % interval == 0:
                    fn(ctx)

            # Collect from probes
            for probe in sim._probes:
                probe.on_step(ctx, step)

            if progress and total_steps:
                _print_progress(step, total_steps)

        elapsed = time.time() - t0
        if progress:
            print()  # newline after progress bar

        # Finalize probes
        data: dict[str, list[dict]] = {}
        for probe in sim._probes:
            data[probe.name] = probe.finalize()

        metadata = {
            "name": sim.name,
            "seed": sim.seed,
            "steps": step,
            "elapsed_seconds": round(elapsed, 3),
            "agent_count_final": ctx.agent_count(),
            "events_total": len(ctx._event_log),
        }

        return SimulationResult(data=data, metadata=metadata)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_steps(clock: Any) -> int | None:
    """Estimate total steps for progress display."""
    from worldforge.time.calendar import CalendarClock
    from worldforge.core.clock import DiscreteClock
    if isinstance(clock, DiscreteClock):
        return clock._steps
    if isinstance(clock, CalendarClock):
        total_td = clock._end - clock._start
        return max(1, round(total_td / clock.step))
    return None


def _times_match(now: Any, shock_time: Any) -> bool:
    """Check if current time matches a shock's scheduled time."""
    if shock_time is None:
        return False
    # datetime comparison
    try:
        from datetime import datetime
        if isinstance(now, datetime) and isinstance(shock_time, (datetime, str)):
            if isinstance(shock_time, str):
                from datetime import date
                shock_dt = datetime.fromisoformat(shock_time)
            else:
                shock_dt = shock_time
            # Match at day granularity for calendar clocks
            return now.date() == shock_dt.date()
    except Exception:
        pass
    # Integer comparison
    return now == shock_time


def _print_progress(step: int, total: int) -> None:
    pct = step / total
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {step}/{total} ({pct:.0%})", end="", flush=True)
