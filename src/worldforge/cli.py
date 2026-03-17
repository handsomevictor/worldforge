"""worldforge CLI — run scenarios and inspect simulation results."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run a built-in scenario and print a summary."""
    from worldforge.agent import _reset_id_counter
    _reset_id_counter(1)

    scenario_map = {
        "ecommerce":           _run_ecommerce,
        "epidemic":            _run_epidemic,
        "fintech":             _run_fintech,
        "saas":                _run_saas,
        "iot":                 _run_iot,
        "supply_chain":        _run_supply_chain,
        "social_network":      _run_social_network,
        "market_microstructure": _run_market,
        "rideshare":           _run_rideshare,
        "game_economy":        _run_game_economy,
        "org_dynamics":        _run_org_dynamics,
        "energy_grid":         _run_energy_grid,
    }

    name = args.scenario
    if name not in scenario_map:
        print(f"Unknown scenario: {name!r}. Available: {list(scenario_map)}", file=sys.stderr)
        sys.exit(1)

    print(f"Running scenario: {name}  (seed={args.seed})")
    result = scenario_map[name](args)
    print(result.summary())

    if args.output:
        result.to_json(args.output)
        print(f"Output written to: {args.output}")


def _run_ecommerce(args: argparse.Namespace):
    from worldforge.scenarios import ecommerce_world
    return ecommerce_world(
        n_users=args.n_agents,
        duration=f"{args.steps} days",
        seed=args.seed,
    ).run(progress=True)


def _run_epidemic(args: argparse.Namespace):
    from worldforge.scenarios import epidemic_world
    return epidemic_world(
        population=args.n_agents,
        duration_days=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_fintech(args: argparse.Namespace):
    from worldforge.scenarios import fintech_world
    return fintech_world(
        n_users=args.n_agents,
        duration_days=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_saas(args: argparse.Namespace):
    from worldforge.scenarios import saas_world
    return saas_world(
        n_users=args.n_agents,
        duration_days=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_iot(args: argparse.Namespace):
    from worldforge.scenarios import iot_world
    return iot_world(
        n_sensors=args.n_agents,
        duration_steps=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_supply_chain(args: argparse.Namespace):
    from worldforge.scenarios import supply_chain_world
    return supply_chain_world(
        n_retailers=args.n_agents,
        duration_days=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_social_network(args: argparse.Namespace):
    from worldforge.scenarios import social_network_world
    return social_network_world(
        n_users=args.n_agents,
        duration_steps=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_market(args: argparse.Namespace):
    from worldforge.scenarios import market_microstructure_world
    return market_microstructure_world(
        n_noise_traders=args.n_agents,
        duration_steps=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_rideshare(args: argparse.Namespace):
    from worldforge.scenarios import rideshare_world
    return rideshare_world(
        n_riders=args.n_agents,
        steps=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_game_economy(args: argparse.Namespace):
    from worldforge.scenarios import game_economy_world
    return game_economy_world(
        n_players=args.n_agents,
        steps=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_org_dynamics(args: argparse.Namespace):
    from worldforge.scenarios import org_dynamics_world
    return org_dynamics_world(
        n_employees=args.n_agents,
        steps=args.steps,
        seed=args.seed,
    ).run(progress=True)


def _run_energy_grid(args: argparse.Namespace):
    from worldforge.scenarios import energy_grid_world
    return energy_grid_world(
        n_consumers=args.n_agents,
        steps=args.steps,
        seed=args.seed,
    ).run(progress=True)


def cmd_list(args: argparse.Namespace) -> None:
    """List available built-in scenarios."""
    scenarios = [
        ("ecommerce",            "E-commerce user behavior (purchases, churn)"),
        ("epidemic",             "SIR epidemic spreading"),
        ("fintech",              "Banking/fintech user lifecycle"),
        ("saas",                 "SaaS product subscription dynamics"),
        ("iot",                  "IoT sensor time series with anomalies"),
        ("supply_chain",         "Retail inventory & order fulfillment"),
        ("social_network",       "Opinion dynamics on a social graph"),
        ("market_microstructure","Limit order book with market makers"),
        ("rideshare",            "Ride-sharing platform (drivers + riders, surge pricing)"),
        ("game_economy",         "Virtual game economy (players, items, inflation)"),
        ("org_dynamics",         "Employee org: hiring, attrition, promotion"),
        ("energy_grid",          "Power grid: generators, consumers, storage, shortages"),
    ]
    print("Available scenarios:")
    for name, desc in scenarios:
        print(f"  {name:<26} {desc}")


def cmd_info(args: argparse.Namespace) -> None:
    """Print worldforge version and environment info."""
    from worldforge import __version__
    import numpy as np
    print(f"worldforge {__version__}")
    print(f"Python     {sys.version.split()[0]}")
    print(f"numpy      {np.__version__}")
    for pkg in ("pandas", "polars", "networkx", "scipy"):
        try:
            mod = __import__(pkg)
            print(f"{pkg:<11}{mod.__version__}")
        except ImportError:
            print(f"{pkg:<11}(not installed)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def app() -> None:
    parser = argparse.ArgumentParser(
        prog="worldforge",
        description="worldforge — industrial-grade simulation framework",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # run
    run_p = sub.add_parser("run", help="Run a built-in scenario")
    run_p.add_argument("scenario", help="Scenario name (see 'worldforge list')")
    run_p.add_argument("--seed",     type=int, default=42,   help="Random seed (default: 42)")
    run_p.add_argument("--n-agents", type=int, default=100,  dest="n_agents",
                       help="Number of agents/users (default: 100)")
    run_p.add_argument("--steps",    type=int, default=30,   help="Number of steps (default: 30)")
    run_p.add_argument("--output",   type=str, default=None, help="Output directory for JSON results")

    # list
    sub.add_parser("list", help="List available scenarios")

    # info
    sub.add_parser("info", help="Print version and environment info")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    app()
