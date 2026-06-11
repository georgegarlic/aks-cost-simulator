#!/usr/bin/env python3
"""
Azure infrastructure cost simulator for virtual assistants.
Compares AKS (Ideal UX / Economy UX) vs Azure OpenAI API.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import azure_pricing
import loader
import output as out
import ui
from config import APP_NAME, VERSION, APP_DATE, AUTHOR
from simulator import simulate_all


HELP_EPILOG = """
 =======================================================================
                           DESCRIPTION
=======================================================================

This simulator estimates monthly Azure infrastructure costs for
LLM-powered virtual assistants. It compares three scenarios:

  1) AKS Ideal UX     -> NVIDIA A100 GPU (max throughput, lowest latency)
  2) AKS Economy UX   -> NVIDIA A10 GPU  (best cost/performance ratio)
  3) Azure OpenAI API -> Pay-as-you-go, no infrastructure

Unit prices (VM/h, storage, network, API tokens) are obtained from
two sources in this order:
  1) Azure Retail Prices API (real-time HTTPS query)
  2) Built-in defaults (fallback if the API does not respond)

The simulator ALWAYS queries the official Azure API for VM, AKS,
storage, load balancer, monitor, and container registry prices. Azure
OpenAI API prices (per token) are read from built-in defaults.

The model sizes GPU nodes across three time bands (office hours base,
peak demand, off-hours) and applies a high-availability multiplier
to compute costs.

It also includes Monte Carlo simulation varying traffic, throughput,
prices, and peak duration to obtain P50/P90/P95 percentiles.

=======================================================================
                              INPUT
=======================================================================

Parameters are built-in with sensible defaults for virtual assistants. An optional Excel
file may override built-in defaults if provided with --excel.

=======================================================================
                              OUTPUT
=======================================================================

  * resultados_simulacion.xlsx  -> Excel with 3 sheets (Results,
                                   Original Comparison, Summary)
  * resultados_simulacion.csv   -> CSV with UTF-8 BOM encoding
  * comparativa_costes.png      -> Chart with breakdown, MC distribution
                                   and total comparison (optional, requires
                                   matplotlib)

Result columns: scenario, nodes (system/gpu base/peak/off-hours),
cost breakdown (system, gpu, storage, lb, monitor, acr),
total cost, cost per user, cost per conversation, and Monte Carlo
percentiles (P50, P90, P95, mean, std dev).

=======================================================================
                          USAGE EXAMPLES
=======================================================================

  python main.py                                -> full run
  python main.py --excel "my_file.xlsx"         -> custom Excel
  python main.py --mc-iterations 50000          -> more precise MC
  python main.py --ha-factor 1.0 --overhead 0   -> no margins
  python main.py --no-chart                     -> skip chart
  python main.py -v                             -> verbose logging
  python main.py --version                      -> show version
  python main.py --apiazure                     -> show Azure API prices

=======================================================================
"""


def print_azure_sources() -> None:
    ui.info("Querying Azure Retail Prices API...")
    prices = azure_pricing.fetch_azure_prices(eur_usd_rate=0.92)
    azure_pricing.show_azure_prices(prices)
    ui.info("Source: https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices")
    ui.info("Region: West Europe | Currency: USD (converted to EUR)")
    ui.info("Type: Pay-as-you-go (Consumption)" +
            " | Prices: Linux (lowest available)")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=f"{APP_NAME} v{VERSION} — {APP_DATE}",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--excel",
        default="",
        help="Optional Excel file to override built-in parameters",
    )
    p.add_argument(
        "--output-xlsx",
        default="resultados_simulacion.xlsx",
        help="Output path for results Excel (default: %(default)s)",
    )
    p.add_argument(
        "--output-csv",
        default="resultados_simulacion.csv",
        help="Output path for results CSV (default: %(default)s)",
    )
    p.add_argument(
        "--chart",
        default="comparativa_costes.png",
        help="Output path for chart PNG (default: %(default)s)",
    )
    p.add_argument(
        "--mc-iterations",
        type=int,
        default=10_000,
        help="Monte Carlo iterations (default: %(default)s)",
    )
    p.add_argument(
        "--ha-factor",
        type=float,
        default=1.15,
        help="High availability factor on GPU cost (default: %(default)s)",
    )
    p.add_argument(
        "--overhead",
        type=float,
        default=0.1,
        help="Overhead factor on peak hours (default: %(default)s)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed DEBUG logging",
    )
    p.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip chart generation",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="Show simulator version and exit",
    )
    p.add_argument(
        "--apiazure",
        action="store_true",
        help="Fetch and display current Azure Retail Prices API prices",
    )
    return p.parse_args(argv)


def print_version() -> None:
    print(f"{APP_NAME} v{VERSION} ({APP_DATE})")
    print(f"Author: {AUTHOR}")
    print("Python 3.14+ | pandas, numpy, openpyxl, matplotlib")
    print("License: Internal use")


_last_mc = ""


def _progress_handler(step_type: str, *args):
    global _last_mc
    if step_type == "scenario":
        _last_mc = ""
        idx, total, name = args
        desc = name[:22] + (name[22:] and "..")
        print(f"  {ui.CYAN}{ui.ARROW}{ui.RESET} {desc}  ", end="", flush=True)
    elif step_type == "mc":
        current, total = args
        pct = current / total
        width = 25
        filled = int(width * pct)
        bar = ui.BAR_FULL * filled + ui.BAR_EMPTY * (width - filled)
        msg = f"\r    {bar} {current:>{len(str(total))}}/{total} {pct*100:4.0f}%"
        sys.stdout.write(msg)
        sys.stdout.flush()
        if current == total:
            sys.stdout.write(f"\r{' ' * 60}\r")
            sys.stdout.flush()
            print(f"  {ui.GREEN}{ui.CHECK}{ui.RESET}")


SHORT_NAMES = {
    "AKS UX Ideal": "AKS Ideal",
    "AKS UX Economico": "AKS Economico",
    "API Azure OpenAI": "API OpenAI",
}


def main() -> int:
    args = parse_args(sys.argv[1:])

    if args.version:
        print_version()
        return 0

    if args.apiazure:
        print_azure_sources()
        return 0

    setup_logging(args.verbose)

    if args.verbose:
        logger = logging.getLogger("main")

    excel_path = Path(args.excel) if args.excel else None
    if excel_path and excel_path.exists() and excel_path.is_file():
        try:
            data = loader.extract_all(str(excel_path))
        except Exception as e:
            logger.error("Error reading Excel: %s. Using built-in defaults.", e)
            data = loader.build_default_data()
    else:
        if args.verbose:
            logger.info("No Excel provided. Using built-in defaults.")
        data = loader.build_default_data()

    ui.info("Fetching updated prices from Azure API...")
    exchange_rate = data["api_config"].eur_usd_rate
    azure_prices = azure_pricing.fetch_azure_prices(eur_usd_rate=exchange_rate)
    if azure_prices.success:
        applied = azure_pricing.apply_azure_prices(
            data["infra_ideal"], data["infra_economica"],
            data["api_config"], azure_prices,
        )
        ui.done(f"Azure API prices applied: {applied} items")
    else:
        ui.info(f"Azure API: {azure_prices.error}. Using default prices.")

    lp = data["load_profile"]
    ui.business_dashboard(lp)

    # ------------------------------------------------------------------
    # PHASE 2: Deterministic simulation + Monte Carlo
    # ------------------------------------------------------------------
    ui.step(2, 5, "Calculating scenarios...")
    ui.info("Comparing 3 scenarios: AKS Ideal (A100) | "
            "AKS Economico (A10) | API OpenAI")
    ui.info("Monte Carlo: P50=median, P90=90% below that cost")

    df_results = simulate_all(
        data,
        mc_iterations=args.mc_iterations,
        ha_factor=args.ha_factor,
        overhead_factor=args.overhead,
        progress_callback=_progress_handler,
    )

    # ------------------------------------------------------------------
    # PHASE 3: Export results
    # ------------------------------------------------------------------
    ui.step(3, 5, "Exporting results...")
    out.export_excel(df_results, data.get("comparativa"), args.output_xlsx)
    ui.done(f"Excel -> {args.output_xlsx}")

    out.export_csv(df_results, args.output_csv)
    ui.done(f"CSV   -> {args.output_csv}")

    # ------------------------------------------------------------------
    # PHASE 4: Chart
    # ------------------------------------------------------------------
    if not args.no_chart:
        ui.step(4, 5, "Generating comparison chart...")
        out.generate_chart(df_results, args.chart, data)
        ui.done(f"Chart -> {args.chart}")
    else:
        ui.step(4, 5, "Chart skipped (--no-chart)")
        ui.done("Skipped")

    # ------------------------------------------------------------------
    # PHASE 5: Final summary
    # ------------------------------------------------------------------
    ui.step(5, 5, "Results")

    table_cols = ["Scenario", "Total", "GPU", "API LLM",
                   "MC P50", "MC P90", "Nodes"]
    table_rows = []
    for _, row in df_results.iterrows():
        short_name = SHORT_NAMES.get(row["scenario"], row["scenario"])
        table_rows.append([
            short_name,
            f'{row["total_cost_eur"]:,.0f}',
            f'{row["aks_gpu_cost_eur"]:,.0f}',
            f'{row["api_llm_cost_eur"]:,.0f}',
            f'{row["mc_p50_eur"]:,.0f}',
            f'{row["mc_p90_eur"]:,.0f}',
            str(int(row.get("gpu_peak_nodes", 0))),
        ])
    ui.rich_table(table_cols, table_rows,
                  title="COST COMPARISON (EUR/month)",
                  col_widths=[14, 9, 9, 9, 9, 9, 7])

    best_idx = df_results["total_cost_eur"].idxmin()
    best = df_results.iloc[best_idx]
    best_name = SHORT_NAMES.get(best["scenario"], best["scenario"])
    ui.info(f"Best option: '{best_name}' at "
            f"{best['total_cost_eur']:,.0f} EUR/month "
            f"(MC P50: {best['mc_p50_eur']:,.0f})")

    if args.verbose:
        logger.info("Simulation completed successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
