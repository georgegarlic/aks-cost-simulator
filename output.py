import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed. Charts disabled.")


def export_excel(df_results: pd.DataFrame, df_comparison: Optional[pd.DataFrame],
                 path: str) -> str:
    path = Path(path).resolve()
    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Results", index=False)
        if df_comparison is not None and not df_comparison.empty:
            df_comparison.to_excel(writer, sheet_name="Original Comparison", index=False)
        df_summary = df_results[[
            "scenario", "total_cost_eur", "cost_per_user_eur",
            "cost_per_conversation_eur",
            "mc_p50_eur", "mc_p90_eur", "mc_p95_eur"
        ]]
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
    logger.info("Excel exported: %s", path)
    return str(path)


def export_csv(df: pd.DataFrame, path: str) -> str:
    path = Path(path).resolve()
    df.to_csv(str(path), index=False, encoding="utf-8-sig")
    logger.info("CSV exported: %s", path)
    return str(path)


def generate_chart(df: pd.DataFrame, path: str,
                   data: Optional[dict] = None) -> Optional[str]:
    if not HAS_MPL:
        logger.warning("matplotlib not available, skipping chart")
        return None

    path = Path(path).resolve()

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    blue = "#2c6b9e"
    red = "#d35400"
    green = "#27ae60"

    fig = plt.figure(figsize=(16, 7), facecolor="#fafafa")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.3],
                           height_ratios=[1, 1.2], hspace=0.3, wspace=0.25)

    n = len(df)

    # Panel (0,0): business data table
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_facecolor("white")

    if data:
        lp = data.get("load_profile")
        if lp:
            requests_day = lp.users * lp.interactions_per_user_day
            biz_rows = [
                ("Users", f"{lp.users:,}"),
                ("Requests/day", f"{requests_day:,}"),
                ("Tokens in/month", f"{lp.input_tokens_per_month:,}"),
                ("Tokens out/month", f"{lp.output_tokens_per_month:,}"),
            ]
            if df is not None and len(df) > 0:
                biz_rows.append(("", ""))
                for i in range(min(n, 3)):
                    name = df.iloc[i]["scenario"].split()[:2]
                    biz_rows.append((
                        f"{' '.join(name)}",
                        f'{df.iloc[i]["total_cost_eur"]:,.0f} EUR'
                    ))
                best = (
                    f'{df.iloc[df["total_cost_eur"].idxmin()]["total_cost_eur"]:,.0f}'
                )
                biz_rows.append(("Best cost", f"{best} EUR"))

            y0 = 0.95
            inc = 0.09
            ax0.text(0, y0 + 0.02, "INPUT DATA", fontsize=9,
                     fontweight="bold", color=blue)
            for i, (a, b) in enumerate(biz_rows):
                yy = y0 - (i + 1) * inc
                if a == "":
                    continue
                ax0.text(0, yy, a, fontsize=8, color="#34495e", va="top")
                ax0.text(0.65, yy, b, fontsize=8, fontweight="bold",
                         color=blue, va="top")

    # Panel (0,1): Throughput per scenario
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("white")

    throughputs = []
    tp_names = []
    if data:
        for key, name in [("infra_ideal", "Ideal"), ("infra_economica", "Econ.")]:
            infra = data.get(key)
            if infra:
                throughputs.append(infra.throughput_tok_s_per_pod)
                tp_names.append(name)

    if throughputs:
        t_bars = ax1.barh(tp_names, throughputs, color=[blue, green],
                          height=0.4, edgecolor="white")
        for bar, val in zip(t_bars, throughputs):
            ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                     f"{val} tok/s", va="center", fontsize=9, fontweight="bold",
                     color="#2c3e50")
        ax1.set_xlim(0, max(throughputs) * 1.4)
    ax1.set_title("GPU Node Throughput", fontsize=10, fontweight="bold",
                  color="#2c3e50")
    ax1.grid(axis="x", alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (0,2): Total cost per scenario
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("white")

    short_names = []
    for name in df["scenario"].tolist():
        parts = name.split()
        short = " ".join(parts[:2]) if len(parts) > 2 else name
        short_names.append(short)
    totals = [df.iloc[i].get("total_cost_eur", 0) for i in range(n)]
    bar_colors = [blue, green, red][:n]

    bars2 = ax2.bar(range(n), totals, color=bar_colors, width=0.5,
                    edgecolor="white")
    for bar, val in zip(bars2, totals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(totals) * 0.01,
                 f"{val:,.0f} EUR", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="#2c3e50")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(short_names, fontsize=9)
    ax2.set_title("Total monthly cost", fontsize=10, fontweight="bold",
                  color="#2c3e50")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)
    ax2.set_ylim(0, max(totals) * 1.2)

    # Bottom panel (1,:): comparison table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    ax3.set_facecolor("white")

    headers = ["Scenario", "Total EUR", "GPU EUR", "API LLM EUR",
               "Peak Nodes", "MC P50", "MC P90"]
    cell_data = []
    for _, row in df.iterrows():
        name = row["scenario"]
        parts = name.split()
        short = " ".join(parts[:2]) if len(parts) > 2 else name
        cell_data.append([
            short,
            f'{row["total_cost_eur"]:,.0f}',
            f'{row["aks_gpu_cost_eur"]:,.0f}',
            f'{row["api_llm_cost_eur"]:,.0f}',
            str(int(row.get("gpu_peak_nodes", 0))),
            f'{row["mc_p50_eur"]:,.0f}',
            f'{row["mc_p90_eur"]:,.0f}',
        ])

    tbl = ax3.table(cellText=cell_data, colLabels=headers,
                    cellLoc="center", loc="center",
                    colWidths=[0.15, 0.12, 0.12, 0.12, 0.10, 0.12, 0.12])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    for (r, c), cel in tbl.get_celld().items():
        cel.set_edgecolor("#cccccc")
        if r == 0:
            cel.set_text_props(fontweight="bold", color="white")
            cel.set_facecolor(blue)
        elif r % 2 == 0:
            cel.set_facecolor("#f0f0f0")
        else:
            cel.set_facecolor("white")
        if c == 0 and r > 0:
            cel.set_text_props(fontweight="bold")
        if r > 0 and c in (1, 2, 3, 5, 6):
            try:
                v = float(cell_data[r - 1][c].replace(",", ""))
                if v >= 50_000:
                    cel.set_text_props(color=red, fontweight="bold")
                elif v >= 20_000:
                    cel.set_text_props(color="#d68910", fontweight="bold")
                else:
                    cel.set_text_props(color=green, fontweight="bold")
            except ValueError:
                pass

    ax3.set_title("DETAILED COMPARISON (EUR/month)", fontsize=12,
                  fontweight="bold", color=blue, pad=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.8)

    fig.savefig(str(path), dpi=150, bbox_inches="tight",
                facecolor="#fafafa")
    plt.close(fig)
    logger.info("Chart exported: %s", path)
    return str(path)
