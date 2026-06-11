import sys
from typing import List, Optional, Any

from rich.console import Console
from rich.table import Table as RichTable
from rich.text import Text

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

CHECK = "+"
CROSS = "x"
ARROW = ">"
BAR_FULL = "#"
BAR_EMPTY = "."
BULLET = "*"

_console = Console(width=120)


def step(num: int, total: int, text: str) -> None:
    print(f"\n{YELLOW}[{num}/{total}]{RESET} {BOLD}{text}{RESET}")


def done(text: str, value: str = "") -> None:
    line = f"  {GREEN}{CHECK}{RESET} {text}"
    if value:
        line += f"  {CYAN}{value}{RESET}"
    print(line)


def error(text: str) -> None:
    print(f"  {RED}{CROSS}{RESET} {text}")


def info(text: str) -> None:
    print(f"  {BLUE}{BULLET}{RESET} {text}")


def subtitle(text: str) -> None:
    print(f"\n  {BOLD}{text}{RESET}")


def separator() -> None:
    print(f"  {BOLD}{'=' * 50}{RESET}")


def header(text: str) -> None:
    print(f"\n{BOLD}{'=' * 55}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'=' * 55}{RESET}")


def summary_row(label: str, value: str) -> None:
    print(f"  {YELLOW}{label:<25}{RESET} {CYAN}{value}{RESET}")


def _rich_color(val: float) -> str:
    if val == 0:
        return "green"
    if val < 20_000:
        return "green"
    elif val < 50_000:
        return "yellow"
    else:
        return "red"


def _fmt(val: float) -> str:
    if val >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    return f"{val:,.0f}"


def rich_table(columns: List[str], rows: List[List[str]],
               title: str = "", footer: str = "",
               col_widths: Optional[List[int]] = None) -> None:
    t = RichTable(
        title=title or None,
        title_style="bold white on blue",
        header_style="bold white on #2c6b9e",
        border_style="#5dade2",
        show_lines=True,
        padding=(0, 2),
    )
    for i, col in enumerate(columns):
        kwargs = {"justify": "center", "no_wrap": True}
        if col_widths and i < len(col_widths):
            kwargs["width"] = col_widths[i]
        t.add_column(col, **kwargs)

    for i, row in enumerate(rows):
        styled_row = []
        for j, val in enumerate(row):
            if j == 0:
                styled_row.append(Text(val, style="bold"))
            else:
                try:
                    n = float(val.replace(",", ""))
                    c = _rich_color(n)
                    styled_row.append(Text(val, style=f"bold {c}"))
                except (ValueError, AttributeError):
                    styled_row.append(val)
        t.add_row(*styled_row)

    _console.print(t)


def business_dashboard(lp: Any) -> None:
    t = RichTable(
        title="[bold]BUSINESS DATA[/]",
        title_style="bold white on #1a5276",
        header_style="bold white on #2c6b9e",
        border_style="#5dade2",
        show_lines=False,
        padding=(0, 3),
    )
    t.add_column("Metric", style="bold", justify="left")
    t.add_column("Value", justify="right")

    requests_day = lp.users * lp.interactions_per_user_day
    requests_month = requests_day * lp.working_days_per_month

    data_rows = [
        ("Active users", f"{lp.users:,}"),
        ("Interactions/user/day", f"{lp.interactions_per_user_day:,}"),
        ("Total requests/day", f"{requests_day:,}"),
        ("Total requests/month", f"{requests_month:,}"),
        ("Input tokens/month", f"{lp.input_tokens_per_month:,}"),
        ("Output tokens/month", f"{lp.output_tokens_per_month:,}"),
        ("Office hours/day", str(lp.office_hours_per_day)),
        ("Peak hours/day", str(lp.peak_hours_per_day)),
        ("Working days/month", str(lp.working_days_per_month)),
    ]

    for label, val in data_rows:
        t.add_row(label, Text(val, style="bold cyan"))

    _console.print(t)
