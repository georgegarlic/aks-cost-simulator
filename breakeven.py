"""
Inflection point / breakeven analysis for scenario comparison.
Sweeps a parameter (e.g. users) and finds where each scenario becomes optimal.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from model import LoadProfile, AKSInfrastructure, APIConfig, ScenarioResult
from pricing import compute_aks_cost, compute_api_cost, finalize_result
from sizing import apply_sizing


PARAM_LABELS = {
    "users": "Active Users",
    "interactions_per_user_day": "Interactions / User / Day",
    "input_tokens_per_interaction": "Input Tokens / Interaction",
    "output_tokens_per_interaction": "Output Tokens / Interaction",
}

SCENARIO_NAMES = [
    "AKS UX Ideal",
    "AKS UX Economico",
    "API Azure OpenAI",
]


def _build_load(base: LoadProfile, param: str, value: float) -> LoadProfile:
    kwargs = {
        "users": base.users,
        "interactions_per_user_day": base.interactions_per_user_day,
        "input_tokens_per_interaction": base.input_tokens_per_interaction,
        "output_tokens_per_interaction": base.output_tokens_per_interaction,
        "working_days_per_month": base.working_days_per_month,
        "office_hours_per_day": base.office_hours_per_day,
        "peak_hours_per_day": base.peak_hours_per_day,
        "concurrent_user_ratio": base.concurrent_user_ratio,
        "peak_multiplier": base.peak_multiplier,
    }
    kwargs[param] = int(value) if param == "users" or param == "interactions_per_user_day" else value
    return LoadProfile(**kwargs)


def _copy_infra(infra: AKSInfrastructure) -> AKSInfrastructure:
    import copy
    return copy.deepcopy(infra)


def _compute_for_load(
    infra_ideal: AKSInfrastructure,
    infra_economy: AKSInfrastructure,
    api: APIConfig,
    load: LoadProfile,
    ha_factor: float,
    overhead_factor: float,
) -> Tuple[float, float, float]:
    ideal_res = compute_aks_cost(apply_sizing(_copy_infra(infra_ideal), load), load, ha_factor, overhead_factor)
    total_ideal = finalize_result(ideal_res, load).total_cost

    eco_res = compute_aks_cost(apply_sizing(_copy_infra(infra_economy), load), load, ha_factor, overhead_factor)
    total_eco = finalize_result(eco_res, load).total_cost

    api_res = compute_api_cost(api, load)
    total_api = finalize_result(api_res, load).total_cost

    return total_ideal, total_eco, total_api


def _find_crossover(
    x_vals: List[float],
    cost_a: List[float],
    cost_b: List[float],
) -> Optional[float]:
    for i in range(1, len(x_vals)):
        if (cost_a[i - 1] - cost_b[i - 1]) * (cost_a[i] - cost_b[i]) < 0:
            x1, x2 = x_vals[i - 1], x_vals[i]
            y1_a, y2_a = cost_a[i - 1], cost_a[i]
            y1_b, y2_b = cost_b[i - 1], cost_b[i]
            if y2_a - y1_a == 0 and y2_b - y1_b == 0:
                return (x1 + x2) / 2
            t = (y1_b - y1_a) / ((y2_a - y1_a) - (y2_b - y1_b))
            return x1 + t * (x2 - x1)
    return None


def analyze_inflection(
    data: dict,
    param: str = "users",
    low_pct: float = 0.05,
    high_pct: float = 3.0,
    steps: int = 100,
    ha_factor: float = 1.15,
    overhead_factor: float = 0.1,
) -> dict:
    base_load: LoadProfile = data["load_profile"]
    infra_ideal: AKSInfrastructure = data["infra_ideal"]
    infra_economy: AKSInfrastructure = data["infra_economica"]
    api: APIConfig = data["api_config"]

    base_val = getattr(base_load, param)
    lo = base_val * low_pct
    hi = base_val * high_pct
    x_vals = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]

    ideal_costs: List[float] = []
    eco_costs: List[float] = []
    api_costs: List[float] = []

    for x in x_vals:
        load = _build_load(base_load, param, x)
        c_ideal, c_eco, c_api = _compute_for_load(
            infra_ideal, infra_economy, api, load, ha_factor, overhead_factor,
        )
        ideal_costs.append(c_ideal)
        eco_costs.append(c_eco)
        api_costs.append(c_api)

    crossovers = []
    cross_api_eco = _find_crossover(x_vals, api_costs, eco_costs)
    if cross_api_eco is not None:
        crossovers.append({
            "parameter": param,
            "value": cross_api_eco,
            "label": f"API OpenAI vs AKS Economy: {cross_api_eco:,.0f} {param}",
            "type": "API -> Economy",
        })
    cross_eco_ideal = _find_crossover(x_vals, eco_costs, ideal_costs)
    if cross_eco_ideal is not None:
        crossovers.append({
            "parameter": param,
            "value": cross_eco_ideal,
            "label": f"AKS Economy vs AKS Ideal: {cross_eco_ideal:,.0f} {param}",
            "type": "Economy -> Ideal",
        })
    cross_api_ideal = _find_crossover(x_vals, api_costs, ideal_costs)
    if cross_api_ideal is not None:
        crossovers.append({
            "parameter": param,
            "value": cross_api_ideal,
            "label": f"API OpenAI vs AKS Ideal: {cross_api_ideal:,.0f} {param}",
            "type": "API → Ideal",
        })

    best_ranges: List[Dict] = []
    current_best = None
    range_start = x_vals[0]
    for i, x in enumerate(x_vals):
        costs = [ideal_costs[i], eco_costs[i], api_costs[i]]
        best_idx = int(np.argmin(costs))
        best_name = SCENARIO_NAMES[best_idx]
        if best_name != current_best:
            if current_best is not None:
                best_ranges.append({
                    "scenario": current_best,
                    "from": range_start,
                    "to": x_vals[i - 1],
                    "from_label": f"{range_start:,.0f}",
                    "to_label": f"{x_vals[i - 1]:,.0f}",
                })
            current_best = best_name
            range_start = x
    if current_best is not None:
        best_ranges.append({
            "scenario": current_best,
            "from": range_start,
            "to": x_vals[-1],
            "from_label": f"{range_start:,.0f}",
            "to_label": f"{x_vals[-1]:,.0f}",
        })

    df = pd.DataFrame({
        param: x_vals,
        "param_label": [f"{v:,.0f}" for v in x_vals],
        "AKS UIdeal": ideal_costs,
        "AKS UEconomico": eco_costs,
        "API OpenAI": api_costs,
    })

    return {
        "parameter": param,
        "param_label": PARAM_LABELS.get(param, param),
        "base_value": base_val,
        "sweep_data": df,
        "crossovers": crossovers,
        "best_ranges": best_ranges,
    }
