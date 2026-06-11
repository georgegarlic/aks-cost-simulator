import logging
import random
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from model import (
    LoadProfile, AKSInfrastructure, APIConfig,
    ScenarioResult, MonteCarloDistribution,
)
from pricing import compute_aks_cost, compute_api_cost, finalize_result
from sizing import apply_sizing

logger = logging.getLogger(__name__)


def _triangular_sample(base: float, variation: float) -> float:
    low = base * (1 - variation)
    high = base * (1 + variation)
    return random.triangular(low, high, base)


def _vary_load_profile(load: LoadProfile, traffic_var: float, peak_var: float,
                       throughput_var: float) -> LoadProfile:
    return LoadProfile(
        users=max(1, int(_triangular_sample(load.users, traffic_var))),
        interactions_per_user_day=max(
            1, int(_triangular_sample(load.interactions_per_user_day, traffic_var))
        ),
        input_tokens_per_interaction=max(
            1, int(_triangular_sample(load.input_tokens_per_interaction, throughput_var))
        ),
        output_tokens_per_interaction=max(
            1, int(_triangular_sample(load.output_tokens_per_interaction, throughput_var))
        ),
        working_days_per_month=load.working_days_per_month,
        office_hours_per_day=load.office_hours_per_day,
        peak_hours_per_day=max(0.1, _triangular_sample(load.peak_hours_per_day, peak_var)),
        concurrent_user_ratio=load.concurrent_user_ratio,
        peak_multiplier=load.peak_multiplier,
    )


def _vary_infra(infra: AKSInfrastructure, throughput_var: float,
                pricing_var: float) -> AKSInfrastructure:
    import copy
    infra_var = AKSInfrastructure(
        name=infra.name,
        system_nodepool=copy.deepcopy(infra.system_nodepool),
        inference_nodepool=copy.deepcopy(infra.inference_nodepool),
        throughput_tok_s_per_pod=max(
            10, _triangular_sample(infra.throughput_tok_s_per_pod, throughput_var)
        ),
        gpu_utilization=infra.gpu_utilization,
        safety_factor=infra.safety_factor,
        base_replicas=infra.base_replicas,
        peak_replicas=infra.peak_replicas,
        off_hours_replicas=infra.off_hours_replicas,
        storage_cost_per_month=infra.storage_cost_per_month,
        lb_cost_per_month=infra.lb_cost_per_month,
        monitor_cost_per_month=infra.monitor_cost_per_month,
        acr_cost_per_month=infra.acr_cost_per_month,
    )
    infra_var.system_nodepool.price_per_hour = max(
        0.01, _triangular_sample(infra.system_nodepool.price_per_hour, pricing_var)
    )
    infra_var.inference_nodepool.price_per_hour = max(
        0.01, _triangular_sample(infra.inference_nodepool.price_per_hour, pricing_var)
    )
    return infra_var


def _vary_api(api: APIConfig, pricing_var: float) -> APIConfig:
    return APIConfig(
        name=api.name,
        model=api.model,
        input_price_per_1m_tokens_usd=max(
            0.01, _triangular_sample(api.input_price_per_1m_tokens_usd, pricing_var)
        ),
        output_price_per_1m_tokens_usd=max(
            0.01, _triangular_sample(api.output_price_per_1m_tokens_usd, pricing_var)
        ),
        eur_usd_rate=api.eur_usd_rate,
    )


def simulate_scenario_deterministic(
    infra: Optional[AKSInfrastructure],
    api: Optional[APIConfig],
    load: LoadProfile,
    ha_factor: float = 1.15,
    overhead_factor: float = 0.1,
    resize: bool = False,
) -> ScenarioResult:
    if infra is not None:
        if resize:
            infra = apply_sizing(infra, load)
        result = compute_aks_cost(infra, load, ha_factor, overhead_factor)
    elif api is not None:
        result = compute_api_cost(api, load)
    else:
        raise ValueError("Must provide either AKS infra or API config")
    return finalize_result(result, load)


def monte_carlo_scenario(
    infra: Optional[AKSInfrastructure],
    api: Optional[APIConfig],
    load: LoadProfile,
    iterations: int = 10_000,
    traffic_var: float = 0.20,
    peak_var: float = 0.30,
    throughput_var: float = 0.10,
    pricing_var: float = 0.05,
    ha_factor: float = 1.15,
    overhead_factor: float = 0.1,
    progress_callback: Optional[Callable] = None,
) -> MonteCarloDistribution:
    samples: List[float] = []
    for i in range(iterations):
        if progress_callback:
            progress_callback(i + 1, iterations)
        load_var = _vary_load_profile(load, traffic_var, peak_var, throughput_var)

        if infra is not None:
            infra_var = _vary_infra(infra, throughput_var, pricing_var)
            result = simulate_scenario_deterministic(
                infra_var, None, load_var, ha_factor, overhead_factor,
                resize=True,
            )
        elif api is not None:
            api_var = _vary_api(api, pricing_var)
            result = simulate_scenario_deterministic(
                None, api_var, load_var, ha_factor, overhead_factor,
            )
        else:
            continue

        samples.append(result.total_cost)

    samples.sort()
    n = len(samples)
    dist = MonteCarloDistribution()
    dist.samples = samples
    dist.p50 = float(samples[int(n * 0.50)]) if n else 0.0
    dist.p90 = float(samples[int(n * 0.90)]) if n else 0.0
    dist.p95 = float(samples[int(n * 0.95)]) if n else 0.0
    dist.mean = float(np.mean(samples)) if samples else 0.0
    dist.std_dev = float(np.std(samples)) if samples else 0.0
    dist.minimum = float(samples[0]) if samples else 0.0
    dist.maximum = float(samples[-1]) if samples else 0.0
    return dist


def simulate_all(
    data: dict,
    mc_iterations: int = 10_000,
    ha_factor: float = 1.15,
    overhead_factor: float = 0.1,
    resize: bool = False,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    load = data["load_profile"]
    api = data["api_config"]
    rows = []

    scenarios = [
        ("AKS UX Ideal", data["infra_ideal"], None),
        ("AKS UX Economico", data["infra_economica"], None),
        ("API Azure OpenAI", None, api),
    ]

    for idx, (name, infra, api_cfg) in enumerate(scenarios, 1):
        if progress_callback:
            progress_callback("scenario", idx, len(scenarios), name)

        result = simulate_scenario_deterministic(
            infra, api_cfg, load, ha_factor, overhead_factor,
            resize=resize,
        )

        def _mc_progress(current, total):
            if progress_callback:
                progress_callback("mc", current, total)

        dist = monte_carlo_scenario(
            infra, api_cfg, load,
            iterations=mc_iterations,
            ha_factor=ha_factor,
            overhead_factor=overhead_factor,
            progress_callback=_mc_progress,
        )

        rows.append({
            "scenario": name,
            "description": result.description,
            "system_nodes": result.system_nodes,
            "gpu_base_nodes": result.gpu_base_nodes,
            "gpu_peak_nodes": result.gpu_peak_nodes,
            "gpu_off_hours_nodes": result.gpu_off_hours_nodes,
            "aks_system_cost_eur": round(result.aks_system_cost, 2),
            "aks_gpu_cost_eur": round(result.aks_gpu_cost, 2),
            "storage_cost_eur": round(result.storage_cost, 2),
            "lb_cost_eur": round(result.lb_cost, 2),
            "monitor_cost_eur": round(result.monitor_cost, 2),
            "acr_cost_eur": round(result.acr_cost, 2),
            "aks_total_cost_eur": round(result.aks_total_cost, 2),
            "api_llm_cost_eur": round(result.api_llm_cost, 2),
            "total_cost_eur": round(result.total_cost, 2),
            "cost_per_user_eur": round(result.cost_per_user, 4),
            "cost_per_conversation_eur": round(result.cost_per_conversation, 6),
            "mc_p50_eur": round(dist.p50, 2),
            "mc_p90_eur": round(dist.p90, 2),
            "mc_p95_eur": round(dist.p95, 2),
            "mc_mean_eur": round(dist.mean, 2),
            "mc_std_dev_eur": round(dist.std_dev, 2),
        })

    return pd.DataFrame(rows)
