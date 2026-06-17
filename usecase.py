import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from simulator import simulate_all

logger = logging.getLogger(__name__)


@dataclass
class InfoSource:
    name: str = ""
    source_type: str = "sharepoint"
    complexity: str = "medium"
    data_volume_gb: float = 10.0
    update_frequency: str = "daily"

    @property
    def integration_cost(self) -> float:
        return _source_integration_cost(self.source_type, self.complexity)

    @property
    def monthly_maintenance(self) -> float:
        base = self.integration_cost * 0.10 / 12
        freq_mult = _frequency_multiplier(self.update_frequency)
        return round(base * freq_mult, 2)


@dataclass
class UseCaseResult:
    sources: List[InfoSource] = field(default_factory=list)

    source_integration_capex: float = 0.0
    capabilities_capex: float = 0.0
    compliance_capex: float = 0.0
    total_capex: float = 0.0

    source_maintenance_opex: float = 0.0
    capabilities_opex: float = 0.0
    infrastructure_opex: float = 0.0
    total_opex_monthly: float = 0.0

    total_year_1: float = 0.0
    total_year_3: float = 0.0

    infrastructure_df: Optional[pd.DataFrame] = None


SOURCE_INTEGRATION_TABLE = {
    "sharepoint":   {"low": 3000, "medium": 5000, "high": 8000},
    "database":     {"low": 4000, "medium": 6500, "high": 10000},
    "web_scraping": {"low": 3500, "medium": 5500, "high": 9000},
    "api":          {"low": 2500, "medium": 4500, "high": 7000},
    "pdf_dynamic":  {"low": 2000, "medium": 3500, "high": 6000},
}

FREQUENCY_MULTIPLIERS = {
    "realtime": 1.05,
    "hourly":   1.03,
    "daily":    1.0,
    "weekly":   0.98,
    "monthly":  0.95,
}


def _source_integration_cost(source_type: str, complexity: str) -> float:
    row = SOURCE_INTEGRATION_TABLE.get(source_type, {})
    return row.get(complexity, 5000)


def _frequency_multiplier(frequency: str) -> float:
    return FREQUENCY_MULTIPLIERS.get(frequency, 1.0)


CAPABILITY_COSTS = {
    "agentic_ai":     {"capex": 8500, "opex_monthly": 850},
    "anonymization":  {"capex": 3500, "opex_monthly": 350},
    "sso":            {"capex": 4000, "opex_monthly": 200},
}

ENS_COSTS = {
    "none":  {"capex": 0},
    "basic":  {"capex": 1000},
    "medium": {"capex": 2500},
    "high":   {"capex": 5000},
}


def calculate_usecase_cost(
    sources: List[InfoSource],
    enabled_capabilities: List[str],
    ens_level: str,
    business_params: Optional[dict] = None,
    deployment: str = "economy",
) -> UseCaseResult:
    result = UseCaseResult(sources=sources)

    # --- CAPEX: Source integration ---
    result.source_integration_capex = sum(s.integration_cost for s in sources)

    # --- CAPEX: Capabilities ---
    cap_capex = 0
    cap_opex = 0
    for cap in enabled_capabilities:
        c = CAPABILITY_COSTS.get(cap, {})
        cap_capex += c.get("capex", 0)
        cap_opex += c.get("opex_monthly", 0)
    result.capabilities_capex = cap_capex
    result.capabilities_opex = cap_opex

    # --- CAPEX: Compliance ---
    result.compliance_capex = ENS_COSTS.get(ens_level, {}).get("capex", 0)

    # --- Total CAPEX ---
    result.total_capex = (
        result.source_integration_capex
        + result.capabilities_capex
        + result.compliance_capex
    )

    # --- OPEX: Source maintenance ---
    result.source_maintenance_opex = sum(s.monthly_maintenance for s in sources)

    # --- OPEX: Infrastructure (from existing simulator) ---
    if business_params:
        try:
            data = _build_infra_data(business_params, deployment)
            df = simulate_all(data, mc_iterations=100, resize=True)
            result.infrastructure_df = df
            target_scenario = _scenario_name(deployment)
            row = df[df["scenario"] == target_scenario]
            if not row.empty:
                result.infrastructure_opex = round(float(row.iloc[0]["total_cost_eur"]), 2)
            else:
                result.infrastructure_opex = 0
        except Exception as e:
            logger.warning("Infrastructure simulation failed: %s", e)
            result.infrastructure_opex = 0

    # --- Total OPEX monthly ---
    result.total_opex_monthly = (
        result.source_maintenance_opex
        + result.capabilities_opex
        + result.infrastructure_opex
    )

    # --- TCO ---
    result.total_year_1 = result.total_capex + result.total_opex_monthly * 12
    result.total_year_3 = result.total_capex + result.total_opex_monthly * 36

    return result


def _scenario_name(deployment: str) -> str:
    return {
        "ideal": "AKS UX Ideal",
        "economy": "AKS UX Economico",
        "api": "API Azure OpenAI",
    }.get(deployment, "AKS UX Economico")


def _build_infra_data(business_params: dict, deployment: str) -> dict:
    from loader import build_default_data
    data = build_default_data()

    lp = data["load_profile"]
    lp.users = int(business_params.get("users", 5000))
    lp.interactions_per_user_day = int(business_params.get("interactions_per_user_day", 20))
    lp.input_tokens_per_interaction = int(business_params.get("input_tokens_per_interaction", 1000))
    lp.output_tokens_per_interaction = int(business_params.get("output_tokens_per_interaction", 400))
    lp.working_days_per_month = int(business_params.get("working_days_per_month", 22))
    lp.office_hours_per_day = int(business_params.get("office_hours_per_day", 10))
    lp.peak_hours_per_day = float(business_params.get("peak_hours_per_day", 1.0))
    lp.concurrent_user_ratio = float(business_params.get("concurrent_user_ratio", 0.15))
    lp.peak_multiplier = float(business_params.get("peak_multiplier", 2.0))

    return data
