import logging
from typing import Dict, Tuple

import pandas as pd

from model import LoadProfile, NodepoolConfig, AKSInfrastructure, APIConfig
from config import (
    DEFAULT_USERS, DEFAULT_INTERACTIONS_PER_USER_DAY,
    DEFAULT_INPUT_TOKENS, DEFAULT_OUTPUT_TOKENS,
    DEFAULT_WORKING_DAYS, DEFAULT_OFFICE_HOURS,
    DEFAULT_PEAK_HOURS, DEFAULT_CONCURRENT_RATIO,
    DEFAULT_PEAK_MULTIPLIER,
    DEFAULT_SYSTEM_VM, DEFAULT_SYSTEM_PRICE, DEFAULT_SYSTEM_NODES,
    DEFAULT_IDEAL_GPU_VM, DEFAULT_IDEAL_GPU_PRICE, DEFAULT_IDEAL_THROUGHPUT,
    DEFAULT_ECO_GPU_VM, DEFAULT_ECO_GPU_PRICE, DEFAULT_ECO_THROUGHPUT,
    DEFAULT_STORAGE_IDEAL, DEFAULT_STORAGE_ECO,
    DEFAULT_LB, DEFAULT_MONITOR_IDEAL, DEFAULT_MONITOR_ECO,
    DEFAULT_ACR_IDEAL, DEFAULT_ACR_ECO,
    DEFAULT_API_MODEL, DEFAULT_API_INPUT_PRICE, DEFAULT_API_OUTPUT_PRICE,
    DEFAULT_EUR_USD,
)

logger = logging.getLogger(__name__)


def build_default_data() -> dict:
    load_profile = LoadProfile(
        users=DEFAULT_USERS,
        interactions_per_user_day=DEFAULT_INTERACTIONS_PER_USER_DAY,
        input_tokens_per_interaction=DEFAULT_INPUT_TOKENS,
        output_tokens_per_interaction=DEFAULT_OUTPUT_TOKENS,
        working_days_per_month=DEFAULT_WORKING_DAYS,
        office_hours_per_day=DEFAULT_OFFICE_HOURS,
        peak_hours_per_day=DEFAULT_PEAK_HOURS,
        concurrent_user_ratio=DEFAULT_CONCURRENT_RATIO,
        peak_multiplier=DEFAULT_PEAK_MULTIPLIER,
    )

    infra_ideal = AKSInfrastructure(name="LLM on AKS (Ideal UX)")
    infra_ideal.system_nodepool = NodepoolConfig(
        vm_type=DEFAULT_SYSTEM_VM, base_office_nodes=DEFAULT_SYSTEM_NODES,
        price_per_hour=DEFAULT_SYSTEM_PRICE,
    )
    infra_ideal.inference_nodepool = NodepoolConfig(
        vm_type=DEFAULT_IDEAL_GPU_VM,
        base_office_nodes=3, peak_nodes=10, off_hours_nodes=1,
        price_per_hour=DEFAULT_IDEAL_GPU_PRICE,
    )
    infra_ideal.throughput_tok_s_per_pod = DEFAULT_IDEAL_THROUGHPUT
    infra_ideal.gpu_utilization = 0.75
    infra_ideal.safety_factor = 1.0
    infra_ideal.base_replicas = 3
    infra_ideal.peak_replicas = 10
    infra_ideal.off_hours_replicas = 1
    infra_ideal.storage_cost_per_month = DEFAULT_STORAGE_IDEAL
    infra_ideal.lb_cost_per_month = DEFAULT_LB
    infra_ideal.monitor_cost_per_month = DEFAULT_MONITOR_IDEAL
    infra_ideal.acr_cost_per_month = DEFAULT_ACR_IDEAL

    infra_economica = AKSInfrastructure(name="LLM on AKS (Economy UX)")
    infra_economica.system_nodepool = NodepoolConfig(
        vm_type=DEFAULT_SYSTEM_VM, base_office_nodes=DEFAULT_SYSTEM_NODES,
        price_per_hour=DEFAULT_SYSTEM_PRICE,
    )
    infra_economica.inference_nodepool = NodepoolConfig(
        vm_type=DEFAULT_ECO_GPU_VM,
        base_office_nodes=5, peak_nodes=20, off_hours_nodes=1,
        price_per_hour=DEFAULT_ECO_GPU_PRICE,
    )
    infra_economica.throughput_tok_s_per_pod = DEFAULT_ECO_THROUGHPUT
    infra_economica.gpu_utilization = 0.75
    infra_economica.safety_factor = 1.0
    infra_economica.base_replicas = 5
    infra_economica.peak_replicas = 20
    infra_economica.off_hours_replicas = 1
    infra_economica.storage_cost_per_month = DEFAULT_STORAGE_ECO
    infra_economica.lb_cost_per_month = DEFAULT_LB
    infra_economica.monitor_cost_per_month = DEFAULT_MONITOR_ECO
    infra_economica.acr_cost_per_month = DEFAULT_ACR_ECO

    api_config = APIConfig(
        name="API Azure OpenAI",
        model=DEFAULT_API_MODEL,
        input_price_per_1m_tokens_usd=DEFAULT_API_INPUT_PRICE,
        output_price_per_1m_tokens_usd=DEFAULT_API_OUTPUT_PRICE,
        eur_usd_rate=DEFAULT_EUR_USD,
    )

    return {
        "load_profile": load_profile,
        "infra_ideal": infra_ideal,
        "infra_economica": infra_economica,
        "api_config": api_config,
        "comparativa": pd.DataFrame(),
    }


def extract_all(path: str = "") -> dict:
    return build_default_data()
