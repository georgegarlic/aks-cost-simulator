import logging

from config import HOURS_PER_MONTH
from model import LoadProfile, AKSInfrastructure, APIConfig, ScenarioResult

logger = logging.getLogger(__name__)


def monthly_vm_cost(price_per_hour: float, nodes: int, hours: float) -> float:
    return nodes * price_per_hour * hours


def compute_aks_cost(
    infra: AKSInfrastructure,
    load: LoadProfile,
    ha_factor: float = 1.15,
    overhead_factor: float = 0.1,
) -> ScenarioResult:
    r = ScenarioResult(scenario=infra.name)
    r.description = f"AKS with {infra.inference_nodepool.vm_type}"

    r.system_nodes = infra.system_nodepool.base_office_nodes
    r.gpu_base_nodes = infra.inference_nodepool.base_office_nodes
    r.gpu_peak_nodes = infra.inference_nodepool.peak_nodes
    r.gpu_off_hours_nodes = infra.inference_nodepool.off_hours_nodes

    total_hours = HOURS_PER_MONTH
    office_hours = load.office_hours_per_month
    peak_hours = load.peak_hours_per_month
    off_hours = total_hours - office_hours

    r.aks_system_cost = monthly_vm_cost(
        infra.system_nodepool.price_per_hour, r.system_nodes, total_hours
    )

    gpu_off_cost = monthly_vm_cost(
        infra.inference_nodepool.price_per_hour, r.gpu_off_hours_nodes, off_hours
    )
    gpu_base_cost = monthly_vm_cost(
        infra.inference_nodepool.price_per_hour,
        max(0, r.gpu_base_nodes - r.gpu_off_hours_nodes),
        office_hours,
    )
    gpu_peak_cost = monthly_vm_cost(
        infra.inference_nodepool.price_per_hour,
        max(0, r.gpu_peak_nodes - r.gpu_base_nodes),
        peak_hours * (1 + overhead_factor),
    )

    r.aks_gpu_cost = (gpu_off_cost + gpu_base_cost + gpu_peak_cost) * ha_factor
    r.storage_cost = infra.storage_cost_per_month
    r.lb_cost = infra.lb_cost_per_month
    r.monitor_cost = infra.monitor_cost_per_month
    r.acr_cost = infra.acr_cost_per_month
    r.aks_total_cost = (
        r.aks_system_cost + r.aks_gpu_cost
        + r.storage_cost + r.lb_cost
        + r.monitor_cost + r.acr_cost
    )
    return r


def compute_api_cost(
    api: APIConfig,
    load: LoadProfile,
) -> ScenarioResult:
    r = ScenarioResult(scenario="API Azure OpenAI")
    r.description = "Pay-as-you-go; no infrastructure"

    input_cost_usd = load.input_tokens_per_month * api.input_price_per_1m_tokens_usd / 1_000_000
    output_cost_usd = load.output_tokens_per_month * api.output_price_per_1m_tokens_usd / 1_000_000
    r.api_llm_cost = (input_cost_usd + output_cost_usd) * api.eur_usd_rate
    r.aks_total_cost = 0.0
    r.total_cost = r.api_llm_cost
    return r


def finalize_result(r: ScenarioResult, load: LoadProfile) -> ScenarioResult:
    r.total_cost = r.aks_total_cost + r.api_llm_cost
    if load.users > 0:
        r.cost_per_user = r.total_cost / load.users
    total_conversations = (
        load.users * load.interactions_per_user_day * load.working_days_per_month
    )
    if total_conversations > 0:
        r.cost_per_conversation = r.total_cost / total_conversations
    return r
