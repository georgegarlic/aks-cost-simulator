import logging
from dataclasses import dataclass, field
import copy
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoadProfile:
    users: int = 5000
    interactions_per_user_day: int = 40
    input_tokens_per_interaction: int = 1500
    output_tokens_per_interaction: int = 500
    working_days_per_month: int = 22
    office_hours_per_day: int = 10
    peak_hours_per_day: float = 0.5
    concurrent_user_ratio: float = 0.15
    peak_multiplier: float = 2.0

    @property
    def office_hours_per_month(self) -> int:
        return self.working_days_per_month * self.office_hours_per_day

    @property
    def peak_hours_per_month(self) -> float:
        return self.working_days_per_month * self.peak_hours_per_day

    @property
    def requests_per_day(self) -> int:
        return self.users * self.interactions_per_user_day

    @property
    def avg_requests_per_hour(self) -> float:
        total = self.requests_per_day
        return total / self.office_hours_per_day if self.office_hours_per_day else 0.0

    @property
    def input_tokens_per_month(self) -> int:
        return (self.users * self.interactions_per_user_day
                * self.input_tokens_per_interaction * self.working_days_per_month)

    @property
    def output_tokens_per_month(self) -> int:
        return (self.users * self.interactions_per_user_day
                * self.output_tokens_per_interaction * self.working_days_per_month)

    @property
    def total_tokens_per_month(self) -> int:
        return self.input_tokens_per_month + self.output_tokens_per_month


@dataclass
class NodepoolConfig:
    vm_type: str = ""
    cpu_per_node: int = 0
    ram_gb_per_node: float = 0.0
    price_per_hour: float = 0.0
    base_office_nodes: int = 0
    peak_nodes: int = 0
    off_hours_nodes: int = 0


@dataclass
class AKSInfrastructure:
    name: str = ""
    system_nodepool: NodepoolConfig = field(default_factory=NodepoolConfig)
    inference_nodepool: NodepoolConfig = field(default_factory=NodepoolConfig)
    throughput_tok_s_per_pod: float = 0.0
    gpu_utilization: float = 0.75
    safety_factor: float = 1.0
    base_replicas: int = 0
    peak_replicas: int = 0
    off_hours_replicas: int = 0
    storage_cost_per_month: float = 0.0
    lb_cost_per_month: float = 0.0
    monitor_cost_per_month: float = 0.0
    acr_cost_per_month: float = 0.0


@dataclass
class APIConfig:
    name: str = "API Azure OpenAI"
    model: str = ""
    input_price_per_1m_tokens_usd: float = 0.0
    output_price_per_1m_tokens_usd: float = 0.0
    eur_usd_rate: float = 0.92


@dataclass
class ScenarioResult:
    scenario: str = ""
    description: str = ""
    system_nodes: int = 0
    gpu_base_nodes: int = 0
    gpu_peak_nodes: int = 0
    gpu_off_hours_nodes: int = 0
    aks_system_cost: float = 0.0
    aks_gpu_cost: float = 0.0
    storage_cost: float = 0.0
    lb_cost: float = 0.0
    monitor_cost: float = 0.0
    acr_cost: float = 0.0
    aks_total_cost: float = 0.0
    api_llm_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_user: float = 0.0
    cost_per_conversation: float = 0.0


@dataclass
class MonteCarloDistribution:
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    mean: float = 0.0
    std_dev: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    samples: List[float] = field(default_factory=list)
