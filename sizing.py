import logging
import math
from typing import Tuple

from model import LoadProfile, AKSInfrastructure

logger = logging.getLogger(__name__)


def effective_throughput_per_pod(infra: AKSInfrastructure) -> float:
    return infra.throughput_tok_s_per_pod * infra.gpu_utilization * infra.safety_factor


def _concurrent_interactions_per_hour(load: LoadProfile) -> float:
    concurrent_users = load.users * load.concurrent_user_ratio
    return concurrent_users * load.interactions_per_user_day / load.office_hours_per_day


def size_gpu_nodes_by_throughput(
    infra: AKSInfrastructure,
    load: LoadProfile,
) -> Tuple[int, int, int]:
    tok_s_pod = effective_throughput_per_pod(infra)
    tokens_per_interaction = (
        load.input_tokens_per_interaction + load.output_tokens_per_interaction
    )
    int_per_hour = _concurrent_interactions_per_hour(load)

    if tok_s_pod <= 0 or int_per_hour <= 0:
        return 1, 1, 1

    base_tok_s = int_per_hour * tokens_per_interaction / 3600
    peak_tok_s = base_tok_s * load.peak_multiplier

    base_nodes = max(1, math.ceil(base_tok_s / tok_s_pod))
    peak_nodes = max(base_nodes, math.ceil(peak_tok_s / tok_s_pod))
    off_nodes = 1

    return base_nodes, peak_nodes, off_nodes


def apply_sizing(
    infra: AKSInfrastructure,
    load: LoadProfile,
) -> AKSInfrastructure:
    base, peak, off = size_gpu_nodes_by_throughput(infra, load)
    infra.inference_nodepool.base_office_nodes = base
    infra.inference_nodepool.peak_nodes = peak
    infra.inference_nodepool.off_hours_nodes = off
    infra.base_replicas = base
    infra.peak_replicas = peak
    infra.off_hours_replicas = off
    logger.info(
        "Sizing %s: base=%d, peak=%d, off=%d",
        infra.name, base, peak, off,
    )
    return infra
