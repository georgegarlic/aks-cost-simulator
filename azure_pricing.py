"""
Queries the Azure Retail Prices API in real-time to get official
prices for VM, AKS, storage and networking in West Europe.
Results are used to override the input Excel prices.

The API only returns prices in USD. They are converted to EUR using
the exchange rate from the Excel config (eur_usd_rate).
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from model import AKSInfrastructure, APIConfig

logger = logging.getLogger(__name__)

AZURE_RETAIL_API = "https://prices.azure.com/api/retail/prices"
API_VERSION = "2023-01-01-preview"
REGION = "westeurope"


@dataclass
class AzurePriceItem:
    key: str = ""
    label: str = ""
    unit_price_usd: float = 0.0
    unit_price_eur: float = 0.0
    unit: str = ""
    source: str = "API Azure"


@dataclass
class AzurePricesResult:
    items: List[AzurePriceItem] = None
    success: bool = False
    error: str = ""

    def __post_init__(self):
        if self.items is None:
            self.items = []


def _query_api(odata_filter: str) -> List[dict]:
    from urllib.parse import quote
    encoded = quote(odata_filter, safe="'()")
    url = f"{AZURE_RETAIL_API}?api-version={API_VERSION}&$filter={encoded}"
    all_items: List[dict] = []
    page = 0
    while url and page < 20:
        try:
            req = Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "simulador-costes-azure/1.0",
            })
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            all_items.extend(data.get("Items", []))
            url = data.get("NextPageLink", "")
            page += 1
        except (URLError, json.JSONDecodeError, OSError) as e:
            logger.warning("Azure API error (page %d): %s", page, e)
            break
    return all_items


def _best_price(items: List[dict], region: str = REGION,
                price_type: str = "Consumption") -> Optional[float]:
    best = None
    for it in items:
        meter = it.get("meterName", "")
        if ("Low Priority" in meter or "Spot" in meter
                or "LowPriority" in meter):
            continue
        if (it.get("type") == price_type
                and it.get("armRegionName", "").lower() == region.lower()
                and it.get("currencyCode") == "USD"):
            up = it.get("unitPrice")
            if up is not None and (best is None or up < best):
                best = up
    return best


def _to_eur(usd: float, eur_usd_rate: float) -> float:
    return usd / eur_usd_rate if eur_usd_rate > 0 else usd / 0.92


def fetch_azure_prices(eur_usd_rate: float = 0.92) -> AzurePricesResult:
    result = AzurePricesResult()
    items: List[AzurePriceItem] = []

    queries = [
        ("vm_nc24ads_a100_v4", "VM NC24ads_A100_v4 (GPU A100)",
         "armSkuName eq 'Standard_NC24ads_A100_v4' "
         "and serviceName eq 'Virtual Machines'",
         "1 Hour"),
        ("vm_nv12ads_a10_v5", "VM NV12ads_A10_v5 (GPU A10)",
         "armSkuName eq 'Standard_NV12ads_A10_v5' "
         "and serviceName eq 'Virtual Machines'",
         "1 Hour"),
        ("vm_d8ds_v5", "VM Standard_D8ds_v5 (system)",
         "armSkuName eq 'Standard_D8ds_v5' "
         "and serviceName eq 'Virtual Machines'",
         "1 Hour"),
        ("aks_cluster", "AKS Cluster (Standard Uptime SLA)",
         "serviceName eq 'Azure Kubernetes Service' "
         "and meterName eq 'Standard Uptime SLA'",
         "1 Hour"),
        ("monitor_logs", "Log Analytics (Data Retention)",
         "serviceName eq 'Log Analytics' "
         "and meterName eq 'Analytics Logs Data Retention'",
         "1 GB/Month"),
        ("container_registry", "Container Registry (Basic)",
         "serviceName eq 'Container Registry' "
         "and meterName eq 'Basic Registry Unit'",
         "1 Day"),
    ]

    for key, label, o_filter, unit in queries:
        try:
            raw = _query_api(o_filter)
            price_usd = _best_price(raw)
            if price_usd is not None:
                price_eur = _to_eur(price_usd, eur_usd_rate)
                items.append(AzurePriceItem(
                    key=key, label=label,
                    unit_price_usd=price_usd,
                    unit_price_eur=price_eur,
                    unit=unit, source="API Azure",
                ))
                logger.debug("Azure API: %s = %.4f USD (%.4f EUR)",
                             label, price_usd, price_eur)
            else:
                logger.info("Azure API: no price for %s in %s/USD",
                            label, REGION)
        except Exception as e:
            logger.warning("Error querying %s: %s", label, e)

    result.items = items
    result.success = len(items) > 0
    if not items:
        result.error = ("Could not fetch prices from Azure API "
                        "(no connection or SKUs not found)")
    return result


def apply_azure_prices(
    infra_ideal: AKSInfrastructure,
    infra_economica: AKSInfrastructure,
    api_config: APIConfig,
    prices_azure: AzurePricesResult,
) -> int:
    applied = 0
    by_key: Dict[str, float] = {}
    for item in prices_azure.items:
        by_key[item.key] = item.unit_price_eur

    if "vm_nc24ads_a100_v4" in by_key:
        infra_ideal.inference_nodepool.price_per_hour = by_key["vm_nc24ads_a100_v4"]
        applied += 1
    if "vm_nv12ads_a10_v5" in by_key:
        infra_economica.inference_nodepool.price_per_hour = by_key["vm_nv12ads_a10_v5"]
        applied += 1
    if "vm_d8ds_v5" in by_key:
        val = by_key["vm_d8ds_v5"]
        infra_ideal.system_nodepool.price_per_hour = val
        infra_economica.system_nodepool.price_per_hour = val
        applied += 1
    if "monitor_logs" in by_key:
        val = by_key["monitor_logs"] * 100
        infra_ideal.monitor_cost_per_month = val
        infra_economica.monitor_cost_per_month = val
        applied += 1
    if "container_registry" in by_key:
        val = by_key["container_registry"] * 30
        infra_ideal.acr_cost_per_month = val
        infra_economica.acr_cost_per_month = val
        applied += 1

    return applied


def show_azure_prices(prices: AzurePricesResult) -> None:
    import ui
    if not prices.success:
        ui.error(prices.error)
        return
    cols = ["Resource", "Price USD", "Price EUR", "Unit"]
    rows = []
    for it in prices.items:
        rows.append([
            it.label,
            f"{it.unit_price_usd:.4f}",
            f"{it.unit_price_eur:.4f}",
            it.unit,
        ])
    ui.rich_table(cols, rows,
                  title="PRICES FROM AZURE RETAIL PRICES API",
                  col_widths=[36, 14, 14, 14])
