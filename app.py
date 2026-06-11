"""
Streamlit web app for Azure infrastructure cost simulation.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import io, base64, datetime

matplotlib.use("Agg")

from model import LoadProfile, NodepoolConfig, AKSInfrastructure, APIConfig
from simulator import simulate_all
from config import (
    DEFAULT_USERS, DEFAULT_INTERACTIONS_PER_USER_DAY,
    DEFAULT_INPUT_TOKENS, DEFAULT_OUTPUT_TOKENS,
    DEFAULT_WORKING_DAYS, DEFAULT_OFFICE_HOURS,
    DEFAULT_PEAK_HOURS, DEFAULT_CONCURRENT_RATIO,
    DEFAULT_PEAK_MULTIPLIER,
    DEFAULT_SYSTEM_PRICE,
    DEFAULT_IDEAL_GPU_PRICE, DEFAULT_IDEAL_THROUGHPUT,
    DEFAULT_ECO_GPU_PRICE, DEFAULT_ECO_THROUGHPUT,
    DEFAULT_STORAGE_IDEAL, DEFAULT_STORAGE_ECO,
    DEFAULT_LB, DEFAULT_MONITOR_IDEAL, DEFAULT_MONITOR_ECO,
    DEFAULT_ACR_IDEAL, DEFAULT_ACR_ECO,
    DEFAULT_API_MODEL, DEFAULT_API_INPUT_PRICE, DEFAULT_API_OUTPUT_PRICE,
    DEFAULT_EUR_USD, DEFAULT_GPU_UTILIZATION, DEFAULT_SAFETY_FACTOR,
)

LANG = {
    "en": {
    },
    "es": {
        "Quick preset": "Perfil rápido",
        "Auto-fill business parameters.": "Rellena automáticamente los parámetros de negocio.",
        "Load Profile": "Perfil de carga",
        "Users": "Usuarios",
        "Active users interacting with the assistant.": "Usuarios activos que interactúan con el asistente.",
        "Interactions/user/day": "Interacciones/usuario/día",
        "Avg conversations per user per day.": "Media de conversaciones por usuario al día.",
        "Input tokens/interaction": "Tokens entrada/interacción",
        "Avg prompt tokens per interaction.": "Media de tokens de entrada por interacción.",
        "Output tokens/interaction": "Tokens salida/interacción",
        "Avg response tokens per interaction.": "Media de tokens de respuesta por interacción.",
        "Working days/month": "Días laborables/mes",
        "Business days per month.": "Días laborables al mes.",
        "Office hours/day": "Horas oficina/día",
        "Hours of normal operation.": "Horas de operación normal.",
        "Peak hours/day": "Horas pico/día",
        "Hours of concentrated peak demand.": "Horas de demanda pico concentrada.",
        "Concurrent user ratio": "Ratio de usuarios concurrentes",
        "Fraction of total users active simultaneously. 15% = 750 concurrent at 5k users.": "Fracción de usuarios activos simultáneamente. 15% = 750 concurrentes con 5k usuarios.",
        "Peak multiplier": "Multiplicador pico",
        "How many times more concurrent users during peak (2 = double).": "Veces más usuarios concurrentes durante el pico (2 = doble).",
        "Pricing & simulation settings": "Precios y simulación",
        "GPU": "GPU",
        "A100 GPU/hr": "A100 GPU/h",
        "A10 GPU/hr": "A10 GPU/h",
        "API": "API",
        "Model": "Modelo",
        "Input $/1M tok": "Entrada $/1M tok",
        "Output $/1M tok": "Salida $/1M tok",
        "Simulation": "Simulación",
        "MC iterations": "Iteraciones MC",
        "HA factor": "Factor HA",
        "Overhead": "Overhead",
        "Exchange": "Cambio",
        "EUR/USD": "EUR/USD",
        "Machine details & settings": "Detalles de máquina",
        "Sizing": "Dimensionado",
        "System node/hr": "Nodo sistema/h",
        "A100 throughput (tok/s)": "Rendimiento A100 (tok/s)",
        "A10 throughput (tok/s)": "Rendimiento A10 (tok/s)",
        "GPU utilization": "Utilización GPU",
        "Safety factor": "Factor de seguridad",
        "Infrastructure (EUR/mo)": "Infraestructura (EUR/mes)",
        "Ideal storage": "Almacenamiento Ideal",
        "Ideal LB": "LB Ideal",
        "Ideal monitor": "Monitor Ideal",
        "Ideal ACR": "ACR Ideal",
        "Eco storage": "Almacenamiento Eco",
        "Eco LB": "LB Eco",
        "Eco monitor": "Monitor Eco",
        "Eco ACR": "ACR Eco",
        "Running simulation...": "Ejecutando simulación...",
        "WINNER": "GANADOR",
        "Concurrent": "Concurrentes",
        "Tokens/mo": "Tokens/mes",
        "Office/Peak hrs": "Horas oficina/pico",
        "Days/mo": "Días/mes",
        "GPU nodes dominate AKS cost. System, Storage, LB, Monitor, and ACR are fixed overheads independent of traffic.": "Los nodos GPU dominan el coste AKS. Sistema, Almacenamiento, LB, Monitor y ACR son costes fijos independientes del tráfico.",
        "API costs {0}x more than the cheapest AKS option. API has no infrastructure cost but pays per token.": "API cuesta {0}x más que la opción AKS más barata. API no tiene coste de infraestructura pero paga por token.",
        "Detailed comparison": "Comparativa detallada",
        "AKS nodes auto-sized. Total = GPU VM + System VM + Storage + LB + Monitor + ACR + API.": "Nodos AKS auto-dimensionados. Total = GPU VM + System VM + Storage + LB + Monitor + ACR + API.",
        "CSV": "CSV",
        "PDF Report": "Informe PDF",
        "Simulation": "Simulación",
        "Azure Pricing": "Precios Azure",
        "Azure Retail Prices (real-time from API)": "Precios minoristas Azure (tiempo real desde API)",
        "Prices fetched from https://prices.azure.com/api/retail/prices. Region: West Europe. USD converted to EUR at configured rate.": "Precios obtenidos de https://prices.azure.com/api/retail/prices. Región: West Europe. USD convertido a EUR al tipo configurado.",
        "Fetched {0} prices from Azure Retail Prices API": "Obtenidos {0} precios de la API de precios Azure",
        "Azure API unavailable: {0}": "API Azure no disponible: {0}",
        "Azure SKUs being considered": "SKUs Azure consideradas",
        "AKS Infrastructure costs (monthly)": "Costes infraestructura AKS (mensuales)",
        "Azure OpenAI API config": "Configuración API Azure OpenAI",
        "Azure API query details": "Detalles de consulta API Azure",
        "Resource": "Recurso",
        "Price USD": "Precio USD",
        "Price EUR": "Precio EUR",
        "Unit": "Unidad",
        "Source": "Fuente",
        "Scenario Part": "Parte del escenario",
        "VM Type": "Tipo VM",
        "Price/hr (EUR)": "Precio/h (EUR)",
        "Nodes": "Nodos",
        "Scenario": "Escenario",
        "Storage (EUR)": "Almacenamiento (EUR)",
        "Load Balancer (EUR)": "Balanceador (EUR)",
        "Monitor (EUR)": "Monitor (EUR)",
        "ACR (EUR)": "ACR (EUR)",
        "Input $/1M tokens": "Entrada $/1M tokens",
        "Output $/1M tokens": "Salida $/1M tokens",
        "EUR/USD rate": "Tipo EUR/USD",
        "exported on": "exportado el",
        "Language": "Idioma",
        "Select interface language.": "Selecciona el idioma de la interfaz.",
        "Peak Nodes": "Nodos Pico",
        "Total (EUR)": "Total (EUR)",
        "GPU Cost (EUR)": "Coste GPU (EUR)",
        "API LLM (EUR)": "API LLM (EUR)",
        "Annual (EUR)": "Anual (EUR)",
        "API ~0.3k": "API ~0.3k",
        "API ~0.8k": "API ~0.8k",
        "API ~1.5k, Eco ~2.2k": "API ~1.5k, Eco ~2.2k",
        "API ~2.0k, Eco ~2.4k": "API ~2.0k, Eco ~2.4k",
        "API ~2.4k, Eco ~2.5k": "API ~2.4k, Eco ~2.5k",
        "API ~3.0k, Eco ~2.8k": "API ~3.0k, Eco ~2.8k",
        "Eco ~3.8k, API ~4.5k": "Eco ~3.8k, API ~4.5k",
        "Eco ~5.2k, API ~8.0k": "Eco ~5.2k, API ~8.0k",
        "Eco ~6.5k, API ~12k": "Eco ~6.5k, API ~12k",
        "Eco ~8k, Ideal ~9k, API ~15k": "Eco ~8k, Ideal ~9k, API ~15k",
        "Eco ~9.5k, Ideal ~10.5k, API ~20k": "Eco ~9.5k, Ideal ~10.5k, API ~20k",
        "Eco ~11k, Ideal ~12k, API ~25k": "Eco ~11k, Ideal ~12k, API ~25k",
        "Eco ~13k, Ideal ~14k, API ~35k": "Eco ~13k, Ideal ~14k, API ~35k",
        "Eco ~16k, Ideal ~17k, API ~45k": "Eco ~16k, Ideal ~17k, API ~45k",
        "Eco ~35k, Ideal ~36k, API ~80k": "Eco ~35k, Ideal ~36k, API ~80k",
        "Eco ~50k, Ideal ~52k, API ~130k": "Eco ~50k, Ideal ~52k, API ~130k",
        "Eco ~68k, Ideal ~70k, API ~180k": "Eco ~68k, Ideal ~70k, API ~180k",
        "Ideal ~140k, Eco ~145k, API ~260k": "Ideal ~140k, Eco ~145k, API ~260k",
        "Ideal ~320k, Eco ~340k, API ~550k": "Ideal ~320k, Eco ~340k, API ~550k",
        "Ideal ~540k, Eco ~580k, API ~900k": "Ideal ~540k, Eco ~580k, API ~900k",
        "Ideal ~750k, Eco ~810k, API ~1.38M": "Ideal ~750k, Eco ~810k, API ~1.38M",
    },
    "gl": {
        "Azure AKS + LLM Cost Simulator": "Simulador de Custos Azure AKS + LLM",
        "Compare AKS (Ideal/Economy) vs Azure OpenAI API for virtual assistants": "Compara AKS (Ideal/Economy) vs Azure OpenAI API para asistentes virtuais",
        "Quick preset": "Perfil rápido",
        "Auto-fill business parameters.": "Enche automaticamente os parámetros de negocio.",
        "Load Profile": "Perfil de carga",
        "Users": "Usuarios",
        "Active users interacting with the assistant.": "Usuarios activos que interactúan co asistente.",
        "Interactions/user/day": "Interaccións/usuario/día",
        "Avg conversations per user per day.": "Media de conversas por usuario ao día.",
        "Input tokens/interaction": "Tokens entrada/interacción",
        "Avg prompt tokens per interaction.": "Media de tokens de entrada por interacción.",
        "Output tokens/interaction": "Tokens saída/interacción",
        "Avg response tokens per interaction.": "Media de tokens de resposta por interacción.",
        "Working days/month": "Días laborables/mes",
        "Business days per month.": "Días laborables ao mes.",
        "Office hours/day": "Horas oficina/día",
        "Hours of normal operation.": "Horas de operación normal.",
        "Peak hours/day": "Horas pico/día",
        "Hours of concentrated peak demand.": "Horas de demanda pico concentrada.",
        "Concurrent user ratio": "Ratio de usuarios concurrentes",
        "Fraction of total users active simultaneously. 15% = 750 concurrent at 5k users.": "Fracción de usuarios activos simultaneamente. 15% = 750 concurrentes con 5k usuarios.",
        "Peak multiplier": "Multiplicador pico",
        "How many times more concurrent users during peak (2 = double).": "Veces máis usuarios concurrentes durante o pico (2 = dobre).",
        "Pricing & simulation settings": "Prezos e simulación",
        "GPU": "GPU",
        "A100 GPU/hr": "A100 GPU/h",
        "A10 GPU/hr": "A10 GPU/h",
        "API": "API",
        "Model": "Modelo",
        "Input $/1M tok": "Entrada $/1M tok",
        "Output $/1M tok": "Saída $/1M tok",
        "Simulation": "Simulación",
        "MC iterations": "Iteracións MC",
        "HA factor": "Factor HA",
        "Overhead": "Overhead",
        "Exchange": "Cambio",
        "EUR/USD": "EUR/USD",
        "Machine details & settings": "Detalles de máquina",
        "Sizing": "Dimensionado",
        "System node/hr": "Nodo sistema/h",
        "A100 throughput (tok/s)": "Rendemento A100 (tok/s)",
        "A10 throughput (tok/s)": "Rendemento A10 (tok/s)",
        "GPU utilization": "Utilización GPU",
        "Safety factor": "Factor de seguridade",
        "Infrastructure (EUR/mo)": "Infraestrutura (EUR/mes)",
        "Ideal storage": "Almacenamento Ideal",
        "Ideal LB": "LB Ideal",
        "Ideal monitor": "Monitor Ideal",
        "Ideal ACR": "ACR Ideal",
        "Eco storage": "Almacenamento Eco",
        "Eco LB": "LB Eco",
        "Eco monitor": "Monitor Eco",
        "Eco ACR": "ACR Eco",
        "Running simulation...": "Executando simulación...",
        "WINNER": "GAÑADOR",
        "Concurrent": "Concurrentes",
        "Tokens/mo": "Tokens/mes",
        "Office/Peak hrs": "Horas oficina/pico",
        "Days/mo": "Días/mes",
        "GPU nodes dominate AKS cost. System, Storage, LB, Monitor, and ACR are fixed overheads independent of traffic.": "Os nodos GPU dominan o custo AKS. Sistema, Almacenamento, LB, Monitor e ACR son custos fixos independentes do tráfico.",
        "API costs {0}x more than the cheapest AKS option. API has no infrastructure cost but pays per token.": "API custa {0}x máis que a opción AKS máis barata. API non ten custo de infraestrutura pero paga por token.",
        "Detailed comparison": "Comparativa detallada",
        "AKS nodes auto-sized. Total = GPU VM + System VM + Storage + LB + Monitor + ACR + API.": "Nodos AKS auto-dimensionados. Total = GPU VM + System VM + Storage + LB + Monitor + ACR + API.",
        "CSV": "CSV",
        "PDF Report": "Informe PDF",
        "Simulation": "Simulación",
        "Azure Pricing": "Prezos Azure",
        "Azure Retail Prices (real-time from API)": "Prezos minoristas Azure (tempo real desde API)",
        "Prices fetched from https://prices.azure.com/api/retail/prices. Region: West Europe. USD converted to EUR at configured rate.": "Prezos obtidos de https://prices.azure.com/api/retail/prices. Rexión: West Europe. USD convertido a EUR ao tipo configurado.",
        "Fetched {0} prices from Azure Retail Prices API": "Obtidos {0} prezos da API de prezos Azure",
        "Azure API unavailable: {0}": "API Azure non dispoñible: {0}",
        "Azure SKUs being considered": "SKUs Azure consideradas",
        "AKS Infrastructure costs (monthly)": "Custos infraestrutura AKS (mensuais)",
        "Azure OpenAI API config": "Configuración API Azure OpenAI",
        "Azure API query details": "Detalles de consulta API Azure",
        "Resource": "Recurso",
        "Price USD": "Prezo USD",
        "Price EUR": "Prezo EUR",
        "Unit": "Unidade",
        "Source": "Fonte",
        "Scenario Part": "Parte do escenario",
        "VM Type": "Tipo VM",
        "Price/hr (EUR)": "Prezo/h (EUR)",
        "Nodes": "Nodos",
        "Scenario": "Escenario",
        "Storage (EUR)": "Almacenamento (EUR)",
        "Load Balancer (EUR)": "Balanceador (EUR)",
        "Monitor (EUR)": "Monitor (EUR)",
        "ACR (EUR)": "ACR (EUR)",
        "Input $/1M tokens": "Entrada $/1M tokens",
        "Output $/1M tokens": "Saída $/1M tokens",
        "EUR/USD rate": "Tipo EUR/USD",
        "exported on": "exportado o",
        "Language": "Idioma",
        "Select interface language.": "Selecciona o idioma da interface.",
        "Peak Nodes": "Nodos Pico",
        "Total (EUR)": "Total (EUR)",
        "GPU Cost (EUR)": "Custo GPU (EUR)",
        "API LLM (EUR)": "API LLM (EUR)",
        "Annual (EUR)": "Anual (EUR)",
        "API ~0.3k": "API ~0.3k",
        "API ~0.8k": "API ~0.8k",
        "API ~1.5k, Eco ~2.2k": "API ~1.5k, Eco ~2.2k",
        "API ~2.0k, Eco ~2.4k": "API ~2.0k, Eco ~2.4k",
        "API ~2.4k, Eco ~2.5k": "API ~2.4k, Eco ~2.5k",
        "API ~3.0k, Eco ~2.8k": "API ~3.0k, Eco ~2.8k",
        "Eco ~3.8k, API ~4.5k": "Eco ~3.8k, API ~4.5k",
        "Eco ~5.2k, API ~8.0k": "Eco ~5.2k, API ~8.0k",
        "Eco ~6.5k, API ~12k": "Eco ~6.5k, API ~12k",
        "Eco ~8k, Ideal ~9k, API ~15k": "Eco ~8k, Ideal ~9k, API ~15k",
        "Eco ~9.5k, Ideal ~10.5k, API ~20k": "Eco ~9.5k, Ideal ~10.5k, API ~20k",
        "Eco ~11k, Ideal ~12k, API ~25k": "Eco ~11k, Ideal ~12k, API ~25k",
        "Eco ~13k, Ideal ~14k, API ~35k": "Eco ~13k, Ideal ~14k, API ~35k",
        "Eco ~16k, Ideal ~17k, API ~45k": "Eco ~16k, Ideal ~17k, API ~45k",
        "Eco ~35k, Ideal ~36k, API ~80k": "Eco ~35k, Ideal ~36k, API ~80k",
        "Eco ~50k, Ideal ~52k, API ~130k": "Eco ~50k, Ideal ~52k, API ~130k",
        "Eco ~68k, Ideal ~70k, API ~180k": "Eco ~68k, Ideal ~70k, API ~180k",
        "Ideal ~140k, Eco ~145k, API ~260k": "Ideal ~140k, Eco ~145k, API ~260k",
        "Ideal ~320k, Eco ~340k, API ~550k": "Ideal ~320k, Eco ~340k, API ~550k",
        "Ideal ~540k, Eco ~580k, API ~900k": "Ideal ~540k, Eco ~580k, API ~900k",
        "Ideal ~750k, Eco ~810k, API ~1.38M": "Ideal ~750k, Eco ~810k, API ~1.38M",
    },
}


def _(text):
    raw = st.session_state.get("lang", "en") if hasattr(st, "session_state") and st.session_state else "en"
    lang = raw.lower()
    return LANG.get(lang, {}).get(text, text)


st.set_page_config(
    page_title="Azure AKS + LLM Cost Simulator",
    page_icon="\u2601",
    layout="wide",
    initial_sidebar_state="expanded",
)

COMPACT_CSS = """
<style>
    .main > div { padding: 0rem 1rem; }
    .stButton > button { font-size: 0.8rem; padding: 0.2rem 0.8rem; }
    .stMetric { font-size: 0.8rem; }
    .stMetric label { font-size: 0.75rem !important; }
    .stMetric .metric-value { font-size: 1.1rem !important; }
    .stHeader { font-size: 1rem !important; }
    .stSubheader { font-size: 0.9rem !important; }
    .stCaption { font-size: 0.75rem !important; }
    .stMarkdown { font-size: 0.8rem; }
    .stDataFrame { font-size: 0.75rem; }
    .row-widget.stNumberInput input { font-size: 0.8rem; padding: 0.2rem; }
    section[data-testid="stSidebar"] .stNumberInput input { font-size: 0.75rem; padding: 0.15rem; }
    section[data-testid="stSidebar"] .stHeader { font-size: 0.85rem !important; }
    section[data-testid="stSidebar"] label { font-size: 0.7rem !important; }
    div[data-testid="stExpander"] { font-size: 0.8rem; }
    div.stTabs button { font-size: 0.8rem; padding: 0.3rem 0.8rem; }
</style>
"""
st.markdown(COMPACT_CSS, unsafe_allow_html=True)


def build_data_from_ui() -> dict:
    lp = LoadProfile(
        users=st.session_state.get("users", DEFAULT_USERS),
        interactions_per_user_day=st.session_state.get("interactions_per_user_day", DEFAULT_INTERACTIONS_PER_USER_DAY),
        input_tokens_per_interaction=st.session_state.get("input_tokens_per_interaction", DEFAULT_INPUT_TOKENS),
        output_tokens_per_interaction=st.session_state.get("output_tokens_per_interaction", DEFAULT_OUTPUT_TOKENS),
        working_days_per_month=st.session_state.get("working_days_per_month", DEFAULT_WORKING_DAYS),
        office_hours_per_day=st.session_state.get("office_hours_per_day", DEFAULT_OFFICE_HOURS),
        peak_hours_per_day=st.session_state.get("peak_hours_per_day", DEFAULT_PEAK_HOURS),
        concurrent_user_ratio=st.session_state.get("concurrent_user_ratio", DEFAULT_CONCURRENT_RATIO),
        peak_multiplier=st.session_state.get("peak_multiplier", DEFAULT_PEAK_MULTIPLIER),
    )

    infra_ideal = AKSInfrastructure(name="LLM on AKS (Ideal UX)")
    infra_ideal.system_nodepool = NodepoolConfig(
        vm_type="Standard_D8ds_v5", base_office_nodes=1,
        price_per_hour=st.session_state.get("system_price", DEFAULT_SYSTEM_PRICE),
    )
    infra_ideal.inference_nodepool = NodepoolConfig(
        vm_type="Standard_NC24ads_A100_v4",
        base_office_nodes=3, peak_nodes=10, off_hours_nodes=1,
        price_per_hour=st.session_state.get("ideal_gpu_price", DEFAULT_IDEAL_GPU_PRICE),
    )
    infra_ideal.throughput_tok_s_per_pod = st.session_state.get("ideal_throughput", DEFAULT_IDEAL_THROUGHPUT)
    infra_ideal.gpu_utilization = st.session_state.get("gpu_utilization", DEFAULT_GPU_UTILIZATION)
    infra_ideal.safety_factor = st.session_state.get("safety_factor", DEFAULT_SAFETY_FACTOR)
    infra_ideal.base_replicas = 3
    infra_ideal.peak_replicas = 10
    infra_ideal.off_hours_replicas = 1
    infra_ideal.storage_cost_per_month = st.session_state.get("ideal_storage", DEFAULT_STORAGE_IDEAL)
    infra_ideal.lb_cost_per_month = st.session_state.get("ideal_lb", DEFAULT_LB)
    infra_ideal.monitor_cost_per_month = st.session_state.get("ideal_monitor", DEFAULT_MONITOR_IDEAL)
    infra_ideal.acr_cost_per_month = st.session_state.get("ideal_acr", DEFAULT_ACR_IDEAL)

    infra_economy = AKSInfrastructure(name="LLM on AKS (Economy UX)")
    infra_economy.system_nodepool = NodepoolConfig(
        vm_type="Standard_D8ds_v5", base_office_nodes=1,
        price_per_hour=st.session_state.get("system_price", DEFAULT_SYSTEM_PRICE),
    )
    infra_economy.inference_nodepool = NodepoolConfig(
        vm_type="Standard_NV12ads_A10_v5",
        base_office_nodes=5, peak_nodes=20, off_hours_nodes=1,
        price_per_hour=st.session_state.get("eco_gpu_price", DEFAULT_ECO_GPU_PRICE),
    )
    infra_economy.throughput_tok_s_per_pod = st.session_state.get("eco_throughput", DEFAULT_ECO_THROUGHPUT)
    infra_economy.gpu_utilization = st.session_state.get("gpu_utilization", DEFAULT_GPU_UTILIZATION)
    infra_economy.safety_factor = st.session_state.get("safety_factor", DEFAULT_SAFETY_FACTOR)
    infra_economy.base_replicas = 5
    infra_economy.peak_replicas = 20
    infra_economy.off_hours_replicas = 1
    infra_economy.storage_cost_per_month = st.session_state.get("eco_storage", DEFAULT_STORAGE_ECO)
    infra_economy.lb_cost_per_month = st.session_state.get("eco_lb", DEFAULT_LB)
    infra_economy.monitor_cost_per_month = st.session_state.get("eco_monitor", DEFAULT_MONITOR_ECO)
    infra_economy.acr_cost_per_month = st.session_state.get("eco_acr", DEFAULT_ACR_ECO)

    api = APIConfig(
        name="API Azure OpenAI",
        model=st.session_state.get("api_model", DEFAULT_API_MODEL),
        input_price_per_1m_tokens_usd=st.session_state.get("api_input_price", DEFAULT_API_INPUT_PRICE),
        output_price_per_1m_tokens_usd=st.session_state.get("api_output_price", DEFAULT_API_OUTPUT_PRICE),
        eur_usd_rate=st.session_state.get("eur_usd_rate", DEFAULT_EUR_USD),
    )

    return {
        "load_profile": lp,
        "infra_ideal": infra_ideal,
        "infra_economica": infra_economy,
        "api_config": api,
        "comparativa": pd.DataFrame(),
    }


PRESETS = {
    "100": {"users": 100, "interactions_per_user_day": 8, "input_tokens_per_interaction": 400, "output_tokens_per_interaction": 150, "working_days_per_month": 22, "office_hours_per_day": 8, "peak_hours_per_day": 0.25, "concurrent_user_ratio": 0.30, "peak_multiplier": 1.5, "desc": "dept. pequeño"},
    "200": {"users": 200, "interactions_per_user_day": 10, "input_tokens_per_interaction": 500, "output_tokens_per_interaction": 200, "working_days_per_month": 22, "office_hours_per_day": 8, "peak_hours_per_day": 0.25, "concurrent_user_ratio": 0.25, "peak_multiplier": 1.5, "desc": "dept. mediano"},
    "500": {"users": 500, "interactions_per_user_day": 12, "input_tokens_per_interaction": 600, "output_tokens_per_interaction": 250, "working_days_per_month": 22, "office_hours_per_day": 9, "peak_hours_per_day": 0.5, "concurrent_user_ratio": 0.20, "peak_multiplier": 1.5, "desc": "división pequeña"},
    "750": {"users": 750, "interactions_per_user_day": 15, "input_tokens_per_interaction": 700, "output_tokens_per_interaction": 300, "working_days_per_month": 22, "office_hours_per_day": 9, "peak_hours_per_day": 0.5, "concurrent_user_ratio": 0.18, "peak_multiplier": 1.5, "desc": "división mediana"},
    "1k": {"users": 1000, "interactions_per_user_day": 20, "input_tokens_per_interaction": 1000, "output_tokens_per_interaction": 400, "working_days_per_month": 22, "office_hours_per_day": 9, "peak_hours_per_day": 0.5, "concurrent_user_ratio": 0.18, "peak_multiplier": 1.5, "desc": "división grande"},
    "2k": {"users": 2000, "interactions_per_user_day": 25, "input_tokens_per_interaction": 1200, "output_tokens_per_interaction": 400, "working_days_per_month": 22, "office_hours_per_day": 10, "peak_hours_per_day": 1.0, "concurrent_user_ratio": 0.15, "peak_multiplier": 2.0, "desc": "organización pequeña"},
    "3k": {"users": 3000, "interactions_per_user_day": 30, "input_tokens_per_interaction": 1500, "output_tokens_per_interaction": 400, "working_days_per_month": 22, "office_hours_per_day": 10, "peak_hours_per_day": 1.0, "concurrent_user_ratio": 0.15, "peak_multiplier": 2.0, "desc": "organización mediana"},
    "6k": {"users": 6000, "interactions_per_user_day": 30, "input_tokens_per_interaction": 2000, "output_tokens_per_interaction": 400, "working_days_per_month": 22, "office_hours_per_day": 10, "peak_hours_per_day": 2.0, "concurrent_user_ratio": 0.15, "peak_multiplier": 2.0, "desc": "organización grande (por defecto)"},
    "10k": {"users": 10000, "interactions_per_user_day": 35, "input_tokens_per_interaction": 2000, "output_tokens_per_interaction": 500, "working_days_per_month": 22, "office_hours_per_day": 11, "peak_hours_per_day": 2.0, "concurrent_user_ratio": 0.15, "peak_multiplier": 2.0, "desc": "empresa"},
    "20k": {"users": 20000, "interactions_per_user_day": 40, "input_tokens_per_interaction": 2500, "output_tokens_per_interaction": 500, "working_days_per_month": 22, "office_hours_per_day": 12, "peak_hours_per_day": 2.0, "concurrent_user_ratio": 0.18, "peak_multiplier": 2.5, "desc": "corporación"},
    "35k": {"users": 35000, "interactions_per_user_day": 45, "input_tokens_per_interaction": 2500, "output_tokens_per_interaction": 600, "working_days_per_month": 22, "office_hours_per_day": 12, "peak_hours_per_day": 3.0, "concurrent_user_ratio": 0.20, "peak_multiplier": 2.5, "desc": "gran corporación"},
}

PRESET_KEYS = list(PRESETS.keys())


def apply_preset(name: str):
    p = PRESETS.get(name)
    if p:
        for k, v in p.items():
            if k != "desc":
                st.session_state[k] = v
                st.session_state[f"{k}_s"] = v
                st.session_state[f"{k}_n"] = v


def _sync_s_to_n(key):
    st.session_state[f"{key}_n"] = st.session_state[f"{key}_s"]
    st.session_state[key] = st.session_state[f"{key}_s"]


def _sync_n_to_s(key):
    st.session_state[f"{key}_s"] = st.session_state[f"{key}_n"]
    st.session_state[key] = st.session_state[f"{key}_n"]


def slider_preciso(label, min_value, max_value, default, step, key, fmt="%d", fmt_n=None,
                    help=None, label_visibility="visible"):
    if fmt_n is None:
        fmt_n = fmt
    s_key = f"{key}_s"
    n_key = f"{key}_n"
    if s_key not in st.session_state:
        st.session_state[s_key] = default
    if n_key not in st.session_state:
        st.session_state[n_key] = default

    col1, col2 = st.columns([4, 1])
    with col1:
        st.slider(
            label, min_value=min_value, max_value=max_value,
            value=st.session_state[s_key], step=step, key=s_key,
            format=fmt, help=help, label_visibility=label_visibility,
            on_change=lambda k=key: _sync_s_to_n(k),
        )
    with col2:
        st.number_input(
            label if label_visibility == "visible" else "",
            min_value=min_value, max_value=max_value,
            value=st.session_state[n_key], step=step, key=n_key,
            format=fmt_n, label_visibility="collapsed",
            on_change=lambda k=key: _sync_n_to_s(k),
        )


def render_sidebar():
    with st.sidebar:
        st.markdown(_("**Load Profile**"))

        slider_preciso(_("Users"), 1, 500000, 5000, 1, "users",
                       help=_("Active users interacting with the assistant."))
        slider_preciso(_("Interactions/user/day"), 1, 200, 40, 1, "interactions_per_user_day",
                       help=_("Avg conversations per user per day."))
        slider_preciso(_("Input tokens/interaction"), 100, 16000, 1500, 100, "input_tokens_per_interaction",
                       help=_("Avg prompt tokens per interaction."))
        slider_preciso(_("Output tokens/interaction"), 50, 8000, 500, 50, "output_tokens_per_interaction",
                       help=_("Avg response tokens per interaction."))
        slider_preciso(_("Working days/month"), 1, 31, 22, 1, "working_days_per_month",
                       help=_("Business days per month."))
        slider_preciso(_("Office hours/day"), 1, 24, 10, 1, "office_hours_per_day",
                       help=_("Hours of normal operation."))
        slider_preciso(_("Peak hours/day"), 0.1, 8.0, 0.5, 0.1, "peak_hours_per_day", fmt="%.1f",
                       help=_("Hours of concentrated peak demand."))
        slider_preciso(_("Concurrent user ratio"), 0.01, 1.0, 0.15, 0.01, "concurrent_user_ratio", fmt="%.0f%%", fmt_n="%.2f",
                       help=_("Fraction of total users active simultaneously. 15% = 750 concurrent at 5k users."))
        slider_preciso(_("Peak multiplier"), 1.0, 5.0, 2.0, 0.5, "peak_multiplier", fmt="%.1f",
                       help=_("How many times more concurrent users during peak (2 = double)."))


def fmt_eur(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M EUR"
    return f"{v:,.0f} EUR"


def fmt_period(v: float) -> str:
    yr = v * 12
    if yr >= 1_000_000:
        return f"{fmt_eur(v)}/mo ({yr/1_000_000:.2f}M/yr)"
    return f"{fmt_eur(v)}/mo ({yr:,.0f}/yr)"


def render_controls():
    with st.expander(_("Pricing & simulation settings"), expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(_("_GPU_"))
            st.number_input(_("A100 GPU/hr"), min_value=0.01, value=6.0, key="ideal_gpu_price", step=0.1, format="%.2f", label_visibility="collapsed")
            st.number_input(_("A10 GPU/hr"), min_value=0.01, value=2.2, key="eco_gpu_price", step=0.1, format="%.2f", label_visibility="collapsed")
        with col2:
            st.markdown(_("_API_"))
            st.text_input(_("Model"), value="gpt-4o", key="api_model", label_visibility="collapsed")
            st.number_input(_("Input $/1M tok"), min_value=0.0, value=2.5, key="api_input_price", step=0.1, format="%.2f", label_visibility="collapsed")
            st.number_input(_("Output $/1M tok"), min_value=0.0, value=10.0, key="api_output_price", step=0.1, format="%.2f", label_visibility="collapsed")
        with col3:
            st.markdown(_("_Simulation_"))
            st.number_input(_("MC iterations"), min_value=100, value=5000, key="mc_iterations", step=100, label_visibility="collapsed")
            st.number_input(_("HA factor"), min_value=1.0, max_value=2.0, value=1.15, key="ha_factor", step=0.01, format="%.2f", label_visibility="collapsed")
            st.number_input(_("Overhead"), min_value=0.0, max_value=1.0, value=0.10, key="overhead_factor", step=0.05, format="%.2f", label_visibility="collapsed")
        with col4:
            st.markdown(_("_Exchange_"))
            st.number_input(_("EUR/USD"), min_value=0.5, max_value=2.0, value=0.92, key="eur_usd_rate", step=0.01, format="%.2f", label_visibility="collapsed")

        with st.expander(_("Machine details & settings")):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(_("_Sizing_"))
                st.number_input(_("System node/hr"), min_value=0.01, value=0.8, key="system_price", step=0.01, format="%.2f", label_visibility="collapsed")
                st.number_input(_("A100 throughput (tok/s)"), min_value=1, value=350, key="ideal_throughput", step=10, label_visibility="collapsed")
                st.number_input(_("A10 throughput (tok/s)"), min_value=1, value=120, key="eco_throughput", step=10, label_visibility="collapsed")
                st.number_input(_("GPU utilization"), min_value=0.0, max_value=1.0, value=0.75, key="gpu_utilization", step=0.05, label_visibility="collapsed")
                st.number_input(_("Safety factor"), min_value=0.0, max_value=2.0, value=1.0, key="safety_factor", step=0.05, label_visibility="collapsed")
            with c2:
                st.markdown(_("_Infrastructure (EUR/mo)_"))
                st.number_input(_("Ideal storage"), min_value=0, value=200, key="ideal_storage", step=10, label_visibility="collapsed")
                st.number_input(_("Ideal LB"), min_value=0, value=60, key="ideal_lb", step=5, label_visibility="collapsed")
                st.number_input(_("Ideal monitor"), min_value=0, value=150, key="ideal_monitor", step=10, label_visibility="collapsed")
                st.number_input(_("Ideal ACR"), min_value=0, value=140, key="ideal_acr", step=10, label_visibility="collapsed")
                st.number_input(_("Eco storage"), min_value=0, value=140, key="eco_storage", step=10, label_visibility="collapsed")
                st.number_input(_("Eco LB"), min_value=0, value=60, key="eco_lb", step=5, label_visibility="collapsed")
                st.number_input(_("Eco monitor"), min_value=0, value=120, key="eco_monitor", step=10, label_visibility="collapsed")
                st.number_input(_("Eco ACR"), min_value=0, value=100, key="eco_acr", step=10, label_visibility="collapsed")


def generate_report_html(lp, df, data, mc_iterations, ha_factor, overhead_factor, df_sorted):
    buf_left_eco = io.BytesIO()
    buf_left_ideal = io.BytesIO()
    buf_right = io.BytesIO()

    categories = ["System", "GPU", "Storage", "LB", "Monitor", "ACR"]
    for sc_filter, sc_color, buf in [
        ("Economico", "#27ae60", buf_left_eco), ("Ideal", "#2c6b9e", buf_left_ideal),
    ]:
        row = df[df["scenario"].str.contains(sc_filter, case=False)].iloc[0]
        vals = [row.get(c, 0) for c in ["aks_system_cost_eur", "aks_gpu_cost_eur",
                 "storage_cost_eur", "lb_cost_eur", "monitor_cost_eur", "acr_cost_eur"]]
        fig, ax = plt.subplots(figsize=(5, 2.2), facecolor="#fafafa")
        ax.bar(categories, vals, color=sc_color, width=0.6, edgecolor="white")
        for bar, val in zip(ax.patches, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:,.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(row["scenario"], fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=200)
        plt.close(fig)
        buf.seek(0)

    names = df["scenario"].tolist()
    totals = df["total_cost_eur"].tolist()
    bar_colors = ["#2c6b9e", "#27ae60", "#d35400"]
    fig2, ax2 = plt.subplots(figsize=(6, 3), facecolor="#fafafa")
    ax2.bar(names, totals, color=bar_colors, width=0.5, edgecolor="white")
    for bar, val in zip(ax2.patches, totals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val:,.0f} EUR", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)
    ax2.set_ylim(0, max(totals) * 1.2)
    plt.tight_layout()
    fig2.savefig(buf_right, format="png", dpi=200)
    plt.close(fig2)
    buf_right.seek(0)

    def img_b64(buf):
        return base64.b64encode(buf.read()).decode("utf-8")

    img_left_eco_b64 = img_b64(buf_left_eco)
    img_left_ideal_b64 = img_b64(buf_left_ideal)
    img_right_b64 = img_b64(buf_right)

    best_row = df_sorted.iloc[0]
    second_row = df_sorted.iloc[1]
    third_row = df_sorted.iloc[2]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lang_label = st.session_state.get("lang", "en").lower()
    t = lambda s: LANG.get(lang_label, {}).get(s, s)

    html = f"""<!DOCTYPE html>
<html lang="{lang_label}">
<head><meta charset="utf-8"><title>{t("Azure AKS + LLM Cost Simulator")}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 15px; color: #222; }}
h1 {{ color: #1557a0; font-size: 1.1rem; margin: 6px 0; }}
h2 {{ color: #333; font-size: 0.9rem; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin: 6px 0; }}
table {{ border-collapse: collapse; margin: 4px 0; font-size: 0.7rem; }}
th, td {{ border: 1px solid #ccc; padding: 1px 4px; text-align: left; }}
th {{ background: #1557a0; color: #fff; font-weight: 600; }}
tr:nth-child(even) {{ background: #f7f9fc; }}
tr:nth-child(odd) {{ background: #fff; }}
tr:hover {{ background: #e8f0fe; }}
.winner {{ background: #d4edda; border: 2px solid #1a6b3c; border-radius: 4px; padding: 6px 10px; text-align: center; font-size: 0.9rem; font-weight: bold; color: #1a6b3c; margin: 6px 0; }}
.caption {{ font-size: 0.65rem; color: #666; margin-top: 2px; }}
.chart {{ margin: 4px 0; }}
.footer {{ font-size: 0.6rem; color: #999; margin-top: 10px; text-align: center; }}
</style></head>
<body>
<h1>{t("Azure AKS + LLM Cost Simulator")}</h1>
<p>{t("exported on")}: {now_str}</p>

<div class="winner">{t("WINNER")}: {best_row["scenario"]} — {fmt_period(best_row["total_cost_eur"])}</div>
<p>{second_row["scenario"]}: {fmt_period(second_row["total_cost_eur"])} | {third_row["scenario"]}: {fmt_period(third_row["total_cost_eur"])}</p>

<h2>{t("Load Profile")}</h2>
<table>
<tr><td>{t("Users")}</td><td>{lp.users:,}</td></tr>
<tr><td>{t("Interactions/user/day")}</td><td>{lp.interactions_per_user_day}</td></tr>
<tr><td>{t("Input tokens/interaction")}</td><td>{lp.input_tokens_per_interaction}</td></tr>
<tr><td>{t("Output tokens/interaction")}</td><td>{lp.output_tokens_per_interaction}</td></tr>
<tr><td>{t("Working days/month")}</td><td>{lp.working_days_per_month}</td></tr>
<tr><td>{t("Office hours/day")}</td><td>{lp.office_hours_per_day}</td></tr>
<tr><td>{t("Peak hours/day")}</td><td>{lp.peak_hours_per_day}</td></tr>
<tr><td>{t("Concurrent user ratio")}</td><td>{lp.concurrent_user_ratio:.0%}</td></tr>
<tr><td>{t("Peak multiplier")}</td><td>{lp.peak_multiplier}</td></tr>
</table>

<h2>{t("Machine details & settings")}</h2>
<table>
<tr><th>{t("GPU")}</th><th>{t("API")}</th></tr>
<tr><td>A100 GPU/hr: {st.session_state.get("ideal_gpu_price", 6.0):.2f} EUR</td><td>{t("Model")}: {st.session_state.get("api_model", "gpt-4o")}</td></tr>
<tr><td>A10 GPU/hr: {st.session_state.get("eco_gpu_price", 2.2):.2f} EUR</td><td>{t("Input $/1M tok")}: {st.session_state.get("api_input_price", 2.5):.2f}</td></tr>
<tr><td>{t("System node/hr")}: {st.session_state.get("system_price", 0.8):.2f} EUR</td><td>{t("Output $/1M tok")}: {st.session_state.get("api_output_price", 10.0):.2f}</td></tr>
<tr><td>{t("A100 throughput (tok/s)")}: {st.session_state.get("ideal_throughput", 350)}</td><td>{t("EUR/USD rate")}: {st.session_state.get("eur_usd_rate", 0.92)}</td></tr>
<tr><td>{t("A10 throughput (tok/s)")}: {st.session_state.get("eco_throughput", 120)}</td><td></td></tr>
</table>

<h2>{t("Costs")}</h2>
<div class="chart"><img src="data:image/png;base64,{img_left_eco_b64}" style="width:45%;display:inline-block" />
<img src="data:image/png;base64,{img_left_ideal_b64}" style="width:45%;display:inline-block" /></div>
<div class="chart"><img src="data:image/png;base64,{img_right_b64}" style="width:70%" /></div>

<table>
<tr><th>{t("Scenario")}</th><th>{t("Total (EUR)")}</th><th>Annual (EUR)</th><th>{t("GPU Cost (EUR)")}</th><th>{t("API LLM (EUR)")}</th><th>{t("Peak Nodes")}</th></tr>"""
    for _, row in df.iterrows():
        ann = row['total_cost_eur'] * 12
        html += f"<tr><td>{row['scenario']}</td><td>{row['total_cost_eur']:,.0f}</td><td>{ann:,.0f}</td><td>{row['aks_gpu_cost_eur']:,.0f}</td><td>{row['api_llm_cost_eur']:,.0f}</td><td>{row['gpu_peak_nodes']}</td></tr>"
    html += """</table>

<div class="footer">Azure AKS + LLM Cost Simulator</div>
</body></html>"""
    return html


def tab_simulation(data: dict):
    lp = data["load_profile"]
    mc_iterations = st.session_state.get("mc_iterations", 5000)
    ha_factor = st.session_state.get("ha_factor", 1.15)
    overhead_factor = st.session_state.get("overhead_factor", 0.1)

    cols = st.columns(5)
    with cols[0]:
        st.metric(_("Users"), f"{lp.users:,}")
    with cols[1]:
        st.metric(_("Concurrent"), f"{int(lp.users * lp.concurrent_user_ratio):,}")
    with cols[2]:
        st.metric(_("Tokens/mo"), f"{lp.total_tokens_per_month:,}")
    with cols[3]:
        st.metric(_("Office/Peak hrs"), f"{lp.office_hours_per_day}h / {lp.peak_hours_per_day}h")
    with cols[4]:
        st.metric(_("Days/mo"), str(lp.working_days_per_month))

    with st.spinner(_("Running simulation...")):
        df = simulate_all(
            data,
            mc_iterations=mc_iterations,
            ha_factor=ha_factor,
            overhead_factor=overhead_factor,
            resize=True,
        )

    df_sorted = df.sort_values("total_cost_eur")
    best_row = df_sorted.iloc[0]
    second_row = df_sorted.iloc[1]
    third_row = df_sorted.iloc[2]

    c_win, c_rest = st.columns(2)
    with c_win:
        st.success(
            f"### {_('WINNER')}: {best_row['scenario']}\n\n"
            f"### {fmt_period(best_row['total_cost_eur'])}"
        )
    with c_rest:
        st.info(
            f"{second_row['scenario']}: {fmt_period(second_row['total_cost_eur'])}  \n"
            f"{third_row['scenario']}: {fmt_period(third_row['total_cost_eur'])}"
        )

    col_left, col_right = st.columns([1, 1.6])
    with col_left:
        categories = ["System", "GPU", "Storage", "LB", "Monitor", "ACR"]
        for sc_name, sc_filter, sc_color in [("Economico", "Economico", "#27ae60"), ("Ideal", "Ideal", "#2c6b9e")]:
            row = df[df["scenario"].str.contains(sc_filter, case=False)].iloc[0]
            vals = [
                row.get("aks_system_cost_eur", 0), row.get("aks_gpu_cost_eur", 0),
                row.get("storage_cost_eur", 0), row.get("lb_cost_eur", 0),
                row.get("monitor_cost_eur", 0), row.get("acr_cost_eur", 0),
            ]
            fig, ax = plt.subplots(figsize=(2.5, 0.9), facecolor="#fafafa")
            bars = ax.bar(categories, vals, color=sc_color, width=0.6, edgecolor="white", linewidth=0.3)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f"{val:,.0f}", ha="center", va="bottom", fontsize=2.5, fontweight="bold")
            ax.set_title(row["scenario"], fontsize=4.5, fontweight="bold")
            ax.tick_params(axis="x", labelsize=3, pad=1)
            ax.tick_params(axis="y", labelsize=3, pad=1)
            ax.grid(axis="y", alpha=0.1, linewidth=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.caption(
            _("GPU nodes dominate AKS cost. System, Storage, LB, Monitor, and ACR are fixed overheads independent of traffic.")
        )

    with col_right:
        names = df["scenario"].tolist()
        totals = df["total_cost_eur"].tolist()
        bar_colors = ["#2c6b9e", "#27ae60", "#d35400"]
        fig2, ax2 = plt.subplots(figsize=(3.5, 1.5), facecolor="#fafafa")
        bars = ax2.bar(names, totals, color=bar_colors, width=0.5, edgecolor="white", linewidth=0.4)
        for bar, val in zip(bars, totals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:,.0f}/mo\n({val*12:,.0f}/yr)", ha="center", va="bottom", fontsize=4.5, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(axis="x", labelsize=4, pad=1)
        ax2.tick_params(axis="y", labelsize=4, pad=1)
        ax2.grid(axis="y", alpha=0.1, linewidth=0.3)
        ax2.set_ylim(0, max(totals) * 1.2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        ratio_api = totals[2] / totals[0] if totals[0] > 0 else 0
        st.caption(
            _("API costs {0}x more than the cheapest AKS option. API has no infrastructure cost but pays per token.").format(f"{ratio_api:.1f}")
        )

    with st.expander(_("Detailed comparison")):
        st.caption(
            _("AKS nodes auto-sized. Total = GPU VM + System VM + Storage + LB + Monitor + ACR + API.")
        )
        display_cols = [
            "scenario", "total_cost_eur", "aks_gpu_cost_eur", "api_llm_cost_eur",
            "gpu_peak_nodes",
        ]
        display_names = {
            "scenario": _("Scenario"), "total_cost_eur": "Total (EUR)",
            "aks_gpu_cost_eur": "GPU Cost (EUR)", "api_llm_cost_eur": "API LLM (EUR)",
            "gpu_peak_nodes": _("Peak Nodes"),
        }
        st.dataframe(
            df[display_cols].copy().rename(columns=display_names),
            use_container_width=True, hide_index=True,
        )

    col_csv, col_html = st.columns([1, 1])
    with col_csv:
        csv_data = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(_("Export CSV"), csv_data, "simulation_results.csv", mime="text/csv")
    with col_html:
        html_data = generate_report_html(lp, df, data, mc_iterations, ha_factor, overhead_factor, df_sorted)
        st.download_button(_("Export HTML"), html_data, "simulation_report.html", mime="text/html")
    st.caption("To save as PDF: open the HTML file in a browser and press Ctrl+P, then select 'Save as PDF'.")


def tab_azure_pricing(data: dict):
    st.markdown(_("**Azure Retail Prices (real-time from API)**"))
    st.caption(
        _("Prices fetched from https://prices.azure.com/api/retail/prices. Region: West Europe. USD converted to EUR at configured rate.")
    )

    import azure_pricing
    exchange_rate = data["api_config"].eur_usd_rate
    prices = azure_pricing.fetch_azure_prices(eur_usd_rate=exchange_rate)

    if prices.success:
        st.success(_("Fetched {0} prices from Azure Retail Prices API").format(len(prices.items)))
        rows = []
        for it in prices.items:
            rows.append({
                _("Resource"): it.label,
                _("Price USD"): f"{it.unit_price_usd:.4f}",
                _("Price EUR"): f"{it.unit_price_eur:.4f}",
                _("Unit"): it.unit,
                _("Source"): it.source,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning(_("Azure API unavailable: {0}").format(prices.error))

    st.markdown(_("**Azure SKUs being considered**"))
    sku_rows = []
    for label, infra_key, nodepool_key in [
        ("Ideal Inference VM", "infra_ideal", "inference_nodepool"),
        ("Ideal System VM", "infra_ideal", "system_nodepool"),
        ("Economy Inference VM", "infra_economica", "inference_nodepool"),
        ("Economy System VM", "infra_economica", "system_nodepool"),
    ]:
        infra = data.get(infra_key)
        if infra:
            npool = getattr(infra, nodepool_key)
            sku_rows.append({
                _("Scenario Part"): label,
                _("VM Type"): npool.vm_type,
                _("Price/hr (EUR)"): f"{npool.price_per_hour:.4f}",
                _("Nodes"): f"{npool.base_office_nodes}/{npool.peak_nodes}/{npool.off_hours_nodes}",
            })
    if sku_rows:
        st.dataframe(pd.DataFrame(sku_rows), use_container_width=True, hide_index=True)

    st.markdown(_("**AKS Infrastructure costs (monthly)**"))
    infra_costs = []
    for label, infra_key in [("Ideal", "infra_ideal"), ("Economy", "infra_economica")]:
        infra = data.get(infra_key)
        if infra:
            infra_costs.append({
                _("Scenario"): label,
                _("Storage (EUR)"): infra.storage_cost_per_month,
                _("Load Balancer (EUR)"): infra.lb_cost_per_month,
                _("Monitor (EUR)"): infra.monitor_cost_per_month,
                _("ACR (EUR)"): infra.acr_cost_per_month,
            })
    if infra_costs:
        st.dataframe(pd.DataFrame(infra_costs), use_container_width=True, hide_index=True)

    st.markdown(_("**Azure OpenAI API config**"))
    api = data.get("api_config")
    if api:
        api_rows = pd.DataFrame([{
            _("Model"): api.model,
            _("Input $/1M tokens"): api.input_price_per_1m_tokens_usd,
            _("Output $/1M tokens"): api.output_price_per_1m_tokens_usd,
            _("EUR/USD rate"): api.eur_usd_rate,
        }])
        st.dataframe(api_rows, use_container_width=True, hide_index=True)

    if prices.success:
        with st.expander(_("Azure API query details")):
            st.markdown("""
**API endpoint:** `https://prices.azure.com/api/retail/prices`  
**Region:** West Europe | **Price type:** Pay-as-you-go
**VM SKUs queried:** `Standard_NC24ads_A100_v4`, `Standard_NV12ads_A10_v5`, `Standard_D8ds_v5`
""")


def main():
    if "_prev_preset" not in st.session_state:
        st.session_state._prev_preset = None
        st.session_state._preset_sel = "6k"

    col_lang_spacer, col_lang = st.columns([8, 1])
    with col_lang:
        st.segmented_control(
            _("Language"), options=["EN", "ES", "GL"], key="lang", default="ES",
            selection_mode="single", label_visibility="collapsed",
        )

    st.markdown(_("**Azure AKS + LLM Cost Simulator**"))
    st.caption(_("Compare AKS (Ideal/Economy) vs Azure OpenAI API for virtual assistants"))

    if "_azure_defaults" not in st.session_state:
        rate = st.session_state.get("eur_usd_rate", 0.92)
        import azure_pricing
        prices = azure_pricing.fetch_azure_prices(eur_usd_rate=rate)
        if prices.success:
            azure_map = {
                "ideal_gpu_price": "vm_nc24ads_a100_v4",
                "eco_gpu_price": "vm_nv12ads_a10_v5",
                "system_price": "vm_d8ds_v5",
            }
            by_key = {it.key: it.unit_price_eur for it in prices.items}
            defaults = {}
            for ss_key, az_key in azure_map.items():
                if az_key in by_key:
                    defaults[ss_key] = round(by_key[az_key], 4)
            if "monitor_logs" in by_key:
                defaults["ideal_monitor"] = round(by_key["monitor_logs"] * 100, 2)
                defaults["eco_monitor"] = round(by_key["monitor_logs"] * 100, 2)
            if "container_registry" in by_key:
                defaults["ideal_acr"] = round(by_key["container_registry"] * 30, 2)
                defaults["eco_acr"] = round(by_key["container_registry"] * 30, 2)
            st.session_state._azure_defaults = defaults
            for k, v in defaults.items():
                if k not in st.session_state:
                    st.session_state[k] = v
                    st.session_state[f"{k}_s"] = v
                    st.session_state[f"{k}_n"] = v
        else:
            st.session_state._azure_defaults = {}

    if st.session_state._azure_defaults:
        azure_keys = st.session_state._azure_defaults
        st.caption(
            f"GPU prices loaded from Azure: A100 = {azure_keys.get('ideal_gpu_price', 6.0):.2f} EUR/h, "
            f"A10 = {azure_keys.get('eco_gpu_price', 2.2):.2f} EUR/h, "
            f"System = {azure_keys.get('system_price', 0.8):.2f} EUR/h"
        )

    current_preset = st.segmented_control(
        _("Quick preset"), options=list(PRESET_KEYS),
        format_func=lambda k: k.split(" - ", 1)[1] if " - " in k else k,
        key="_preset_sel", default="6k",
        selection_mode="single", help=_("Auto-fill business parameters."),
    )
    p = PRESETS.get(current_preset)
    if p:
        st.caption(_(p["desc"]))
    if current_preset != st.session_state._prev_preset:
        apply_preset(current_preset)
        st.session_state._prev_preset = current_preset
        st.rerun()

    render_sidebar()
    render_controls()
    data = build_data_from_ui()

    tab1, tab2 = st.tabs([_("Simulation"), _("Azure Pricing")])
    with tab1:
        tab_simulation(data)
    with tab2:
        tab_azure_pricing(data)


if __name__ == "__main__":
    main()
