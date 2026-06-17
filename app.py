"""
Streamlit web app for Azure infrastructure cost simulation.
Refactored: centralized state, fixed presets, robust CdU sync, improved visuals.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import io
import base64
import datetime

matplotlib.use("Agg")

from model import LoadProfile, NodepoolConfig, AKSInfrastructure, APIConfig
from simulator import simulate_all
from usecase import InfoSource, calculate_usecase_cost
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
    "en": {},
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
        "Use Case Simulation": "Coste por Caso de Uso",
        "CAPEX (one-time)": "CAPEX (único)",
        "Source integration": "Integración de fuentes",
        "OPEX (monthly)": "OPEX (mensual)",
        "Source maintenance": "Mantenimiento de fuentes",
        "Capabilities maintenance": "Mantenimiento capacidades",
        "Year 1": "Año 1",
        "Year 3": "Año 3",
        "CAPEX Breakdown": "Desglose CAPEX",
        "Monthly OPEX Breakdown": "Desglose OPEX mensual",
        "CdU": "CdU",
        "sources": "fuentes",
        "sin caps.": "sin caps.",
    },
    "gl": {
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
        "System node/hr": "Nodo sistema/h",
        "A100 throughput (tok/s)": "Rendemento A100 (tok/s)",
        "A10 throughput (tok/s)": "Rendemento A10 (tok/s)",
        "GPU utilization": "Utilización GPU",
        "Safety factor": "Factor de seguridade",
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
        "GPU nodes dominate AKS cost.": "Os nodos GPU dominan o custo AKS.",
        "Detailed comparison": "Comparativa detallada",
        "CSV": "CSV",
        "PDF Report": "Informe PDF",
        "Simulation": "Simulación",
        "Azure Pricing": "Prezos Azure",
        "Azure Retail Prices (real-time from API)": "Prezos minoristas Azure (tempo real desde API)",
        "Fetched {0} prices from Azure Retail Prices API": "Obtidos {0} prezos da API de prezos Azure",
        "Azure API unavailable: {0}": "API Azure non dispoñible: {0}",
        "Azure SKUs being considered": "SKUs Azure consideradas",
        "AKS Infrastructure costs (monthly)": "Custos infraestrutura AKS (mensuais)",
        "Azure OpenAI API config": "Configuración API Azure OpenAI",
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
        "Peak Nodes": "Nodos Pico",
        "Total (EUR)": "Total (EUR)",
        "GPU Cost (EUR)": "Custo GPU (EUR)",
        "API LLM (EUR)": "API LLM (EUR)",
        "Annual (EUR)": "Anual (EUR)",
        "Use Case Simulation": "Custo por Caso de Uso",
        "CAPEX (one-time)": "CAPEX (único)",
        "Source integration": "Integración de fontes",
        "OPEX (monthly)": "OPEX (mensual)",
        "Source maintenance": "Mantenemento de fontes",
        "Capabilities maintenance": "Mantenemento capacidades",
        "Year 1": "Ano 1",
        "Year 3": "Ano 3",
        "CAPEX Breakdown": "Desglose CAPEX",
        "Monthly OPEX Breakdown": "Desglose OPEX mensual",
        "CdU": "CdU",
        "sources": "fontes",
        "sin caps.": "sen caps.",
    },
}


def _(text):
    raw = st.session_state.get("lang", "en")
    lang = raw.lower()
    return LANG.get(lang, {}).get(text, text)


st.set_page_config(
    page_title="Simulador de Costes de Asistentes Virtuales",
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
    .cdu-card { background: #ffffff; border: 1px solid #cbd5e1; border-left: 5px solid #2563eb; border-radius: 8px; padding: 0.7rem 1.2rem; margin-bottom: 0.5rem; font-size: 0.9rem; color: #0f172a; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .cdu-card strong { color: #1d4ed8; font-weight: 700; }
    .cdu-card p { margin: 0.2rem 0; line-height: 1.5; }
    .winner-card { background: #d4edda; border: 2px solid #166534; border-radius: 10px; padding: 0.8rem 1.2rem; text-align: center; color: #14532d; }
    .winner-card h3 { margin: 0; color: #166534; font-size: 1rem; }
    .winner-card .winner-amount { margin: 0; color: #14532d; font-weight: 700; font-size: 1.5rem; }
    .runner-card { background: #e8f0fe; border: 1px solid #1e40af; border-radius: 8px; padding: 0.5rem 1rem; color: #1e3a5f; }
    .runner-card p { margin: 0; color: #1e3a5f; }
    div[data-testid="stMetric"] { background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.4rem 0.6rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    div[data-testid="stMetric"] label { color: #334155 !important; font-weight: 600 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.02em; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700 !important; }
    .stAlert { font-size: 0.85rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #94a3b8; border-radius: 6px; overflow: hidden; font-size: 0.8rem; }
    div[data-testid="stDataFrame"] th { background: #1e3a5f; color: #ffffff; font-weight: 700; padding: 0.4rem 0.6rem; text-align: left; border-bottom: 2px solid #0f2440; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em; }
    div[data-testid="stDataFrame"] td { padding: 0.35rem 0.6rem; border-bottom: 1px solid #e2e8f0; color: #0f172a; font-size: 0.8rem; }
    div[data-testid="stDataFrame"] tr:nth-child(even) td { background: #f1f5f9; }
    div[data-testid="stDataFrame"] tr:hover td { background: #dbeafe; }
    .stDataFrame [data-testid="StyledDataFrameDataCell"] { border-bottom: 1px solid #e2e8f0; }
    table { border-collapse: collapse; width: 100%; }
    th { background: #1e3a5f; color: white; font-weight: 700; padding: 0.4rem 0.6rem; text-align: left; }
    td { padding: 0.35rem 0.6rem; border-bottom: 1px solid #e2e8f0; color: #0f172a; }
    tr:nth-child(even) td { background: #f1f5f9; }
    [data-testid="stContainer"] { border-color: #e2e8f0 !important; }
    .cost-table td { color: #0f172a; }
    .cost-table th { background: #1e3a5f; color: #fff; }
    .cost-table .footer { background: #f8fafc; color: #64748b; }
    .cost-table .sc-ideal { color: #1e40af; }
    .cost-table .sc-eco { color: #15803d; }
    .cost-table .sc-api { color: #b45309; }
    .cost-table .year1 { color: #dc2626; }
    @media (prefers-color-scheme: dark) {
        .cdu-card { background: #1e293b; border-color: #334155; color: #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
        .cdu-card strong { color: #60a5fa; }
        .winner-card { background: #14532d; border-color: #22c55e; color: #bbf7d0; }
        .winner-card h3 { color: #4ade80; }
        .winner-card .winner-amount { color: #bbf7d0; }
        .runner-card { background: #1e3a5f; border-color: #3b82f6; color: #bfdbfe; }
        .runner-card p { color: #bfdbfe; }
        div[data-testid="stMetric"] { background: #1e293b; border-color: #334155; }
        div[data-testid="stMetric"] label { color: #94a3b8 !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
        div[data-testid="stDataFrame"] { border-color: #334155; }
        div[data-testid="stDataFrame"] th { background: #0f2440; border-bottom-color: #1e3a5f; }
        div[data-testid="stDataFrame"] td { color: #e2e8f0; border-bottom-color: #334155; }
        div[data-testid="stDataFrame"] tr:nth-child(even) td { background: #1e293b; }
        div[data-testid="stDataFrame"] tr:hover td { background: #334155; }
        td { color: #e2e8f0; border-bottom-color: #334155; }
        tr:nth-child(even) td { background: #1e293b; }
        [data-testid="stContainer"] { border-color: #334155 !important; }
        .cost-table td { color: #e2e8f0; }
        .cost-table th { background: #0f2440; color: #e2e8f0; }
        .cost-table .footer { background: #1e293b; color: #94a3b8; }
        .cost-table .sc-ideal { color: #60a5fa; }
        .cost-table .sc-eco { color: #4ade80; }
        .cost-table .sc-api { color: #fbbf24; }
        .cost-table .year1 { color: #f87171; }
        .cost-table { border-color: #334155 !important; }
    }
</style>
"""
st.markdown(COMPACT_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
STATE_DEFAULTS = {
    "users": DEFAULT_USERS,
    "interactions_per_user_day": DEFAULT_INTERACTIONS_PER_USER_DAY,
    "input_tokens_per_interaction": DEFAULT_INPUT_TOKENS,
    "output_tokens_per_interaction": DEFAULT_OUTPUT_TOKENS,
    "working_days_per_month": DEFAULT_WORKING_DAYS,
    "office_hours_per_day": DEFAULT_OFFICE_HOURS,
    "peak_hours_per_day": DEFAULT_PEAK_HOURS,
    "concurrent_user_ratio": DEFAULT_CONCURRENT_RATIO,
    "peak_multiplier": DEFAULT_PEAK_MULTIPLIER,
    "system_price": DEFAULT_SYSTEM_PRICE,
    "ideal_gpu_price": DEFAULT_IDEAL_GPU_PRICE,
    "ideal_throughput": DEFAULT_IDEAL_THROUGHPUT,
    "eco_gpu_price": DEFAULT_ECO_GPU_PRICE,
    "eco_throughput": DEFAULT_ECO_THROUGHPUT,
    "gpu_utilization": DEFAULT_GPU_UTILIZATION,
    "safety_factor": DEFAULT_SAFETY_FACTOR,
    "ideal_storage": DEFAULT_STORAGE_IDEAL,
    "ideal_lb": DEFAULT_LB,
    "ideal_monitor": DEFAULT_MONITOR_IDEAL,
    "ideal_acr": DEFAULT_ACR_IDEAL,
    "eco_storage": DEFAULT_STORAGE_ECO,
    "eco_lb": DEFAULT_LB,
    "eco_monitor": DEFAULT_MONITOR_ECO,
    "eco_acr": DEFAULT_ACR_ECO,
    "api_model": DEFAULT_API_MODEL,
    "api_input_price": DEFAULT_API_INPUT_PRICE,
    "api_output_price": DEFAULT_API_OUTPUT_PRICE,
    "eur_usd_rate": DEFAULT_EUR_USD,
    "mc_iterations": 5000,
    "ha_factor": 1.15,
    "overhead_factor": 0.1,
    "lang": "ES",
    "_preset_sel": "750",
    "_uc_ens": "medium",
    "_uc_cap_agentic": False,
    "_uc_cap_anon": False,
    "_uc_cap_sso": False,
    "_uc_src_sharepoint_cnt": 6,
    "_uc_src_database_cnt": 4,
    "_uc_src_web_scraping_cnt": 4,
    "_uc_src_api_cnt": 4,
    "_uc_src_pdf_dynamic_cnt": 2,
    "_uc_src_sharepoint_vol": 10.0,
    "_uc_src_database_vol": 10.0,
    "_uc_src_web_scraping_vol": 10.0,
    "_uc_src_api_vol": 10.0,
    "_uc_src_pdf_dynamic_vol": 10.0,
    "_uc_src_sharepoint_freq": "daily",
    "_uc_src_database_freq": "daily",
    "_uc_src_web_scraping_freq": "daily",
    "_uc_src_api_freq": "daily",
    "_uc_src_pdf_dynamic_freq": "daily",
}

CDU_SOURCE_KEYS = ["sharepoint", "database", "web_scraping", "api", "pdf_dynamic"]


def _gs(key, default):
    return st.session_state.get(key, default)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "100": {"users": 100, "interactions_per_user_day": 8, "input_tokens_per_interaction": 400, "output_tokens_per_interaction": 150, "working_days_per_month": 22, "office_hours_per_day": 8, "peak_hours_per_day": 0.25, "concurrent_user_ratio": 0.30, "peak_multiplier": 1.5, "desc": "dept. pequeño",
     "_uc_src_sharepoint_cnt":2,"_uc_src_database_cnt":1,"_uc_src_web_scraping_cnt":1,"_uc_src_api_cnt":1,"_uc_src_pdf_dynamic_cnt":1,"_uc_cap_agentic":False,"_uc_cap_anon":False,"_uc_cap_sso":True,"_uc_ens":"basic"},
    "200": {"users": 200, "interactions_per_user_day": 10, "input_tokens_per_interaction": 500, "output_tokens_per_interaction": 200, "working_days_per_month": 22, "office_hours_per_day": 8, "peak_hours_per_day": 0.25, "concurrent_user_ratio": 0.25, "peak_multiplier": 1.5, "desc": "dept. mediano",
     "_uc_src_sharepoint_cnt":3,"_uc_src_database_cnt":2,"_uc_src_web_scraping_cnt":2,"_uc_src_api_cnt":1,"_uc_src_pdf_dynamic_cnt":1,"_uc_cap_agentic":False,"_uc_cap_anon":False,"_uc_cap_sso":True,"_uc_ens":"basic"},
    "500": {"users": 500, "interactions_per_user_day": 12, "input_tokens_per_interaction": 600, "output_tokens_per_interaction": 250, "working_days_per_month": 22, "office_hours_per_day": 9, "peak_hours_per_day": 0.5, "concurrent_user_ratio": 0.20, "peak_multiplier": 1.5, "desc": "división pequeña",
     "_uc_src_sharepoint_cnt":5,"_uc_src_database_cnt":3,"_uc_src_web_scraping_cnt":3,"_uc_src_api_cnt":3,"_uc_src_pdf_dynamic_cnt":2,"_uc_cap_agentic":True,"_uc_cap_anon":False,"_uc_cap_sso":True,"_uc_ens":"basic"},
    "750": {"users": 750, "interactions_per_user_day": 15, "input_tokens_per_interaction": 700, "output_tokens_per_interaction": 300, "working_days_per_month": 22, "office_hours_per_day": 9, "peak_hours_per_day": 0.5, "concurrent_user_ratio": 0.18, "peak_multiplier": 1.5, "desc": "división mediana (por defecto)",
     "_uc_src_sharepoint_cnt":6,"_uc_src_database_cnt":4,"_uc_src_web_scraping_cnt":4,"_uc_src_api_cnt":4,"_uc_src_pdf_dynamic_cnt":2,"_uc_cap_agentic":True,"_uc_cap_anon":True,"_uc_cap_sso":True,"_uc_ens":"medium"},
    "1k": {"users": 1000, "interactions_per_user_day": 20, "input_tokens_per_interaction": 1000, "output_tokens_per_interaction": 400, "working_days_per_month": 22, "office_hours_per_day": 9, "peak_hours_per_day": 0.5, "concurrent_user_ratio": 0.18, "peak_multiplier": 1.5, "desc": "división grande",
     "_uc_src_sharepoint_cnt":8,"_uc_src_database_cnt":5,"_uc_src_web_scraping_cnt":5,"_uc_src_api_cnt":4,"_uc_src_pdf_dynamic_cnt":3,"_uc_cap_agentic":True,"_uc_cap_anon":True,"_uc_cap_sso":True,"_uc_ens":"medium"},
    "2k": {"users": 2000, "interactions_per_user_day": 25, "input_tokens_per_interaction": 1200, "output_tokens_per_interaction": 400, "working_days_per_month": 22, "office_hours_per_day": 10, "peak_hours_per_day": 1.0, "concurrent_user_ratio": 0.15, "peak_multiplier": 2.0, "desc": "organización pequeña",
     "_uc_src_sharepoint_cnt":10,"_uc_src_database_cnt":6,"_uc_src_web_scraping_cnt":6,"_uc_src_api_cnt":6,"_uc_src_pdf_dynamic_cnt":4,"_uc_cap_agentic":True,"_uc_cap_anon":True,"_uc_cap_sso":True,"_uc_ens":"medium"},
    "3k": {"users": 3000, "interactions_per_user_day": 30, "input_tokens_per_interaction": 1500, "output_tokens_per_interaction": 400, "working_days_per_month": 22, "office_hours_per_day": 10, "peak_hours_per_day": 1.0, "concurrent_user_ratio": 0.15, "peak_multiplier": 2.0, "desc": "organización mediana",
     "_uc_src_sharepoint_cnt":12,"_uc_src_database_cnt":8,"_uc_src_web_scraping_cnt":8,"_uc_src_api_cnt":8,"_uc_src_pdf_dynamic_cnt":4,"_uc_cap_agentic":True,"_uc_cap_anon":True,"_uc_cap_sso":True,"_uc_ens":"medium"},
    "6k": {"users": 6000, "interactions_per_user_day": 30, "input_tokens_per_interaction": 2000, "output_tokens_per_interaction": 400, "working_days_per_month": 22, "office_hours_per_day": 10, "peak_hours_per_day": 2.0, "concurrent_user_ratio": 0.15, "peak_multiplier": 2.0, "desc": "organización grande",
     "_uc_src_sharepoint_cnt":15,"_uc_src_database_cnt":10,"_uc_src_web_scraping_cnt":10,"_uc_src_api_cnt":10,"_uc_src_pdf_dynamic_cnt":5,"_uc_cap_agentic":True,"_uc_cap_anon":True,"_uc_cap_sso":True,"_uc_ens":"high"},
    "10k": {"users": 10000, "interactions_per_user_day": 35, "input_tokens_per_interaction": 2000, "output_tokens_per_interaction": 500, "working_days_per_month": 22, "office_hours_per_day": 11, "peak_hours_per_day": 2.0, "concurrent_user_ratio": 0.15, "peak_multiplier": 2.0, "desc": "empresa",
     "_uc_src_sharepoint_cnt":18,"_uc_src_database_cnt":12,"_uc_src_web_scraping_cnt":12,"_uc_src_api_cnt":12,"_uc_src_pdf_dynamic_cnt":6,"_uc_cap_agentic":True,"_uc_cap_anon":True,"_uc_cap_sso":True,"_uc_ens":"high"},
    "20k": {"users": 20000, "interactions_per_user_day": 40, "input_tokens_per_interaction": 2500, "output_tokens_per_interaction": 500, "working_days_per_month": 22, "office_hours_per_day": 12, "peak_hours_per_day": 2.0, "concurrent_user_ratio": 0.18, "peak_multiplier": 2.5, "desc": "corporación",
     "_uc_src_sharepoint_cnt":22,"_uc_src_database_cnt":15,"_uc_src_web_scraping_cnt":15,"_uc_src_api_cnt":14,"_uc_src_pdf_dynamic_cnt":8,"_uc_cap_agentic":True,"_uc_cap_anon":True,"_uc_cap_sso":True,"_uc_ens":"high"},
    "35k": {"users": 35000, "interactions_per_user_day": 45, "input_tokens_per_interaction": 2500, "output_tokens_per_interaction": 600, "working_days_per_month": 22, "office_hours_per_day": 12, "peak_hours_per_day": 3.0, "concurrent_user_ratio": 0.20, "peak_multiplier": 2.5, "desc": "gran corporación",
     "_uc_src_sharepoint_cnt":25,"_uc_src_database_cnt":18,"_uc_src_web_scraping_cnt":18,"_uc_src_api_cnt":16,"_uc_src_pdf_dynamic_cnt":10,"_uc_cap_agentic":True,"_uc_cap_anon":True,"_uc_cap_sso":True,"_uc_ens":"high"},
}

PRESET_KEYS = list(PRESETS.keys())


def apply_preset(name: str):
    p = PRESETS.get(name)
    if p:
        for k, v in p.items():
            if k != "desc":
                st.session_state[k] = v
        st.session_state._uc_profile_prev = None


# ---------------------------------------------------------------------------
# CdU parameters
# ---------------------------------------------------------------------------
CDU_PROFILES = {
    "simple":  {"label": "Simple",  "desc": "~10 fuentes",  "sources": {"sharepoint":3, "database":2, "web_scraping":2, "api":2, "pdf_dynamic":1}},
    "medio":   {"label": "Medio",   "desc": "~20 fuentes",  "sources": {"sharepoint":6, "database":4, "web_scraping":4, "api":4, "pdf_dynamic":2}},
    "grande":  {"label": "Grande",  "desc": "~40 fuentes",  "sources": {"sharepoint":12, "database":8, "web_scraping":8, "api":8, "pdf_dynamic":4}},
    "complejo":{"label": "Complejo","desc": "50+ fuentes",  "sources": {"sharepoint":15, "database":10, "web_scraping":12, "api":10, "pdf_dynamic":5}},
}


def _apply_cdu_profile(prof_key):
    p = CDU_PROFILES.get(prof_key)
    if p:
        for sk, sv in p["sources"].items():
            st.session_state[f"_uc_src_{sk}_cnt"] = sv


def _render_cdu_params():
    if "_uc_profile_prev" not in st.session_state:
        st.session_state._uc_profile_prev = None

    prof_key = st.segmented_control(
        "Perfil CdU", options=list(CDU_PROFILES.keys()),
        format_func=lambda k: CDU_PROFILES[k]["label"], key="_uc_profile",
        selection_mode="single",
    )
    p = CDU_PROFILES.get(prof_key)
    if p:
        st.caption(p["desc"])

    if prof_key and prof_key != st.session_state.get("_uc_profile_prev"):
        _apply_cdu_profile(prof_key)
        st.session_state._uc_profile_prev = prof_key

    st.markdown("**Fuentes**")
    for key, label in [("sharepoint","SharePoint/Alfresco"), ("database","BD Oracle/SQL"),
                        ("web_scraping","Web Scraping"), ("api","API REST"),
                        ("pdf_dynamic","PDF Dinámico")]:
        cnt = st.slider(label, 0, 50, value=st.session_state.get(f"_uc_src_{key}_cnt", 0), key=f"_uc_src_{key}_cnt")
        if cnt > 0:
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Vol. (GB)", 0.0, 100000.0, value=st.session_state.get(f"_uc_src_{key}_vol", 10.0), step=10.0, key=f"_uc_src_{key}_vol")
            with c2:
                st.selectbox("Frecuencia", ["realtime","hourly","daily","weekly","monthly"],
                    format_func=lambda x: {"realtime":"Tiempo real","hourly":"Cada hora","daily":"Diaria","weekly":"Semanal","monthly":"Mensual"}[x],
                    key=f"_uc_src_{key}_freq")

    st.markdown("**Capacidades**")
    ccc = st.columns(3)
    with ccc[0]: st.checkbox("IA Agéntica\n8.500€", key="_uc_cap_agentic")
    with ccc[1]: st.checkbox("Anonimización\n3.500€", key="_uc_cap_anon")
    with ccc[2]: st.checkbox("Autenticación SSO\n4.000€", key="_uc_cap_sso")

    st.markdown("**Cumplimiento**")
    st.select_slider("ENS", options=["none", "basic", "medium", "high"],
        format_func=lambda x: {"none":"Ninguno","basic":"Básico","medium":"Medio","high":"Alto"}[x],
        value=st.session_state.get("_uc_ens", "medium"), key="_uc_ens")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("**Infra/Cloud**")
        st.slider("Usuarios", 100, 35000, value=_gs("users", DEFAULT_USERS), step=100, key="users",
                  help="Usuarios activos que interactúan con el asistente.")
        st.slider("Interacciones/usuario/día", 1, 200, value=_gs("interactions_per_user_day", DEFAULT_INTERACTIONS_PER_USER_DAY), step=1, key="interactions_per_user_day")
        st.slider("Tokens entrada/interacción", 100, 16000, value=_gs("input_tokens_per_interaction", DEFAULT_INPUT_TOKENS), step=100, key="input_tokens_per_interaction")
        st.slider("Tokens salida/interacción", 50, 8000, value=_gs("output_tokens_per_interaction", DEFAULT_OUTPUT_TOKENS), step=50, key="output_tokens_per_interaction")
        st.slider("Días laborables/mes", 1, 31, value=_gs("working_days_per_month", DEFAULT_WORKING_DAYS), step=1, key="working_days_per_month")
        st.slider("Horas oficina/día", 1, 24, value=_gs("office_hours_per_day", DEFAULT_OFFICE_HOURS), step=1, key="office_hours_per_day")
        st.slider("Horas pico/día", 0.1, 8.0, value=_gs("peak_hours_per_day", DEFAULT_PEAK_HOURS), step=0.1, key="peak_hours_per_day")
        st.slider("Ratio concurrencia", 0.01, 1.0, value=_gs("concurrent_user_ratio", DEFAULT_CONCURRENT_RATIO), step=0.01, key="concurrent_user_ratio")
        st.slider("Multiplicador pico", 1.0, 5.0, value=_gs("peak_multiplier", DEFAULT_PEAK_MULTIPLIER), step=0.5, key="peak_multiplier")

        st.markdown("---")
        st.markdown("**CdU**")
        _render_cdu_params()


# ---------------------------------------------------------------------------
# Build data from UI state
# ---------------------------------------------------------------------------
def build_data_from_ui() -> dict:
    lp = LoadProfile(
        users=_gs("users", DEFAULT_USERS),
        interactions_per_user_day=_gs("interactions_per_user_day", DEFAULT_INTERACTIONS_PER_USER_DAY),
        input_tokens_per_interaction=_gs("input_tokens_per_interaction", DEFAULT_INPUT_TOKENS),
        output_tokens_per_interaction=_gs("output_tokens_per_interaction", DEFAULT_OUTPUT_TOKENS),
        working_days_per_month=_gs("working_days_per_month", DEFAULT_WORKING_DAYS),
        office_hours_per_day=_gs("office_hours_per_day", DEFAULT_OFFICE_HOURS),
        peak_hours_per_day=_gs("peak_hours_per_day", DEFAULT_PEAK_HOURS),
        concurrent_user_ratio=_gs("concurrent_user_ratio", DEFAULT_CONCURRENT_RATIO),
        peak_multiplier=_gs("peak_multiplier", DEFAULT_PEAK_MULTIPLIER),
    )

    infra_ideal = AKSInfrastructure(name="LLM on AKS (Ideal UX)")
    infra_ideal.system_nodepool = NodepoolConfig(
        vm_type="Standard_D8ds_v5", base_office_nodes=1,
        price_per_hour=_gs("system_price", DEFAULT_SYSTEM_PRICE),
    )
    infra_ideal.inference_nodepool = NodepoolConfig(
        vm_type="Standard_NC24ads_A100_v4",
        base_office_nodes=3, peak_nodes=10, off_hours_nodes=1,
        price_per_hour=_gs("ideal_gpu_price", DEFAULT_IDEAL_GPU_PRICE),
    )
    infra_ideal.throughput_tok_s_per_pod = _gs("ideal_throughput", DEFAULT_IDEAL_THROUGHPUT)
    infra_ideal.gpu_utilization = _gs("gpu_utilization", DEFAULT_GPU_UTILIZATION)
    infra_ideal.safety_factor = _gs("safety_factor", DEFAULT_SAFETY_FACTOR)
    infra_ideal.base_replicas = 3
    infra_ideal.peak_replicas = 10
    infra_ideal.off_hours_replicas = 1
    infra_ideal.storage_cost_per_month = _gs("ideal_storage", DEFAULT_STORAGE_IDEAL)
    infra_ideal.lb_cost_per_month = _gs("ideal_lb", DEFAULT_LB)
    infra_ideal.monitor_cost_per_month = _gs("ideal_monitor", DEFAULT_MONITOR_IDEAL)
    infra_ideal.acr_cost_per_month = _gs("ideal_acr", DEFAULT_ACR_IDEAL)

    infra_economy = AKSInfrastructure(name="LLM on AKS (Economy UX)")
    infra_economy.system_nodepool = NodepoolConfig(
        vm_type="Standard_D8ds_v5", base_office_nodes=1,
        price_per_hour=_gs("system_price", DEFAULT_SYSTEM_PRICE),
    )
    infra_economy.inference_nodepool = NodepoolConfig(
        vm_type="Standard_NV12ads_A10_v5",
        base_office_nodes=5, peak_nodes=20, off_hours_nodes=1,
        price_per_hour=_gs("eco_gpu_price", DEFAULT_ECO_GPU_PRICE),
    )
    infra_economy.throughput_tok_s_per_pod = _gs("eco_throughput", DEFAULT_ECO_THROUGHPUT)
    infra_economy.gpu_utilization = _gs("gpu_utilization", DEFAULT_GPU_UTILIZATION)
    infra_economy.safety_factor = _gs("safety_factor", DEFAULT_SAFETY_FACTOR)
    infra_economy.base_replicas = 5
    infra_economy.peak_replicas = 20
    infra_economy.off_hours_replicas = 1
    infra_economy.storage_cost_per_month = _gs("eco_storage", DEFAULT_STORAGE_ECO)
    infra_economy.lb_cost_per_month = _gs("eco_lb", DEFAULT_LB)
    infra_economy.monitor_cost_per_month = _gs("eco_monitor", DEFAULT_MONITOR_ECO)
    infra_economy.acr_cost_per_month = _gs("eco_acr", DEFAULT_ACR_ECO)

    api = APIConfig(
        name="API Azure OpenAI",
        model=_gs("api_model", DEFAULT_API_MODEL),
        input_price_per_1m_tokens_usd=_gs("api_input_price", DEFAULT_API_INPUT_PRICE),
        output_price_per_1m_tokens_usd=_gs("api_output_price", DEFAULT_API_OUTPUT_PRICE),
        eur_usd_rate=_gs("eur_usd_rate", DEFAULT_EUR_USD),
    )

    return {
        "load_profile": lp,
        "infra_ideal": infra_ideal,
        "infra_economica": infra_economy,
        "api_config": api,
        "comparativa": pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_eur(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M €"
    return f"{v:,.0f} €"


def fmt_period(v: float) -> str:
    yr = v * 12
    if yr >= 1_000_000:
        return f"{fmt_eur(v)}/mes ({yr/1_000_000:.2f}M €/año)"
    return f"{fmt_eur(v)}/mes ({yr:,.0f} €/año)"


# ---------------------------------------------------------------------------
# Build InfoSource list from UI state
# ---------------------------------------------------------------------------
def _build_cdu_custom_pricing():
    from usecase import SOURCE_INTEGRATION_TABLE, FREQUENCY_MULTIPLIERS, CAPABILITY_COSTS, ENS_COSTS
    src_table = {}
    for sk in ["sharepoint","database","web_scraping","api","pdf_dynamic"]:
        src_table[sk] = {}
        for cmp in ["low","medium","high"]:
            src_table[sk][cmp] = _gs(f"_cdu_src_{sk}_{cmp}", SOURCE_INTEGRATION_TABLE[sk][cmp])
    freq_mult = {}
    for fk in FREQUENCY_MULTIPLIERS:
        freq_mult[fk] = _gs(f"_cdu_freq_{fk}", FREQUENCY_MULTIPLIERS[fk])
    cap_costs = {}
    for ck in CAPABILITY_COSTS:
        cap_costs[ck] = {
            "capex": _gs(f"_cdu_cap_{ck}_capex", CAPABILITY_COSTS[ck]["capex"]),
            "opex_monthly": _gs(f"_cdu_cap_{ck}_opex", CAPABILITY_COSTS[ck]["opex_monthly"]),
        }
    ens_costs = {}
    for ek in ENS_COSTS:
        ens_costs[ek] = {"capex": _gs(f"_cdu_ens_{ek}_capex", ENS_COSTS[ek]["capex"])}
    maint_pct = _gs("_cdu_maintenance_pct", 10.0) / 100.0
    return {"source_table": src_table, "freq_mult": freq_mult, "maintenance_pct": maint_pct}, cap_costs, ens_costs


def _build_uc_sources(custom_pricing=None):
    sources = []
    for key in CDU_SOURCE_KEYS:
        cnt = st.session_state.get(f"_uc_src_{key}_cnt", 0)
        if cnt > 0:
            for i in range(cnt):
                sources.append(InfoSource(
                    name=f"{key}_{i+1}",
                    source_type=key,
                    complexity="medium",
                    data_volume_gb=st.session_state.get(f"_uc_src_{key}_vol", 10.0),
                    update_frequency=st.session_state.get(f"_uc_src_{key}_freq", "daily"),
                    custom_pricing=custom_pricing,
                ))
    return sources


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def generate_report_html(lp, df, data, mc_iterations, ha_factor, overhead_factor, df_sorted,
                         impl_cdu=0, rec_anual=0, cdu_result=None, cdu_tag=""):
    def _make_bar_chart(sc_filter, sc_color):
        buf = io.BytesIO()
        row = df[df["scenario"].str.contains(sc_filter, case=False)].iloc[0]
        categories = ["System", "GPU", "Storage", "LB", "Monitor", "ACR"]
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
        return buf

    def _make_total_chart():
        buf = io.BytesIO()
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
        fig2.savefig(buf, format="png", dpi=200)
        plt.close(fig2)
        buf.seek(0)
        return buf

    from usecase import SOURCE_INTEGRATION_TABLE, CAPABILITY_COSTS, ENS_COSTS
    buf_left_eco = _make_bar_chart("Economico", "#27ae60")
    buf_left_ideal = _make_bar_chart("Ideal", "#2c6b9e")
    buf_right = _make_total_chart()

    def img_b64(buf):
        return base64.b64encode(buf.read()).decode("utf-8")

    best_row = df_sorted.iloc[0]
    second_row = df_sorted.iloc[1]
    third_row = df_sorted.iloc[2]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lang_label = st.session_state.get("lang", "en").lower()
    t = lambda s: LANG.get(lang_label, {}).get(s, s)

    return f"""<!DOCTYPE html>
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
<tr><td>A100 GPU/hr: {_gs('ideal_gpu_price', DEFAULT_IDEAL_GPU_PRICE):.2f} EUR</td><td>{t("Model")}: {_gs('api_model', DEFAULT_API_MODEL)}</td></tr>
<tr><td>A10 GPU/hr: {_gs('eco_gpu_price', DEFAULT_ECO_GPU_PRICE):.2f} EUR</td><td>{t("Input $/1M tok")}: {_gs('api_input_price', DEFAULT_API_INPUT_PRICE):.2f}</td></tr>
<tr><td>{t("System node/hr")}: {_gs('system_price', DEFAULT_SYSTEM_PRICE):.2f} EUR</td><td>{t("Output $/1M tok")}: {_gs('api_output_price', DEFAULT_API_OUTPUT_PRICE):.2f}</td></tr>
<tr><td>{t("A100 throughput (tok/s)")}: {_gs('ideal_throughput', DEFAULT_IDEAL_THROUGHPUT)}</td><td>{t("EUR/USD rate")}: {_gs('eur_usd_rate', DEFAULT_EUR_USD)}</td></tr>
<tr><td>{t("A10 throughput (tok/s)")}: {_gs('eco_throughput', DEFAULT_ECO_THROUGHPUT)}</td><td></td></tr>
</table>

<h2>{t("Costs")}</h2>
<div class="chart"><img src="data:image/png;base64,{img_b64(buf_left_eco)}" style="width:45%;display:inline-block" />
<img src="data:image/png;base64,{img_b64(buf_left_ideal)}" style="width:45%;display:inline-block" /></div>
<div class="chart"><img src="data:image/png;base64,{img_b64(buf_right)}" style="width:70%" /></div>

<table>
<tr><th>{t("Scenario")}</th><th>{t("Total (EUR)")}</th><th>{t("Annual (EUR)")}</th><th>{t("GPU Cost (EUR)")}</th><th>{t("API LLM (EUR)")}</th><th>{t("Peak Nodes")}</th></tr>""" + \
    "".join(f"<tr><td>{r['scenario']}</td><td>{r['total_cost_eur']:,.0f}</td><td>{r['total_cost_eur']*12:,.0f}</td><td>{r['aks_gpu_cost_eur']:,.0f}</td><td>{r['api_llm_cost_eur']:,.0f}</td><td>{r['gpu_peak_nodes']}</td></tr>" for _, r in df.iterrows()) + \
    """</table>

<h2>CdU — Caso de Uso</h2>
<p style="font-size:0.75rem;background:#eef5ff;padding:4px 8px;border-radius:4px;">""" + cdu_tag + """</p>
<table>
<tr><th>Concepto</th><th>Importe</th></tr>
<tr><td>Integración de fuentes (CAPEX)</td><td>{:,.0f} €</td></tr>
<tr><td>Capacidades (CAPEX)</td><td>{:,.0f} €</td></tr>
<tr><td>ENS / Cumplimiento (CAPEX)</td><td>{:,.0f} €</td></tr>
<tr><td style="font-weight:bold">Total implantación (CAPEX)</td><td style="font-weight:bold">{:,.0f} €</td></tr>
<tr><td>Mantenimiento fuentes (OPEX/mes)</td><td>{:,.0f} €</td></tr>
<tr><td>Capacidades (OPEX/mes)</td><td>{:,.0f} €</td></tr>
<tr><td style="font-weight:bold">Total recurrente/mes</td><td style="font-weight:bold">{:,.0f} €</td></tr>
<tr><td style="font-weight:bold">Total recurrente/año</td><td style="font-weight:bold">{:,.0f} €</td></tr>
</table>
""".format(
    cdu_result.source_integration_capex if cdu_result else 0,
    cdu_result.capabilities_capex if cdu_result else 0,
    cdu_result.compliance_capex if cdu_result else 0,
    impl_cdu,
    cdu_result.source_maintenance_opex if cdu_result else 0,
    cdu_result.capabilities_opex if cdu_result else 0,
    (cdu_result.source_maintenance_opex + cdu_result.capabilities_opex) if cdu_result else 0,
    rec_anual,
) + """

<h2>CdU — Fuentes</h2>
<table>
<tr><th>Fuente</th><th>Cantidad</th><th>Vol. (GB)</th><th>Frecuencia</th></tr>""" + \
"".join(f"<tr><td>{k}</td><td>{st.session_state.get(f'_uc_src_{k}_cnt', 0)}</td>"
        f"<td>{st.session_state.get(f'_uc_src_{k}_vol', 10.0)}</td>"
        f"<td>{st.session_state.get(f'_uc_src_{k}_freq', 'daily')}</td></tr>"
        for k in ["sharepoint","database","web_scraping","api","pdf_dynamic"]
        if st.session_state.get(f"_uc_src_{k}_cnt", 0) > 0) + \
"""</table>

<h2>CdU — Precios</h2>
<table>
<tr><th>Concepto</th><th>Valor</th></tr>""" + \
"".join(
    "<tr><td>{}</td><td>{}</td></tr>".format(
        {"sharepoint":"SharePoint/Alfresco","database":"BD Oracle/SQL",
         "web_scraping":"Web Scraping","api":"API REST","pdf_dynamic":"PDF Dinámico"}[sk] + f" ({cmp})",
        f"{_gs(f'_cdu_src_{sk}_{cmp}', SOURCE_INTEGRATION_TABLE[sk][cmp]):,.0f} €"
    )
    for sk in ["sharepoint","database","web_scraping","api","pdf_dynamic"]
    for cmp in ["low","medium","high"]
) + \
"".join(
    f"<tr><td>Multiplicador frecuencia {fl}</td><td>{_gs(f'_cdu_freq_{fk}', fv):.2f}</td></tr>"
    for fk, fv in {"realtime":"Tiempo real","hourly":"Cada hora","daily":"Diaria","weekly":"Semanal","monthly":"Mensual"}.items()
) + \
"".join(
    f"<tr><td>Capacidad {cl} CAPEX</td><td>{_gs(f'_cdu_cap_{ck}_capex', CAPABILITY_COSTS[ck]['capex']):,.0f} €</td></tr>"
    f"<tr><td>Capacidad {cl} OPEX/mes</td><td>{_gs(f'_cdu_cap_{ck}_opex', CAPABILITY_COSTS[ck]['opex_monthly']):,.0f} €</td></tr>"
    for ck, cl in [("agentic_ai","IA Agéntica"),("anonymization","Anonimización"),("sso","SSO")]
) + \
"".join(
    f"<tr><td>ENS {el} CAPEX</td><td>{_gs(f'_cdu_ens_{ek}_capex', ENS_COSTS[ek]['capex']):,.0f} €</td></tr>"
    for ek, el in [("none","Ninguno"),("basic","Básico"),("medium","Medio"),("high","Alto")]
) + \
f"""<tr><td>Mantenimiento anual</td><td>{_gs('_cdu_maintenance_pct', 10.0):.1f} %</td></tr>
</table>
<div class="footer">Azure AKS + LLM Cost Simulator</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Simulation tab
# ---------------------------------------------------------------------------
def tab_simulation(data: dict, df=None, df_totales=None, impl_cdu=0, rec_anual=0, cdu_result=None, cdu_tag=""):
    lp = data["load_profile"]
    mc_iterations = _gs("mc_iterations", 5000)
    ha_factor = _gs("ha_factor", 1.15)
    overhead_factor = _gs("overhead_factor", 0.1)

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

    if df is None:
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

    c_win, c_rest = st.columns([1, 1])
    with c_win:
        st.markdown(
            f'<div class="winner-card"><h3>{_("WINNER")}: {best_row["scenario"]}</h3>'
            f'<p class="winner-amount">{fmt_period(best_row["total_cost_eur"])}</p></div>',
            unsafe_allow_html=True,
        )
    with c_rest:
        st.markdown(
            f'<div class="runner-card"><p style="margin:0">{second_row["scenario"]}: {fmt_period(second_row["total_cost_eur"])}</p>'
            f'<p style="margin:0">{third_row["scenario"]}: {fmt_period(third_row["total_cost_eur"])}</p></div>',
            unsafe_allow_html=True,
        )

    col_left, col_right = st.columns([1, 1.4])
    with col_left:
        categories = ["System", "GPU", "Storage", "LB", "Monitor", "ACR"]
        for sc_name, sc_filter, sc_color in [("Economico", "Economico", "#27ae60"), ("Ideal", "Ideal", "#2c6b9e")]:
            row = df[df["scenario"].str.contains(sc_filter, case=False)].iloc[0]
            vals = [
                row.get("aks_system_cost_eur", 0), row.get("aks_gpu_cost_eur", 0),
                row.get("storage_cost_eur", 0), row.get("lb_cost_eur", 0),
                row.get("monitor_cost_eur", 0), row.get("acr_cost_eur", 0),
            ]
            fig, ax = plt.subplots(figsize=(4, 1.5), facecolor="#fafafa")
            ax.bar(categories, vals, color=sc_color, width=0.6, edgecolor="white", linewidth=0.5)
            for bar, val in zip(ax.patches, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f"{val:,.0f}", ha="center", va="bottom", fontsize=5, fontweight="bold")
            ax.set_title(row["scenario"], fontsize=7, fontweight="bold")
            ax.tick_params(axis="x", labelsize=5, pad=1)
            ax.tick_params(axis="y", labelsize=5, pad=1)
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
        fig2, ax2 = plt.subplots(figsize=(5, 2.5), facecolor="#fafafa")
        ax2.bar(names, totals, color=bar_colors, width=0.5, edgecolor="white", linewidth=0.5)
        for bar, val in zip(ax2.patches, totals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:,.0f}/mes\n({val*12:,.0f}/año)", ha="center", va="bottom", fontsize=6, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(axis="x", labelsize=6, pad=1)
        ax2.tick_params(axis="y", labelsize=6, pad=1)
        ax2.grid(axis="y", alpha=0.15, linewidth=0.3)
        ax2.set_ylim(0, max(totals) * 1.25)
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

    with st.container(border=True):
        st.markdown("**Exportar resultados**")
        col_xls, col_html = st.columns([1, 1])
        with col_xls:
            buf_xls = io.BytesIO()
            ens_txt_det = {"none":"Ninguno","basic":"Básico","medium":"Medio","high":"Alto"}.get(st.session_state.get("_uc_ens","medium"),"Medio")
            caps_det = []
            if st.session_state.get("_uc_cap_agentic"): caps_det.append("IA Agéntica (8.500€)")
            if st.session_state.get("_uc_cap_anon"): caps_det.append("Anonimización (3.500€)")
            if st.session_state.get("_uc_cap_sso"): caps_det.append("Autenticación SSO (4.000€)")
            total_src_xls = sum(st.session_state.get(f"_uc_src_{k}_cnt", 0) for k in CDU_SOURCE_KEYS)
            sources_det = []
            for k in CDU_SOURCE_KEYS:
                c = st.session_state.get(f"_uc_src_{k}_cnt", 0)
                if c > 0:
                    v = st.session_state.get(f"_uc_src_{k}_vol", 10.0)
                    f = st.session_state.get(f"_uc_src_{k}_freq", "daily")
                    sources_det.append({"Fuente": k, "Cantidad": c, "Vol. (GB)": v, "Frecuencia": f})
            df_sources = pd.DataFrame(sources_det) if sources_det else pd.DataFrame()
            with pd.ExcelWriter(buf_xls, engine="openpyxl") as writer:
                if df_totales is not None and not df_totales.empty:
                    df_totales.to_excel(writer, sheet_name="Resumen", index=False)
                df_sim = df[["scenario","total_cost_eur","aks_gpu_cost_eur","api_llm_cost_eur",
                             "aks_system_cost_eur","storage_cost_eur","lb_cost_eur",
                             "monitor_cost_eur","acr_cost_eur","gpu_peak_nodes","mc_p50_eur","mc_p90_eur"]].copy()
                df_sim.columns = ["Escenario","Total €","GPU €","API LLM €","Sistema €",
                                  "Almacenamiento €","LB €","Monitor €","ACR €","Nodos Pico","MC P50 €","MC P90 €"]
                df_sim.to_excel(writer, sheet_name="Simulación", index=False)
                lp_data = pd.DataFrame([{
                    "Usuarios": lp.users,
                    "Interacciones/usuario/día": lp.interactions_per_user_day,
                    "Tokens entrada/interacción": lp.input_tokens_per_interaction,
                    "Tokens salida/interacción": lp.output_tokens_per_interaction,
                    "Días laborables/mes": lp.working_days_per_month,
                    "Horas oficina/día": lp.office_hours_per_day,
                    "Horas pico/día": lp.peak_hours_per_day,
                    "Ratio concurrencia": lp.concurrent_user_ratio,
                    "Multiplicador pico": lp.peak_multiplier,
                    "Concurrentes estimados": int(lp.users * lp.concurrent_user_ratio),
                    "Tokens/mes": lp.total_tokens_per_month,
                }])
                lp_data.to_excel(writer, sheet_name="Perfil Carga", index=False)
                cdu_data = pd.DataFrame([{
                    "Total fuentes": total_src_xls,
                    "Capacidades": ", ".join(caps_det) if caps_det else "Ninguna",
                    "ENS": ens_txt_det,
                    "CAPEX implantación €": impl_cdu,
                    "OPEX recurrente/año €": rec_anual,
                }])
                cdu_data.to_excel(writer, sheet_name="CdU", index=False)
                if not df_sources.empty:
                    df_sources.to_excel(writer, sheet_name="Fuentes CdU", index=False)
                from usecase import SOURCE_INTEGRATION_TABLE, CAPABILITY_COSTS, ENS_COSTS, FREQUENCY_MULTIPLIERS
                rows_p = []
                def ap(cat, param, val):
                    rows_p.append({"Categoría": cat, "Parámetro": param, "Valor": val})
                ap("InfraCloud", "A100 GPU/h", f"{_gs('ideal_gpu_price', DEFAULT_IDEAL_GPU_PRICE):.2f} €")
                ap("InfraCloud", "A10 GPU/h", f"{_gs('eco_gpu_price', DEFAULT_ECO_GPU_PRICE):.2f} €")
                ap("InfraCloud", "Nodo sistema/h", f"{_gs('system_price', DEFAULT_SYSTEM_PRICE):.2f} €")
                ap("InfraCloud", "Rendimiento A100 (tok/s)", str(_gs('ideal_throughput', DEFAULT_IDEAL_THROUGHPUT)))
                ap("InfraCloud", "Rendimiento A10 (tok/s)", str(_gs('eco_throughput', DEFAULT_ECO_THROUGHPUT)))
                ap("InfraCloud", "Utilización GPU", f"{_gs('gpu_utilization', DEFAULT_GPU_UTILIZATION):.0%}")
                ap("InfraCloud", "Factor seguridad", f"{_gs('safety_factor', DEFAULT_SAFETY_FACTOR):.2f}")
                ap("InfraCloud", "Storage Ideal/mes", f"{_gs('ideal_storage', DEFAULT_STORAGE_IDEAL):.0f} €")
                ap("InfraCloud", "Storage Eco/mes", f"{_gs('eco_storage', DEFAULT_STORAGE_ECO):.0f} €")
                ap("InfraCloud", "LB/mes", f"{_gs('ideal_lb', DEFAULT_LB):.0f} €")
                ap("InfraCloud", "Monitor Ideal/mes", f"{_gs('ideal_monitor', DEFAULT_MONITOR_IDEAL):.0f} €")
                ap("InfraCloud", "Monitor Eco/mes", f"{_gs('eco_monitor', DEFAULT_MONITOR_ECO):.0f} €")
                ap("InfraCloud", "ACR Ideal/mes", f"{_gs('ideal_acr', DEFAULT_ACR_IDEAL):.0f} €")
                ap("InfraCloud", "ACR Eco/mes", f"{_gs('eco_acr', DEFAULT_ACR_ECO):.0f} €")
                ap("InfraCloud", "Modelo API", _gs('api_model', DEFAULT_API_MODEL))
                ap("InfraCloud", "Input $/1M tok", f"{_gs('api_input_price', DEFAULT_API_INPUT_PRICE):.2f}")
                ap("InfraCloud", "Output $/1M tok", f"{_gs('api_output_price', DEFAULT_API_OUTPUT_PRICE):.2f}")
                ap("InfraCloud", "EUR/USD", f"{_gs('eur_usd_rate', DEFAULT_EUR_USD):.2f}")
                ap("InfraCloud", "HA factor", f"{_gs('ha_factor', 1.15):.2f}")
                ap("InfraCloud", "Overhead pico", f"{_gs('overhead_factor', 0.1):.0%}")
                ap("InfraCloud", "MC iteraciones", str(_gs("mc_iterations", 5000)))
                for sk in ["sharepoint","database","web_scraping","api","pdf_dynamic"]:
                    sl = {"sharepoint":"SharePoint/Alfresco","database":"BD Oracle/SQL",
                          "web_scraping":"Web Scraping","api":"API REST","pdf_dynamic":"PDF Dinámico"}[sk]
                    for cmp, cl in [("low","baja"),("medium","media"),("high","alta")]:
                        v = _gs(f"_cdu_src_{sk}_{cmp}", SOURCE_INTEGRATION_TABLE[sk][cmp])
                        ap("CdU", f"Integración {sl} ({cl})", f"{v:,.0f} €")
                maint_pct = _gs("_cdu_maintenance_pct", 10.0)
                ap("CdU", "Mantenimiento anual (%)", f"{maint_pct:.1f} %")
                for fk, fv in FREQUENCY_MULTIPLIERS.items():
                    fl = {"realtime":"Tiempo real","hourly":"Cada hora","daily":"Diaria","weekly":"Semanal","monthly":"Mensual"}
                    v = _gs(f"_cdu_freq_{fk}", fv)
                    ap("CdU", f"Multiplicador frecuencia {fl.get(fk,fk)}", f"{v:.2f}")
                for ck, cl in [("agentic_ai","IA Agéntica"),("anonymization","Anonimización"),("sso","SSO")]:
                    capex = _gs(f"_cdu_cap_{ck}_capex", CAPABILITY_COSTS[ck]["capex"])
                    opex = _gs(f"_cdu_cap_{ck}_opex", CAPABILITY_COSTS[ck]["opex_monthly"])
                    ap("CdU", f"Capacidad {cl} CAPEX", f"{capex:,.0f} €")
                    ap("CdU", f"Capacidad {cl} OPEX/mes", f"{opex:,.0f} €")
                for ek, el in [("none","Ninguno"),("basic","Básico"),("medium","Medio"),("high","Alto")]:
                    v = _gs(f"_cdu_ens_{ek}_capex", ENS_COSTS[ek]["capex"])
                    ap("CdU", f"ENS {el} CAPEX", f"{v:,.0f} €")
                precios = pd.DataFrame(rows_p)
                precios.to_excel(writer, sheet_name="Precios", index=False)
            buf_xls.seek(0)
            st.download_button("📥 Exportar Excel", data=buf_xls, file_name="costes_completos.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True, type="primary")
        with col_html:
            html_data = generate_report_html(lp, df, data, mc_iterations, ha_factor, overhead_factor, df_sorted,
                                              impl_cdu=impl_cdu, rec_anual=rec_anual, cdu_result=cdu_result, cdu_tag=cdu_tag)
            st.download_button("📄 Exportar HTML", html_data, "simulation_report.html", mime="text/html",
                               use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Azure Pricing tab
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    col_lang_spacer, col_lang = st.columns([8, 1])
    with col_lang:
        st.segmented_control(
            _("Language"), options=["EN", "ES", "GL"], key="lang", default="ES",
            selection_mode="single", label_visibility="collapsed",
        )

    st.markdown(
        '<span title="Comparativa AKS (Ideal/Económico) vs Azure OpenAI API + costes de implantación por caso de uso">'
        '**Simulador de Costes de Asistentes Virtuales**</span>',
        unsafe_allow_html=True,
    )

    # Azure defaults
    if "_azure_defaults" not in st.session_state:
        rate = _gs("eur_usd_rate", DEFAULT_EUR_USD)
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
                st.session_state[k] = v
        else:
            st.session_state._azure_defaults = {}

    if _gs("_azure_defaults", None):
        azure_keys = _gs("_azure_defaults", {})
        st.markdown(
            f"<span style='font-size:0.7rem;color:#888' title='GPU prices loaded from Azure: "
            f"A100 = {azure_keys.get('ideal_gpu_price', 6.0):.2f} EUR/h, "
            f"A10 = {azure_keys.get('eco_gpu_price', 2.2):.2f} EUR/h, "
            f"System = {azure_keys.get('system_price', 0.8):.2f} EUR/h'></span>",
            unsafe_allow_html=True,
        )

    # Apply default preset on first run
    if "_preset_applied" not in st.session_state:
        apply_preset("750")
        st.session_state._preset_applied = True

    # Top row: Preset selector + CdU summary card (same frame)
    with st.container(border=True):
        col_pre, col_cdu = st.columns([1.5, 1])
        with col_pre:
            st.segmented_control(
                _("Quick preset"), options=list(PRESET_KEYS),
                format_func=lambda k: {"100":"100","200":"200","500":"500","750":"750",
                                       "1k":"1k","2k":"2k","3k":"3k","6k":"6k",
                                       "10k":"10k","20k":"20k","35k":"35k"}.get(k, k),
                key="_preset_sel", default="750",
                selection_mode="single", help=_("Auto-fill business parameters."),
                on_change=lambda: apply_preset(st.session_state._preset_sel),
            )
            p = PRESETS.get(_gs("_preset_sel", "750"))
            if p:
                st.caption(f"{p['users']:,.0f} usuarios")
        with col_cdu:
            total_src = sum(st.session_state.get(f"_uc_src_{k}_cnt", 0) for k in CDU_SOURCE_KEYS)
            caps_list = []
            if st.session_state.get("_uc_cap_agentic"): caps_list.append("IA Agéntica")
            if st.session_state.get("_uc_cap_anon"): caps_list.append("Anonimización")
            if st.session_state.get("_uc_cap_sso"): caps_list.append("SSO")
            ens_txt = {"none":"Ninguno","basic":"Básico","medium":"Medio","high":"Alto"}.get(st.session_state.get("_uc_ens","medium"),"Medio")
            st.markdown(
                f'<div style="padding:0.2rem 0;text-align:right;font-size:0.85rem;">'
                f'<strong>CdU</strong>: {total_src} {_("sources")} &middot; {", ".join(caps_list) if caps_list else _("sin caps.")} &middot; ENS {ens_txt}'
                f'</div>',
                unsafe_allow_html=True,
            )

    render_sidebar()
    data = build_data_from_ui()

    # Compute CdU costs
    cdu_pricing, cap_costs, ens_costs = _build_cdu_custom_pricing()
    sources = _build_uc_sources(custom_pricing=cdu_pricing)
    enabled_caps = [k for k, v in [("agentic_ai","_uc_cap_agentic"),("anonymization","_uc_cap_anon"),("sso","_uc_cap_sso")] if st.session_state.get(v)]
    ens_level = st.session_state.get("_uc_ens", "medium")

    mc_iter = _gs("mc_iterations", 5000)
    cdu_result = calculate_usecase_cost(sources, enabled_caps, ens_level, business_params=None, deployment="economy",
                                         capability_costs=cap_costs, ens_costs=ens_costs)
    impl_cdu = cdu_result.total_capex
    rec_anual = (cdu_result.source_maintenance_opex + cdu_result.capabilities_opex) * 12

    # Combined cost table with visual styling
    with st.container(border=True):
        st.markdown("**Costes combinados — CdU + Infraestructura Cloud**")
        with st.spinner("Simulando infraestructura cloud..."):
            df_infra = simulate_all(data, mc_iterations=mc_iter, resize=True)

        total_src = sum(st.session_state.get(f"_uc_src_{k}_cnt", 0) for k in CDU_SOURCE_KEYS)
        caps_names = []
        if st.session_state.get("_uc_cap_agentic"): caps_names.append("IA Agéntica")
        if st.session_state.get("_uc_cap_anon"): caps_names.append("Anonimización")
        if st.session_state.get("_uc_cap_sso"): caps_names.append("SSO")
        ens_nombre = {"none":"-","basic":"ENS Básico","medium":"ENS Medio","high":"ENS Alto"}.get(st.session_state.get("_uc_ens","medium"),"ENS Medio")
        cdu_tag = f"{total_src} fuentes · {ens_nombre}"
        if caps_names:
            cdu_tag += f" · {', '.join(caps_names)}"

        # Build HTML table + df_totales for Excel
        iconos = {"AKS UX Ideal":"\U0001f680", "AKS UX Economico":"\u2699\ufe0f", "API Azure OpenAI":"\u2601\ufe0f"}
        colores = {"AKS UX Ideal":"#1e40af", "AKS UX Economico":"#15803d", "API Azure OpenAI":"#b45309"}
        html_rows = ""
        rows_totales = []
        for _i, r in df_infra.iterrows():
            sc = r["scenario"]
            infra_anual = r["total_cost_eur"] * 12
            ano_impl = impl_cdu + rec_anual + infra_anual
            ano_sig = rec_anual + infra_anual
            rows_totales.append({
                "Escenario": cdu_tag + f"\n+ {sc}",
                "Implant. CdU": f"{impl_cdu:,.0f} €",
                "Recurrente/año": f"{rec_anual:,.0f} €",
                "Infra Cloud/año": f"{infra_anual:,.0f} €",
                "Año implantación": f"{ano_impl:,.0f} €",
                "Año siguiente": f"{ano_sig:,.0f} €",
            })
            icono = iconos.get(sc, "")
            cls = {"AKS UX Ideal":"ideal", "AKS UX Economico":"eco", "API Azure OpenAI":"api"}.get(sc, "")
            bg = colores.get(sc, "#333")
            html_rows += f"""<tr>
                <td class="sc-{cls}" style="font-weight:700;background:{bg}18">{icono} {sc}</td>
                <td style="text-align:right;font-weight:600">{impl_cdu:,.0f} €</td>
                <td style="text-align:right;font-weight:600">{rec_anual:,.0f} €</td>
                <td class="sc-{cls}" style="text-align:right;font-weight:700">{infra_anual:,.0f} €</td>
                <td class="year1" style="text-align:right;font-weight:700">{ano_impl:,.0f} €</td>
                <td style="text-align:right;font-weight:600">{ano_sig:,.0f} €</td>
            </tr>"""
        df_totales = pd.DataFrame(rows_totales)

        html_table = f"""<div class="cost-table" style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px;font-size:0.8rem;">
        <table style="width:100%;border-collapse:collapse;">
        <thead>
        <tr class="cost-table">
            <th style="padding:0.5rem 0.6rem;text-align:left;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.03em;">Escenario</th>
            <th style="padding:0.5rem 0.6rem;text-align:right;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.03em;">CdU Implant.</th>
            <th style="padding:0.5rem 0.6rem;text-align:right;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.03em;">CdU Recur./año</th>
            <th style="padding:0.5rem 0.6rem;text-align:right;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.03em;">Infra Cloud/año</th>
            <th style="padding:0.5rem 0.6rem;text-align:right;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.03em;">Año implantación</th>
            <th style="padding:0.5rem 0.6rem;text-align:right;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.03em;">Año siguiente</th>
        </tr>
        </thead>
        <tbody>
        {html_rows}
        </tbody>
        </table>
        <div class="footer" style="padding:0.4rem 0.6rem;font-size:0.7rem;border-top:1px solid #e2e8f0;">
            CdU: {cdu_tag} &middot; CAPEX único + recurrencia + infraestructura cloud
        </div>
        </div>"""

        col_tab, col_res = st.columns([1.5, 1])
        with col_tab:
            st.markdown(html_table, unsafe_allow_html=True)
        with col_res:
            st.markdown(f'<div class="cdu-card">'
                        f'<p><strong>CdU CAPEX (único):</strong> {impl_cdu:,.0f} €</p>'
                        f'<p><strong>CdU recurrente/año:</strong> {rec_anual:,.0f} €</p>'
                        f'<p><strong>Total infra cloud/año:</strong> '
                        f'{df_infra["total_cost_eur"].min()*12:,.0f} - {df_infra["total_cost_eur"].max()*12:,.0f} €</p>'
                        f'</div>',
                        unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Infraestructura Cloud — Simulación**")
        tab_simulation(data, df=df_infra, df_totales=df_totales, impl_cdu=impl_cdu, rec_anual=rec_anual,
                       cdu_result=cdu_result, cdu_tag=cdu_tag)

    with st.expander("Precios y simulación"):
        tab_infra, tab_cdu = st.tabs(["Infra/Cloud", "CdU"])

        with tab_infra:
            st.caption("Precios obtenidos de Azure Retail Prices API (West Europe). No configurables.")
            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                st.metric("A100 GPU/h", f"{_gs('ideal_gpu_price', DEFAULT_IDEAL_GPU_PRICE):.2f} €")
                st.metric("A10 GPU/h", f"{_gs('eco_gpu_price', DEFAULT_ECO_GPU_PRICE):.2f} €")
                st.metric("Nodo sistema/h", f"{_gs('system_price', DEFAULT_SYSTEM_PRICE):.2f} €")
            with ic2:
                st.metric("Modelo API", _gs("api_model", DEFAULT_API_MODEL))
                st.metric("Input $/1M tok", f"{_gs('api_input_price', DEFAULT_API_INPUT_PRICE):.2f}")
                st.metric("Output $/1M tok", f"{_gs('api_output_price', DEFAULT_API_OUTPUT_PRICE):.2f}")
            with ic3:
                st.metric("EUR/USD", f"{_gs('eur_usd_rate', DEFAULT_EUR_USD):.2f}")
                st.metric("HA factor", f"{_gs('ha_factor', 1.15):.2f}")
                st.metric("Overhead pico", f"{_gs('overhead_factor', 0.1):.0%}")
            st.caption("Rendimiento y costes fijos")
            ic4, ic5 = st.columns(2)
            with ic4:
                st.metric("A100 tok/s", str(_gs("ideal_throughput", DEFAULT_IDEAL_THROUGHPUT)))
                st.metric("A10 tok/s", str(_gs("eco_throughput", DEFAULT_ECO_THROUGHPUT)))
                st.metric("Utilización GPU", f"{_gs('gpu_utilization', DEFAULT_GPU_UTILIZATION):.0%}")
                st.metric("Factor seguridad", f"{_gs('safety_factor', DEFAULT_SAFETY_FACTOR):.2f}")
                st.metric("MC iteraciones", str(_gs("mc_iterations", 5000)))
            with ic5:
                st.metric("Storage Ideal/mes", f"{int(_gs('ideal_storage', DEFAULT_STORAGE_IDEAL))} €")
                st.metric("Storage Eco/mes", f"{int(_gs('eco_storage', DEFAULT_STORAGE_ECO))} €")
                st.metric("LB/mes", f"{int(_gs('ideal_lb', DEFAULT_LB))} €")
                st.metric("Monitor Ideal/mes", f"{int(_gs('ideal_monitor', DEFAULT_MONITOR_IDEAL))} €")
                st.metric("Monitor Eco/mes", f"{int(_gs('eco_monitor', DEFAULT_MONITOR_ECO))} €")
                st.metric("ACR Ideal/mes", f"{int(_gs('ideal_acr', DEFAULT_ACR_IDEAL))} €")
                st.metric("ACR Eco/mes", f"{int(_gs('eco_acr', DEFAULT_ACR_ECO))} €")

        with tab_cdu:
            from usecase import SOURCE_INTEGRATION_TABLE, CAPABILITY_COSTS, ENS_COSTS, FREQUENCY_MULTIPLIERS
            st.caption("Costes de integración de fuentes — configurables")
            src_labels = {"sharepoint":"SharePoint/Alfresco","database":"BD Oracle/SQL",
                         "web_scraping":"Web Scraping","api":"API REST","pdf_dynamic":"PDF Dinámico"}
            cmp_labels = {"low":"Baja","medium":"Media","high":"Alta"}
            for sk, sl in src_labels.items():
                with st.container(border=True):
                    st.markdown(f"**{sl}**")
                    cols = st.columns(3)
                    for i, cmp in enumerate(["low","medium","high"]):
                        key = f"_cdu_src_{sk}_{cmp}"
                        default = SOURCE_INTEGRATION_TABLE[sk][cmp]
                        with cols[i]:
                            st.number_input(
                                f"{cmp_labels[cmp]}", 0, 100000,
                                value=_gs(key, default), step=500, key=key,
                                help=f"Coste de integración {sl} - {cmp_labels[cmp]}"
                            )

            st.markdown("---")
            st.caption("Costes de capacidades — configurables")
            cap_labels = {"agentic_ai":"IA Agéntica","anonymization":"Anonimización","sso":"Autenticación SSO"}
            for ck, cl in cap_labels.items():
                with st.container(border=True):
                    st.markdown(f"**{cl}**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.number_input("CAPEX", 0, 100000,
                            value=_gs(f"_cdu_cap_{ck}_capex", CAPABILITY_COSTS[ck]["capex"]), step=500, key=f"_cdu_cap_{ck}_capex")
                    with c2:
                        st.number_input("OPEX/mes", 0, 10000,
                            value=_gs(f"_cdu_cap_{ck}_opex", CAPABILITY_COSTS[ck]["opex_monthly"]), step=100, key=f"_cdu_cap_{ck}_opex")

            st.markdown("---")
            st.caption("Costes ENS — configurables")
            ens_labels = {"none":"Ninguno","basic":"Básico","medium":"Medio","high":"Alto"}
            cols_ens = st.columns(4)
            for i, (ek, el) in enumerate(ens_labels.items()):
                with cols_ens[i]:
                    st.number_input(f"ENS {el}", 0, 50000,
                        value=_gs(f"_cdu_ens_{ek}_capex", ENS_COSTS[ek]["capex"]), step=500, key=f"_cdu_ens_{ek}_capex")

            st.markdown("---")
            st.caption("Soporte y mantenimiento — configurable")
            st.number_input("% mantenimiento anual sobre coste integración",
                0.0, 50.0, value=_gs("_cdu_maintenance_pct", 10.0), step=0.5, format="%.1f",
                key="_cdu_maintenance_pct",
                help="Porcentaje del coste de integración que se aplica anualmente en concepto de soporte y mantenimiento.")

            st.markdown("---")
            st.caption("Multiplicadores de frecuencia de actualización — configurables")
            cols_freq = st.columns(len(FREQUENCY_MULTIPLIERS))
            for i, (fk, fv) in enumerate(FREQUENCY_MULTIPLIERS.items()):
                freq_labels = {"realtime":"Tiempo real","hourly":"Cada hora","daily":"Diaria","weekly":"Semanal","monthly":"Mensual"}
                with cols_freq[i]:
                    st.number_input(freq_labels.get(fk, fk), 0.5, 2.0,
                        value=_gs(f"_cdu_freq_{fk}", fv), step=0.01, format="%.2f", key=f"_cdu_freq_{fk}")

    with st.expander("**Azure Pricing**", expanded=False):
        tab_azure_pricing(data)


if __name__ == "__main__":
    main()
