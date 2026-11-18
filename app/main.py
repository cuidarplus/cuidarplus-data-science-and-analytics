"""Dashboard executivo - Citizen Intelligence SP.

Interface Streamlit sobre o pipeline de ``src/``. Três áreas:

1. Visão Geral - indicadores (ISR, FRT, TSC) e análise exploratória.
2. Simulador Preditivo - inferência com os modelos ``.pkl`` treinados.
3. Recomendações técnicas - síntese, limitações e métricas dos modelos.

Execução (raiz do projeto): ``streamlit run app/main.py``.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, viz
from src.features import (
    ORDEM_CHUVA,
    ORDEM_DIAS,
    adicionar_features_temporais,
    calcular_frt,
    calcular_isr,
    calcular_kpis_gerais,
    calcular_taxa_sinistros_por_clima,
    preparar_features,
)
from src.modeling import ajustar_probabilidade_prior, carregar_metadata, carregar_modelo
from src.preprocessing import carregar_consolidado, carregar_horario

st.set_page_config(layout="wide", page_title="Citizen Intelligence SP", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Tema visual (escuro / claro)
# --------------------------------------------------------------------------- #
# O tema é escolhido na sidebar (vinculado à URL via ``?tema=``) e vale só para
# a sessão do usuário. O CSS é montado a partir de tokens; os componentes
# nativos herdam do config.toml (base escura) e são sobrescritos aqui para que
# os dois temas fiquem consistentes. Os gráficos usam ``viz.PALETAS[tema]``.
TEMAS = {"Escuro": "escuro", "Claro": "claro"}
TEMA = TEMAS.get(st.session_state.get("tema") or st.query_params.get("tema") or "Escuro", "escuro")
P: dict[str, Any] = viz.PALETAS[TEMA]

TOKENS = {
    "escuro": """
    --bg: #0a0a0a; --bg-sidebar: #0f0f11; --bg-sidebar-topo: #121215;
    --card: #1a1a1a; --card-soft: #121214; --campo: #141416;
    --linha: #26262a; --linha-suave: #1d1d21; --linha-forte: #3a3a40;
    --texto: #f2f4f7; --texto-2: #c4c4c9; --texto-3: #888888; --texto-4: #5a5a62;
    --acento: #ff4f00; --acento-2: #ff6a1f; --acento-texto: #ff7a33; --acento-suave: rgba(255, 79, 0, 0.10); --acento-borda: rgba(255, 79, 0, 0.40);
    --tag-texto: #ffb38a; --magenta: #ed145b; --magenta-suave: rgba(237, 20, 91, 0.16);
    --ambar: #ffb020; --ambar-texto: #ffb020; --ambar-fundo: rgba(255, 176, 32, 0.05); --ambar-borda: rgba(255, 176, 32, 0.22);
    --code-bg: #17171a; --code-texto: #ff8a4c; --linha-hover: rgba(255, 255, 255, 0.02);
    --glow: 0 0 0 1px rgba(255, 79, 0, 0.35), 0 10px 32px -12px rgba(255, 79, 0, 0.45);
    --glow-magenta: 0 0 0 1px rgba(237, 20, 91, 0.35), 0 10px 32px -12px rgba(237, 20, 91, 0.45);
    --sombra-card: none; --halo: rgba(255, 79, 0, 0.16);
    --grad-1: rgba(255, 79, 0, 0.07); --grad-2: rgba(237, 20, 91, 0.05);
    --botao-bg: #0f0f11; --botao-texto-hover: #0a0a0a; --scroll: #2a2a2e;
    --menu-sombra: 0 16px 40px -12px rgba(0, 0, 0, 0.8);
    """,
    "claro": """
    --bg: #f4f4f2; --bg-sidebar: #ffffff; --bg-sidebar-topo: #fbfbfa;
    --card: #ffffff; --card-soft: #ffffff; --campo: #ffffff;
    --linha: #e2e2e6; --linha-suave: #ebebee; --linha-forte: #c4c4cc;
    --texto: #111113; --texto-2: #3f3f46; --texto-3: #6b6b73; --texto-4: #a1a1aa;
    --acento: #e94800; --acento-2: #ff5a14; --acento-texto: #c23c00; --acento-suave: rgba(233, 72, 0, 0.08); --acento-borda: rgba(233, 72, 0, 0.35);
    --tag-texto: #a83400; --magenta: #c8104f; --magenta-suave: rgba(200, 16, 79, 0.10);
    --ambar: #d97706; --ambar-texto: #92400e; --ambar-fundo: #fffaf0; --ambar-borda: #f5dfb3;
    --code-bg: #f4f4f5; --code-texto: #b93a06; --linha-hover: rgba(17, 17, 19, 0.025);
    --glow: 0 0 0 1px rgba(233, 72, 0, 0.28), 0 12px 28px -14px rgba(233, 72, 0, 0.40);
    --glow-magenta: 0 0 0 1px rgba(200, 16, 79, 0.28), 0 12px 28px -14px rgba(200, 16, 79, 0.40);
    --sombra-card: 0 1px 2px rgba(16, 16, 20, 0.04), 0 1px 1px rgba(16, 16, 20, 0.02); --halo: rgba(233, 72, 0, 0.09);
    --grad-1: rgba(233, 72, 0, 0.045); --grad-2: rgba(200, 16, 79, 0.03);
    --botao-bg: #ffffff; --botao-texto-hover: #ffffff; --scroll: #d4d4d8;
    --menu-sombra: 0 16px 40px -16px rgba(16, 16, 20, 0.25);
    """,
}

CSS_BASE = """
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

  :root {
    --fonte-display: 'Space Grotesk', 'Inter', sans-serif;
    --fonte-texto: 'Inter', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
    --ease: cubic-bezier(0.22, 1, 0.36, 1);
  }

  /* ---------- Chrome do Streamlit: removido ---------- */
  #MainMenu, footer, div[data-testid="stDecoration"], div[data-testid="stToolbar"], div[data-testid="stStatusWidget"],
  div[data-testid="stAppDeployButton"], .stAppDeployButton, div[data-testid="stToolbarActions"] { display: none !important; }
  header[data-testid="stHeader"] { background: transparent !important; height: 2.25rem; pointer-events: none; }
  header[data-testid="stHeader"] button { pointer-events: auto; color: var(--texto-3) !important; }

  /* ---------- Base ---------- */
  html, body, .stApp { font-family: var(--fonte-texto); -webkit-font-smoothing: antialiased; }
  .stApp {
    background:
      radial-gradient(1200px 420px at 100% -10%, var(--grad-1), transparent 60%),
      radial-gradient(900px 380px at -10% 110%, var(--grad-2), transparent 60%),
      var(--bg) !important;
    color: var(--texto);
  }
  .stApp::before {
    content: ""; position: fixed; top: 0; left: 0; right: 0; height: 2px; z-index: 999999;
    background: linear-gradient(90deg, var(--acento) 0%, var(--magenta) 55%, transparent 100%);
  }
  ::selection { background: var(--acento-borda); color: var(--texto); }
  .block-container { padding-top: 1.75rem !important; padding-bottom: 3rem; max-width: 1440px; }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--scroll); border-radius: 8px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--acento); }

  /* ---------- Tipografia ---------- */
  div[data-testid="stMarkdownContainer"] h1 {
    font-family: var(--fonte-display) !important; font-size: 2.1rem !important; font-weight: 700 !important; text-transform: uppercase;
    letter-spacing: -0.03em; line-height: 1.05; color: var(--texto) !important; padding: 0 !important; margin: 0 0 0.35rem 0 !important;
  }
  div[data-testid="stMarkdownContainer"] h2 {
    font-family: var(--fonte-display) !important; font-size: 1.15rem !important; font-weight: 600 !important; letter-spacing: -0.01em;
    color: var(--texto) !important; padding: 0 !important; margin: 2rem 0 0.6rem 0 !important; display: flex; align-items: center; gap: 0.6rem;
  }
  div[data-testid="stMarkdownContainer"] h2::before {
    content: ""; width: 14px; height: 3px; border-radius: 2px; flex: none; background: linear-gradient(90deg, var(--acento), var(--magenta));
  }
  div[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--fonte-display) !important; font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--texto-3) !important; padding: 0 !important; margin: 1.1rem 0 0.45rem 0 !important;
  }
  div[data-testid="stMarkdownContainer"] h1 a, div[data-testid="stMarkdownContainer"] h2 a, div[data-testid="stMarkdownContainer"] h3 a,
  div[data-testid="stHeadingActionElements"] { display: none !important; }
  div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li { color: var(--texto-2); font-size: 0.92rem; line-height: 1.6; }
  div[data-testid="stMarkdownContainer"] strong { color: var(--texto); font-weight: 600; }
  div[data-testid="stMarkdownContainer"] a { color: var(--acento-texto); text-decoration: none; border-bottom: 1px solid var(--acento-borda); transition: border-color .2s; }
  div[data-testid="stMarkdownContainer"] a:hover { border-bottom-color: var(--acento); }
  div[data-testid="stMarkdownContainer"] code { background: var(--code-bg); color: var(--code-texto); border: 1px solid var(--linha-suave); border-radius: 4px; padding: 0.05rem 0.35rem; font-size: 0.82em; }
  .subtitulo { color: var(--texto-3) !important; font-size: 0.9rem !important; margin-bottom: 1rem !important; max-width: 920px; }
  .nota { color: var(--texto-3) !important; font-size: 0.8rem !important; line-height: 1.55; margin-top: 0.35rem; }
  .nota b, .nota i { color: var(--texto-2); }
  .katex { color: var(--texto); font-size: 1.05em; }
  div[data-testid="stCaptionContainer"] p { color: var(--texto-3) !important; }

  /* ---------- Tabelas (markdown e HTML) ---------- */
  div[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; margin: 0.4rem 0 0.8rem 0; font-size: 0.86rem; }
  div[data-testid="stMarkdownContainer"] th {
    background: var(--card-soft); color: var(--texto-3); font-family: var(--fonte-display); font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; font-size: 0.72rem; text-align: left; padding: 9px 12px; border: 0; border-bottom: 1px solid var(--linha);
  }
  div[data-testid="stMarkdownContainer"] td { color: var(--texto-2); padding: 9px 12px; border: 0; border-bottom: 1px solid var(--linha-suave); vertical-align: top; }
  div[data-testid="stMarkdownContainer"] tr { border: 0 !important; }
  div[data-testid="stMarkdownContainer"] tr:hover td { background: var(--linha-hover); }
  table.tabela th, table.tabela td { padding: 8px 10px !important; font-size: 0.8rem; }
  table.tabela td { white-space: nowrap; }
  table.tabela th { white-space: normal; line-height: 1.3; vertical-align: bottom; }
  .tabela-wrap { background: var(--card-soft); border: 1px solid var(--linha-suave); border-radius: 8px; overflow: auto; box-shadow: var(--sombra-card); margin: 0.2rem 0 0.6rem 0; }
  .tabela-wrap table.tabela { margin: 0 !important; }
  .tabela-wrap thead th { position: sticky; top: 0; z-index: 1; }
  table.tabela td:not(:first-child), table.tabela th:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; }
  table.tabela td:first-child { color: var(--texto); font-weight: 500; }
  table.tabela tr:last-child td { border-bottom: 0; }

  /* ---------- Cards: métricas ---------- */
  div[data-testid="stMetric"] {
    position: relative; background: var(--card); border: 1px solid var(--linha-suave); border-top: 2px solid var(--acento);
    border-radius: 8px; padding: 14px 16px 12px 16px; overflow: hidden; box-shadow: var(--sombra-card);
    transition: transform .35s var(--ease), box-shadow .35s var(--ease);
  }
  div[data-testid="stMetric"]::after {
    content: ""; position: absolute; inset: 0; pointer-events: none; opacity: 0; transition: opacity .35s var(--ease);
    background: linear-gradient(180deg, var(--acento-suave), transparent 55%);
  }
  div[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: var(--glow); }
  div[data-testid="stMetric"]:hover::after { opacity: 1; }
  div[data-testid="stMetric"] label p {
    font-family: var(--fonte-display); font-size: 0.68rem !important; color: var(--texto-3) !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
  }
  div[data-testid="stMetricValue"] {
    font-family: var(--fonte-display); font-size: 1.7rem !important; font-weight: 600; color: var(--texto) !important; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] p {
    font-family: var(--fonte-display); font-size: 1.7rem !important; font-weight: 600; color: var(--texto) !important; line-height: 1.15;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
  }
  div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    font-family: var(--fonte-display); font-size: 0.64rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--texto-3) !important; white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.3;
  }
  div[data-testid="stMetric"] [data-testid="stMetricLabel"] .stMarkdownEllipsis { overflow: visible !important; white-space: normal !important; }
  div[data-testid="stMetricDelta"] { font-size: 0.74rem !important; }
  div[data-testid="stMetricDelta"] p { font-size: 0.74rem !important; color: inherit !important; }

  /* ---------- Cards: gráficos, formulário, expanders ---------- */
  div[data-testid="stPlotlyChart"] {
    background: var(--card-soft); border: 1px solid var(--linha-suave); border-radius: 8px; padding: 6px 8px 2px 8px; box-shadow: var(--sombra-card);
    transition: border-color .35s var(--ease), box-shadow .35s var(--ease);
  }
  div[data-testid="stPlotlyChart"]:hover { border-color: var(--acento-borda); box-shadow: 0 12px 36px -18px var(--acento-borda); }
  div[data-testid="stForm"] {
    background: var(--card-soft); border: 1px solid var(--linha-suave); border-top: 2px solid var(--magenta); border-radius: 8px;
    padding: 18px 22px 10px 22px; box-shadow: var(--sombra-card);
  }
  div[data-testid="stExpander"] details {
    background: var(--card-soft); border: 1px solid var(--linha-suave) !important; border-radius: 8px; box-shadow: var(--sombra-card); transition: border-color .3s var(--ease);
  }
  div[data-testid="stExpander"] details:hover, div[data-testid="stExpander"] details[open] { border-color: var(--linha) !important; }
  div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * { color: var(--texto) !important; }
  div[data-testid="stExpander"] summary p { font-family: var(--fonte-display); font-weight: 500; letter-spacing: 0.01em; }
  div[data-testid="stExpander"] summary:hover p { color: var(--acento-texto) !important; }
  div[data-testid="stExpander"] summary [data-testid="stExpanderIcon"], div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] { color: var(--texto-3) !important; }

  /* ---------- Resultado do simulador ---------- */
  .resultado {
    position: relative; background: var(--card); border: 1px solid var(--linha-suave); border-left: 3px solid var(--acento);
    border-radius: 8px; padding: 16px 20px; overflow: hidden; box-shadow: var(--sombra-card); transition: box-shadow .35s var(--ease);
  }
  .resultado::after {
    content: ""; position: absolute; right: -60px; top: -60px; width: 180px; height: 180px; border-radius: 50%;
    background: radial-gradient(closest-side, var(--halo), transparent); pointer-events: none;
  }
  .resultado:hover { box-shadow: var(--glow); }
  .resultado .rotulo { font-family: var(--fonte-display); font-size: 0.7rem; color: var(--texto-3); font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; }
  .resultado .valor { font-family: var(--fonte-display); font-size: 2.5rem; font-weight: 700; color: var(--texto); line-height: 1.1; margin: 6px 0 4px 0; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
  .resultado .delta { font-size: 0.84rem; color: var(--texto-2); }
  .resultado .detalhe { font-size: 0.78rem; color: var(--texto-3); margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--linha-suave); }
  .resultado.teal { border-left-color: var(--magenta); }
  .resultado.teal::after { background: radial-gradient(closest-side, var(--magenta-suave), transparent); }
  .resultado.teal:hover { box-shadow: var(--glow-magenta); }

  /* ---------- Aviso técnico ---------- */
  .aviso { background: var(--ambar-fundo); border: 1px solid var(--ambar-borda); border-left: 3px solid var(--ambar); border-radius: 8px; padding: 14px 18px; margin: 0.8rem 0 1rem 0; }
  .aviso .titulo { font-family: var(--fonte-display); font-size: 0.72rem; color: var(--ambar-texto); font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px; }
  .aviso p { color: var(--texto-2) !important; font-size: 0.85rem !important; line-height: 1.6; margin: 0 0 0.4rem 0; }
  .aviso b { color: var(--texto); }

  /* ---------- Abas (react-aria) ---------- */
  div[data-testid="stTabs"] div[role="tablist"] { gap: 1.8rem; background: transparent !important; }
  div[data-testid="stTabs"] div[role="tablist"]::after { background: var(--linha-suave) !important; height: 1px !important; }
  div[data-testid="stTab"] { padding: 0.55rem 0 !important; background: transparent !important; }
  div[data-testid="stTab"] p {
    font-family: var(--fonte-display); font-size: 0.8rem !important; font-weight: 600 !important; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--texto-3) !important; transition: color .25s var(--ease);
  }
  div[data-testid="stTab"]:hover p { color: var(--texto-2) !important; }
  div[data-testid="stTab"][aria-selected="true"] p { color: var(--texto) !important; }
  div[data-testid="stTab"] .react-aria-SelectionIndicator { background: linear-gradient(90deg, var(--acento), var(--magenta)) !important; height: 2px !important; }

  /* ---------- Entradas (react-aria) ---------- */
  [data-testid="stWidgetLabel"] p {
    font-family: var(--fonte-display); font-size: 0.68rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--texto-3) !important;
  }
  [data-testid="stSelectbox"] div[role="group"], [data-testid="stMultiSelect"] div[role="group"], [data-testid="stDateInputField"] {
    background: var(--campo) !important; border: 1px solid var(--linha) !important; border-radius: 6px !important;
    transition: border-color .25s var(--ease), box-shadow .25s var(--ease);
  }
  [data-testid="stSelectbox"] div[role="group"]:hover, [data-testid="stMultiSelect"] div[role="group"]:hover, [data-testid="stDateInputField"]:hover { border-color: var(--linha-forte) !important; }
  [data-testid="stSelectbox"] div[role="group"]:focus-within, [data-testid="stMultiSelect"] div[role="group"]:focus-within, [data-testid="stDateInputField"]:focus-within {
    border-color: var(--acento) !important; box-shadow: 0 0 0 3px var(--acento-suave);
  }
  [data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input, [data-testid="stDateInputField"] [role="spinbutton"], [data-testid="stDateInputField"] span {
    color: var(--texto) !important; -webkit-text-fill-color: var(--texto);
  }
  [data-testid="stSelectbox"] button, [data-testid="stMultiSelect"] button:not([aria-label^="Remove"]), [data-testid="stDateInput"] button { color: var(--texto-3) !important; }
  [data-testid="stMultiSelect"] span[data-tag] { background: var(--acento-suave) !important; border: 1px solid var(--acento-borda) !important; border-radius: 4px !important; }
  [data-testid="stMultiSelect"] span[data-tag] > span { color: var(--tag-texto) !important; font-family: var(--fonte-display); font-size: 0.76rem; font-weight: 500; letter-spacing: 0.03em; }
  [data-testid="stMultiSelect"] span[data-tag] button { color: var(--tag-texto) !important; opacity: 0.75; }
  [data-testid="stMultiSelect"] span[data-tag] button:hover { opacity: 1; }
  [data-testid="stRadioOption"] p, [data-testid="stCheckbox"] p { color: var(--texto-2) !important; }
  [data-testid="stRadioOption"]:not([data-selected="true"]) > div > div:not([data-testid]) { background: var(--campo) !important; border: 1px solid var(--linha-forte) !important; }
  [data-testid="stRadioOption"]:not([data-selected="true"]) > div > div:not([data-testid]) > div { background: transparent !important; }
  [data-testid="stSliderTickBar"] p { color: var(--texto-4) !important; font-size: 0.72rem !important; }
  [data-testid="stSliderThumbValue"] p { font-family: var(--fonte-display); color: var(--acento) !important; font-weight: 600; }

  /* menus suspensos, calendário e tooltips (portais fora do app) */
  [data-testid$="Dropdown"] {
    background: var(--card) !important; border: 1px solid var(--linha) !important; border-radius: 8px !important; box-shadow: var(--menu-sombra) !important;
  }
  [data-testid$="Dropdown"] [role="listbox"] { background: transparent !important; }
  [data-testid$="Dropdown"] [role="option"], [data-testid$="Dropdown"] [role="option"] * { color: var(--texto-2) !important; }
  [data-testid$="Dropdown"] [role="option"][data-focused="true"], [data-testid$="Dropdown"] [role="option"][data-hovered="true"], [data-testid$="Dropdown"] [role="option"]:hover {
    background: var(--acento-suave) !important;
  }
  [data-testid$="Dropdown"] [role="option"][data-focused="true"] *, [data-testid$="Dropdown"] [role="option"][data-selected="true"] * { color: var(--texto) !important; }
  [role="dialog"]:has([role="grid"]) { background: var(--card) !important; border: 1px solid var(--linha) !important; border-radius: 8px !important; box-shadow: var(--menu-sombra) !important; color: var(--texto) !important; }
  [role="dialog"]:has([role="grid"]) * { color: var(--texto-2); }
  [role="dialog"]:has([role="grid"]) h2, [role="dialog"]:has([role="grid"]) button { color: var(--texto) !important; }
  [role="dialog"] [role="gridcell"] [data-selected="true"], [role="dialog"] [role="gridcell"] [aria-selected="true"] { color: #ffffff !important; }
  [data-testid="stTooltipContent"], div[role="tooltip"] { background: var(--card) !important; color: var(--texto-2) !important; border: 1px solid var(--linha); box-shadow: var(--menu-sombra); }
  [data-testid="stTooltipContent"] p, div[role="tooltip"] p { color: var(--texto-2) !important; font-size: 0.8rem !important; }
  [data-testid="stTooltipIcon"] svg, [data-testid="stTooltipHoverTarget"] svg { color: var(--texto-4) !important; }

  /* blocos de código, alertas, spinner */
  div[data-testid="stCode"] pre, div[data-testid="stCode"] code { background: var(--code-bg) !important; color: var(--texto) !important; }
  div[data-testid="stCode"] pre { border: 1px solid var(--linha-suave); border-radius: 8px; }
  div[data-testid="stAlertContainer"] { border-radius: 8px; }
  div[data-testid="stSpinner"] p, div[data-testid="stSpinner"] { color: var(--texto-3) !important; }

  /* ---------- Botão de ação ---------- */
  button[kind="primary"], button[kind="primaryFormSubmit"], button[data-testid="stBaseButton-primaryFormSubmit"], button[data-testid="stBaseButton-primary"] {
    background-color: var(--botao-bg) !important; background-image: linear-gradient(90deg, var(--acento), var(--acento-2)) !important;
    background-repeat: no-repeat !important; background-size: 0% 100% !important; background-position: left center !important;
    border: 1px solid var(--acento) !important; border-radius: 6px !important; color: var(--acento) !important;
    transition: background-size .45s var(--ease), color .3s var(--ease), box-shadow .35s var(--ease), transform .2s var(--ease) !important;
  }
  button[kind="primary"] p, button[kind="primaryFormSubmit"] p, button[data-testid="stBaseButton-primaryFormSubmit"] p, button[data-testid="stBaseButton-primary"] p {
    font-family: var(--fonte-display); font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.8rem; color: inherit !important;
  }
  button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background-size: 100% 100% !important; color: var(--botao-texto-hover) !important; box-shadow: 0 0 24px -4px var(--acento-borda);
  }
  button[kind="primary"]:active, button[data-testid="stBaseButton-primaryFormSubmit"]:active { transform: scale(0.985); }
  button[kind="primary"]:focus-visible, button[data-testid="stBaseButton-primaryFormSubmit"]:focus-visible { outline: 2px solid var(--magenta) !important; outline-offset: 2px; }
  button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    background: transparent !important; border: 1px solid var(--linha) !important; border-radius: 6px !important; color: var(--texto-2) !important;
    transition: border-color .25s var(--ease), color .25s var(--ease) !important;
  }
  button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover { border-color: var(--acento) !important; color: var(--acento-texto) !important; }
  button[kind="secondary"] p, button[data-testid="stBaseButton-secondary"] p { font-family: var(--fonte-display); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.1em; color: inherit !important; }

  /* ---------- Sidebar: painel de controle ---------- */
  section[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--bg-sidebar-topo) 0%, var(--bg-sidebar) 40%) !important; border-right: 1px solid var(--linha-suave); }
  section[data-testid="stSidebar"] > div { padding-top: 0.5rem; }
  section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { height: 2rem; min-height: 2rem; padding-bottom: 0; }
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button, section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button { color: var(--texto-3) !important; }
  section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p { font-size: 0.82rem; color: var(--texto-3); line-height: 1.55; }
  section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h3 {
    font-size: 0.64rem !important; letter-spacing: 0.18em; color: var(--texto-3) !important; margin: 1.5rem 0 0.55rem 0 !important; display: flex; align-items: center; gap: 0.5rem;
  }
  section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h3::before {
    content: ""; width: 6px; height: 6px; background: var(--acento); border-radius: 1px; box-shadow: 0 0 8px var(--acento-borda); flex: none;
  }
  section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h3::after { content: ""; flex: 1; height: 1px; background: var(--linha-suave); }
  section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.55rem; }

  .marca { padding: 0.2rem 0 1rem 0; border-bottom: 1px solid var(--linha-suave); }
  .marca .sigla { display: inline-flex; align-items: center; gap: 0.45rem; font-family: var(--fonte-display); font-size: 0.62rem; font-weight: 600; letter-spacing: 0.2em; text-transform: uppercase; color: var(--acento-texto); }
  .marca .sigla::before { content: ""; width: 18px; height: 2px; background: linear-gradient(90deg, var(--acento), var(--magenta)); }
  .marca .nome { font-family: var(--fonte-display); font-size: 1.32rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.05; color: var(--texto); text-transform: uppercase; margin: 0.45rem 0 0.4rem 0; }
  .marca .desc { font-size: 0.78rem; color: var(--texto-3); line-height: 1.5; }

  .painel { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--linha-suave); border: 1px solid var(--linha-suave); border-radius: 8px; overflow: hidden; margin-top: 0.2rem; }
  .painel > div { background: var(--card-soft); padding: 9px 11px; }
  .painel .k { font-family: var(--fonte-display); font-size: 0.58rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--texto-4); }
  .painel .v { font-family: var(--fonte-display); font-size: 0.98rem; font-weight: 600; color: var(--texto); font-variant-numeric: tabular-nums; margin-top: 2px; }
  .painel .v small { font-size: 0.68rem; color: var(--texto-3); font-weight: 500; }
  .painel .v.acento { color: var(--acento-texto); }

  /* navegação: radio vira lista de itens com marcador ativo */
  .st-key-pagina, .st-key-pagina [data-testid="stRadio"], .st-key-pagina div[role="radiogroup"], .st-key-pagina div[role="radiogroup"] > div { width: 100% !important; }
  .st-key-pagina div[role="radiogroup"] { gap: 2px !important; }
  .st-key-pagina [data-testid="stRadioOption"] {
    width: 100%; margin: 0; padding: 0.55rem 0.75rem; border-radius: 6px; border-left: 2px solid transparent; cursor: pointer;
    transition: background .2s var(--ease), border-color .2s var(--ease);
  }
  .st-key-pagina [data-testid="stRadioOption"] > div > div:not([data-testid]) { display: none !important; }
  .st-key-pagina [data-testid="stRadioOption"] p { font-family: var(--fonte-display); font-size: 0.86rem !important; font-weight: 500; letter-spacing: 0.02em; color: var(--texto-3) !important; }
  .st-key-pagina [data-testid="stRadioOption"]:hover { background: var(--linha-hover); }
  .st-key-pagina [data-testid="stRadioOption"]:hover p { color: var(--texto) !important; }
  .st-key-pagina [data-testid="stRadioOption"][data-selected="true"] { background: var(--acento-suave); border-left-color: var(--acento); }
  .st-key-pagina [data-testid="stRadioOption"][data-selected="true"] p { color: var(--texto) !important; font-weight: 600; }

  /* seletor de tema (segmented control) */
  .st-key-tema div[role="radiogroup"] { width: 100%; }
  .st-key-tema button {
    flex: 1; border-radius: 6px !important; background: var(--campo) !important; border: 1px solid var(--linha) !important; color: var(--texto-3) !important;
    transition: border-color .25s var(--ease), color .25s var(--ease), background .25s var(--ease) !important;
  }
  .st-key-tema button:hover { border-color: var(--linha-forte) !important; color: var(--texto) !important; }
  .st-key-tema button[aria-checked="true"] { background: var(--acento-suave) !important; border-color: var(--acento) !important; color: var(--acento-texto) !important; }
  .st-key-tema button p { color: inherit !important; }
  .st-key-tema button p { font-family: var(--fonte-display); font-size: 0.72rem !important; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; }

  section[data-testid="stSidebar"] .nota { font-size: 0.72rem !important; margin-top: 0.1rem; }
  .recorte { margin-bottom: 0.35rem; font-size: 0.74rem; color: var(--texto-3); line-height: 1.5; padding: 8px 10px; border: 1px dashed var(--linha); border-radius: 6px; }
  .recorte b { color: var(--texto); font-weight: 600; font-variant-numeric: tabular-nums; }
  .equipe { margin: 0.2rem 0 0.4rem; padding: 10px 12px; background: var(--card); border: 1px solid var(--linha); border-radius: 8px; }
  .equipe .titulo { font-family: var(--fonte-display); font-size: 0.6rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: var(--texto-4); margin-bottom: 0.35rem; }
  .equipe .membro { display: flex; flex-direction: column; gap: 1px; font-size: 0.78rem; color: var(--texto); line-height: 1.4; padding: 5px 0; }
  .equipe .membro + .membro { border-top: 1px solid var(--linha-suave); }
  .equipe .rm { font-family: var(--fonte-display); font-size: 0.66rem; color: var(--acento-texto); letter-spacing: 0.08em; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .rodape { margin-top: 1.4rem; padding-top: 0.8rem; border-top: 1px solid var(--linha-suave); font-family: var(--fonte-display); font-size: 0.6rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--texto-4); line-height: 1.8; }

  @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition: none !important; } }
"""
st.markdown(f"<style>:root {{{TOKENS[TEMA]}}}{CSS_BASE}</style>", unsafe_allow_html=True)

INTEGRANTES: list[tuple[str, str]] = [
    ("José Ribeiro dos Santos Neto", "567692"),
    ("Danilo Roberto Anterio dos Santos", "566966"),
]

ROTULO_SEVERIDADE = {"SEM_VITIMA": "Sem vítima", "COM_VITIMA": "Com vítima", "FATAL": "Fatal"}
ROTULO_CHUVA = {"SEM_CHUVA": "Sem chuva", "FRACA": "Fraca (≤ 2,5 mm)", "MODERADA": "Moderada (≤ 10 mm)", "FORTE": "Forte (> 10 mm)"}
LARGURA = "stretch"


def fmt_int(valor: float) -> str:
    """Formata inteiro com separador de milhar em padrão pt-BR."""
    return f"{int(valor):,}".replace(",", ".")


def fmt_num(valor: float, casas: int = 2) -> str:
    """Formata número decimal em padrão pt-BR."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_delta(valor: float, casas: int = 1) -> str:
    """Formata variação com sinal explícito em padrão pt-BR."""
    return ("+" if valor >= 0 else "−") + fmt_num(abs(valor), casas)


def tabela_html(dados: pd.DataFrame | pd.Series, casas: int = 2, indice: bool = True, alvo: Any = None, altura: int | None = None) -> None:
    """Renderiza uma tabela HTML estilizada pelo tema (substitui ``st.dataframe``).

    ``st.dataframe`` desenha em canvas com as cores do config.toml e não acompanha
    a troca de tema por sessão; a tabela HTML herda os tokens CSS do tema ativo.
    Números são formatados em padrão pt-BR (inteiros com milhar, decimais com ``casas``).
    """
    df = dados.to_frame() if isinstance(dados, pd.Series) else dados.copy()
    df = df.reset_index() if indice else df.reset_index(drop=True)

    def formatar(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "–"
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, (int, np.integer)):
            return fmt_int(v)
        if isinstance(v, (float, np.floating)):
            return fmt_num(float(v), casas)
        return str(v)

    html = df.map(formatar).to_html(classes="tabela", border=0, index=False, escape=True).replace("\n", "")
    estilo = f' style="max-height:{altura}px"' if altura else ""
    (alvo or st).markdown(f'<div class="tabela-wrap"{estilo}>{html}</div>', unsafe_allow_html=True)


def nota(texto: str) -> None:
    """Renderiza um parágrafo de nota técnica em fonte reduzida."""
    st.markdown(f'<p class="nota">{texto}</p>', unsafe_allow_html=True)


def aviso(titulo: str, paragrafos: list[str]) -> None:
    """Renderiza um bloco de aviso técnico (disclaimer) com título e parágrafos."""
    corpo = "".join(f"<p>{p}</p>" for p in paragrafos)
    st.markdown(f'<div class="aviso"><div class="titulo">{titulo}</div>{corpo}</div>', unsafe_allow_html=True)


FEATURES_CALENDARIO = ("hora", "dia_semana_num", "mes", "fim_de_semana", "feriado")
FEATURES_CLIMA = ("precipitacao", "temperatura", "umidade", "vento", "pressao")


def participacao_ganho(importancias: dict[str, float], prefixos: tuple[str, ...] = (), contem: str | None = None) -> float:
    """Fração do ganho total (XGBoost) atribuída a um grupo de atributos.

    O grupo é definido por prefixo do nome do atributo (``hora`` cobre também
    ``hora_pico``) ou por substring (útil para dummies ``*_NAO DISPONIVEL``).
    """
    total = sum(importancias.values())
    if not total:
        return float("nan")
    grupo = sum(v for k, v in importancias.items() if (prefixos and k.startswith(prefixos)) or (contem and contem in k))
    return grupo / total


@st.cache_data(show_spinner="Carregando bases processadas...")
def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê os parquets processados e aplica a engenharia de atributos."""
    return preparar_features(carregar_consolidado()), preparar_features(carregar_horario())


@st.cache_resource(show_spinner="Carregando modelos...")
def carregar_modelos() -> tuple[Any, Any, dict[str, Any]]:
    """Carrega os dois pipelines ``.pkl`` e o JSON de metadados."""
    return carregar_modelo(config.MODELO_CLASSIFICACAO_PKL), carregar_modelo(config.MODELO_REGRESSAO_PKL), carregar_metadata()


def layout(fig: go.Figure, titulo: str, altura: int = 380, **kwargs: Any) -> go.Figure:
    """Aplica o tema corporativo e remove bordas/gridlines secundárias."""
    fig.update_layout(**P["layout"])
    fig.update_layout(height=altura, title_text=titulo, **kwargs)
    eixo = dict(tickfont_color=P["texto_sec"], title_font_color=P["texto_sec"], zeroline=False)
    fig.update_xaxes(showgrid=False, showline=False, mirror=False, minor_showgrid=False, **eixo)
    fig.update_yaxes(showline=False, mirror=False, minor_showgrid=False, gridcolor=P["grade"], gridwidth=1, **eixo)
    return fig


def grafico_sinistros_por_hora(df: pd.DataFrame) -> go.Figure:
    por_hora = df.groupby(["hora", "severidade"]).size().rename("n").reset_index()
    fig = go.Figure()
    for sev in ["SEM_VITIMA", "COM_VITIMA", "FATAL"]:
        d = por_hora[por_hora.severidade == sev]
        fig.add_trace(go.Scatter(
            x=d.hora, y=d.n, mode="lines", name=ROTULO_SEVERIDADE[sev],
            line=dict(color=P["severidade"][sev], width=2),
            hovertemplate="Hora %{x:02d}:00 · %{y:,.0f} sinistros<extra>" + ROTULO_SEVERIDADE[sev] + "</extra>",
        ))
    for ini, fim in ((7, 10), (17, 20)):
        fig.add_vrect(x0=ini - 0.5, x1=fim - 0.5, fillcolor=P["faixa"], line_width=0)
    return layout(fig, "Sinistros por hora do dia e severidade (faixas: horário de pico)",
                  xaxis_title="Hora", yaxis_title="Sinistros", hovermode="x unified",
                  xaxis=dict(tickmode="linear", dtick=2, showgrid=False))


def grafico_lentidao_por_hora(horario: pd.DataFrame) -> go.Figure:
    h = horario.dropna(subset=["km_lentidao_total"]).copy()
    h["tipo"] = np.where((h.fim_de_semana == 1) | (h.feriado == 1), "Fim de semana / feriado", "Dia útil")
    perfil = h.groupby(["tipo", "hora"]).km_lentidao_total.agg(
        media="mean", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75)).reset_index()
    fig = go.Figure()
    for i, tipo in enumerate(["Dia útil", "Fim de semana / feriado"]):
        d = perfil[perfil.tipo == tipo]
        cor = P["series"][i]
        fig.add_trace(go.Scatter(x=pd.concat([d.hora, d.hora[::-1]]), y=pd.concat([d.p75, d.p25[::-1]]), fill="toself",
                                 fillcolor=cor, opacity=0.12, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=d.hora, y=d.media, mode="lines", name=tipo, line=dict(color=cor, width=2),
                                 hovertemplate="Hora %{x:02d}:00 · %{y:.2f} km<extra>" + tipo + "</extra>"))
    return layout(fig, "Lentidão média por hora (faixa: quartis 25–75 %)", xaxis_title="Hora", yaxis_title="km de lentidão",
                  hovermode="x unified", xaxis=dict(tickmode="linear", dtick=2, showgrid=False))


def grafico_heatmap_semana(df: pd.DataFrame) -> go.Figure:
    """Heatmap de incidência: dia da semana (Y) × hora do dia (X), cor = volume de sinistros."""
    mapa = (df.groupby(["dia_semana", "hora"], observed=True).size().unstack("hora")
            .reindex(index=ORDEM_DIAS, columns=range(24)).fillna(0))
    participacao = mapa.values / mapa.values.sum() * 100
    fig = go.Figure(go.Heatmap(
        z=mapa.values, x=mapa.columns, y=mapa.index, customdata=participacao,
        colorscale=P["sequencial"], xgap=2, ygap=2,
        colorbar=dict(title=dict(text="Sinistros", font=dict(size=11)), thickness=8, outlinewidth=0, len=0.85, tickfont=dict(size=10)),
        hovertemplate="<b>%{y}, %{x:02d}h</b><br>%{z:,.0f} sinistros · %{customdata:.2f} % do total<extra></extra>",
    ))
    return layout(fig, "Incidência temporal de sinistros: dia da semana × hora", altura=340,
                  xaxis=dict(title="Hora do dia", tickmode="linear", dtick=2, showgrid=False, zeroline=False),
                  yaxis=dict(autorange="reversed", showgrid=False, zeroline=False))


def grafico_lentidao_chuva_hora(horario: pd.DataFrame) -> go.Figure:
    """Perfil horário de dias úteis: lentidão média em horas secas vs. com chuva, com precipitação ao fundo."""
    h = horario[(horario.fim_de_semana == 0) & (horario.feriado == 0)]
    chuva = h.dropna(subset=["precipitacao_mm"]).groupby("hora").agg(
        mm=("precipitacao_mm", "mean"), freq=("chovendo", lambda s: 100 * s.astype(float).mean())).reindex(range(24))
    lent = (h.dropna(subset=["km_lentidao_total", "chovendo"])
            .groupby(["hora", "chovendo"]).km_lentidao_total.agg(["mean", "size"]).unstack("chovendo").reindex(range(24)))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=chuva.index, y=chuva.mm, name="Precipitação média (mm/h)", marker_color=P["chuva_fundo"], marker_line_width=0,
        width=0.7, customdata=chuva.freq,
        hovertemplate="Precipitação %{y:.2f} mm/h · chuva em %{customdata:.1f} % das horas<extra></extra>",
    ), secondary_y=True)
    for flag, nome, cor, traco in ((0, "Lentidão: horas secas", P["neutro"], "dot"), (1, "Lentidão: horas com chuva", P["series"][0], "solid")):
        fig.add_trace(go.Scatter(
            x=lent.index, y=lent[("mean", flag)], customdata=lent[("size", flag)], mode="lines", name=nome,
            line=dict(color=cor, width=2.2 if flag else 1.8, dash=traco),
            hovertemplate=f"{nome.split(': ')[1].capitalize()} %{{y:.1f}} km (n = %{{customdata:,.0f}} h)<extra></extra>",
        ), secondary_y=False)

    fig.update_yaxes(title_text="km de lentidão", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="mm/h", showgrid=False, rangemode="tozero", range=[0, float(chuva.mm.max()) * 3],
                     title_font=dict(color=P["texto_sec"]), tickfont=dict(color=P["texto_sec"]), secondary_y=True)
    fig = layout(fig, "Tráfego e clima por hora do dia (dias úteis)", altura=380, hovermode="x unified",
                 xaxis=dict(title="Hora do dia", tickmode="linear", dtick=2, showgrid=False), bargap=0.15)
    fig.update_layout(hoverlabel=dict(namelength=0), xaxis_hoverformat="02d")
    fig.update_xaxes(ticksuffix="h")
    return fig


def grafico_chuva(tsc: pd.DataFrame, isr_geral: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Sinistros por hora observada", "ISR (proporção com vítima)"), horizontal_spacing=0.12)
    faixas = tsc.faixa_chuva.astype(str)
    cores = [P["chuva"][f] for f in faixas]
    rot = [ROTULO_CHUVA[f] for f in faixas]
    fig.add_trace(go.Bar(x=rot, y=tsc.sinistros_por_hora, marker_color=cores, marker_line_width=0, width=0.55,
                         text=[fmt_num(v) for v in tsc.sinistros_por_hora], textposition="outside", textfont=dict(color=P["texto_sec"], size=11),
                         customdata=tsc.horas, hovertemplate="%{x}<br>%{y:.2f} sinistros/h · %{customdata:,.0f} h observadas<extra></extra>", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=rot, y=tsc.isr, marker_color=cores, marker_line_width=0, width=0.55,
                         text=[fmt_num(v, 3) for v in tsc.isr], textposition="outside", textfont=dict(color=P["texto_sec"], size=11),
                         customdata=tsc.sinistros, hovertemplate="%{x}<br>ISR %{y:.3f} · n = %{customdata:,.0f}<extra></extra>", showlegend=False), row=1, col=2)
    fig.add_hline(y=isr_geral, line_dash="dot", line_color=P["neutro"], line_width=1, row=1, col=2,
                  annotation_text=f"ISR geral {fmt_num(isr_geral, 3)}", annotation_position="top left", annotation_font=dict(size=11, color=P["texto_sec"]))
    fig.update_yaxes(range=[0, tsc.sinistros_por_hora.max() * 1.25], row=1, col=1)
    fig.update_yaxes(range=[0.5, 0.72], row=1, col=2)
    fig.update_annotations(font=dict(size=12, color=P["texto"]))
    return layout(fig, "Chuva: frequência e severidade dos sinistros", altura=360)


def grafico_isr_heatmap(isr: pd.DataFrame) -> go.Figure:
    mapa = isr.pivot(index="categoria_via", columns="faixa_chuva", values="isr").reindex(columns=ORDEM_CHUVA)
    mapa = mapa.loc[mapa["SEM_CHUVA"].sort_values().index]
    fig = go.Figure(go.Heatmap(
        z=mapa.values, x=[ROTULO_CHUVA[c] for c in mapa.columns], y=mapa.index, colorscale=P["sequencial"],
        zmin=0.5, zmax=0.75, xgap=2, ygap=2, text=np.where(np.isnan(mapa.values), "", np.vectorize(lambda v: fmt_num(v, 2))(np.nan_to_num(mapa.values))), texttemplate="%{text}",
        textfont=dict(size=11), colorbar=dict(title="ISR", thickness=10, outlinewidth=0),
        hovertemplate="%{y} · %{x}<br>ISR %{z:.3f}<extra></extra>",
    ))
    return layout(fig, "ISR por categoria de via × faixa de chuva (n ≥ 30)", altura=400, yaxis=dict(showgrid=False))


def grafico_frt(frt: pd.DataFrame, coluna: str) -> go.Figure:
    d = frt.sort_values("excedente_km")
    rot = d[coluna].map(ROTULO_SEVERIDADE).fillna(d[coluna]) if coluna == "severidade" else d[coluna]
    fig = go.Figure(go.Bar(
        x=d.excedente_km, y=rot, orientation="h", marker_color=P["series"][0], marker_line_width=0, width=0.5,
        text=[f"+{fmt_num(v, 1)} km · n = {fmt_int(n)}" for v, n in zip(d.excedente_km, d.total_sinistros)],
        textposition="outside", textfont=dict(color=P["texto_sec"], size=11),
        customdata=d[["frt_km_medio", "lentidao_tipica_km", "excedente_pct"]].values,
        hovertemplate="%{y}<br>FRT %{customdata[0]:.2f} km · típico %{customdata[1]:.2f} km · excedente %{customdata[2]:.1f} %<extra></extra>",
    ))
    fig.update_xaxes(range=[0, d.excedente_km.max() * 1.7])
    return layout(fig, "FRT: lentidão excedente em relação ao típico do mesmo dia/hora", altura=320,
                  xaxis_title="km acima da mediana do horário", yaxis=dict(showgrid=False))


def grafico_mensal(horario: pd.DataFrame) -> go.Figure:
    m = horario.assign(mes_ref=horario.datetime.dt.to_period("M").dt.to_timestamp()).groupby("mes_ref").agg(
        sinistros=("n_sinistros", "sum"), lentidao=("km_lentidao_total", "mean"), chuva=("precipitacao_mm", "sum")).reset_index()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Sinistros por mês", "Lentidão média (km)", "Chuva acumulada (mm)"))
    fig.add_trace(go.Scatter(x=m.mes_ref, y=m.sinistros, mode="lines", line=dict(color=P["series"][0], width=2), name="Sinistros",
                             hovertemplate="%{x|%b %Y} · %{y:,.0f} sinistros<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=m.mes_ref, y=m.lentidao, mode="lines", line=dict(color=P["series"][1], width=2), name="Lentidão",
                             hovertemplate="%{x|%b %Y} · %{y:.2f} km<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(x=m.mes_ref, y=m.chuva, marker_color=P["neutro_escuro"], marker_line_width=0, name="Chuva",
                         hovertemplate="%{x|%b %Y} · %{y:.1f} mm<extra></extra>"), row=3, col=1)
    fig.update_annotations(font=dict(size=12, color=P["texto"]))
    return layout(fig, "Evolução mensal", altura=520, showlegend=False)


def secao_visao_geral(consolidado: pd.DataFrame, horario: pd.DataFrame, anos: list[int], zonas: list[str]) -> None:
    sin = consolidado[consolidado.ano.isin(anos) & consolidado.zona.isin(zonas)]
    hor = horario[horario.ano.isin(anos)]
    if sin.empty or hor.empty:
        st.warning("Nenhum registro para os filtros selecionados.")
        return
    kpis = calcular_kpis_gerais(sin, hor)

    st.markdown("## Contexto e metodologia")
    st.markdown(
        "A mobilidade em São Paulo é afetada por dois tipos de fator. Os **estruturais e previsíveis** são a hora do dia, "
        "o dia da semana e o calendário de feriados. Os **exógenos e voláteis** são principalmente a chuva e as "
        "condições atmosféricas. Hoje, a operação de tráfego (posicionamento de agentes, guinchos e viaturas, planos "
        "semafóricos e campanhas de segurança) é planejada em grande parte com escalas fixas. Este estudo responde a duas "
        "perguntas: (i) quanto da variação na extensão de lentidão e na gravidade dos sinistros vem do calendário e "
        "quanto vem do clima; e (ii) se essa variação é previsível o bastante para orientar a alocação de recursos "
        "antes que o evento aconteça."
    )
    st.markdown(
        f"O recorte é o município de São Paulo de **1º de janeiro de 2022 a 31 de dezembro de 2023**, o período em que "
        f"as três fontes se sobrepõem (a série de lentidão da CET disponível cobre apenas esses dois anos). "
        f"Há duas unidades de análise. A **base de sinistros** tem {fmt_int(len(consolidado))} registros, um por sinistro, "
        f"já enriquecidos com o clima e a lentidão da hora em que ocorreram. A **grade horária** tem "
        f"{fmt_int(len(horario))} horas, com contagens de sinistros, lentidão e clima por hora."
    )
    st.markdown(
        """
| Fonte | Conteúdo utilizado | Granularidade original | Papel na análise |
|---|---|---|---|
| **Infosiga-SP** (Detran-SP) | Sinistros com e sem vítima, tipo de sinistro, via, coordenadas e severidade | Evento (data e hora) | Variável-alvo da classificação (`com_vitima`) e numerador do ISR |
| **CET-SP** (Lentidão por trechos) | Extensão em km dos trechos monitorados com lentidão | Instantâneos a cada 30 min | Variável-alvo da regressão (`km_lentidao_total`) e base do FRT |
| **INMET** (estação automática A701, Mirante de Santana) | Precipitação, temperatura, umidade, vento e pressão | Horária, em UTC | Variáveis explicativas climáticas e faixas de chuva |
        """
    )
    with st.expander("Notas metodológicas: integração temporal e tratamento de dados"):
        st.markdown(
            """
- **Correção de fuso horário.** O INMET publica as observações em UTC. Elas foram convertidas para `America/Sao_Paulo` (UTC-3 fixo, já que não há horário de verão desde 2019) antes de qualquer cruzamento. Sem essa correção, a chuva ficaria deslocada 3 h em relação aos sinistros e à lentidão: um temporal das 18 h seria atribuído às 21 h, fora do pico, e o efeito climático medido seria atenuado.
- **Alinhamento temporal via `pd.merge_asof`.** Uma observação do INMET registrada às `H` horas resume o intervalo `(H-1, H]`. Por isso, a hora `H` da grade recebe o registro `H+1`, de modo que o clima atribuído corresponde ao que de fato ocorreu dentro daquela hora. O `merge_asof` usa direção *backward* e tolerância de 1 h: se o registro esperado falta, ele reaproveita o último registro válido até esse limite e, além disso, deixa o clima nulo em vez de propagar valores antigos.
- **Agregação da lentidão.** Os instantâneos de 30 min da CET são agregados pela média dentro de cada hora. Cada sinistro recebe a lentidão da hora em que ocorreu e a da hora seguinte, o que permite medir a retenção posterior ao evento (FRT).
- **Variáveis derivadas.** A chuva acumulada em 3 h (`precipitacao_3h_mm`, janela móvel) distingue chuva persistente, que alaga vias, de pancada isolada. Dia da semana, fim de semana, feriado e hora de pico (07–09 h e 17–19 h em dias úteis) são derivados do carimbo de data e hora local.
- **Ausências.** Clima contínuo (temperatura, umidade, pressão, vento): interpolação linear limitada a 6 h, só entre observações válidas. Precipitação: lacunas de até 3 h preenchidas com 0 mm; lacunas maiores ficam nulas, porque imputar chuva zero em lacunas longas subestimaria o efeito climático. Lentidão: dias sem nenhuma coleta da CET ficam nulos e são excluídos do treino (flag `lentidao_disponivel`). Sinistros sem hora válida (0,08 %) foram removidos.
- **Categorias ausentes.** Via e zona sem informação foram mantidas como a categoria explícita `NAO DISPONIVEL`, sem imputação, porque a ausência não é aleatória (ver a aba Recomendações técnicas).
            """
        )

    st.markdown("## Indicadores do período")
    c = st.columns(6)
    c[0].metric("Sinistros", fmt_int(kpis["total_sinistros"]),
                help="Total de sinistros registrados no Infosiga-SP para os anos e zonas selecionados (com e sem vítima).")
    c[1].metric("ISR geral", f"{fmt_num(kpis['isr_geral'] * 100, 1)} %",
                help="Índice de Severidade Relativa = sinistros com vítima (feridos ou óbito) / total de sinistros registrados. "
                     "Mede a composição da gravidade dos sinistros registrados, não o risco por exposição.")
    c[2].metric("ISR sob chuva", f"{fmt_num(kpis['isr_chuva'] * 100, 1)} %",
                delta=f"{fmt_num((kpis['isr_chuva'] - kpis['isr_seco']) * 100, 1)} p.p. vs. seco", delta_color="inverse",
                help="ISR restrito a sinistros ocorridos em horas com precipitação > 0 mm. A variação compara com o ISR das horas secas, em pontos percentuais.")
    c[3].metric("Sinistros fatais", fmt_int(kpis["total_fatais"]),
                help="Sinistros com ao menos um óbito registrado no Infosiga-SP.")
    c[4].metric("Lentidão média", f"{fmt_num(kpis['lentidao_media_km'], 1)} km",
                help="Média, sobre todas as horas com coleta válida da CET, da extensão somada dos trechos monitorados em lentidão.")
    c[5].metric("Lentidão no pico", f"{fmt_num(kpis['lentidao_pico_km'], 1)} km",
                help="Mesma medida, restrita às horas de pico de dias úteis (07–09 h e 17–19 h).")

    with st.expander("Metodologia dos indicadores (ISR, FRT, TSC)"):
        st.markdown("**ISR: Índice de Severidade Relativa**")
        st.latex(r"\text{ISR}_{g} = \frac{\#\{\text{sinistros com vítima em } g\}}{\#\{\text{sinistros em } g\}}")
        nota(
            "Calculado por grupo <i>g</i> (categoria de via, faixa de chuva ou a combinação das duas). Grupos com menos de 30 "
            "sinistros são descartados para evitar proporções instáveis. <b>Uso na gestão:</b> indica onde e em que condições "
            "um sinistro, uma vez ocorrido, tende a ter vítima. Serve para priorizar atendimento pré-hospitalar e intervenções "
            "de engenharia. <b>Limitação:</b> o denominador são os sinistros registrados, não o volume de tráfego. Por isso o "
            "ISR mede gravidade condicional, não risco por veículo-km."
        )
        st.markdown("**FRT: Fator de Retenção de Tráfego**")
        st.latex(r"\text{FRT}_{c} = \frac{1}{n_c}\sum_{i \in c} \frac{L_{h(i)} + L_{h(i)+1}}{2}"
                 r"\qquad \text{Excedente}_{c} = \text{FRT}_{c} - \frac{1}{n_c}\sum_{i \in c}\tilde{L}_{d(i),\,h(i)}")
        nota(
            "<i>L<sub>h</sub></i> é a lentidão (km) na hora do sinistro, <i>L<sub>h+1</sub></i> a da hora seguinte e "
            "<i>L̃<sub>d,h</sub></i> a mediana histórica de lentidão para o mesmo dia da semana e hora. Comparar com o típico do "
            "horário é essencial: colisões se concentram no pico, então o FRT bruto confundiria horário com efeito do sinistro. "
            "<b>Uso na gestão:</b> o excedente estima quantos km de retenção acima do normal acompanham cada tipo de ocorrência, "
            "o que ajuda a priorizar equipes de remoção. <b>Limitação:</b> a relação não é causal. A lentidão é medida na cidade "
            "inteira, não no trecho do sinistro, e congestionamento também gera colisões (causalidade reversa)."
        )
        st.markdown("**TSC: Taxa de Sinistros por Condição climática**")
        st.latex(r"\text{TSC}_{f} = \frac{\#\{\text{sinistros em horas da faixa } f\}}{\#\{\text{horas observadas na faixa } f\}}")
        nota(
            "Normalizar pelas horas de exposição elimina o viés de 'chove pouco, logo há poucos sinistros na chuva'. Faixas "
            "de chuva: sem chuva (0 mm), fraca (até 2,5 mm/h), moderada (até 10 mm/h), forte (acima de 10 mm/h)."
        )

    st.markdown("## Distribuição temporal")
    g1, g2 = st.columns(2)
    g1.plotly_chart(grafico_sinistros_por_hora(sin), width=LARGURA)
    g2.plotly_chart(grafico_lentidao_por_hora(hor), width=LARGURA)
    st.plotly_chart(grafico_heatmap_semana(sin), width=LARGURA)

    nota(
        "Cor: volume absoluto de sinistros em cada combinação de dia da semana e hora, somado sobre o período e as zonas "
        "selecionados. O tooltip mostra a participação de cada célula no total. A concentração nos fins de tarde de dias úteis "
        "acompanha o volume de tráfego (exposição). Nas madrugadas (00–05 h), a fração de sinistros sobe de 6–9 % nos dias "
        "úteis para 17 % aos sábados e 23 % aos domingos, um deslocamento de perfil de risco que não é explicado por volume."
    )

    st.markdown("## Clima e severidade")
    st.plotly_chart(grafico_lentidao_chuva_hora(hor), width=LARGURA)
    nota(
        "Linhas (eixo esquerdo): lentidão média por hora do dia, separada entre horas com precipitação registrada e horas secas. "
        "Barras ao fundo (eixo direito): precipitação média por hora. A amostra se restringe a dias úteis não feriados, para que "
        "a comparação não seja contaminada pela composição de dias. A distância entre as curvas é maior na dissipação do pico "
        "da manhã (09–11 h) e no pico vespertino (17–19 h), onde chega a cerca de 7 a 10 km, e é praticamente nula no início "
        "da tarde. Ou seja, a chuva amplia o congestionamento justamente onde a rede já opera perto da capacidade. "
        "Horas com chuva são minoria, então a curva correspondente tem menos suporte (n por hora no tooltip). O efeito é "
        "associativo e medido por uma única estação."
    )
    tsc = calcular_taxa_sinistros_por_clima(hor)
    isr = calcular_isr(sin)
    g3, g4 = st.columns([5, 4])
    g3.plotly_chart(grafico_chuva(tsc, kpis["isr_geral"]), width=LARGURA)
    g4.plotly_chart(grafico_isr_heatmap(isr), width=LARGURA)
    nota(
        "<b>Leitura:</b> a chuva aumenta a <i>frequência</i> de sinistros por hora observada, mas reduz ligeiramente a "
        "<i>proporção</i> com vítima. Isso é compatível com velocidades menores sob chuva (mais colisões leves) e também com "
        "maior registro de sinistros sem vítima nessas condições. Na matriz à direita, a variação do ISR entre categorias de via "
        "é maior que entre faixas de precipitação: a via discrimina gravidade melhor que o clima. As células com n < 30 foram "
        "omitidas. As faixas de chuva forte têm suporte bem menor que a faixa seca, portanto as diferenças nelas devem ser lidas "
        "com cautela."
    )

    st.markdown("## Retenção de tráfego")
    agrupamento = st.radio("Agrupar FRT por", ["Tipo de sinistro", "Severidade"], horizontal=True, label_visibility="collapsed")
    col = "tp_sinistro_primario" if agrupamento == "Tipo de sinistro" else "severidade"
    frt = calcular_frt(sin, hor, por_categoria=col)
    g5, g6 = st.columns([1, 1])
    g5.plotly_chart(grafico_frt(frt, col), width=LARGURA)
    with g6:
        st.markdown("### FRT por categoria (km)")
        tabela = frt.set_index(col)[["total_sinistros", "frt_km_hora", "frt_km_hora_seguinte", "frt_km_medio", "lentidao_tipica_km", "excedente_km"]]
        tabela.columns = ["n", "hora do sinistro", "hora seguinte", "FRT médio", "típico", "excedente"]
        tabela_html(tabela.rename_axis("categoria"), casas=2)
        nota(
            "Colunas: n = sinistros na categoria; hora do sinistro e hora seguinte = lentidão média (km) nessas horas; "
            "FRT médio = média das duas; típico = mediana histórica do mesmo dia da semana e hora; excedente = FRT médio - típico. "
            "Categorias com menos de 30 sinistros são omitidas."
        )

    st.markdown("## Tendência")
    st.plotly_chart(grafico_mensal(hor), width=LARGURA)
    nota(
        "Com dois anos de série há apenas dois ciclos sazonais. Isso não basta para separar tendência estrutural de "
        "sazonalidade ou de choques pontuais. Por isso os modelos usam o mês como atributo sazonal e não incluem termo de tendência."
    )

    with st.expander("Tabelas de indicadores"):
        st.markdown("### ISR por categoria de via × faixa de chuva")
        tabela_html(isr, casas=3, indice=False, altura=420)
        st.markdown("### Taxa de sinistros por faixa de chuva")
        tabela_html(tsc, casas=3, indice=False)


ROTULO_FEATURE: dict[str, tuple[str, str]] = {
    "hora": ("Hora do dia", "Calendário"),
    "hora_pico": ("Hora de pico", "Calendário"),
    "dia_semana_num": ("Dia da semana", "Calendário"),
    "fim_de_semana": ("Fim de semana", "Calendário"),
    "feriado": ("Feriado", "Calendário"),
    "mes": ("Mês", "Calendário"),
    "precipitacao_mm": ("Chuva na hora", "Clima"),
    "precipitacao_3h_mm": ("Chuva acumulada 3 h", "Clima"),
    "temperatura_c": ("Temperatura", "Clima"),
    "umidade_pct": ("Umidade relativa", "Clima"),
    "vento_m_s": ("Vento", "Clima"),
    "pressao_mb": ("Pressão atmosférica", "Clima"),
    "tipo_via": ("Tipo de via", "Via e localização"),
    "categoria_via": ("Categoria da via", "Via e localização"),
    "zona": ("Zona", "Via e localização"),
}


def importancia_agregada(importancias: dict[str, float], features: list[str]) -> pd.DataFrame:
    """Agrega o ganho das colunas one-hot (``categoria_via_RUA`` etc.) na variável de origem.

    O ganho do XGBoost é aditivo entre colunas, então a soma das dummies equivale à
    contribuição da variável categórica original. O resultado é normalizado para 100 %.
    """
    linhas = []
    for f in features:
        cols = {k: v for k, v in importancias.items() if k == f or k.startswith(f + "_")}
        if f == "hora":  # ``hora_pico`` é variável própria, não dummy de ``hora``
            cols = {k: v for k, v in cols.items() if k == "hora"}
        rotulo, grupo = ROTULO_FEATURE.get(f, (f, "Outros"))
        nd = sum(v for k, v in cols.items() if "NAO DISPONIVEL" in k)
        linhas.append({"atributo": rotulo, "grupo": grupo, "ganho": sum(cols.values()), "n_colunas": len(cols), "ganho_nd": nd})
    df = pd.DataFrame(linhas)
    total = df.ganho.sum()
    df["pct"] = 100 * df.ganho / total if total else 0.0
    df["pct_nd"] = 100 * df.ganho_nd / total if total else 0.0
    return df.sort_values("pct")


def grafico_importancia(imp: pd.DataFrame, titulo: str) -> go.Figure:
    """Barras horizontais de importância (ganho %) coloridas por grupo de atributo."""
    fig = go.Figure()
    cores = {"Calendário": P["series"][0], "Clima": P["series"][1], "Via e localização": P["neutro"]}
    for grupo, cor in cores.items():
        d = imp[imp.grupo == grupo]
        if d.empty:
            continue
        detalhe = [f"{n} colunas one-hot · {fmt_num(p, 1)} p.p. em NAO DISPONIVEL" if n > 1 else "atributo numérico"
                   for n, p in zip(d.n_colunas, d.pct_nd)]
        fig.add_trace(go.Bar(
            x=d.pct, y=d.atributo, orientation="h", name=grupo, marker_color=cor, marker_line_width=0, width=0.62,
            text=[f"{fmt_num(v, 1)} %" for v in d.pct], textposition="outside", textfont=dict(size=11, color=P["texto_sec"]),
            customdata=np.column_stack([np.full(len(d), grupo), detalhe]), cliponaxis=False,
            hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>%{x:.1f} % do ganho total · %{customdata[1]}<extra></extra>",
        ))
    fig = layout(fig, titulo, altura=max(300, 26 * len(imp) + 90),
                 xaxis=dict(title="% do ganho total", showgrid=False, showticklabels=False, range=[0, imp.pct.max() * 1.25]),
                 yaxis=dict(showgrid=False, categoryorder="array", categoryarray=list(imp.atributo)),
                 barmode="overlay", margin=dict(l=8, r=24, t=64, b=24))
    fig.update_layout(legend=dict(itemclick=False, itemdoubleclick=False))
    return fig


def montar_entrada(data_ref: date, hora: int, clima: dict[str, float], via: dict[str, str]) -> pd.DataFrame:
    """Converte os campos do formulário em um DataFrame com as features dos modelos."""
    base = pd.DataFrame({"datetime": [datetime.combine(data_ref, datetime.min.time()).replace(hour=hora)]})
    base = adicionar_features_temporais(base)
    for k, v in {**clima, **via}.items():
        base[k] = v
    return base


def caixa_resultado(rotulo: str, valor: str, delta: str, detalhe: str, classe: str = "") -> None:
    """Renderiza uma caixa de destaque estática para um resultado numérico."""
    st.markdown(
        f'<div class="resultado {classe}"><div class="rotulo">{rotulo}</div><div class="valor">{valor}</div>'
        f'<div class="delta">{delta}</div><div class="detalhe">{detalhe}</div></div>',
        unsafe_allow_html=True,
    )


def secao_simulador(consolidado: pd.DataFrame, horario: pd.DataFrame) -> None:
    st.markdown("## Simulador Preditivo")
    st.markdown('<p class="subtitulo">Inferência com os modelos treinados: probabilidade de vítima em um sinistro e extensão de lentidão esperada na cidade para o cenário informado.</p>', unsafe_allow_html=True)
    try:
        clf, reg, meta = carregar_modelos()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    faixas = meta["faixas_numericas"]
    categorias = meta["categorias_via"]
    mc, mr = meta["classificacao"], meta["regressao"]
    rt, rh, rb = mr["metricas_temporal"], mr["metricas_teste"], mr["baseline_teste"]
    ct, ch, cb = mc["metricas_temporal"], mc["metricas_teste"], mc["baseline_teste"]
    rbt, cbt = mr.get("baseline_temporal") or {}, mc.get("baseline_temporal") or {}
    linha_rbt = (
        f"| Baseline linear (Ridge, Out-of-Time) | {fmt_num(rbt['r2'], 3)} | {fmt_num(rbt['rmse'], 1)} km | {fmt_num(rbt['mae'], 1)} km |\n"
        if rbt else ""
    )
    linha_cbt = (
        f"| Baseline (regressão logística, Out-of-Time) | {fmt_num(cbt['roc_auc'], 3)} | {fmt_num(cbt['balanced_accuracy'], 3)} | {fmt_num(cbt['f1_macro'], 3)} |\n"
        if cbt else ""
    )

    st.markdown("### Especificação e desempenho dos modelos")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            f"""
**Regressão de lentidão: XGBoost (`XGBRegressor`)**

Estima a extensão total, em km, dos trechos monitorados pela CET em lentidão na hora informada. Usa {len(mr['features'])} atributos de calendário e clima. Não há atributos de via, porque o alvo é agregado para a cidade inteira. O treino usou {fmt_int(mr['n_treino'])} horas com coleta válida.

| Validação | R² | RMSE | MAE |
|---|---|---|---|
| **Out-of-Time** (treino 2022, teste 2023, n = {fmt_int(rt['n_teste'])} h) | **{fmt_num(rt['r2'], 3)}** | **{fmt_num(rt['rmse'], 1)} km** | {fmt_num(rt['mae'], 1)} km |
{linha_rbt}| Holdout aleatório (20 %) | {fmt_num(rh['r2'], 3)} | {fmt_num(rh['rmse'], 1)} km | {fmt_num(rh['mae'], 1)} km |
| Baseline linear (Ridge, holdout) | {fmt_num(rb['r2'], 3)} | {fmt_num(rb['rmse'], 1)} km | {fmt_num(rb['mae'], 1)} km |
            """
        )
        nota(
            f"A validação Out-of-Time (OOT) é a referência para uso operacional, porque reproduz a situação real: treinar com o "
            f"passado e prever um ano ainda não visto. O holdout aleatório mistura horas vizinhas entre treino e teste. Como horas "
            f"adjacentes são autocorrelacionadas, isso infla o desempenho. A queda de R² de {fmt_num(rh['r2'], 2)} para "
            f"{fmt_num(rt['r2'], 2)} indica deriva entre os anos: a lentidão média sobe para {fmt_num(rt['media_alvo'], 1)} km em 2023. "
            f"O RMSE OOT de {fmt_num(rt['rmse'], 1)} km tem a mesma ordem de grandeza da média do alvo. O modelo separa bem cenários "
            f"(pico vs. vale, dia útil vs. feriado), mas não substitui a medição em tempo real."
        )
    with e2:
        st.markdown(
            f"""
**Classificação de severidade: XGBoost (`XGBClassifier`)**

Estima a probabilidade de que um sinistro, **dado que ocorreu**, tenha vítima. Não é a probabilidade de ocorrer um sinistro. Usa {len(mc['features'])} atributos de calendário, clima, tipo e categoria de via e zona. O desbalanceamento é tratado com `scale_pos_weight` = {fmt_num(mc.get('scale_pos_weight', 1.0), 3)}, e a probabilidade é reajustada para a prevalência original antes de ser exibida.

| Validação | ROC-AUC | Acurácia balanceada | F1 macro |
|---|---|---|---|
| **Out-of-Time** (treino 2022, teste 2023, n = {fmt_int(ct['n_teste'])}) | **{fmt_num(ct['roc_auc'], 3)}** | {fmt_num(ct['balanced_accuracy'], 3)} | {fmt_num(ct['f1_macro'], 3)} |
{linha_cbt}| Holdout aleatório (20 %) | {fmt_num(ch['roc_auc'], 3)} | {fmt_num(ch['balanced_accuracy'], 3)} | {fmt_num(ch['f1_macro'], 3)} |
| Baseline (regressão logística, holdout) | {fmt_num(cb['roc_auc'], 3)} | {fmt_num(cb['balanced_accuracy'], 3)} | {fmt_num(cb['f1_macro'], 3)} |
            """
        )
        nota(
            f"O ganho do XGBoost sobre a regressão logística é de apenas {fmt_num(ch['roc_auc'] - cb['roc_auc'], 3)} em ROC-AUC no holdout"
            + (
                (f" e de {fmt_num(ct['roc_auc'] - cbt['roc_auc'], 3)} no Out-of-Time. " if ct["roc_auc"] >= cbt["roc_auc"]
                 else f"; no Out-of-Time, a logística fica {fmt_num(cbt['roc_auc'] - ct['roc_auc'], 3)} à frente. ")
                if cbt else ". "
            )
            + 
            "Quando um modelo não linear de alta capacidade quase empata com um linear, a restrição está na informação disponível, "
            "não no algoritmo. Trocar de algoritmo ou ajustar hiperparâmetros não deve mudar esse quadro."
        )

    st.markdown("### Importância dos atributos nas decisões do XGBoost")
    imp_r = importancia_agregada(mr["importancia_features"], mr["features"])
    imp_c = importancia_agregada(mc["importancia_features"], mc["features"])
    f1, f2 = st.columns(2)
    f1.plotly_chart(grafico_importancia(imp_r, "Regressão de lentidão"), width=LARGURA)
    f2.plotly_chart(grafico_importancia(imp_c, "Classificação de severidade"), width=LARGURA)
    grupos_r = imp_r.groupby("grupo").pct.sum()
    grupos_c = imp_c.groupby("grupo").pct.sum()
    nota(
        f"Métrica: ganho (<i>gain</i>) médio das divisões em que cada atributo é usado, normalizado para 100 %. Para variáveis "
        f"categóricas, o ganho das colunas one-hot foi somado na variável de origem. "
        f"<b>Regressão:</b> calendário {fmt_num(grupos_r.get('Calendário', 0), 0)} %, clima {fmt_num(grupos_r.get('Clima', 0), 0)} %. "
        f"A lentidão é governada pela rotina da cidade, e o clima atua como modulador. "
        f"<b>Classificação:</b> via e localização {fmt_num(grupos_c.get('Via e localização', 0), 0)} %, das quais "
        f"{fmt_num(imp_c.pct_nd.sum(), 0)} p.p. vêm das categorias NAO DISPONIVEL, ou seja, de registros incompletos e não do "
        f"risco da via (ver aba Recomendações técnicas). Clima soma {fmt_num(grupos_c.get('Clima', 0), 0)} %, distribuído de "
        f"forma quase uniforme entre as variáveis, o padrão típico de atributos com sinal fraco. "
        f"<b>Ressalva:</b> importância por ganho mede o uso do atributo pela árvore, não a direção nem a magnitude causal do efeito. "
        f"Ela favorece atributos contínuos e de alta cardinalidade e se divide entre atributos correlacionados (por exemplo, hora "
        f"e hora de pico; chuva na hora e chuva acumulada)."
    )

    aviso(
        "Limite estatístico da classificação de severidade",
        [
            "A gravidade de um sinistro depende sobretudo de fatores ausentes deste conjunto de dados: velocidade no impacto, "
            "tipo e massa dos veículos, envolvimento de pedestre, ciclista ou motociclista, uso de cinto e capacete, "
            "consumo de álcool e tempo de atendimento. Clima, calendário e categoria da via são aproximações distantes dessas "
            "causas. O sinal disponível é, portanto, <b>fraco por natureza</b>.",
            f"Na validação OOT, o ROC-AUC de {fmt_num(ct['roc_auc'], 2)} significa o seguinte: se sortearmos um sinistro com vítima e "
            f"outro sem vítima, o modelo atribui probabilidade maior ao primeiro em cerca de {fmt_num(ct['roc_auc'] * 100, 0)} % "
            "dos pares (50 % equivale ao acaso). A probabilidade exibida deve ser lida como <b>comparação relativa entre "
            "cenários</b> (por exemplo, marginal à noite vs. rua local à tarde). Ela não é uma estimativa individual confiável e "
            "não deve ser usada para triagem de ocorrências nem para decisões sobre casos isolados.",
            "Atributos que aumentariam a acurácia, como tipo de veículo, número de envolvidos e perfil das vítimas, foram excluídos "
            "de propósito. Eles só são conhecidos depois do sinistro, e usá-los seria vazamento de informação (<i>data leakage</i>): "
            "o modelo pareceria mais preciso do que poderia ser no momento da decisão.",
        ],
    )

    with st.form("simulador"):
        st.markdown("### Cenário")
        c1, c2, c3, c4 = st.columns(4)
        data_ref = c1.date_input("Data", value=date(2023, 3, 15), min_value=date(2022, 1, 1), max_value=date(2030, 12, 31))
        hora = c2.slider("Hora", 0, 23, 18)
        tipo_via = c3.selectbox("Tipo de via", categorias["tipo_via"], index=categorias["tipo_via"].index("VIAS URBANAS"))
        categoria_via = c4.selectbox("Categoria da via", categorias["categoria_via"], index=categorias["categoria_via"].index("AVENIDA"))

        c5, c6, c7, c8 = st.columns(4)
        zona = c5.selectbox("Zona", categorias["zona"], index=categorias["zona"].index("CENTRO"))
        precipitacao = c6.slider("Chuva na hora (mm)", 0.0, 50.0, 0.0, 0.5)
        precipitacao_3h = c7.slider("Chuva acumulada 3 h (mm)", 0.0, 100.0, 0.0, 0.5)
        temperatura = c8.slider("Temperatura (°C)", 5.0, 40.0, float(round(faixas["temperatura_c"]["mediana"], 1)), 0.5)

        c9, c10, c11, c12 = st.columns(4)
        umidade = c9.slider("Umidade relativa (%)", 15, 100, int(faixas["umidade_pct"]["mediana"]))
        vento = c10.slider("Vento (m/s)", 0.0, 10.0, float(round(faixas["vento_m_s"]["mediana"], 1)), 0.1)
        pressao = c11.slider("Pressão (mB)", 910.0, 945.0, float(round(faixas["pressao_mb"]["mediana"], 1)), 0.5)
        c12.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        enviado = c12.form_submit_button("Executar inferência", type="primary", width=LARGURA)

    if not enviado:
        nota(
            "Dia da semana, fim de semana, feriado, mês e hora de pico são derivados da data e da hora informadas. "
            "Os modelos não têm termo de tendência: uma data futura é tratada pelo padrão sazonal do mesmo mês e dia da semana "
            "observado em 2022 e 2023."
        )
        return

    precipitacao_3h = max(precipitacao_3h, precipitacao)
    clima = {"precipitacao_mm": precipitacao, "precipitacao_3h_mm": precipitacao_3h, "temperatura_c": temperatura,
             "umidade_pct": float(umidade), "vento_m_s": vento, "pressao_mb": pressao}
    via = {"tipo_via": tipo_via, "categoria_via": categoria_via, "zona": zona}
    entrada = montar_entrada(data_ref, hora, clima, via)
    entrada_seca = montar_entrada(data_ref, hora, {**clima, "precipitacao_mm": 0.0, "precipitacao_3h_mm": 0.0}, via)

    peso = float(meta["classificacao"].get("scale_pos_weight", 1.0))
    feats_c, feats_r = meta["classificacao"]["features"], meta["regressao"]["features"]
    with st.spinner("Processando inferência..."):
        try:
            p_vitima = ajustar_probabilidade_prior(float(clf.predict_proba(entrada[feats_c])[0, 1]), peso)
            p_vitima_seco = ajustar_probabilidade_prior(float(clf.predict_proba(entrada_seca[feats_c])[0, 1]), peso)
            km = float(max(reg.predict(entrada[feats_r])[0], 0))
            km_seco = float(max(reg.predict(entrada_seca[feats_r])[0], 0))
        except Exception as exc:
            st.error(f"Falha na inferência: {exc}")
            return

    isr_geral = float(consolidado.com_vitima.mean())
    tipico = horario.dropna(subset=["km_lentidao_total"])
    tipico = tipico[(tipico.dia_semana_num == int(entrada.dia_semana_num[0])) & (tipico.hora == hora)].km_lentidao_total
    km_tipico = float(tipico.median()) if len(tipico) else float("nan")

    st.markdown("### Resultado")
    r1, r2 = st.columns(2)
    with r1:
        caixa_resultado(
            "Probabilidade de vítima",
            f"{fmt_num(p_vitima * 100, 1)} %",
            f"{fmt_delta((p_vitima - isr_geral) * 100)} p.p. em relação à média histórica ({fmt_num(isr_geral * 100, 1)} %)",
            f"Mesmo cenário sem chuva: {fmt_num(p_vitima_seco * 100, 1)} %. Leitura relativa entre cenários; "
            f"ROC-AUC Out-of-Time = {fmt_num(ct['roc_auc'], 3)} (sinal fraco, ver aviso acima).",
        )
    with r2:
        delta = (f"{fmt_delta(km - km_tipico)} km em relação à mediana histórica deste dia/hora ({fmt_num(km_tipico, 1)} km)"
                 if not np.isnan(km_tipico) else "Sem referência histórica para este dia/hora")
        caixa_resultado(
            "Lentidão estimada",
            f"{fmt_num(km, 1)} km",
            delta,
            f"Mesmo cenário sem chuva: {fmt_num(km_seco, 1)} km. Horas acima de 80 km são classificadas como extremas; "
            f"Erro típico Out-of-Time: RMSE {fmt_num(rt['rmse'], 1)} km · MAE {fmt_num(rt['mae'], 1)} km.",
            classe="teal",
        )

    st.markdown("### Leitura dos resultados")
    st.markdown(
        """
- **Comparação com e sem chuva.** O cenário "sem chuva" mantém todos os demais atributos e zera apenas a precipitação. É uma comparação contrafactual *ceteris paribus* dentro do modelo, útil para isolar a contribuição marginal da chuva segundo o que o modelo aprendeu. Não é uma estimativa causal.
- **Referência histórica.** A lentidão é comparada com a mediana observada para o mesmo dia da semana e hora em 2022 e 2023. A probabilidade de vítima é comparada com o ISR médio da base.
- **Escopo espacial.** A lentidão estimada é um total para toda a cidade. Ela não muda com o tipo de via, a categoria ou a zona escolhidos, que só influenciam o classificador de severidade.
        """
    )
    fora_faixa = [
        rotulo for chave, valor, rotulo in (
            ("precipitacao_mm", precipitacao, "chuva na hora"),
            ("precipitacao_3h_mm", precipitacao_3h, "chuva acumulada 3 h"),
            ("temperatura_c", temperatura, "temperatura"),
            ("umidade_pct", float(umidade), "umidade"),
            ("vento_m_s", vento, "vento"),
            ("pressao_mb", pressao, "pressão"),
        ) if not faixas[chave]["min"] <= valor <= faixas[chave]["max"]
    ]
    if fora_faixa:
        aviso(
            "Extrapolação fora do domínio de treino",
            [f"Os valores informados para {', '.join(fora_faixa)} estão fora da faixa observada no treino. Modelos baseados em "
             "árvores mantêm a previsão constante além do último ponto de corte aprendido. Nesse caso o resultado reflete o "
             "valor extremo observado, não o valor informado, e deve ser tratado como indicativo."],
        )
    if categoria_via == "NAO DISPONIVEL" or tipo_via == "NAO DISPONIVEL" or zona == "NAO DISPONIVEL":
        nota(
            "<b>Atenção:</b> a categoria NAO DISPONIVEL representa registros incompletos no Infosiga, não um tipo de via real. "
            "O efeito dela no classificador reflete o padrão de preenchimento do cadastro (ver aba Recomendações técnicas) e não "
            "deve ser interpretado como risco viário."
        )

    with st.expander("Vetor de entrada enviado aos modelos"):
        tabela_html(entrada[feats_c].T.rename(columns={0: "valor"}).rename_axis("atributo").astype(str))


def secao_recomendacoes(meta: dict[str, Any] | None) -> None:
    st.markdown("## Recomendações técnicas")
    st.markdown('<p class="subtitulo">Sumário executivo: principais achados, vieses conhecidos da base, propostas de política pública derivadas dos modelos e métricas de validação.</p>', unsafe_allow_html=True)

    cal_reg = clima_reg = clima_clf = nd_clf = r2_oot = auc_oot = float("nan")
    if meta:
        imp_r, imp_c = meta["regressao"]["importancia_features"], meta["classificacao"]["importancia_features"]
        cal_reg = participacao_ganho(imp_r, FEATURES_CALENDARIO)
        clima_reg = participacao_ganho(imp_r, FEATURES_CLIMA)
        clima_clf = participacao_ganho(imp_c, FEATURES_CLIMA)
        nd_clf = participacao_ganho(imp_c, contem="NAO DISPONIVEL")
        r2_oot = meta["regressao"]["metricas_temporal"]["r2"]
        auc_oot = meta["classificacao"]["metricas_temporal"]["roc_auc"]

    st.markdown("### Sumário executivo")
    k = st.columns(4)
    k[0].metric("Ganho do calendário (lentidão)", f"{fmt_num(cal_reg * 100, 0)} %",
                help="Parcela do ganho total do XGBRegressor atribuída a hora, hora de pico, dia da semana, fim de semana, feriado e mês.")
    k[1].metric("Ganho do clima (lentidão)", f"{fmt_num(clima_reg * 100, 0)} %",
                help="Parcela do ganho total do XGBRegressor atribuída a precipitação (hora e 3 h), temperatura, umidade, vento e pressão.")
    k[2].metric("R² Out-of-Time (lentidão)", fmt_num(r2_oot, 2),
                help="Coeficiente de determinação do modelo treinado em 2022 e avaliado em 2023.")
    k[3].metric("ROC-AUC Out-of-Time (severidade)", fmt_num(auc_oot, 2),
                help="Capacidade de ordenar sinistros com e sem vítima no ano não visto (0,50 = acaso).")
    st.markdown(
        f"""
- **A lentidão é majoritariamente previsível pelo calendário.** Cerca de {fmt_num(cal_reg * 100, 0)} % do ganho do modelo de regressão vem de hora, dia e feriado, e o modelo explica metade da variância de um ano não visto (R² {fmt_num(r2_oot, 2)}). Isso permite planejar escalas operacionais com antecedência.
- **O clima modula, mas não determina.** A chuva responde por uma parcela pequena do ganho ({fmt_num(clima_reg * 100, 0)} %), mas concentra o efeito nos piores cenários: chuva persistente no pico dobra a frequência de horas com lentidão extrema. Esse é o caso em que a resposta tática agrega valor.
- **A gravidade dos sinistros não é previsível com precisão a partir de clima e calendário** (ROC-AUC {fmt_num(auc_oot, 2)}). As alavancas sobre a severidade são a via (engenharia e velocidade) e o comportamento (fiscalização), não a previsão meteorológica.
- **A qualidade do registro é um achado em si.** As categorias `NAO DISPONIVEL` respondem por {fmt_num(nd_clf * 100, 0)} % do ganho do classificador, mais que todo o bloco climático ({fmt_num(clima_clf * 100, 0)} %). Parte do que o modelo aprende é o padrão de preenchimento do cadastro, não o risco viário.
        """
    )

    st.markdown("### Evidências")
    st.markdown(
        f"""
1. **Calendário domina a lentidão; o clima modula.** Hora, dia útil e feriado concentram cerca de {fmt_num(cal_reg * 100, 0)} % do ganho do modelo de regressão. Chuva nas últimas 3 h eleva a mediana da lentidão no pico em cerca de 8 % e dobra a frequência de horas extremas (> 80 km).
2. **A chuva aumenta a frequência de sinistros (+17 % por hora observada), não a gravidade.** O ISR cai de 65 % para cerca de 60 % sob chuva. A explicação mais plausível é uma combinação de velocidades menores (colisões mais leves) e maior registro de sinistros sem vítima sob chuva.
3. **A via é o principal determinante observável da severidade.** Rodovias e marginais têm ISR ≈ 0,72; ruas locais, ≈ 0,61. A diferença entre categorias de via supera o efeito de qualquer faixa de chuva, o que é consistente com a velocidade operacional como mecanismo.
4. **Sinistros fatais concentram-se entre 20 h e 6 h** (49 % dos fatais contra 30 % dos sinistros em geral), com madrugadas de sábado e domingo acima dos dias úteis. O padrão é compatível com velocidade elevada em vias livres e com consumo de álcool.
5. **Colisões têm o maior excedente de retenção** (FRT 2,6 km acima do típico do horário) e o maior volume (37 % dos sinistros), o que as torna prioridade na resposta operacional.
        """
    )

    st.markdown("### Vieses e qualidade dos dados")
    st.markdown(
        f"""
**1. Subnotificação de sinistros sem vítima.** Em sistemas com notificação completa, sinistros sem vítima costumam ser a maioria. Nesta base, o ISR geral fica em torno de 65 %, um forte indício de que o Infosiga capta sobretudo sinistros com vítima. Sinistros apenas materiais dependem de boletim de ocorrência voluntário e são resolvidos com frequência entre as partes ou via seguradora. Consequências: (i) o ISR observado é um **limite superior** da severidade real e não deve ser comparado diretamente com bases internacionais; (ii) a classe "sem vítima" do classificador é uma amostra seletiva, não representativa; (iii) diferenças de ISR entre grupos podem refletir diferenças de propensão a registrar, não de gravidade.

**2. A categoria "via não disponível" não é aleatória.** Registros sem tipo de via, categoria ou coordenadas foram mantidos como a categoria explícita `NAO DISPONIVEL`, porque a ausência é informativa: ela se concentra nos sinistros sem vítima, cujo cadastro é mais incompleto. Por isso essas dummies estão entre os atributos de maior ganho do classificador ({fmt_num(nd_clf * 100, 0)} % do total). É um sinal espúrio para fins de política pública: o modelo aprende que "registro incompleto indica menor gravidade", o que diz respeito ao processo de registro, não ao risco da via. Excluir esses registros, por outro lado, enviesaria a amostra para os sinistros graves. A decisão foi mantê-los e explicitar a limitação.

**3. Ausência de registro de lentidão não significa via livre.** A série da CET registra apenas os trechos **monitorados** que ultrapassam o limiar de lentidão da companhia. Em dias com coleta, horas sem nenhum trecho registrado foram tratadas como 0 km. Isso é uma premissa de modelagem: significa "nenhum trecho monitorado acima do limiar", não ausência de congestionamento na cidade, já que vias locais e corredores não monitorados ficam fora da série. Dias sem nenhuma coleta (cerca de 8 % do período) foram deixados nulos e excluídos do treino. Imputar zero nesses dias ensinaria ao modelo que falhas de coleta equivalem a trânsito livre.

**4. Outros vieses e limitações.**
- **Estação meteorológica única** (A701, zona norte). Chuvas convectivas localizadas, típicas do verão paulistano, não são capturadas com precisão espacial. O erro de medição na variável explicativa atenua o coeficiente (*attenuation bias*), de modo que o efeito climático estimado é um **limite inferior**.
- **Lentidão agregada à cidade.** O modelo de regressão não distingue corredores nem regiões, e o FRT associa cada sinistro à lentidão da cidade inteira, não do trecho afetado.
- **ISR sem denominador de exposição.** Sem volume de tráfego por via e hora, não é possível calcular sinistros por veículo-km. O ISR mede gravidade condicional, não risco.
- **Validação cruzada aleatória superestima o desempenho** em séries horárias autocorrelacionadas. A validação Out-of-Time (2022 para 2023) é a estimativa conservadora adotada como referência.
- **Período curto** (dois ciclos anuais), sem cobertura de eventos atípicos. A zona geográfica foi derivada por regra geométrica sobre as coordenadas, não pelo zoneamento oficial.
        """
    )

    st.markdown("### Propostas de política pública")
    st.markdown(
        """
| Achado do modelo | Proposta acionável | Responsável | Indicador de acompanhamento |
|---|---|---|---|
| Lentidão explicada principalmente pelo calendário; R² Out-of-Time ≈ 0,50 | **Alocação dinâmica de viaturas, agentes e guinchos.** Substituir escalas fixas por escalas por dia da semana, hora e feriado, dimensionadas pela lentidão prevista, com antecipação de efetivo nas janelas 07–09 h e 17–19 h dos dias úteis e redução em feriados. | CET | Tempo médio de chegada ao local; excedente de FRT por ocorrência |
| Chuva acumulada no pico dobra a frequência de horas com lentidão extrema | **Protocolo de chuva com gatilho objetivo.** Com acumulado de 3 h acima de 2,5 mm entre 16 h e 20 h, pré-posicionar equipes de remoção e agentes nos corredores com histórico de lentidão extrema, integrando o alerta ao CGE e à Defesa Civil. | CET, CGE, Defesa Civil | Horas com lentidão acima de 80 km em dias de chuva |
| Picos de dias úteis concentram lentidão; colisões têm o maior excedente de retenção | **Gestão semafórica por horário crítico.** Planos de temporização dedicados aos picos nos corredores arteriais e marginais, com prioridade de escoamento no sentido dominante, e um plano de contingência para chuva (ciclos e entreverdes revisados) acionado pela central semafórica. | CET (centrais semafóricas) | km de lentidão nos corredores no pico; sinistros por hora de pico |
| Marginais, rodovias urbanas e avenidas de trânsito rápido têm ISR ≈ 0,72 | **Gestão de velocidade e engenharia viária** nas categorias de maior ISR: revisão de limites, fiscalização eletrônica contínua e tratamento de pontos críticos. | Prefeitura, DER, concessionárias | ISR por categoria de via; óbitos por km de via |
| 49 % dos fatais ocorrem entre 20 h e 6 h, com pico nas madrugadas de fim de semana | **Fiscalização de velocidade e alcoolemia** focada nas noites de sexta a domingo, nos corredores de maior velocidade operacional. | Polícia Militar, Detran-SP, CET | Óbitos noturnos de fim de semana |
| Chuva eleva o volume de colisões, não a gravidade | **Campanhas de chuva** orientadas a distância de seguimento e frenagem, veiculadas junto com os alertas meteorológicos. | Prefeitura, Detran-SP | Sinistros por hora de chuva (TSC) |
| Registros incompletos dominam o sinal do classificador | **Governança do dado.** Tornar obrigatórios via e coordenada no registro de sinistros, ampliar a captura de sinistros sem vítima e publicar a lentidão por corredor. | Detran-SP, SSP, CET | Percentual de registros com `NAO DISPONIVEL` |
        """
    )
    nota(
        "As propostas usam apenas o que os modelos sustentam com robustez: a previsibilidade da lentidão pelo calendário e o "
        "efeito do clima sobre volume e retenção. Nenhuma proposta depende da probabilidade individual de vítima do "
        "classificador, cujo poder discriminativo não sustenta decisões caso a caso."
    )

    st.markdown("### Próximos passos analíticos")
    st.markdown(
        """
- Incorporar a rede pluviométrica do CGE (dezenas de pontos no município) para medir a chuva por região e reduzir o viés de atenuação.
- Modelar a lentidão por corredor, cruzando o trecho da CET com a geolocalização do sinistro, para estimar a retenção local causada por cada ocorrência.
- Incluir variáveis de exposição (contagens volumétricas da CET) para converter o ISR em taxa de risco por veículo-km.
- Calibrar formalmente as probabilidades do classificador (curva de confiabilidade e *Brier score*) e monitorar a deriva anual com revalidação Out-of-Time a cada novo ano de dados.
        """
    )

    if meta:
        st.markdown("### Métricas dos modelos")
        mc, mr = meta["classificacao"]["metricas_teste"], meta["regressao"]["metricas_teste"]
        tc, tr = meta["classificacao"]["metricas_temporal"], meta["regressao"]["metricas_temporal"]
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Classificação de severidade (XGBClassifier, classes balanceadas)**")
            tabela_html(pd.DataFrame({
                "Teste (holdout 20 %)": [f"{mc['f1']:.3f}", f"{mc['f1_macro']:.3f}", f"{mc['roc_auc']:.3f}", f"{mc['balanced_accuracy']:.3f}"],
                "Temporal (2022 → 2023)": [f"{tc.get('f1', float('nan')):.3f}", f"{tc.get('f1_macro', float('nan')):.3f}", f"{tc.get('roc_auc', float('nan')):.3f}", f"{tc.get('balanced_accuracy', float('nan')):.3f}"],
            }, index=pd.Index(["F1", "F1 macro", "ROC-AUC", "Acurácia balanceada"], name="métrica")))
            cm = np.array(mc["matriz_confusao"])
            st.markdown(f'<p class="nota">Matriz de confusão (teste): VN {fmt_int(cm[0, 0])} · FP {fmt_int(cm[0, 1])} · FN {fmt_int(cm[1, 0])} · VP {fmt_int(cm[1, 1])}.</p>', unsafe_allow_html=True)
        with m2:
            st.markdown("**Regressão de lentidão (XGBRegressor)**")
            tabela_html(pd.DataFrame({
                "Teste (holdout 20 %)": [f"{mr['rmse']:.2f} km", f"{mr['mae']:.2f} km", f"{mr['r2']:.3f}"],
                "Temporal (2022 → 2023)": [f"{tr.get('rmse', float('nan')):.2f} km", f"{tr.get('mae', float('nan')):.2f} km", f"{tr.get('r2', float('nan')):.3f}"],
            }, index=pd.Index(["RMSE", "MAE", "R²"], name="métrica")))
            st.markdown(f'<p class="nota">Baseline Ridge no teste: R² {meta["regressao"]["baseline_teste"]["r2"]:.3f}. Validação cruzada 5-fold apenas no conjunto de treino.</p>', unsafe_allow_html=True)
        nota(
            "<b>Como ler as métricas.</b> A coluna Temporal é a estimativa de referência: o modelo treinado só com 2022 é avaliado "
            "em 2023, como aconteceria em produção. O holdout aleatório mistura horas e dias vizinhos entre treino e teste e "
            "tende a superestimar o desempenho em séries temporais. Na classificação, o F1 da classe positiva parece alto porque "
            "a classe com vítima é majoritária na base, por causa da subnotificação. ROC-AUC e acurácia balanceada são as "
            "métricas que refletem o poder discriminativo real."
        )


REFERENCIAS: list[dict[str, str]] = [
    {
        "fonte": "Infosiga SP (Detran-SP)",
        "uso": "Sinistros de trânsito com e sem vítimas, pessoas e veículos (arquivos sinistros/pessoas/veiculos 2022-2024).",
        "url": "https://www.infosiga.sp.gov.br/",
        "nota": "Portal de dados abertos do sistema Infosiga; os CSVs usam separador ';' e codificação Latin-1.",
    },
    {
        "fonte": "CET - Companhia de Engenharia de Tráfego de São Paulo",
        "uso": "Trechos em lentidão a cada 30 minutos (lentidaotrechos2022.csv, lentidao_trechos_2023.csv).",
        "url": "https://www.cetsp.com.br/",
        "nota": "Série 'Lentidão - Trechos' publicada pela CET e distribuída pelo Portal de Dados Abertos da Prefeitura de São Paulo.",
    },
    {
        "fonte": "Portal de Dados Abertos da Prefeitura de São Paulo",
        "uso": "Catálogo onde a série de lentidão da CET é publicada.",
        "url": "http://dados.prefeitura.sp.gov.br/",
        "nota": "Busque por 'lentidão' no catálogo para localizar os arquivos anuais.",
    },
    {
        "fonte": "INMET - Dados Históricos das Estações Automáticas",
        "uso": "Observações horárias da estação A701 (São Paulo - Mirante de Santana): precipitação, temperatura, umidade, vento, pressão e radiação.",
        "url": "https://portal.inmet.gov.br/dadoshistoricos",
        "nota": "Os registros são publicados em UTC; o pipeline converte para America/Sao_Paulo. O arquivo automaticas_processed.parquet consolida as estações automáticas do país.",
    },
    {
        "fonte": "INMET - BDMEP (Banco de Dados Meteorológicos)",
        "uso": "Consulta alternativa de séries históricas por estação.",
        "url": "https://bdmep.inmet.gov.br/",
        "nota": "Requer cadastro; útil para validar valores pontuais.",
    },
]

BIBLIOTECAS: list[tuple[str, str]] = [
    ("pandas / pyarrow", "https://pandas.pydata.org/"),
    ("scikit-learn", "https://scikit-learn.org/"),
    ("XGBoost", "https://xgboost.readthedocs.io/"),
    ("Streamlit", "https://docs.streamlit.io/"),
    ("Plotly", "https://plotly.com/python/"),
    ("Seaborn", "https://seaborn.pydata.org/"),
]


def ler_texto(caminho: Path) -> str | None:
    """Lê um arquivo de texto do projeto; devolve ``None`` se não existir."""
    try:
        return caminho.read_text(encoding="utf-8")
    except OSError:
        return None


def secao_documentacao(meta: dict[str, Any] | None) -> None:
    st.markdown("## Documentação")
    st.markdown('<p class="subtitulo">Arquitetura do pipeline, dicionário de dados, modelo relacional, especificação dos modelos e referências das fontes.</p>', unsafe_allow_html=True)
    d1, d2, d3, d4, d5 = st.tabs(["Projeto", "Dicionário de dados", "Modelo relacional", "Modelos", "Referências"])

    with d1:
        st.markdown("### Objetivo")
        st.markdown(
            "Quantificar o impacto de fatores climáticos e temporais na severidade de sinistros de trânsito e na extensão "
            "de congestionamentos no município de São Paulo (2022–2023), com indicadores e dois modelos preditivos."
        )
        st.markdown("### Pipeline")
        st.markdown(
            """
| Etapa | Módulo | Entrada | Saída |
|---|---|---|---|
| 1. Pré-processamento | `src/preprocessing.py` | `data/raw/*` | `data/processed/mobilidade_sp_consolidado.parquet`, `mobilidade_sp_horario.parquet` |
| 2. Indicadores | `src/features.py` | parquets processados | `data/processed/kpi_*.csv` |
| 3. Modelagem | `src/modeling.py` | parquets processados | `modelo_classificacao_severidade.pkl`, `modelo_regressao_lentidao.pkl`, `modelos_metadata.json` |
| 4. Relatório | `notebooks/01_relatorio_analise_mobilidade.ipynb` | módulos de `src/` | análise exploratória e crítica dos modelos |
| 5. Dashboard | `app/main.py` | parquets, `.pkl`, metadata | esta aplicação |
            """
        )
        st.markdown("### Integração temporal")
        st.markdown(
            "As três fontes são alinhadas pela hora local. Uma observação INMET carimbada em `H` resume o intervalo `(H-1, H]`; "
            "a hora `H` da grade recebe o registro `H+1` via `pd.merge_asof` (direção *backward*, tolerância de 1 h). "
            "A lentidão é a média dos instantâneos de 30 min dentro da hora; sinistros recebem clima e lentidão da hora em que ocorreram."
        )
        st.markdown("### Tratamento de ausências")
        st.markdown(
            """
- Clima contínuo (temperatura, umidade, pressão, vento): interpolação linear limitada a 6 h.
- Precipitação: lacunas de até 3 h preenchidas com 0 mm; lacunas maiores permanecem nulas.
- Lentidão: horas sem registro em dias com coleta = 0 km; dias sem nenhuma coleta (58 dias) = nulo, com flag `lentidao_disponivel`.
- Sinistros sem hora válida (0,08 %) removidos; contagens nulas do Infosiga interpretadas como zero.
            """
        )
        st.markdown("### Execução")
        st.code("brew install libomp\nmake setup\nmake pipeline\nmake notebook\nmake run-app", language="bash")

    with d2:
        texto = ler_texto(config.DOCS_DIR / "dicionario_de_dados.md")
        if texto:
            st.markdown(texto.split("\n", 1)[1] if texto.startswith("#") else texto)
        else:
            st.warning("Arquivo docs/dicionario_de_dados.md não encontrado.")

    with d3:
        st.markdown("### Modelo relacional")
        diagrama = config.PROJECT_ROOT / "arquitetura.svg"
        if diagrama.exists():
            st.image(diagrama, width="stretch", alt="Diagrama do modelo relacional")
        else:
            st.warning("Arquivo arquitetura.svg não encontrado na raiz do projeto. Exporte o SVG pelo preview DBML do VS Code.")

    with d4:
        if not meta:
            st.warning("modelos_metadata.json não encontrado. Execute `make pipeline`.")
        else:
            mc, mr = meta["classificacao"], meta["regressao"]
            st.markdown("### Especificação")
            st.markdown(
                f"""
| | Classificação | Regressão |
|---|---|---|
| Alvo | `{mc['alvo']}` (1 = sinistro com vítima) | `{mr['alvo']}` (km de lentidão na hora) |
| Algoritmo | {mc['nome']} com `scale_pos_weight` = {mc.get('scale_pos_weight', 1):.3f} | {mr['nome']} |
| Baseline | Regressão logística | Ridge |
| Treino / teste | {fmt_int(mc['n_treino'])} / {fmt_int(mc['n_teste'])} | {fmt_int(mr['n_treino'])} / {fmt_int(mr['n_teste'])} |
| Validação | 5-fold estratificada no treino + holdout + temporal 2022 → 2023 | 5-fold no treino + holdout + temporal 2022 → 2023 |
| Artefato | `modelo_classificacao_severidade.pkl` | `modelo_regressao_lentidao.pkl` |
                """
            )
            st.markdown("### Atributos de entrada")
            e1, e2 = st.columns(2)
            tabela_html(pd.DataFrame({"classificação": mc["features"]}), indice=False, alvo=e1)
            tabela_html(pd.DataFrame({"regressão": mr["features"]}), indice=False, alvo=e2)
            st.markdown("### Importância dos atributos (ganho)")
            i1, i2 = st.columns(2)
            tabela_html(pd.Series(mc["importancia_features"], name="classificação").head(12).rename_axis("atributo"), casas=3, alvo=i1)
            tabela_html(pd.Series(mr["importancia_features"], name="regressão").head(12).rename_axis("atributo"), casas=3, alvo=i2)
            st.markdown("### Validação cruzada (treino)")
            v1, v2 = st.columns(2)
            tabela_html(pd.DataFrame(mc["metricas_cv"]).T.rename_axis("métrica"), casas=3, alvo=v1)
            tabela_html(pd.DataFrame(mr["metricas_cv"]).T.rename_axis("métrica"), casas=3, alvo=v2)

    with d5:
        st.markdown("### Fontes de dados")
        for ref in REFERENCIAS:
            st.markdown(f"**{ref['fonte']}** — [{ref['url']}]({ref['url']})")
            st.markdown(f"{ref['uso']}")
            st.markdown(f'<p class="nota">{ref["nota"]}</p>', unsafe_allow_html=True)
        st.markdown("### Bibliotecas")
        st.markdown("\n".join(f"- [{nome}]({url})" for nome, url in BIBLIOTECAS))
        st.markdown("### Documentos do projeto")
        st.markdown(
            """
- `docs/dicionario_de_dados.md` — variáveis finais, tipos e unidades.
- `docs/arquitetura.dbml` — modelo relacional dos dados processados.
- `notebooks/01_relatorio_analise_mobilidade.ipynb` — análise exploratória e avaliação crítica dos modelos.
- `README.md` — instruções de execução.
            """
        )
        st.markdown('<p class="nota">Os links apontam para os portais oficiais das fontes; a página específica de cada conjunto pode mudar com o tempo.</p>', unsafe_allow_html=True)


def barra_lateral(consolidado: pd.DataFrame, horario: pd.DataFrame, meta: dict[str, Any] | None) -> tuple[str, list[int], list[str]]:
    """Sidebar em blocos: marca, navegação, recorte, status da base/modelos, aparência e referência."""
    anos_disp = sorted(int(a) for a in consolidado.ano.unique())
    zonas_disp = sorted(consolidado.zona.unique())
    st.session_state.setdefault("filtro_anos", anos_disp)
    st.session_state.setdefault("filtro_zonas", zonas_disp)

    def restaurar_filtros() -> None:
        st.session_state["filtro_anos"] = anos_disp
        st.session_state["filtro_zonas"] = zonas_disp

    with st.sidebar:
        st.markdown(
            '<div class="marca"><div class="sigla">Mobilidade urbana · SP</div>'
            '<div class="nome">Citizen<br>Intelligence SP</div>'
            '<div class="desc">Severidade de sinistros e lentidão sob fatores climáticos e temporais. Município de São Paulo, 2022–2023.</div></div>',
            unsafe_allow_html=True,
        )
        membros = "".join(f'<div class="membro"><span>{nome}</span><span class="rm">RM {rm}</span></div>' for nome, rm in INTEGRANTES)
        st.markdown(f'<div class="equipe"><div class="titulo">Integrantes</div>{membros}</div>', unsafe_allow_html=True)

        st.markdown("### Navegação")
        pagina = st.radio("Página", ["Painel analítico", "Documentação"], key="pagina", label_visibility="collapsed")

        anos, zonas = anos_disp, zonas_disp
        if pagina == "Painel analítico":
            st.markdown("### Recorte da visão geral")
            anos = st.multiselect("Ano", anos_disp, key="filtro_anos", persist_state="session")
            zonas = st.multiselect("Zona", zonas_disp, key="filtro_zonas", persist_state="session",
                                   help="Zona derivada das coordenadas do sinistro. NAO DISPONIVEL = registro sem coordenadas.")
            n = int((consolidado.ano.isin(anos) & consolidado.zona.isin(zonas)).sum())
            st.markdown(
                f'<div class="recorte">Recorte ativo: <b>{fmt_int(n)}</b> sinistros ({fmt_num(100 * n / len(consolidado), 1)} % da base) · '
                f'<b>{len(anos)}</b>/{len(anos_disp)} anos · <b>{len(zonas)}</b>/{len(zonas_disp)} zonas.<br>'
                "Simulador e Recomendações usam sempre a base completa.</div>",
                unsafe_allow_html=True,
            )
            completo = set(anos) == set(anos_disp) and set(zonas) == set(zonas_disp)
            st.button("Restaurar filtros", on_click=restaurar_filtros, disabled=completo, width="stretch")

        st.markdown("### Base e modelos")
        r2 = auc = "–"
        if meta:
            r2 = fmt_num(meta["regressao"]["metricas_temporal"]["r2"], 2)
            auc = fmt_num(meta["classificacao"]["metricas_temporal"]["roc_auc"], 2)
        st.markdown(
            '<div class="painel">'
            f'<div><div class="k">Sinistros</div><div class="v">{fmt_int(len(consolidado))}</div></div>'
            f'<div><div class="k">Grade horária</div><div class="v">{fmt_int(len(horario))} <small>h</small></div></div>'
            f'<div><div class="k">R² lentidão</div><div class="v acento">{r2}</div></div>'
            f'<div><div class="k">AUC severidade</div><div class="v">{auc}</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        nota("Métricas na validação Out-of-Time (treino 2022, teste 2023).")

        st.markdown("### Aparência")
        st.segmented_control("Tema", list(TEMAS), key="tema", default="Escuro", required=True, bind="query-params",
                             label_visibility="collapsed", width="stretch")

        st.markdown("### Referência")
        with st.expander("Glossário de indicadores"):
            st.markdown(
                "**ISR** · Índice de Severidade Relativa: sinistros com vítima / total de sinistros.\n\n"
                "**FRT** · Fator de Retenção de Tráfego: lentidão média na hora do sinistro e na seguinte, comparada ao típico do horário.\n\n"
                "**TSC** · Taxa de Sinistros por Condição climática: sinistros por hora observada em cada faixa de chuva."
            )
        with st.expander("Fontes de dados"):
            st.markdown(
                "**Infosiga-SP** (Detran-SP): sinistros.\n\n"
                "**CET-SP**: lentidão por trechos, instantâneos de 30 min.\n\n"
                "**INMET** A701, Mirante de Santana: clima horário (UTC convertido para horário local)."
            )
        st.markdown('<div class="rodape">Global Solution · FIAP<br>XGBoost · validação Out-of-Time</div>', unsafe_allow_html=True)

    return pagina, anos, zonas


def main() -> None:
    try:
        consolidado, horario = carregar_dados()
    except FileNotFoundError as exc:
        st.error(f"{exc} Execute `make pipeline` na raiz do projeto.")
        st.stop()
    try:
        meta: dict[str, Any] | None = carregar_metadata()
    except FileNotFoundError:
        meta = None

    pagina, anos, zonas = barra_lateral(consolidado, horario, meta)

    st.markdown("# Citizen Intelligence SP")
    st.markdown('<p class="subtitulo">Análise preditiva de mobilidade urbana: impacto de clima e calendário na severidade de sinistros e na extensão de congestionamentos.</p>', unsafe_allow_html=True)

    if pagina == "Documentação":
        secao_documentacao(meta)
        return

    if not anos or not zonas:
        st.warning("Selecione ao menos um ano e uma zona na barra lateral.")
        st.stop()

    aba1, aba2, aba3 = st.tabs(["Visão Geral", "Simulador Preditivo", "Recomendações técnicas"])
    with aba1:
        secao_visao_geral(consolidado, horario, anos, zonas)
    with aba2:
        secao_simulador(consolidado, horario)
    with aba3:
        secao_recomendacoes(meta)


if __name__ == "__main__":
    main()
