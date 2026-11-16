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

from src import config, viz  # noqa: E402
from src.features import (  # noqa: E402
    ORDEM_CHUVA,
    ORDEM_DIAS,
    adicionar_features_temporais,
    calcular_frt,
    calcular_isr,
    calcular_kpis_gerais,
    calcular_taxa_sinistros_por_clima,
    preparar_features,
)
from src.modeling import ajustar_probabilidade_prior, carregar_metadata, carregar_modelo  # noqa: E402
from src.preprocessing import carregar_consolidado, carregar_horario  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuração da página e estilo
# --------------------------------------------------------------------------- #
st.set_page_config(layout="wide", page_title="Citizen Intelligence SP", initial_sidebar_state="expanded")

CSS = """
<style>
  /* Remove chrome padrão do Streamlit */
  #MainMenu, footer { visibility: hidden; height: 0; }
  header[data-testid="stHeader"] { background: transparent; height: 0; }
  div[data-testid="stDecoration"], div[data-testid="stToolbar"], div[data-testid="stStatusWidget"] { display: none; }

  /* Tipografia e espaçamento */
  .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1440px; }
  div[data-testid="stMarkdownContainer"] h1 { font-size: 1.6rem !important; font-weight: 600 !important; letter-spacing: -0.01em; color: #111827 !important; padding: 0 !important; margin: 0 0 0.2rem 0 !important; }
  div[data-testid="stMarkdownContainer"] h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #111827 !important; padding: 0 !important; margin: 1.3rem 0 0.4rem 0 !important; }
  div[data-testid="stMarkdownContainer"] h3 { font-size: 0.92rem !important; font-weight: 600 !important; color: #374151 !important; padding: 0 !important; margin: 0.8rem 0 0.3rem 0 !important; }
  div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li { color: #374151; font-size: 0.92rem; line-height: 1.5; }
  .subtitulo { color: #6b7280 !important; font-size: 0.88rem !important; margin-bottom: 0.8rem !important; }
  .nota { color: #6b7280 !important; font-size: 0.8rem !important; line-height: 1.45; margin-top: 0.25rem; }

  /* Cards de KPI */
  div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px 16px; }
  div[data-testid="stMetric"] label p { font-size: 0.78rem !important; color: #6b7280 !important; font-weight: 500; }
  div[data-testid="stMetricValue"] { font-size: 1.45rem !important; font-weight: 600; color: #111827; }
  div[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

  /* Painéis */
  div[data-testid="stForm"] { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 16px 20px 8px 20px; }
  div[data-testid="stExpander"] { border: 1px solid #e5e7eb; border-radius: 6px; }

  /* Caixas de resultado do simulador */
  .resultado { background: #f9fafb; border: 1px solid #e5e7eb; border-left: 3px solid #2f5597; border-radius: 6px; padding: 14px 18px; }
  .resultado .rotulo { font-size: 0.76rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .resultado .valor { font-size: 1.9rem; font-weight: 600; color: #111827; line-height: 1.2; margin: 4px 0; }
  .resultado .delta { font-size: 0.82rem; color: #374151; }
  .resultado .detalhe { font-size: 0.8rem; color: #6b7280; margin-top: 8px; }
  .resultado.teal { border-left-color: #0f9688; }

  /* Abas */
  button[data-baseweb="tab"] p { font-size: 0.9rem !important; font-weight: 500 !important; }
  div[data-baseweb="tab-highlight"] { background-color: #2f5597 !important; }
  div[data-baseweb="tab-border"] { background-color: #e5e7eb !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #f9fafb; border-right: 1px solid #e5e7eb; }
  section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h1 { font-size: 1.05rem !important; margin-bottom: 0.4rem !important; }
  section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h3 { font-size: 0.74rem !important; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280 !important; margin-top: 1.1rem !important; }
  section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p { font-size: 0.84rem; }
  span[data-baseweb="tag"] { background-color: #e5e9f2 !important; color: #1f2937 !important; border-radius: 4px !important; }
  span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg { color: #1f2937 !important; fill: #1f2937 !important; }

  /* Tabelas e botões */
  div[data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 6px; }
  button[kind="primary"], button[data-testid="stBaseButton-primaryFormSubmit"] { background-color: #2f5597 !important; border-color: #2f5597 !important; border-radius: 4px !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ROTULO_SEVERIDADE = {"SEM_VITIMA": "Sem vítima", "COM_VITIMA": "Com vítima", "FATAL": "Fatal"}
ROTULO_CHUVA = {"SEM_CHUVA": "Sem chuva", "FRACA": "Fraca (≤ 2,5 mm)", "MODERADA": "Moderada (≤ 10 mm)", "FORTE": "Forte (> 10 mm)"}
LARGURA = "stretch"  # equivalente atual de use_container_width=True (parâmetro descontinuado no Streamlit >= 1.50)


def fmt_int(valor: float) -> str:
    """Formata inteiro com separador de milhar em padrão pt-BR."""
    return f"{int(valor):,}".replace(",", ".")


def fmt_num(valor: float, casas: int = 2) -> str:
    """Formata número decimal em padrão pt-BR."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_delta(valor: float, casas: int = 1) -> str:
    """Formata variação com sinal explícito em padrão pt-BR."""
    return ("+" if valor >= 0 else "−") + fmt_num(abs(valor), casas)


# --------------------------------------------------------------------------- #
# Carregamento (cacheado)
# --------------------------------------------------------------------------- #
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
    fig.update_layout(**viz.PLOTLY_LAYOUT_CORP)          # tema base
    fig.update_layout(height=altura, title_text=titulo, **kwargs)  # ajustes específicos do gráfico
    fig.update_xaxes(showgrid=False, showline=False, mirror=False, minor_showgrid=False)
    fig.update_yaxes(showline=False, mirror=False, minor_showgrid=False, gridcolor=viz.GRADE_CORP, gridwidth=1)
    return fig


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def grafico_sinistros_por_hora(df: pd.DataFrame) -> go.Figure:
    por_hora = df.groupby(["hora", "severidade"]).size().rename("n").reset_index()
    fig = go.Figure()
    for sev in ["SEM_VITIMA", "COM_VITIMA", "FATAL"]:
        d = por_hora[por_hora.severidade == sev]
        fig.add_trace(go.Scatter(
            x=d.hora, y=d.n, mode="lines", name=ROTULO_SEVERIDADE[sev],
            line=dict(color=viz.COR_SEVERIDADE_CORP[sev], width=2),
            hovertemplate="Hora %{x:02d}:00 · %{y:,.0f} sinistros<extra>" + ROTULO_SEVERIDADE[sev] + "</extra>",
        ))
    for ini, fim in ((7, 10), (17, 20)):
        fig.add_vrect(x0=ini - 0.5, x1=fim - 0.5, fillcolor=viz.GRADE_CORP, opacity=0.55, line_width=0)
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
        cor = viz.CORPORATIVA[i]
        fig.add_trace(go.Scatter(x=pd.concat([d.hora, d.hora[::-1]]), y=pd.concat([d.p75, d.p25[::-1]]), fill="toself",
                                 fillcolor=cor, opacity=0.12, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=d.hora, y=d.media, mode="lines", name=tipo, line=dict(color=cor, width=2),
                                 hovertemplate="Hora %{x:02d}:00 · %{y:.2f} km<extra>" + tipo + "</extra>"))
    return layout(fig, "Lentidão média por hora (faixa: quartis 25–75 %)", xaxis_title="Hora", yaxis_title="km de lentidão",
                  hovermode="x unified", xaxis=dict(tickmode="linear", dtick=2, showgrid=False))


def grafico_heatmap_semana(df: pd.DataFrame) -> go.Figure:
    mapa = df.pivot_table(index="dia_semana", columns="hora", values="id_sinistro", aggfunc="size").reindex(ORDEM_DIAS)
    fig = go.Figure(go.Heatmap(
        z=mapa.values, x=mapa.columns, y=mapa.index, colorscale=viz.SEQUENCIAL_NAVY, xgap=2, ygap=2,
        colorbar=dict(title="Sinistros", thickness=10, outlinewidth=0),
        hovertemplate="%{y} · %{x:02d}:00 · %{z:,.0f} sinistros<extra></extra>",
    ))
    return layout(fig, "Concentração de sinistros: dia da semana × hora", altura=340,
                  xaxis=dict(tickmode="linear", dtick=2, showgrid=False), yaxis=dict(autorange="reversed", showgrid=False))


def grafico_chuva(tsc: pd.DataFrame, isr_geral: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Sinistros por hora observada", "ISR (proporção com vítima)"), horizontal_spacing=0.12)
    faixas = tsc.faixa_chuva.astype(str)
    cores = [viz.COR_CHUVA_CORP[f] for f in faixas]
    rot = [ROTULO_CHUVA[f] for f in faixas]
    fig.add_trace(go.Bar(x=rot, y=tsc.sinistros_por_hora, marker_color=cores, marker_line_width=0, width=0.55,
                         text=[fmt_num(v) for v in tsc.sinistros_por_hora], textposition="outside", textfont=dict(color=viz.TEXTO_SEC_CORP, size=11),
                         customdata=tsc.horas, hovertemplate="%{x}<br>%{y:.2f} sinistros/h · %{customdata:,.0f} h observadas<extra></extra>", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=rot, y=tsc.isr, marker_color=cores, marker_line_width=0, width=0.55,
                         text=[fmt_num(v, 3) for v in tsc.isr], textposition="outside", textfont=dict(color=viz.TEXTO_SEC_CORP, size=11),
                         customdata=tsc.sinistros, hovertemplate="%{x}<br>ISR %{y:.3f} · n = %{customdata:,.0f}<extra></extra>", showlegend=False), row=1, col=2)
    fig.add_hline(y=isr_geral, line_dash="dot", line_color=viz.CHUMBO, line_width=1, row=1, col=2,
                  annotation_text=f"ISR geral {fmt_num(isr_geral, 3)}", annotation_position="top left", annotation_font=dict(size=11, color=viz.TEXTO_SEC_CORP))
    fig.update_yaxes(range=[0, tsc.sinistros_por_hora.max() * 1.25], row=1, col=1)
    fig.update_yaxes(range=[0.5, 0.72], row=1, col=2)
    fig.update_annotations(font=dict(size=12, color=viz.TEXTO_CORP))
    return layout(fig, "Chuva: frequência e severidade dos sinistros", altura=360)


def grafico_isr_heatmap(isr: pd.DataFrame) -> go.Figure:
    mapa = isr.pivot(index="categoria_via", columns="faixa_chuva", values="isr").reindex(columns=ORDEM_CHUVA)
    mapa = mapa.loc[mapa["SEM_CHUVA"].sort_values().index]
    fig = go.Figure(go.Heatmap(
        z=mapa.values, x=[ROTULO_CHUVA[c] for c in mapa.columns], y=mapa.index, colorscale=viz.SEQUENCIAL_NAVY,
        zmin=0.5, zmax=0.75, xgap=2, ygap=2, text=np.round(mapa.values, 2), texttemplate="%{text:.2f}",
        textfont=dict(size=11), colorbar=dict(title="ISR", thickness=10, outlinewidth=0),
        hovertemplate="%{y} · %{x}<br>ISR %{z:.3f}<extra></extra>",
    ))
    return layout(fig, "ISR por categoria de via × faixa de chuva (n ≥ 30)", altura=400, yaxis=dict(showgrid=False))


def grafico_frt(frt: pd.DataFrame, coluna: str) -> go.Figure:
    d = frt.sort_values("excedente_km")
    rot = d[coluna].map(ROTULO_SEVERIDADE).fillna(d[coluna]) if coluna == "severidade" else d[coluna]
    fig = go.Figure(go.Bar(
        x=d.excedente_km, y=rot, orientation="h", marker_color=viz.CORPORATIVA[0], marker_line_width=0, width=0.5,
        text=[f"+{fmt_num(v, 1)} km · n = {fmt_int(n)}" for v, n in zip(d.excedente_km, d.total_sinistros)],
        textposition="outside", textfont=dict(color=viz.TEXTO_SEC_CORP, size=11),
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
    fig.add_trace(go.Scatter(x=m.mes_ref, y=m.sinistros, mode="lines", line=dict(color=viz.CORPORATIVA[0], width=2), name="Sinistros",
                             hovertemplate="%{x|%b %Y} · %{y:,.0f} sinistros<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=m.mes_ref, y=m.lentidao, mode="lines", line=dict(color=viz.CORPORATIVA[1], width=2), name="Lentidão",
                             hovertemplate="%{x|%b %Y} · %{y:.2f} km<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(x=m.mes_ref, y=m.chuva, marker_color=viz.CHUMBO_CLARO, marker_line_width=0, name="Chuva",
                         hovertemplate="%{x|%b %Y} · %{y:.1f} mm<extra></extra>"), row=3, col=1)
    fig.update_annotations(font=dict(size=12, color=viz.TEXTO_CORP))
    return layout(fig, "Evolução mensal", altura=520, showlegend=False)


# --------------------------------------------------------------------------- #
# Seções
# --------------------------------------------------------------------------- #
def secao_visao_geral(consolidado: pd.DataFrame, horario: pd.DataFrame, anos: list[int], zonas: list[str]) -> None:
    sin = consolidado[consolidado.ano.isin(anos) & consolidado.zona.isin(zonas)]
    hor = horario[horario.ano.isin(anos)]
    if sin.empty or hor.empty:
        st.warning("Nenhum registro para os filtros selecionados.")
        return
    kpis = calcular_kpis_gerais(sin, hor)

    st.markdown("## Indicadores do período")
    c = st.columns(6)
    c[0].metric("Sinistros", fmt_int(kpis["total_sinistros"]))
    c[1].metric("ISR geral", f"{fmt_num(kpis['isr_geral'] * 100, 1)} %", help="Índice de Severidade Relativa: proporção de sinistros com vítima.")
    c[2].metric("ISR sob chuva", f"{fmt_num(kpis['isr_chuva'] * 100, 1)} %",
                delta=f"{fmt_num((kpis['isr_chuva'] - kpis['isr_seco']) * 100, 1)} p.p. vs. seco", delta_color="inverse")
    c[3].metric("Sinistros fatais", fmt_int(kpis["total_fatais"]))
    c[4].metric("Lentidão média", f"{fmt_num(kpis['lentidao_media_km'], 1)} km", help="Média de km em lentidão por hora, todas as horas.")
    c[5].metric("Lentidão no pico", f"{fmt_num(kpis['lentidao_pico_km'], 1)} km", help="Média nas horas de pico de dias úteis (07–09 h e 17–19 h).")

    st.markdown("## Distribuição temporal")
    g1, g2 = st.columns(2)
    g1.plotly_chart(grafico_sinistros_por_hora(sin), width=LARGURA)
    g2.plotly_chart(grafico_lentidao_por_hora(hor), width=LARGURA)
    st.plotly_chart(grafico_heatmap_semana(sin), width=LARGURA)

    st.markdown("## Clima e severidade")
    tsc = calcular_taxa_sinistros_por_clima(hor)
    isr = calcular_isr(sin)
    g3, g4 = st.columns([5, 4])
    g3.plotly_chart(grafico_chuva(tsc, kpis["isr_geral"]), width=LARGURA)
    g4.plotly_chart(grafico_isr_heatmap(isr), width=LARGURA)
    st.markdown(
        '<p class="nota">A chuva eleva a frequência de sinistros por hora observada e reduz marginalmente a proporção com vítima. '
        "A categoria da via discrimina a severidade mais do que qualquer faixa de precipitação.</p>",
        unsafe_allow_html=True,
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
        st.dataframe(tabela.round(2), width=LARGURA)

    st.markdown("## Tendência")
    st.plotly_chart(grafico_mensal(hor), width=LARGURA)

    with st.expander("Tabelas de indicadores"):
        st.markdown("### ISR por categoria de via × faixa de chuva")
        st.dataframe(isr.round(3), width=LARGURA, hide_index=True)
        st.markdown("### Taxa de sinistros por faixa de chuva")
        st.dataframe(tsc.round(3), width=LARGURA, hide_index=True)


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
        st.markdown('<p class="nota">Dia da semana, feriado e hora de pico são derivados da data e da hora informadas.</p>', unsafe_allow_html=True)
        return

    precipitacao_3h = max(precipitacao_3h, precipitacao)  # acumulado nunca é menor que a chuva da hora
    clima = {"precipitacao_mm": precipitacao, "precipitacao_3h_mm": precipitacao_3h, "temperatura_c": temperatura,
             "umidade_pct": float(umidade), "vento_m_s": vento, "pressao_mb": pressao}
    via = {"tipo_via": tipo_via, "categoria_via": categoria_via, "zona": zona}
    entrada = montar_entrada(data_ref, hora, clima, via)
    entrada_seca = montar_entrada(data_ref, hora, {**clima, "precipitacao_mm": 0.0, "precipitacao_3h_mm": 0.0}, via)

    # O classificador foi treinado com classes balanceadas; a probabilidade é
    # devolvida à escala da prevalência real para comparação com a média.
    peso = float(meta["classificacao"].get("scale_pos_weight", 1.0))
    feats_c, feats_r = meta["classificacao"]["features"], meta["regressao"]["features"]
    with st.spinner("Processando inferência..."):
        try:
            p_vitima = ajustar_probabilidade_prior(float(clf.predict_proba(entrada[feats_c])[0, 1]), peso)
            p_vitima_seco = ajustar_probabilidade_prior(float(clf.predict_proba(entrada_seca[feats_c])[0, 1]), peso)
            km = float(max(reg.predict(entrada[feats_r])[0], 0))
            km_seco = float(max(reg.predict(entrada_seca[feats_r])[0], 0))
        except Exception as exc:  # noqa: BLE001 - erro de inferência exibido na interface
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
            f"Mesmo cenário sem chuva: {fmt_num(p_vitima_seco * 100, 1)} %. Leitura relativa entre cenários; ROC-AUC do modelo = "
            f"{fmt_num(meta['classificacao']['metricas_teste']['roc_auc'])}.",
        )
    with r2:
        delta = (f"{fmt_delta(km - km_tipico)} km em relação à mediana histórica deste dia/hora ({fmt_num(km_tipico, 1)} km)"
                 if not np.isnan(km_tipico) else "Sem referência histórica para este dia/hora")
        caixa_resultado(
            "Lentidão estimada",
            f"{fmt_num(km, 1)} km",
            delta,
            f"Mesmo cenário sem chuva: {fmt_num(km_seco, 1)} km. Horas acima de 80 km são classificadas como extremas; "
            f"RMSE do modelo = {fmt_num(meta['regressao']['metricas_teste']['rmse'], 1)} km.",
            classe="teal",
        )

    with st.expander("Vetor de entrada enviado aos modelos"):
        st.dataframe(entrada[feats_c].T.rename(columns={0: "valor"}).astype(str), width=LARGURA)


def secao_recomendacoes(meta: dict[str, Any] | None) -> None:
    st.markdown("## Recomendações técnicas")
    st.markdown('<p class="subtitulo">Síntese dos resultados, diretrizes para operação e segurança viária, limitações e métricas dos modelos.</p>', unsafe_allow_html=True)

    st.markdown("### Evidências")
    st.markdown(
        """
1. **Calendário domina a lentidão; o clima modula.** Hora, dia útil e feriado respondem por mais de 85 % do ganho do modelo de regressão. Chuva nas últimas 3 h eleva a mediana do pico em cerca de 8 % e dobra a frequência de horas extremas (> 80 km).
2. **Chuva aumenta a frequência de sinistros (+17 % por hora observada), não a gravidade.** O ISR cai de 65 % para cerca de 60 % sob chuva.
3. **A via é o principal determinante de severidade.** Rodovias e marginais: ISR ≈ 0,72; ruas locais: ≈ 0,61. A diferença entre vias supera o efeito de qualquer faixa de chuva.
4. **Sinistros fatais concentram-se entre 20 h e 6 h** (49 % dos fatais contra 30 % dos sinistros em geral), com madrugadas de sábado e domingo acima dos dias úteis.
5. **Colisões apresentam o maior excedente de retenção** (FRT +2,6 km acima do típico) e o maior volume (37 % dos sinistros).
        """
    )

    st.markdown("### Diretrizes")
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
**Operação de tráfego (CET)**

- Protocolo de chuva no pico: com acumulado de 3 h acima de 2,5 mm entre 16 h e 20 h, antecipar equipes de remoção e agentes nos corredores com histórico de lentidão extrema.
- Priorizar colisões na resposta operacional: maior volume e maior excedente de lentidão.
- Dimensionar escalas por dia/hora/feriado com o modelo de regressão (R² 0,73 no teste; 0,50 em período futuro).
            """
        )
    with b:
        st.markdown(
            """
**Segurança viária (Prefeitura, DER, concessionárias)**

- Gestão de velocidade em marginais, rodovias urbanas e avenidas de trânsito rápido, onde o ISR é maior.
- Fiscalização de velocidade e alcoolemia nas madrugadas de fim de semana.
- Campanhas de chuva orientadas a distância de seguimento e frenagem, dado que o risco adicional é de volume de colisões.
- Completude do registro: 10 % dos sinistros sem tipo e 0,7 % sem via; registros incompletos concentram-se nos sem vítima.
            """
        )

    st.markdown("### Limitações e vieses")
    st.markdown(
        """
- **Estação meteorológica única** (A701, zona norte). Chuvas convectivas localizadas não são capturadas; o efeito climático medido é um limite inferior.
- **Lentidão agregada à cidade.** O simulador não distingue corredores ou regiões. Dias sem coleta da CET (8 %) foram excluídos do treino.
- **Viés de registro no Infosiga.** Sinistros sem vítima dependem de notificação voluntária e têm cadastro mais incompleto; o ISR observado tende a superestimar o real.
- **Classificador de baixo poder discriminativo** (ROC-AUC ≈ 0,60) por restrição deliberada a clima, tempo e via. Variáveis conhecidas apenas após o fato foram excluídas.
- **Validação cruzada aleatória superestima** o desempenho em séries horárias; a validação temporal (2022 → 2023) é a estimativa conservadora.
- **Período de dois anos** sem cobertura de eventos atípicos; zona geográfica derivada por regra geométrica, não pelo zoneamento oficial.
        """
    )

    if meta:
        st.markdown("### Métricas dos modelos")
        mc, mr = meta["classificacao"]["metricas_teste"], meta["regressao"]["metricas_teste"]
        tc, tr = meta["classificacao"]["metricas_temporal"], meta["regressao"]["metricas_temporal"]
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Classificação de severidade (XGBClassifier, classes balanceadas)**")
            st.dataframe(pd.DataFrame({
                "Teste (holdout 20 %)": [f"{mc['f1']:.3f}", f"{mc['f1_macro']:.3f}", f"{mc['roc_auc']:.3f}", f"{mc['balanced_accuracy']:.3f}"],
                "Temporal (2022 → 2023)": [f"{tc.get('f1', float('nan')):.3f}", f"{tc.get('f1_macro', float('nan')):.3f}", f"{tc.get('roc_auc', float('nan')):.3f}", f"{tc.get('balanced_accuracy', float('nan')):.3f}"],
            }, index=["F1", "F1 macro", "ROC-AUC", "Acurácia balanceada"]), width=LARGURA)
            cm = np.array(mc["matriz_confusao"])
            st.markdown(f'<p class="nota">Matriz de confusão (teste): VN {fmt_int(cm[0, 0])} · FP {fmt_int(cm[0, 1])} · FN {fmt_int(cm[1, 0])} · VP {fmt_int(cm[1, 1])}.</p>', unsafe_allow_html=True)
        with m2:
            st.markdown("**Regressão de lentidão (XGBRegressor)**")
            st.dataframe(pd.DataFrame({
                "Teste (holdout 20 %)": [f"{mr['rmse']:.2f} km", f"{mr['mae']:.2f} km", f"{mr['r2']:.3f}"],
                "Temporal (2022 → 2023)": [f"{tr.get('rmse', float('nan')):.2f} km", f"{tr.get('mae', float('nan')):.2f} km", f"{tr.get('r2', float('nan')):.3f}"],
            }, index=["RMSE", "MAE", "R²"]), width=LARGURA)
            st.markdown(f'<p class="nota">Baseline Ridge no teste: R² {meta["regressao"]["baseline_teste"]["r2"]:.3f}. Validação cruzada 5-fold apenas no conjunto de treino.</p>', unsafe_allow_html=True)


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
        st.markdown("### docs/arquitetura.dbml")
        st.markdown('<p class="nota">Modelo relacional lógico dos parquets processados (notação DBML, renderizável em dbdiagram.io).</p>', unsafe_allow_html=True)
        dbml = ler_texto(config.DOCS_DIR / "arquitetura.dbml")
        if dbml:
            st.code(dbml, language="sql")
        else:
            st.warning("Arquivo docs/arquitetura.dbml não encontrado.")

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
            e1.dataframe(pd.DataFrame({"classificação": mc["features"]}), width=LARGURA, hide_index=True)
            e2.dataframe(pd.DataFrame({"regressão": mr["features"]}), width=LARGURA, hide_index=True)
            st.markdown("### Importância dos atributos (ganho)")
            i1, i2 = st.columns(2)
            i1.dataframe(pd.Series(mc["importancia_features"], name="classificação").head(12).round(3), width=LARGURA)
            i2.dataframe(pd.Series(mr["importancia_features"], name="regressão").head(12).round(3), width=LARGURA)
            st.markdown("### Validação cruzada (treino)")
            v1, v2 = st.columns(2)
            v1.dataframe(pd.DataFrame(mc["metricas_cv"]).T.round(3), width=LARGURA)
            v2.dataframe(pd.DataFrame(mr["metricas_cv"]).T.round(3), width=LARGURA)

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


# --------------------------------------------------------------------------- #
# Aplicação
# --------------------------------------------------------------------------- #
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

    with st.sidebar:
        st.markdown("# Citizen Intelligence SP")
        st.markdown("Severidade de sinistros e lentidão sob fatores climáticos e temporais. Município de São Paulo, 2022–2023.")
        st.markdown("### Navegação")
        pagina = st.radio("Página", ["Painel analítico", "Documentação"], label_visibility="collapsed")
        anos_disp = sorted(int(a) for a in consolidado.ano.unique())
        zonas_disp = sorted(consolidado.zona.unique())
        anos, zonas = anos_disp, zonas_disp
        if pagina == "Painel analítico":
            st.markdown("### Filtros da visão geral")
            anos = st.multiselect("Ano", anos_disp, default=anos_disp)
            zonas = st.multiselect("Zona", zonas_disp, default=zonas_disp)
        st.markdown("### Fontes")
        st.markdown("Infosiga-SP (sinistros), CET (lentidão), INMET A701 Mirante de Santana (clima).")
        st.markdown("### Indicadores")
        st.markdown("**ISR** Índice de Severidade Relativa: proporção de sinistros com vítima.")
        st.markdown("**FRT** Fator de Retenção de Tráfego: km de lentidão associados a cada tipo de sinistro, comparados ao típico do horário.")
        st.markdown("**TSC** Taxa de Sinistros por Condição climática: sinistros por hora observada em cada faixa de chuva.")

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
