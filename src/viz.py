"""Paleta e utilitários de visualização compartilhados por notebook e dashboard.

A paleta categórica segue uma ordem fixa validada para daltonismo (pares
adjacentes com separação suficiente). Regras adotadas:

* Categórico: cores atribuídas em ordem fixa, nunca cicladas.
* Sequencial (magnitude, ex.: mapas de calor): um único matiz, azul claro -> escuro.
* Divergente (polaridade, ex.: excedente de lentidão): azul <-> vermelho com
  cinza neutro no centro.
* Texto e eixos sempre em tons neutros, nunca na cor da série.
"""

from __future__ import annotations

from typing import Final

CATEGORICAS: Final[list[str]] = [
    "#2a78d6",  # 1 azul
    "#eb6834",  # 2 laranja
    "#1baf7a",  # 3 água
    "#eda100",  # 4 amarelo
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 verde
    "#6250d6",  # 7 violeta
    "#e34948",  # 8 vermelho
]
SEQUENCIAL: Final[list[str]] = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
DIVERGENTE: Final[list[str]] = ["#1c5cab", "#2a78d6", "#86b6ef", "#f0efec", "#f3a3a2", "#e34948", "#b32a2a"]

# Cores semânticas fixas para severidade (identidade estável em todos os gráficos)
COR_SEVERIDADE: Final[dict[str, str]] = {
    "SEM_VITIMA": CATEGORICAS[0],
    "COM_VITIMA": CATEGORICAS[1],
    "FATAL": CATEGORICAS[7],
}
COR_CHUVA: Final[dict[str, str]] = {
    "SEM_CHUVA": "#9ec5f4",
    "FRACA": "#5598e7",
    "MODERADA": "#256abf",
    "FORTE": "#104281",
}
SURFACE: Final[str] = "#fcfcfb"
TEXTO: Final[str] = "#0b0b0b"
TEXTO_SECUNDARIO: Final[str] = "#52514e"
GRADE: Final[str] = "#e6e5e1"

PLOTLY_LAYOUT: Final[dict] = {
    "template": "plotly_white",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, -apple-system, Helvetica, Arial, sans-serif", "color": TEXTO, "size": 13},
    "colorway": CATEGORICAS,
    "margin": {"l": 48, "r": 24, "t": 56, "b": 48},
    "xaxis": {"gridcolor": GRADE, "zeroline": False},
    "yaxis": {"gridcolor": GRADE, "zeroline": False},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    "hoverlabel": {"bgcolor": "#ffffff", "font": {"color": TEXTO}},
}


def aplicar_tema_seaborn() -> None:
    """Configura matplotlib/seaborn com a paleta e eixos recessivos."""
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette=CATEGORICAS, font_scale=1.0)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRADE,
            "axes.grid": True,
            "grid.color": GRADE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelcolor": TEXTO_SECUNDARIO,
            "xtick.color": TEXTO_SECUNDARIO,
            "ytick.color": TEXTO_SECUNDARIO,
            "figure.dpi": 110,
        }
    )


# --------------------------------------------------------------------------- #
# Paleta corporativa (dashboard executivo)
# --------------------------------------------------------------------------- #
# Tons sóbrios: azul-marinho, teal, cobre e ameixa, em ordem fixa validada
# para daltonismo. Cinza-chumbo é reservado a séries de referência/baseline
# (sempre acompanhado de rótulo direto), nunca a uma série categórica.
CORPORATIVA: Final[list[str]] = ["#2f5597", "#0f9688", "#c46a2e", "#8a5a9a"]
CHUMBO: Final[str] = "#6b7280"
CHUMBO_CLARO: Final[str] = "#9ca3af"
SEQUENCIAL_NAVY: Final[list[str]] = [
    "#e3e9f4", "#c9d5ea", "#aec0df", "#92abd3", "#7696c6", "#5a80b7", "#3f6aa6", "#2f5597", "#264780", "#1d3a69",
]
COR_SEVERIDADE_CORP: Final[dict[str, str]] = {
    "SEM_VITIMA": CHUMBO,
    "COM_VITIMA": CORPORATIVA[0],
    "FATAL": CORPORATIVA[2],
}
COR_CHUVA_CORP: Final[dict[str, str]] = {
    "SEM_CHUVA": "#aec0df",
    "FRACA": "#7696c6",
    "MODERADA": "#3f6aa6",
    "FORTE": "#1d3a69",
}
TEXTO_CORP: Final[str] = "#1f2937"
TEXTO_SEC_CORP: Final[str] = "#6b7280"
GRADE_CORP: Final[str] = "#e5e7eb"

PLOTLY_LAYOUT_CORP: Final[dict] = {
    "template": "plotly_white",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif", "color": TEXTO_CORP, "size": 12},
    "colorway": CORPORATIVA,
    "margin": {"l": 48, "r": 16, "t": 48, "b": 40},
    "title": {"font": {"size": 14, "color": TEXTO_CORP}, "x": 0, "xanchor": "left"},
    "xaxis": {"gridcolor": GRADE_CORP, "showline": False, "zeroline": False, "showgrid": False, "ticks": ""},
    "yaxis": {"gridcolor": GRADE_CORP, "showline": False, "zeroline": False, "ticks": ""},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "font": {"size": 11}},
    "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": GRADE_CORP, "font": {"color": TEXTO_CORP, "size": 12, "family": "Inter, Helvetica, Arial, sans-serif"}},
}


# --------------------------------------------------------------------------- #
# Paleta "telemetria" (dashboard em dark mode)
# --------------------------------------------------------------------------- #
# Fundo quase preto; dados em cores de alta luminância (laranja de sotaque,
# magenta FIAP, violeta, âmbar). Eixos e rótulos em cinza médio (#888888)
# para recuarem visualmente. A escala sequencial parte de um tom próximo ao
# fundo (valores baixos "somem") e sobe por violeta -> magenta -> laranja ->
# âmbar, monotônica em luminância.
LARANJA: Final[str] = "#ff4f00"
MAGENTA: Final[str] = "#ed145b"
VIOLETA: Final[str] = "#a78bfa"
AMBAR: Final[str] = "#ffb020"
RACING: Final[list[str]] = [LARANJA, MAGENTA, VIOLETA, AMBAR]
NEUTRO: Final[str] = "#71717a"
NEUTRO_ESCURO: Final[str] = "#3f3f46"
SEQUENCIAL_TELEMETRIA: Final[list[str]] = [
    "#1c1b22", "#2d1b45", "#47196a", "#6a1878", "#921a74", "#bd1a63", "#e3274a", "#ff4f00", "#ff8a1f", "#ffc247",
]
COR_SEVERIDADE_DARK: Final[dict[str, str]] = {
    "SEM_VITIMA": NEUTRO,
    "COM_VITIMA": LARANJA,
    "FATAL": MAGENTA,
}
COR_CHUVA_DARK: Final[dict[str, str]] = {
    "SEM_CHUVA": NEUTRO_ESCURO,
    "FRACA": "#7c3aed",
    "MODERADA": MAGENTA,
    "FORTE": LARANJA,
}
TEXTO_DARK: Final[str] = "#e8eaed"
TEXTO_SEC_DARK: Final[str] = "#888888"
GRADE_DARK: Final[str] = "#232326"

PLOTLY_LAYOUT_DARK: Final[dict] = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif", "color": TEXTO_SEC_DARK, "size": 12},
    "colorway": RACING,
    "margin": {"l": 48, "r": 16, "t": 52, "b": 40},
    "title": {"font": {"family": "'Space Grotesk', Inter, sans-serif", "size": 14, "color": TEXTO_DARK}, "x": 0, "xanchor": "left"},
    "xaxis": {"gridcolor": GRADE_DARK, "showline": False, "zeroline": False, "showgrid": False, "ticks": "",
              "tickfont": {"color": TEXTO_SEC_DARK}, "title": {"font": {"color": TEXTO_SEC_DARK}}},
    "yaxis": {"gridcolor": GRADE_DARK, "showline": False, "zeroline": False, "ticks": "",
              "tickfont": {"color": TEXTO_SEC_DARK}, "title": {"font": {"color": TEXTO_SEC_DARK}}},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
               "font": {"size": 11, "color": TEXTO_SEC_DARK}, "bgcolor": "rgba(0,0,0,0)"},
    "hoverlabel": {"bgcolor": "#141416", "bordercolor": LARANJA,
                   "font": {"color": TEXTO_DARK, "size": 12, "family": "Inter, Helvetica, Arial, sans-serif"}},
    "modebar": {"bgcolor": "rgba(0,0,0,0)", "color": "#52525b", "activecolor": LARANJA},
}


# --------------------------------------------------------------------------- #
# Paleta "telemetria" (dashboard em light mode)
# --------------------------------------------------------------------------- #
# Mesma identidade (laranja de sotaque + magenta FIAP) sobre fundo off-white.
# O laranja é escurecido para manter contraste sobre branco; a escala
# sequencial inverte o sentido de luminância (claro = pouco, escuro = muito).
LARANJA_CLARO_TEMA: Final[str] = "#e94800"
MAGENTA_CLARO_TEMA: Final[str] = "#c8104f"
RACING_CLARO: Final[list[str]] = [LARANJA_CLARO_TEMA, MAGENTA_CLARO_TEMA, "#7c3aed", "#b7791f"]
SEQUENCIAL_TELEMETRIA_CLARO: Final[list[str]] = [
    "#f6f4f8", "#ecdff2", "#ddc0e6", "#cb9bd8", "#b673c8", "#a14bb6", "#c0267f", "#b0124f", "#7d0b3a", "#4a0624",
]
TEXTO_LIGHT: Final[str] = "#111113"
TEXTO_SEC_LIGHT: Final[str] = "#6b6b73"
GRADE_LIGHT: Final[str] = "#e7e7ea"

PLOTLY_LAYOUT_LIGHT: Final[dict] = {
    **PLOTLY_LAYOUT_DARK,
    "template": "plotly_white",
    "font": {**PLOTLY_LAYOUT_DARK["font"], "color": TEXTO_SEC_LIGHT},
    "colorway": RACING_CLARO,
    "title": {**PLOTLY_LAYOUT_DARK["title"], "font": {**PLOTLY_LAYOUT_DARK["title"]["font"], "color": TEXTO_LIGHT}},
    "xaxis": {**PLOTLY_LAYOUT_DARK["xaxis"], "gridcolor": GRADE_LIGHT, "tickfont": {"color": TEXTO_SEC_LIGHT}, "title": {"font": {"color": TEXTO_SEC_LIGHT}}},
    "yaxis": {**PLOTLY_LAYOUT_DARK["yaxis"], "gridcolor": GRADE_LIGHT, "tickfont": {"color": TEXTO_SEC_LIGHT}, "title": {"font": {"color": TEXTO_SEC_LIGHT}}},
    "legend": {**PLOTLY_LAYOUT_DARK["legend"], "font": {"size": 11, "color": TEXTO_SEC_LIGHT}},
    "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": LARANJA_CLARO_TEMA,
                   "font": {"color": TEXTO_LIGHT, "size": 12, "family": "Inter, Helvetica, Arial, sans-serif"}},
    "modebar": {"bgcolor": "rgba(0,0,0,0)", "color": "#a1a1aa", "activecolor": LARANJA_CLARO_TEMA},
}

# Tokens de gráfico por tema, consumidos pelo dashboard (app/main.py).
PALETAS: Final[dict[str, dict]] = {
    "escuro": {
        "layout": PLOTLY_LAYOUT_DARK, "series": RACING, "neutro": NEUTRO, "neutro_escuro": NEUTRO_ESCURO,
        "sequencial": SEQUENCIAL_TELEMETRIA, "severidade": COR_SEVERIDADE_DARK, "chuva": COR_CHUVA_DARK,
        "texto": TEXTO_DARK, "texto_sec": TEXTO_SEC_DARK, "grade": GRADE_DARK,
        "chuva_fundo": "rgba(167,139,250,0.22)", "faixa": "rgba(255,255,255,0.035)",
    },
    "claro": {
        "layout": PLOTLY_LAYOUT_LIGHT, "series": RACING_CLARO, "neutro": "#8a8a93", "neutro_escuro": "#c4c4cc",
        "sequencial": SEQUENCIAL_TELEMETRIA_CLARO,
        "severidade": {"SEM_VITIMA": "#8a8a93", "COM_VITIMA": LARANJA_CLARO_TEMA, "FATAL": MAGENTA_CLARO_TEMA},
        "chuva": {"SEM_CHUVA": "#d4d4d8", "FRACA": "#8b5cf6", "MODERADA": MAGENTA_CLARO_TEMA, "FORTE": LARANJA_CLARO_TEMA},
        "texto": TEXTO_LIGHT, "texto_sec": TEXTO_SEC_LIGHT, "grade": GRADE_LIGHT,
        "chuva_fundo": "rgba(124,58,237,0.14)", "faixa": "rgba(17,17,19,0.04)",
    },
}
