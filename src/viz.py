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
