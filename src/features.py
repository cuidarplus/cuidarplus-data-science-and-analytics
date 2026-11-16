"""Etapa 2 - Engenharia de atributos e indicadores (KPIs) de mobilidade.

Este módulo é puro (não lê arquivos brutos): recebe os DataFrames produzidos
por :mod:`src.preprocessing` e devolve novas colunas ou tabelas-resumo.

Indicadores definidos aqui:

* **ISR - Índice de Severidade Relativa**: proporção de sinistros com vítima
  sobre o total de sinistros, calculado por combinação de via e condição
  climática. Um ISR de 0,70 em "MARGINAL + chuva forte" significa que 70 %
  dos sinistros nesse contexto tiveram ao menos uma vítima.
* **FRT - Fator de Retenção de Tráfego**: média de quilômetros de lentidão
  (na hora do sinistro e na hora seguinte) por categoria de sinistro,
  acompanhada do excedente em relação à lentidão típica daquele horário e
  dia da semana. Mede quanto congestionamento "acompanha" cada tipo de
  ocorrência.
* **TSC - Taxa de Sinistros por Condição climática** (extra): sinistros por
  hora em cada faixa de chuva, normalizando pela quantidade de horas
  observadas em cada faixa.

Execução direta (``python -m src.features``) grava as tabelas de KPI em
``data/processed/`` para consumo pelo notebook e pelo dashboard.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Final

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes de domínio
# --------------------------------------------------------------------------- #
# Feriados nacionais, estaduais (SP) e municipais (capital) de 2022 e 2023.
# Pontos facultativos de Carnaval e Corpus Christi incluídos porque reduzem
# fortemente o tráfego na cidade.
FERIADOS_SP: Final[frozenset[date]] = frozenset(
    {
        # 2022
        date(2022, 1, 1), date(2022, 1, 25), date(2022, 2, 28), date(2022, 3, 1),
        date(2022, 4, 15), date(2022, 4, 21), date(2022, 5, 1), date(2022, 6, 16),
        date(2022, 7, 9), date(2022, 9, 7), date(2022, 10, 12), date(2022, 11, 2),
        date(2022, 11, 15), date(2022, 11, 20), date(2022, 12, 25),
        # 2023
        date(2023, 1, 1), date(2023, 1, 25), date(2023, 2, 20), date(2023, 2, 21),
        date(2023, 4, 7), date(2023, 4, 21), date(2023, 5, 1), date(2023, 6, 8),
        date(2023, 7, 9), date(2023, 9, 7), date(2023, 10, 12), date(2023, 11, 2),
        date(2023, 11, 15), date(2023, 11, 20), date(2023, 12, 25),
    }
)

HORAS_PICO_MANHA: Final[tuple[int, int]] = (7, 10)   # 07:00-09:59
HORAS_PICO_TARDE: Final[tuple[int, int]] = (17, 20)  # 17:00-19:59

# Faixas de chuva horária (mm/h) inspiradas na classificação do INMET/CGE.
FAIXAS_CHUVA: Final[list[tuple[float, str]]] = [
    (0.0, "SEM_CHUVA"),
    (2.5, "FRACA"),
    (10.0, "MODERADA"),
    (np.inf, "FORTE"),
]
ORDEM_CHUVA: Final[list[str]] = ["SEM_CHUVA", "FRACA", "MODERADA", "FORTE"]
ORDEM_DIAS: Final[list[str]] = [
    "Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo",
]
DIAS_PT: Final[dict[int, str]] = dict(enumerate(ORDEM_DIAS))

# Colunas numéricas/categóricas usadas pelos modelos (compartilhadas com
# ``src.modeling`` e ``app/main.py`` para garantir consistência).
FEATURES_TEMPORAIS: Final[list[str]] = ["hora", "dia_semana_num", "mes", "fim_de_semana", "feriado", "hora_pico"]
FEATURES_CLIMATICAS: Final[list[str]] = [
    "precipitacao_mm", "precipitacao_3h_mm", "temperatura_c", "umidade_pct", "vento_m_s", "pressao_mb",
]
FEATURES_VIA: Final[list[str]] = ["tipo_via", "categoria_via", "zona"]


# --------------------------------------------------------------------------- #
# Engenharia de atributos
# --------------------------------------------------------------------------- #
def adicionar_features_temporais(df: pd.DataFrame, coluna_tempo: str = "datetime") -> pd.DataFrame:
    """Adiciona atributos derivados do timestamp.

    Novas colunas: ``hora``, ``dia_semana_num`` (0 = segunda), ``dia_semana``
    (nome em português), ``mes``, ``ano``, ``fim_de_semana``, ``feriado``,
    ``hora_pico`` (1 = pico manhã/tarde em dia útil) e ``periodo_dia``.

    Args:
        df: DataFrame com uma coluna datetime.
        coluna_tempo: Nome da coluna de tempo.

    Returns:
        Cópia do DataFrame com as novas colunas.

    Raises:
        KeyError: Se ``coluna_tempo`` não existir.
    """
    if coluna_tempo not in df.columns:
        raise KeyError(f"Coluna {coluna_tempo!r} ausente no DataFrame")
    out = df.copy()
    ts = pd.to_datetime(out[coluna_tempo])
    out["hora"] = ts.dt.hour.astype("int8")
    out["dia_semana_num"] = ts.dt.day_of_week.astype("int8")
    out["dia_semana"] = out["dia_semana_num"].map(DIAS_PT)
    out["mes"] = ts.dt.month.astype("int8")
    out["ano"] = ts.dt.year.astype("int16")
    out["feriado"] = ts.dt.date.isin(FERIADOS_SP).astype("int8")
    out["fim_de_semana"] = (out["dia_semana_num"] >= 5).astype("int8")
    dia_util = (out["fim_de_semana"] == 0) & (out["feriado"] == 0)
    pico = out["hora"].between(*HORAS_PICO_MANHA, inclusive="left") | out["hora"].between(*HORAS_PICO_TARDE, inclusive="left")
    out["hora_pico"] = (dia_util & pico).astype("int8")
    out["periodo_dia"] = pd.cut(
        out["hora"], bins=[-1, 5, 11, 17, 23], labels=["MADRUGADA", "MANHA", "TARDE", "NOITE"]
    ).astype("string")
    return out


def classificar_chuva(precipitacao_mm: pd.Series) -> pd.Series:
    """Classifica a precipitação horária em faixas qualitativas.

    Args:
        precipitacao_mm: Precipitação em mm/h (NaN permanece NaN).

    Returns:
        Série categórica ordenada: SEM_CHUVA (0), FRACA (<= 2,5), MODERADA
        (<= 10) ou FORTE (> 10 mm/h).
    """
    limites = [-np.inf] + [lim for lim, _ in FAIXAS_CHUVA]
    rotulos = [nome for _, nome in FAIXAS_CHUVA]
    return pd.cut(precipitacao_mm, bins=limites, labels=rotulos, right=True).astype(
        pd.CategoricalDtype(ORDEM_CHUVA, ordered=True)
    )


def adicionar_features_climaticas(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona ``faixa_chuva``, ``chovendo`` e ``chuva_3h`` a partir do clima.

    Args:
        df: DataFrame com ``precipitacao_mm`` e ``precipitacao_3h_mm``.

    Returns:
        Cópia do DataFrame com as novas colunas.
    """
    out = df.copy()
    out["faixa_chuva"] = classificar_chuva(out["precipitacao_mm"])
    out["chovendo"] = (out["precipitacao_mm"] > 0).astype("Int8").where(out["precipitacao_mm"].notna())
    out["chuva_3h"] = (out["precipitacao_3h_mm"] > 0).astype("Int8").where(out["precipitacao_3h_mm"].notna())
    return out


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as transformações de atributos de uma vez.

    Args:
        df: Base consolidada ou base horária.

    Returns:
        DataFrame com atributos temporais e climáticos adicionados.
    """
    return adicionar_features_climaticas(adicionar_features_temporais(df))


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
def calcular_isr(
    consolidado: pd.DataFrame,
    por_via: str = "categoria_via",
    por_clima: str = "faixa_chuva",
    minimo_sinistros: int = 30,
) -> pd.DataFrame:
    """Calcula o ISR (Índice de Severidade Relativa) por via e condição climática.

    ISR = sinistros com vítima / total de sinistros, em cada combinação
    ``(por_via, por_clima)``. Combinações com menos de ``minimo_sinistros``
    são removidas para evitar proporções instáveis.

    Args:
        consolidado: Base consolidada já com ``faixa_chuva`` (ver
            :func:`preparar_features`).
        por_via: Coluna de via (``categoria_via`` ou ``tipo_via``).
        por_clima: Coluna climática categórica.
        minimo_sinistros: Suporte mínimo por combinação.

    Returns:
        DataFrame com colunas ``[por_via, por_clima, total_sinistros,
        com_vitima, isr, isr_geral, isr_relativo]`` ordenado por ISR
        decrescente. ``isr_relativo`` compara o ISR da célula com o ISR
        geral da base (1,0 = igual à média).

    Raises:
        KeyError: Se as colunas necessárias não existirem.
    """
    for col in (por_via, por_clima, "com_vitima"):
        if col not in consolidado.columns:
            raise KeyError(f"Coluna {col!r} ausente; execute preparar_features() antes.")
    base = consolidado.dropna(subset=[por_clima])
    isr_geral = float(base["com_vitima"].mean())
    tabela = (
        base.groupby([por_via, por_clima], observed=True)
        .agg(total_sinistros=("com_vitima", "size"), com_vitima=("com_vitima", "sum"))
        .reset_index()
    )
    tabela = tabela[tabela["total_sinistros"] >= minimo_sinistros].copy()
    tabela["isr"] = tabela["com_vitima"] / tabela["total_sinistros"]
    tabela["isr_geral"] = isr_geral
    tabela["isr_relativo"] = tabela["isr"] / isr_geral
    return tabela.sort_values("isr", ascending=False).reset_index(drop=True)


def calcular_frt(
    consolidado: pd.DataFrame,
    base_horaria: pd.DataFrame,
    por_categoria: str = "tp_sinistro_primario",
    minimo_sinistros: int = 30,
) -> pd.DataFrame:
    """Calcula o FRT (Fator de Retenção de Tráfego) por categoria de sinistro.

    Para cada sinistro, medimos a lentidão da hora (``km_lentidao_total``) e
    da hora seguinte (``km_lentidao_h1``). O FRT é a média desses valores por
    categoria. Como a lentidão depende fortemente do horário, também
    calculamos o **excedente**: a diferença entre a lentidão observada e a
    lentidão típica (mediana) para o mesmo dia da semana e hora, obtida da
    base horária. Um excedente positivo indica congestionamento acima do
    esperado para aquele horário.

    Args:
        consolidado: Base consolidada com features temporais.
        base_horaria: Base horária com features temporais (para a linha de base).
        por_categoria: Coluna de categoria do sinistro (``tp_sinistro_primario``
            ou ``severidade``).
        minimo_sinistros: Suporte mínimo por categoria.

    Returns:
        DataFrame com ``[por_categoria, total_sinistros, frt_km_hora,
        frt_km_hora_seguinte, frt_km_medio, lentidao_tipica_km,
        excedente_km, excedente_pct]`` ordenado por ``frt_km_medio``.
    """
    for col in ("dia_semana_num", "hora"):
        if col not in consolidado.columns or col not in base_horaria.columns:
            raise KeyError(f"Coluna {col!r} ausente; execute preparar_features() em ambas as bases.")
    tipica = (
        base_horaria.dropna(subset=["km_lentidao_total"])
        .groupby(["dia_semana_num", "hora"])["km_lentidao_total"]
        .median()
        .rename("lentidao_tipica_km")
        .reset_index()
    )
    base = consolidado.dropna(subset=["km_lentidao_total"]).merge(tipica, on=["dia_semana_num", "hora"], how="left")
    base["km_medio"] = base[["km_lentidao_total", "km_lentidao_h1"]].mean(axis=1)
    base["excedente_km"] = base["km_medio"] - base["lentidao_tipica_km"]
    tabela = (
        base.groupby(por_categoria, observed=True)
        .agg(
            total_sinistros=("km_medio", "size"),
            frt_km_hora=("km_lentidao_total", "mean"),
            frt_km_hora_seguinte=("km_lentidao_h1", "mean"),
            frt_km_medio=("km_medio", "mean"),
            lentidao_tipica_km=("lentidao_tipica_km", "mean"),
            excedente_km=("excedente_km", "mean"),
        )
        .reset_index()
    )
    tabela = tabela[tabela["total_sinistros"] >= minimo_sinistros].copy()
    tabela["excedente_pct"] = tabela["excedente_km"] / tabela["lentidao_tipica_km"].replace(0, np.nan) * 100
    return tabela.sort_values("frt_km_medio", ascending=False).reset_index(drop=True)


def calcular_taxa_sinistros_por_clima(base_horaria: pd.DataFrame) -> pd.DataFrame:
    """Calcula sinistros por hora e ISR em cada faixa de chuva (KPI extra).

    Normaliza pela quantidade de horas observadas em cada faixa, o que evita
    o viés de "chove pouco, logo há poucos sinistros na chuva".

    Args:
        base_horaria: Base horária com ``faixa_chuva`` e contagens de sinistros.

    Returns:
        DataFrame com ``[faixa_chuva, horas, sinistros, sinistros_por_hora,
        sinistros_com_vitima_por_hora, isr]``.
    """
    base = base_horaria.dropna(subset=["faixa_chuva"])
    tabela = (
        base.groupby("faixa_chuva", observed=True)
        .agg(horas=("n_sinistros", "size"), sinistros=("n_sinistros", "sum"), com_vitima=("n_sinistros_com_vitima", "sum"))
        .reset_index()
    )
    tabela["sinistros_por_hora"] = tabela["sinistros"] / tabela["horas"]
    tabela["sinistros_com_vitima_por_hora"] = tabela["com_vitima"] / tabela["horas"]
    tabela["isr"] = tabela["com_vitima"] / tabela["sinistros"].replace(0, np.nan)
    return tabela


def calcular_kpis_gerais(consolidado: pd.DataFrame, base_horaria: pd.DataFrame) -> dict[str, float]:
    """Resume os números-chave usados nos cartões do dashboard.

    Args:
        consolidado: Base consolidada com features.
        base_horaria: Base horária com features.

    Returns:
        Dicionário com totais e médias do período.
    """
    com_lentidao = base_horaria.dropna(subset=["km_lentidao_total"])
    chuva = consolidado.dropna(subset=["chovendo"])
    return {
        "total_sinistros": int(len(consolidado)),
        "total_com_vitima": int(consolidado["com_vitima"].sum()),
        "total_fatais": int((consolidado["severidade"] == "FATAL").sum()),
        "isr_geral": float(consolidado["com_vitima"].mean()),
        "isr_chuva": float(chuva.loc[chuva["chovendo"] == 1, "com_vitima"].mean()),
        "isr_seco": float(chuva.loc[chuva["chovendo"] == 0, "com_vitima"].mean()),
        "lentidao_media_km": float(com_lentidao["km_lentidao_total"].mean()),
        "lentidao_pico_km": float(com_lentidao.loc[com_lentidao["hora_pico"] == 1, "km_lentidao_total"].mean()),
        "lentidao_max_km": float(com_lentidao["km_lentidao_total"].max()),
        "horas_com_chuva_pct": float(100 * (base_horaria["precipitacao_mm"] > 0).mean()),
    }


def main() -> None:
    """Gera e grava as tabelas de KPI em ``data/processed``."""
    from src.preprocessing import carregar_consolidado, carregar_horario

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    consolidado = preparar_features(carregar_consolidado())
    horario = preparar_features(carregar_horario())

    isr = calcular_isr(consolidado)
    frt = calcular_frt(consolidado, horario)
    frt_sev = calcular_frt(consolidado, horario, por_categoria="severidade")
    taxa = calcular_taxa_sinistros_por_clima(horario)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    isr.to_csv(PROCESSED_DIR / "kpi_isr.csv", index=False)
    frt.to_csv(PROCESSED_DIR / "kpi_frt_tipo_sinistro.csv", index=False)
    frt_sev.to_csv(PROCESSED_DIR / "kpi_frt_severidade.csv", index=False)
    taxa.to_csv(PROCESSED_DIR / "kpi_taxa_sinistros_clima.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n== ISR por categoria de via x faixa de chuva (top 10)\n", isr.head(10).to_string(index=False))
    print("\n== FRT por tipo de sinistro\n", frt.to_string(index=False))
    print("\n== FRT por severidade\n", frt_sev.to_string(index=False))
    print("\n== Taxa de sinistros por faixa de chuva\n", taxa.to_string(index=False))
    print("\n== KPIs gerais\n", calcular_kpis_gerais(consolidado, horario))


if __name__ == "__main__":
    main()
