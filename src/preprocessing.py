"""Etapa 1 - Preparação e tratamento dos dados brutos (``data/raw`` -> ``data/processed``).

Fontes:
    * ``automaticas_processed.parquet`` - observações horárias das estações
      automáticas do INMET (filtramos apenas a estação A701, Mirante de Santana).
    * ``sinistros_2022-2024.csv`` - sinistros de trânsito do Infosiga-SP
      (com e sem vítimas); filtramos o município de São Paulo.
    * ``lentidaotrechos2022.csv`` e ``lentidao_trechos_2023.csv`` - trechos em
      lentidão medidos pela CET a cada 30 minutos.

Saídas:
    * ``data/processed/mobilidade_sp_consolidado.parquet`` - uma linha por
      sinistro, enriquecida com o clima da hora e a lentidão da hora.
    * ``data/processed/mobilidade_sp_horario.parquet`` - uma linha por hora do
      período, com clima, lentidão total e contagem de sinistros (base para a
      regressão de congestionamento).

Execução: ``python -m src.preprocessing`` a partir da raiz do projeto.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from src.config import (
    CONSOLIDADO_PARQUET,
    ESTACAO_INMET,
    FUSO_LOCAL,
    HORARIO_PARQUET,
    MUNICIPIO_ALVO,
    PERIODO_FIM,
    PERIODO_INICIO,
    PROCESSED_DIR,
    RAW_CLIMA_PARQUET,
    RAW_LENTIDAO_CSVS,
    RAW_SINISTROS_CSV,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Mapeamentos de colunas
# --------------------------------------------------------------------------- #
# Renomeamos as colunas do INMET para português e descartamos as variáveis de
# telemetria da estação (bateria, temperatura da CPU) e as versões max/min da
# hora anterior, redundantes para a análise.
COLUNAS_CLIMA: dict[str, str] = {
    "datetime": "datetime",
    "precipitation_mm": "precipitacao_mm",
    "dry_bulb_temp_C": "temperatura_c",
    "dew_point_temp_C": "ponto_orvalho_c",
    "relative_humidity_pct": "umidade_pct",
    "atm_pressure_station_mB": "pressao_mb",
    "wind_speed_m_s": "vento_m_s",
    "wind_gust_max_m_s": "rajada_m_s",
    "wind_direction_deg": "direcao_vento_deg",
    "global_radiation_kJ_m2": "radiacao_kj_m2",
}

COLUNAS_CLIMA_CONTINUAS: tuple[str, ...] = (
    "temperatura_c",
    "ponto_orvalho_c",
    "umidade_pct",
    "pressao_mb",
    "vento_m_s",
    "rajada_m_s",
    "direcao_vento_deg",
    "radiacao_kj_m2",
)

COLUNAS_GRAVIDADE: tuple[str, ...] = (
    "qtd_gravidade_fatal",
    "qtd_gravidade_grave",
    "qtd_gravidade_leve",
    "qtd_gravidade_ileso",
    "qtd_gravidade_nao_disponivel",
)

COLUNAS_VEICULOS: tuple[str, ...] = (
    "qtd_pedestre",
    "qtd_bicicleta",
    "qtd_motocicleta",
    "qtd_automovel",
    "qtd_onibus",
    "qtd_caminhao",
    "qtd_veic_outros",
    "qtd_veic_nao_disponivel",
)

# Primeira palavra do logradouro -> categoria de via. Os prefixos cobrem ~99%
# dos registros de São Paulo; o restante cai em "OUTROS".
CATEGORIA_VIA_POR_PREFIXO: dict[str, str] = {
    "AVENIDA": "AVENIDA",
    "AV": "AVENIDA",
    "AV.": "AVENIDA",
    "RUA": "RUA",
    "R": "RUA",
    "R.": "RUA",
    "ALAMEDA": "RUA",
    "TRAVESSA": "RUA",
    "MARGINAL": "MARGINAL",
    "SP": "RODOVIA",
    "BR": "RODOVIA",
    "RODOVIA": "RODOVIA",
    "ROD": "RODOVIA",
    "ESTRADA": "ESTRADA",
    "ACESSO": "ACESSO",
    "PONTE": "PONTE_VIADUTO_TUNEL",
    "PTE": "PONTE_VIADUTO_TUNEL",
    "VIADUTO": "PONTE_VIADUTO_TUNEL",
    "VD": "PONTE_VIADUTO_TUNEL",
    "TUNEL": "PONTE_VIADUTO_TUNEL",
    "TÚNEL": "PONTE_VIADUTO_TUNEL",
    "PRAÇA": "PRACA_LARGO",
    "PRACA": "PRACA_LARGO",
    "LARGO": "PRACA_LARGO",
}

REGIOES_CET: tuple[str, ...] = ("CENTRO", "LESTE", "NORTE", "OESTE", "SUL")

# Centro geográfico de referência (Praça da Sé) e raio aproximado da zona
# central, usados para derivar a zona do sinistro a partir das coordenadas.
CENTRO_SP: tuple[float, float] = (-23.5505, -46.6333)
RAIO_CENTRO_GRAUS: float = 0.04  # ~4,4 km


# --------------------------------------------------------------------------- #
# Clima (INMET)
# --------------------------------------------------------------------------- #
def carregar_clima(
    caminho: Path = RAW_CLIMA_PARQUET,
    estacao: str = ESTACAO_INMET,
    inicio: str = PERIODO_INICIO,
    fim: str = PERIODO_FIM,
) -> pd.DataFrame:
    """Carrega e trata as observações horárias de uma estação do INMET.

    O parquet bruto tem ~73 milhões de linhas (todas as estações do país).
    Usamos ``pyarrow.dataset`` com *predicate pushdown* para ler apenas os
    row groups da estação desejada, evitando carregar 2 GB em memória.

    Args:
        caminho: Caminho do parquet do INMET.
        estacao: Código da estação (padrão A701 - Mirante de Santana).
        inicio: Início da janela, em horário local.
        fim: Fim da janela, em horário local.

    Returns:
        DataFrame indexado por posição com a coluna ``datetime`` (horário local,
        sem tz) em grade horária completa e as variáveis climáticas tratadas.

    Raises:
        FileNotFoundError: Se o parquet não existir.
        ValueError: Se a estação não for encontrada no arquivo.
    """
    if not caminho.exists():
        raise FileNotFoundError(f"Parquet do INMET não encontrado: {caminho}")

    dataset = ds.dataset(caminho)
    tabela = dataset.to_table(
        filter=pc.field("station_code") == estacao,
        columns=list(COLUNAS_CLIMA.keys()),
    )
    clima = tabela.to_pandas().rename(columns=COLUNAS_CLIMA)
    if clima.empty:
        raise ValueError(f"Estação {estacao!r} não encontrada em {caminho}")

    # O INMET publica em UTC; convertemos para o horário local de São Paulo
    # para alinhar com as bases da CET e do Infosiga, que usam hora local.
    clima["datetime"] = (
        pd.to_datetime(clima["datetime"], utc=True)
        .dt.tz_convert(FUSO_LOCAL)
        .dt.tz_localize(None)
    )
    clima = clima[(clima["datetime"] >= inicio) & (clima["datetime"] <= fim)]
    clima = clima.sort_values("datetime").drop_duplicates("datetime")

    # Reindexamos para a grade horária completa: as ~200 horas ausentes do
    # período viram linhas com NaN, que são tratadas de forma explícita abaixo
    # em vez de desaparecerem silenciosamente do merge.
    grade = pd.date_range(inicio, fim, freq="h", name="datetime")
    clima = clima.set_index("datetime").reindex(grade)

    # Radiação global aparece como -3.0 à noite (ruído do sensor); fisicamente
    # o valor mínimo é zero.
    clima["radiacao_kj_m2"] = clima["radiacao_kj_m2"].clip(lower=0)

    # Tratamento de nulos -----------------------------------------------------
    # 1) Variáveis contínuas e suaves (temperatura, umidade, pressão, vento):
    #    interpolação linear limitada a 6 horas. Falhas curtas de sensor são
    #    bem aproximadas pela tendência local; buracos maiores permanecem NaN
    #    para não inventar um dia inteiro de clima.
    # 2) Precipitação: NaN em até 3 horas consecutivas vira 0 mm. A hipótese é
    #    que falhas curtas do pluviômetro em geral ocorrem em períodos secos
    #    (em chuva forte o sensor registra). Buracos maiores permanecem NaN.
    # 3) Horas que continuam com NaN após (1) e (2) são sinalizadas em
    #    ``clima_imputado``/``clima_disponivel`` e excluídas apenas no treino
    #    dos modelos, não da base consolidada.
    faltante_antes = clima.isna().any(axis=1)
    clima[list(COLUNAS_CLIMA_CONTINUAS)] = clima[list(COLUNAS_CLIMA_CONTINUAS)].interpolate(
        method="linear", limit=6, limit_area="inside"
    )
    clima["precipitacao_mm"] = _preencher_lacunas_curtas(clima["precipitacao_mm"], limite=3, valor=0.0)
    clima["clima_imputado"] = faltante_antes & ~clima.isna().any(axis=1)

    # Pluviometria acumulada nas últimas 3 h: captura chuva persistente que
    # alaga vias, diferente de uma pancada isolada.
    clima["precipitacao_3h_mm"] = clima["precipitacao_mm"].rolling(3, min_periods=1).sum()

    # Horas em que a estação não reportou nada (lacunas longas) são removidas
    # da tabela de clima: no merge, ``merge_asof`` recua até 1 h para o último
    # registro válido; além disso o clima fica indisponível para a hora.
    sem_registro = clima[list(COLUNAS_CLIMA_CONTINUAS) + ["precipitacao_mm"]].isna().all(axis=1)
    clima = clima[~sem_registro].reset_index()
    logger.info(
        "Clima %s: %d horas válidas (%d removidas por ausência total), %d imputadas",
        estacao,
        len(clima),
        int(sem_registro.sum()),
        int(clima["clima_imputado"].sum()),
    )
    return clima


def _preencher_lacunas_curtas(serie: pd.Series, limite: int, valor: float) -> pd.Series:
    """Preenche com ``valor`` apenas sequências de NaN com tamanho <= ``limite``.

    Args:
        serie: Série ordenada no tempo.
        limite: Tamanho máximo (em posições) da lacuna a preencher.
        valor: Valor usado no preenchimento.

    Returns:
        Série com as lacunas curtas preenchidas e as longas intactas.
    """
    nulo = serie.isna()
    # Identifica blocos consecutivos de NaN e seu tamanho.
    bloco = (nulo != nulo.shift()).cumsum()
    tamanho_bloco = nulo.groupby(bloco).transform("sum")
    preencher = nulo & (tamanho_bloco <= limite)
    return serie.mask(preencher, valor)


# --------------------------------------------------------------------------- #
# Sinistros (Infosiga)
# --------------------------------------------------------------------------- #
def carregar_sinistros(
    caminho: Path = RAW_SINISTROS_CSV,
    municipio: str = MUNICIPIO_ALVO,
    inicio: str = PERIODO_INICIO,
    fim: str = PERIODO_FIM,
) -> pd.DataFrame:
    """Carrega e trata os sinistros do Infosiga para o município alvo.

    Args:
        caminho: Caminho do CSV de sinistros (``;`` como separador, Latin-1).
        municipio: Nome do município como aparece na coluna ``municipio``.
        inicio: Início da janela temporal.
        fim: Fim da janela temporal.

    Returns:
        DataFrame com uma linha por sinistro, timestamp local em ``datetime``,
        alvo binário ``com_vitima`` e atributos de via/tipo tratados.

    Raises:
        FileNotFoundError: Se o CSV não existir.
    """
    if not caminho.exists():
        raise FileNotFoundError(f"CSV de sinistros não encontrado: {caminho}")

    colunas = [
        "id_sinistro", "tipo_registro", "data_sinistro", "hora_sinistro", "dia_da_semana",
        "turno", "logradouro", "tipo_via", "latitude", "longitude", "municipio",
        "administracao", "tp_sinistro_primario", *COLUNAS_GRAVIDADE, *COLUNAS_VEICULOS,
    ]
    sin = pd.read_csv(
        caminho, sep=";", encoding="latin-1", usecols=colunas, low_memory=False, dtype={"hora_sinistro": "string"}
    )
    sin = sin[sin["municipio"] == municipio].drop(columns="municipio").copy()
    logger.info("Sinistros em %s: %d registros brutos", municipio, len(sin))

    # Timestamp: 0,08% dos registros não têm hora. Sem hora não é possível
    # alinhar com clima/lentidão, e a fração é pequena demais para enviesar a
    # análise, então removemos (alternativa descartada: imputar meio do turno,
    # o que distorceria a análise de horário de pico).
    hora_valida = sin["hora_sinistro"].str.match(r"^\d{2}:\d{2}$", na=False)
    logger.info("Removendo %d sinistros sem hora válida", int((~hora_valida).sum()))
    sin = sin[hora_valida].copy()
    sin["datetime"] = pd.to_datetime(
        sin["data_sinistro"] + " " + sin["hora_sinistro"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    sin = sin.dropna(subset=["datetime"])
    sin = sin[(sin["datetime"] >= inicio) & (sin["datetime"] <= fim)]

    # Coordenadas vêm com vírgula decimal; 0,3% sem coordenada permanece NaN
    # (não é usada nos modelos, apenas em mapas).
    for col in ("latitude", "longitude"):
        sin[col] = pd.to_numeric(sin[col].astype("string").str.replace(",", ".", regex=False), errors="coerce").astype("float64")

    # Contagens de vítimas/veículos: NaN no Infosiga significa "zero" (o campo
    # só é preenchido quando há ocorrência daquela categoria), então imputamos 0.
    contagens = [*COLUNAS_GRAVIDADE, *COLUNAS_VEICULOS]
    sin[contagens] = sin[contagens].fillna(0).astype("int16")
    sin["qtd_vitimas"] = sin[["qtd_gravidade_fatal", "qtd_gravidade_grave", "qtd_gravidade_leve"]].sum(axis=1)
    sin["qtd_veiculos"] = sin[list(COLUNAS_VEICULOS)].sum(axis=1)

    # Alvo de severidade. "NOTIFICACAO" é o registro sem vítimas (apenas danos
    # materiais); os demais tipos envolvem ao menos uma vítima.
    sin["com_vitima"] = (sin["tipo_registro"] != "NOTIFICACAO").astype("int8")
    sin["severidade"] = sin["tipo_registro"].map(
        {"NOTIFICACAO": "SEM_VITIMA", "SINISTRO NAO FATAL": "COM_VITIMA", "SINISTRO FATAL": "FATAL"}
    )

    # Categóricas: "NAO DISPONIVEL" é mantido como categoria explícita porque
    # a ausência de informação é frequente (10% em tipo de sinistro) e pode
    # carregar sinal (registros menos completos tendem a ser os sem vítima).
    sin["tipo_via"] = sin["tipo_via"].fillna("NAO DISPONIVEL").str.strip()
    sin["tp_sinistro_primario"] = sin["tp_sinistro_primario"].fillna("NAO DISPONIVEL").str.strip()
    sin["categoria_via"] = classificar_categoria_via(sin["logradouro"])
    sin["zona"] = classificar_zona(sin["latitude"], sin["longitude"])
    sin["dia_da_semana"] = sin["dia_da_semana"].str.strip()

    sin = sin.drop(columns=["data_sinistro", "hora_sinistro"]).sort_values("datetime").reset_index(drop=True)
    logger.info("Sinistros tratados: %d (%.1f%% com vítima)", len(sin), 100 * sin["com_vitima"].mean())
    return sin


def classificar_categoria_via(logradouro: pd.Series) -> pd.Series:
    """Deriva a categoria da via a partir do prefixo do nome do logradouro.

    Args:
        logradouro: Série com o nome do logradouro (pode conter NaN).

    Returns:
        Série categórica com valores como ``AVENIDA``, ``RUA``, ``MARGINAL``,
        ``RODOVIA``, ``PONTE_VIADUTO_TUNEL``, ``OUTROS`` ou ``NAO DISPONIVEL``.
    """
    nome = logradouro.astype("string").str.strip().str.upper()
    prefixo = nome.str.split().str[0]
    categoria = prefixo.map(CATEGORIA_VIA_POR_PREFIXO)
    # Nomes que começam com "AYRTON SENNA", "ANHANGUERA" etc. são rodovias ou
    # marginais conhecidas sem prefixo; tratamos os casos mais frequentes.
    marginal = nome.str.contains("MARGINAL|MARG\\.", regex=True, na=False)
    rodovia = nome.str.contains("RODOVIA|ROD\\.|ANHANGUERA|BANDEIRANTES|IMIGRANTES|ANCHIETA|AYRTON SENNA|DUTRA|REGIS BITTENCOURT|CASTELO BRANCO|RAPOSO TAVARES|FERNAO DIAS|RODOANEL", regex=True, na=False)
    categoria = categoria.mask(categoria.isna() & marginal, "MARGINAL")
    categoria = categoria.mask(categoria.isna() & rodovia, "RODOVIA")
    categoria = categoria.mask(nome.isna() | nome.str.startswith("NAO DISPONIVEL"), "NAO DISPONIVEL")
    return categoria.fillna("OUTROS").astype("string")


def classificar_zona(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    """Deriva a zona da cidade (CENTRO/NORTE/SUL/LESTE/OESTE) pelas coordenadas.

    Regra geométrica simples em torno da Praça da Sé: pontos dentro de um
    quadrado de ~4,4 km são CENTRO; os demais seguem o eixo (latitude ou
    longitude) de maior afastamento. É uma aproximação suficiente para
    análise agregada, não substitui o zoneamento oficial das subprefeituras.

    Args:
        latitude: Latitude em graus decimais (NaN permitido).
        longitude: Longitude em graus decimais (NaN permitido).

    Returns:
        Série de strings com a zona ou ``NAO DISPONIVEL`` sem coordenadas.
    """
    dlat = latitude - CENTRO_SP[0]
    dlon = longitude - CENTRO_SP[1]
    centro = (dlat.abs() < RAIO_CENTRO_GRAUS) & (dlon.abs() < RAIO_CENTRO_GRAUS)
    norte_sul = np.where(dlat > 0, "NORTE", "SUL")
    leste_oeste = np.where(dlon > 0, "LESTE", "OESTE")
    zona = np.where(centro, "CENTRO", np.where(dlat.abs() > dlon.abs(), norte_sul, leste_oeste))
    return pd.Series(zona, index=latitude.index, dtype="string").mask(latitude.isna() | longitude.isna(), "NAO DISPONIVEL")


# --------------------------------------------------------------------------- #
# Lentidão (CET)
# --------------------------------------------------------------------------- #
def carregar_lentidao(
    caminhos: Iterable[Path] = RAW_LENTIDAO_CSVS,
    inicio: str = PERIODO_INICIO,
    fim: str = PERIODO_FIM,
) -> pd.DataFrame:
    """Carrega os trechos em lentidão da CET e agrega em grade horária.

    A CET publica, a cada 30 minutos, a lista de trechos em lentidão com seu
    comprimento em metros. Somamos os trechos de cada instantâneo (km totais na
    cidade) e tiramos a média dos instantâneos da hora, obtendo a extensão
    média de lentidão por hora.

    Args:
        caminhos: CSVs de lentidão (um por ano).
        inicio: Início da janela temporal.
        fim: Fim da janela temporal.

    Returns:
        DataFrame em grade horária completa com ``km_lentidao_total``,
        ``km_lentidao_<regiao>`` para cada região da CET, ``n_trechos_lentidao``
        e a flag ``lentidao_disponivel``.

    Raises:
        FileNotFoundError: Se algum CSV não existir.
    """
    frames: list[pd.DataFrame] = []
    for caminho in caminhos:
        if not caminho.exists():
            raise FileNotFoundError(f"CSV de lentidão não encontrado: {caminho}")
        frames.append(pd.read_csv(caminho, sep=";", encoding="latin-1", usecols=["data", "corredor", "tamanho", "nome_regiao"]))
    bruto = pd.concat(frames, ignore_index=True)
    bruto["datetime"] = pd.to_datetime(bruto["data"], format="%d/%m/%Y %H:%M", errors="coerce")
    bruto = bruto.dropna(subset=["datetime", "tamanho"])
    bruto = bruto[(bruto["datetime"] >= inicio) & (bruto["datetime"] <= fim)]
    bruto["km"] = bruto["tamanho"].astype(float) / 1000.0
    bruto["nome_regiao"] = bruto["nome_regiao"].str.strip().str.upper()
    bruto["hora"] = bruto["datetime"].dt.floor("h")

    # Soma por instantâneo (30 min) e por região, depois média dos instantâneos
    # dentro da hora. Média (e não soma) evita dobrar a métrica em horas com
    # dois instantâneos versus horas com apenas um.
    por_snapshot = (
        bruto.pivot_table(index=["hora", "datetime"], columns="nome_regiao", values="km", aggfunc="sum", fill_value=0.0)
        .reindex(columns=list(REGIOES_CET), fill_value=0.0)
    )
    por_snapshot["km_total"] = por_snapshot.sum(axis=1)
    por_snapshot["n_trechos"] = bruto.groupby(["hora", "datetime"]).size()
    horario = por_snapshot.groupby(level="hora").mean()
    horario.columns = [f"km_lentidao_{c.lower()}" if c in REGIOES_CET else c for c in horario.columns]
    horario = horario.rename(columns={"km_total": "km_lentidao_total", "n_trechos": "n_trechos_lentidao"})

    # Tratamento de ausências --------------------------------------------------
    # A CET registra apenas trechos *em lentidão*: uma hora sem registro em um
    # dia com medições significa "0 km de lentidão" (madrugadas, por exemplo).
    # Já dias inteiros sem nenhum instantâneo (~8% do período) indicam falha
    # de coleta/publicação; nesses dias mantemos NaN e a flag
    # ``lentidao_disponivel = False`` para não ensinar ao modelo que não houve
    # trânsito.
    grade = pd.date_range(inicio, fim, freq="h", name="datetime")
    horario = horario.reindex(grade)
    dias_com_dado = set(horario.dropna(subset=["km_lentidao_total"]).index.normalize().unique())
    dia_disponivel = horario.index.normalize().isin(dias_com_dado)
    colunas_km = [c for c in horario.columns]
    horario.loc[dia_disponivel, colunas_km] = horario.loc[dia_disponivel, colunas_km].fillna(0.0)
    horario["lentidao_disponivel"] = dia_disponivel
    horario = horario.reset_index()
    logger.info(
        "Lentidão: %d horas na grade, %d sem coleta (%d dias)",
        len(horario),
        int((~horario["lentidao_disponivel"]).sum()),
        len(grade.normalize().unique()) - len(dias_com_dado),
    )
    return horario


# --------------------------------------------------------------------------- #
# Consolidação
# --------------------------------------------------------------------------- #
def construir_base_horaria(clima: pd.DataFrame, lentidao: pd.DataFrame, sinistros: pd.DataFrame) -> pd.DataFrame:
    """Monta a tabela em grade horária: clima + lentidão + contagem de sinistros.

    Uma observação do INMET carimbada na hora ``H`` resume o intervalo
    ``(H-1, H]`` (ex.: precipitação acumulada na hora anterior). Já a hora
    ``H`` da grade representa o intervalo ``[H, H+1)``. Por isso o clima da
    hora ``H`` da grade vem do registro INMET ``H+1``; se ele faltar,
    ``merge_asof`` recua até 1 hora (registro ``H``), garantindo cobertura nas
    lacunas curtas do sensor.

    Args:
        clima: Saída de :func:`carregar_clima`.
        lentidao: Saída de :func:`carregar_lentidao`.
        sinistros: Saída de :func:`carregar_sinistros`.

    Returns:
        DataFrame com uma linha por hora e as colunas das três fontes.
    """
    base = lentidao.copy()
    base["datetime_ref_clima"] = base["datetime"] + pd.Timedelta(hours=1)
    base = pd.merge_asof(
        base.sort_values("datetime_ref_clima"),
        clima.sort_values("datetime").rename(columns={"datetime": "datetime_clima"}),
        left_on="datetime_ref_clima",
        right_on="datetime_clima",
        direction="backward",
        tolerance=pd.Timedelta(hours=1),
    )
    base["defasagem_clima_min"] = (base["datetime_ref_clima"] - base["datetime_clima"]).dt.total_seconds().div(60)
    # Clima "disponível" exige as três variáveis centrais dos modelos; horas
    # sem registro em até 1 h ficam com NaN e são excluídas apenas do treino.
    base["clima_disponivel"] = ~base[["precipitacao_mm", "temperatura_c", "umidade_pct"]].isna().any(axis=1)
    base["clima_imputado"] = base["clima_imputado"].fillna(False).astype(bool)
    base = base.drop(columns=["datetime_ref_clima", "datetime_clima"])

    contagem = (
        sinistros.assign(hora=sinistros["datetime"].dt.floor("h"))
        .groupby("hora")
        .agg(n_sinistros=("com_vitima", "size"), n_sinistros_com_vitima=("com_vitima", "sum"), n_vitimas=("qtd_vitimas", "sum"))
        .reset_index()
        .rename(columns={"hora": "datetime"})
    )
    base = base.merge(contagem, on="datetime", how="left")
    for col in ("n_sinistros", "n_sinistros_com_vitima", "n_vitimas"):
        base[col] = base[col].fillna(0).astype("int16")
    return base.sort_values("datetime").reset_index(drop=True)


def construir_base_consolidada(sinistros: pd.DataFrame, base_horaria: pd.DataFrame) -> pd.DataFrame:
    """Enriquece cada sinistro com o clima e a lentidão da hora em que ocorreu.

    O alinhamento usa ``pd.merge_asof`` (direção *backward*, tolerância de 1 h)
    sobre a base horária, que já contém clima e lentidão alinhados: um sinistro
    às 14h35 recebe a linha da hora 14h. Também anexamos a lentidão da hora
    seguinte (``km_lentidao_h1``) para medir o efeito posterior do sinistro.

    Args:
        sinistros: Saída de :func:`carregar_sinistros`.
        base_horaria: Saída de :func:`construir_base_horaria`.

    Returns:
        DataFrame com uma linha por sinistro e colunas das três fontes.
    """
    colunas_hora = [c for c in base_horaria.columns if c not in {"n_sinistros", "n_sinistros_com_vitima", "n_vitimas"}]
    horaria = base_horaria[colunas_hora].sort_values("datetime")
    consolidado = pd.merge_asof(
        sinistros.sort_values("datetime"),
        horaria.rename(columns={"datetime": "datetime_hora"}),
        left_on="datetime",
        right_on="datetime_hora",
        direction="backward",
        tolerance=pd.Timedelta(hours=1),
    )
    proxima = horaria[["datetime", "km_lentidao_total"]].rename(
        columns={"datetime": "datetime_h1", "km_lentidao_total": "km_lentidao_h1"}
    )
    consolidado["datetime_h1"] = consolidado["datetime_hora"] + pd.Timedelta(hours=1)
    consolidado = consolidado.merge(proxima, on="datetime_h1", how="left").drop(columns=["datetime_h1"])
    for col in ("clima_disponivel", "clima_imputado", "lentidao_disponivel"):
        consolidado[col] = consolidado[col].fillna(False).astype(bool)
    return consolidado.reset_index(drop=True)


def executar_pipeline(salvar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa toda a Etapa 1 e (opcionalmente) grava os parquets processados.

    Args:
        salvar: Se ``True``, grava ``CONSOLIDADO_PARQUET`` e ``HORARIO_PARQUET``.

    Returns:
        Tupla ``(consolidado, horario)``.
    """
    clima = carregar_clima()
    sinistros = carregar_sinistros()
    lentidao = carregar_lentidao()
    horario = construir_base_horaria(clima, lentidao, sinistros)
    consolidado = construir_base_consolidada(sinistros, horario)
    if salvar:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        consolidado.to_parquet(CONSOLIDADO_PARQUET, index=False)
        horario.to_parquet(HORARIO_PARQUET, index=False)
        logger.info("Gravado %s (%d linhas) e %s (%d linhas)", CONSOLIDADO_PARQUET.name, len(consolidado), HORARIO_PARQUET.name, len(horario))
    return consolidado, horario


def carregar_consolidado(caminho: Path = CONSOLIDADO_PARQUET) -> pd.DataFrame:
    """Lê a base consolidada já processada.

    Raises:
        FileNotFoundError: Se o pipeline ainda não foi executado.
    """
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Execute `python -m src.preprocessing` primeiro.")
    return pd.read_parquet(caminho)


def carregar_horario(caminho: Path = HORARIO_PARQUET) -> pd.DataFrame:
    """Lê a base horária já processada.

    Raises:
        FileNotFoundError: Se o pipeline ainda não foi executado.
    """
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Execute `python -m src.preprocessing` primeiro.")
    return pd.read_parquet(caminho)


def main() -> None:
    """Ponto de entrada de linha de comando."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    consolidado, horario = executar_pipeline(salvar=True)
    print(f"Consolidado: {consolidado.shape}, Horário: {horario.shape}")


if __name__ == "__main__":
    main()
