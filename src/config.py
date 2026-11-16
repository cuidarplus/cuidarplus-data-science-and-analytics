"""Configurações centrais do projeto: caminhos, constantes e nomes de colunas.

Todos os módulos de ``src/`` importam daqui para que caminhos e parâmetros
fiquem em um único lugar. Os caminhos são resolvidos em relação à raiz do
projeto (pasta que contém ``src/``), o que permite executar os scripts tanto
da raiz (``python -m src.preprocessing``) quanto a partir de ``notebooks/``
ou ``app/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"

RAW_CLIMA_PARQUET: Final[Path] = RAW_DIR / "automaticas_processed.parquet"
RAW_SINISTROS_CSV: Final[Path] = RAW_DIR / "sinistros_2022-2024.csv"
RAW_LENTIDAO_CSVS: Final[tuple[Path, ...]] = (
    RAW_DIR / "lentidaotrechos2022.csv",
    RAW_DIR / "lentidao_trechos_2023.csv",
)

# Saídas da etapa de pré-processamento
CONSOLIDADO_PARQUET: Final[Path] = PROCESSED_DIR / "mobilidade_sp_consolidado.parquet"
HORARIO_PARQUET: Final[Path] = PROCESSED_DIR / "mobilidade_sp_horario.parquet"

# Saídas da etapa de modelagem (modelos ``.pkl`` na raiz, conforme especificação)
MODELO_CLASSIFICACAO_PKL: Final[Path] = PROJECT_ROOT / "modelo_classificacao_severidade.pkl"
MODELO_REGRESSAO_PKL: Final[Path] = PROJECT_ROOT / "modelo_regressao_lentidao.pkl"
MODELOS_METADATA_JSON: Final[Path] = PROJECT_ROOT / "modelos_metadata.json"

# --------------------------------------------------------------------------- #
# Parâmetros de domínio
# --------------------------------------------------------------------------- #
# Estação INMET "SAO PAULO - MIRANTE" (Mirante de Santana), a referência
# climatológica oficial da capital paulista.
ESTACAO_INMET: Final[str] = "A701"

# O INMET publica as observações das estações automáticas em UTC. São Paulo
# está em UTC-3 e não adota horário de verão desde 2019, então o deslocamento
# é fixo para todo o período analisado.
FUSO_LOCAL: Final[str] = "America/Sao_Paulo"

# Janela temporal em que as três fontes se sobrepõem. Os dados de lentidão da
# CET disponíveis cobrem apenas 2022 e 2023; sinistros e clima vão além, mas
# só há valor analítico onde as três bases coexistem.
PERIODO_INICIO: Final[str] = "2022-01-01"
PERIODO_FIM: Final[str] = "2023-12-31 23:59:59"

MUNICIPIO_ALVO: Final[str] = "SAO PAULO"

# Semente única para garantir reprodutibilidade de splits e modelos.
RANDOM_STATE: Final[int] = 42
