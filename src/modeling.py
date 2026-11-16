"""Etapa 3 - Modelagem preditiva: severidade de sinistros e extensão de lentidão.

Modelos:

* **Classificação** (``com_vitima``): prevê se um sinistro terá vítima a
  partir de atributos temporais, climáticos e de via. Algoritmo principal:
  ``XGBClassifier``; baseline: regressão logística.
* **Regressão** (``km_lentidao_total``): estima a extensão média de lentidão
  na cidade em uma hora a partir de atributos temporais e climáticos.
  Algoritmo principal: ``XGBRegressor``; baseline: Ridge.

Protocolo de avaliação (nunca apenas em treino):

1. Divisão treino/teste (80/20, estratificada na classificação).
2. Validação cruzada k-fold (k = 5) *somente no treino* para estimar a
   variância das métricas.
3. Métricas finais no conjunto de teste, intocado durante o treino.
4. Verificação temporal extra: treina em 2022 e testa em 2023, para medir a
   degradação quando o modelo é usado em um período futuro.

Os modelos finais (pipelines completos com pré-processamento) são gravados
em ``.pkl`` na raiz do projeto, acompanhados de ``modelos_metadata.json``
com métricas e lista de atributos, consumidos pelo dashboard.

Execução: ``python -m src.modeling``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    MODELO_CLASSIFICACAO_PKL,
    MODELO_REGRESSAO_PKL,
    MODELOS_METADATA_JSON,
    RANDOM_STATE,
)
from src.features import FEATURES_CLIMATICAS, FEATURES_TEMPORAIS, FEATURES_VIA, preparar_features

logger = logging.getLogger(__name__)

ALVO_CLASSIFICACAO: Final[str] = "com_vitima"
ALVO_REGRESSAO: Final[str] = "km_lentidao_total"
FEATURES_CLASSIFICACAO: Final[list[str]] = [*FEATURES_TEMPORAIS, *FEATURES_CLIMATICAS, *FEATURES_VIA]
FEATURES_REGRESSAO: Final[list[str]] = [*FEATURES_TEMPORAIS, *FEATURES_CLIMATICAS]
N_FOLDS: Final[int] = 5
TEST_SIZE: Final[float] = 0.2


@dataclass
class ResultadoModelo:
    """Métricas e metadados de um modelo treinado."""

    nome: str
    alvo: str
    features: list[str]
    n_treino: int
    n_teste: int
    metricas_cv: dict[str, dict[str, float]] = field(default_factory=dict)
    metricas_teste: dict[str, Any] = field(default_factory=dict)
    metricas_temporal: dict[str, Any] = field(default_factory=dict)
    baseline_teste: dict[str, Any] = field(default_factory=dict)
    importancia_features: dict[str, float] = field(default_factory=dict)
    scale_pos_weight: float = 1.0


# --------------------------------------------------------------------------- #
# Construção de pipelines
# --------------------------------------------------------------------------- #
def _preprocessador(features_numericas: list[str], features_categoricas: list[str], escalar: bool) -> ColumnTransformer:
    """Monta o ``ColumnTransformer`` com one-hot nas categóricas.

    Args:
        features_numericas: Colunas numéricas (passam direto ou são escaladas).
        features_categoricas: Colunas categóricas (one-hot, categorias
            desconhecidas são ignoradas para robustez no simulador).
        escalar: Se ``True``, aplica ``StandardScaler`` às numéricas (útil para
            modelos lineares; desnecessário para árvores).
    """
    numerico = StandardScaler() if escalar else "passthrough"
    transformadores: list[tuple[str, Any, list[str]]] = [("num", numerico, features_numericas)]
    if features_categoricas:
        transformadores.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features_categoricas)
        )
    return ColumnTransformer(transformadores, verbose_feature_names_out=False)


def construir_classificador(
    features: list[str] = FEATURES_CLASSIFICACAO, baseline: bool = False, scale_pos_weight: float = 1.0
) -> Pipeline:
    """Cria o pipeline de classificação de severidade.

    Args:
        features: Lista de atributos de entrada.
        baseline: Se ``True``, usa regressão logística em vez de XGBoost.
        scale_pos_weight: Peso da classe positiva no XGBoost. Usamos
            ``n_negativos / n_positivos`` para balancear: sem isso o modelo
            colapsa em "sempre com vítima" (classe majoritária, 65 %), o que
            maximiza F1 mas é inútil como apoio à decisão.

    Returns:
        ``Pipeline`` não treinado.
    """
    categoricas = [f for f in features if f in FEATURES_VIA]
    numericas = [f for f in features if f not in categoricas]
    if baseline:
        modelo: Any = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    else:
        modelo = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return Pipeline([("prep", _preprocessador(numericas, categoricas, escalar=baseline)), ("modelo", modelo)])


def construir_regressor(features: list[str] = FEATURES_REGRESSAO, baseline: bool = False) -> Pipeline:
    """Cria o pipeline de regressão da extensão de lentidão.

    Args:
        features: Lista de atributos de entrada.
        baseline: Se ``True``, usa Ridge em vez de XGBoost.

    Returns:
        ``Pipeline`` não treinado.
    """
    categoricas = [f for f in features if f in FEATURES_VIA]
    numericas = [f for f in features if f not in categoricas]
    if baseline:
        modelo: Any = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    else:
        modelo = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return Pipeline([("prep", _preprocessador(numericas, categoricas, escalar=baseline)), ("modelo", modelo)])


# --------------------------------------------------------------------------- #
# Preparação dos conjuntos
# --------------------------------------------------------------------------- #
def preparar_dados_classificacao(consolidado: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Seleciona linhas e colunas para o modelo de severidade.

    Mantém apenas sinistros com clima disponível (~3 % são descartados). As
    categorias "NAO DISPONIVEL" de via/zona são mantidas como classe própria,
    pois o simulador também pode receber uma via desconhecida.

    Args:
        consolidado: Base consolidada (com ou sem features; são recalculadas).

    Returns:
        Tupla ``(X, y)``.
    """
    df = preparar_features(consolidado)
    df = df[df["clima_disponivel"]].dropna(subset=FEATURES_CLASSIFICACAO + [ALVO_CLASSIFICACAO])
    X = df[FEATURES_CLASSIFICACAO].copy()
    for col in FEATURES_VIA:
        X[col] = X[col].astype(str)
    y = df[ALVO_CLASSIFICACAO].astype(int)
    X.attrs["ano"] = df["ano"].to_numpy()
    return X, y


def preparar_dados_regressao(base_horaria: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Seleciona linhas e colunas para o modelo de lentidão.

    Mantém apenas horas com clima e lentidão disponíveis (dias sem coleta da
    CET não são "zero trânsito" e por isso ficam fora do treino).

    Args:
        base_horaria: Base horária (features são recalculadas).

    Returns:
        Tupla ``(X, y)``.
    """
    df = preparar_features(base_horaria)
    df = df[df["clima_disponivel"] & df["lentidao_disponivel"]].dropna(subset=FEATURES_REGRESSAO + [ALVO_REGRESSAO])
    X = df[FEATURES_REGRESSAO].copy()
    y = df[ALVO_REGRESSAO].astype(float)
    X.attrs["ano"] = df["ano"].to_numpy()
    return X, y


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def metricas_classificacao(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    """Calcula o conjunto de métricas de classificação binária.

    Returns:
        Dicionário com F1 (positiva e macro), precisão, recall, acurácia,
        acurácia balanceada, ROC-AUC e matriz de confusão ``[[TN, FP], [FN, TP]]``.
    """
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "matriz_confusao": confusion_matrix(y_true, y_pred).tolist(),
    }


def metricas_regressao(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula RMSE, MAE, R² e MAPE-robusto para regressão."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "media_alvo": float(np.mean(y_true)),
    }


def _resumir_cv(cv: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Converte a saída de ``cross_validate`` em média/desvio por métrica."""
    resumo: dict[str, dict[str, float]] = {}
    for chave, valores in cv.items():
        if chave.startswith("test_"):
            nome = chave.removeprefix("test_")
            resumo[nome] = {"media": float(np.mean(valores)), "desvio": float(np.std(valores))}
    return resumo


def _importancia(pipeline: Pipeline) -> dict[str, float]:
    """Extrai importância de atributos (ganho) de um pipeline com XGBoost."""
    modelo = pipeline.named_steps["modelo"]
    if not hasattr(modelo, "feature_importances_"):
        return {}
    nomes = pipeline.named_steps["prep"].get_feature_names_out()
    pares = sorted(zip(nomes, modelo.feature_importances_), key=lambda p: p[1], reverse=True)
    return {str(n): float(v) for n, v in pares}


# --------------------------------------------------------------------------- #
# Treinamento e avaliação
# --------------------------------------------------------------------------- #
def treinar_classificador(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, ResultadoModelo]:
    """Treina, valida e avalia o classificador de severidade.

    Args:
        X: Atributos (saída de :func:`preparar_dados_classificacao`).
        y: Alvo binário.

    Returns:
        Pipeline treinado em *todo* o treino e o :class:`ResultadoModelo`.
    """
    anos = X.attrs.get("ano")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    peso_positivo = float((y_tr == 0).sum() / (y_tr == 1).sum())
    pipeline = construir_classificador(scale_pos_weight=peso_positivo)
    cv = cross_validate(
        pipeline, X_tr, y_tr,
        cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=("f1", "f1_macro", "roc_auc", "balanced_accuracy"),
        n_jobs=1,
    )
    pipeline.fit(X_tr, y_tr)
    proba = pipeline.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    baseline = construir_classificador(baseline=True).fit(X_tr, y_tr)
    proba_b = baseline.predict_proba(X_te)[:, 1]
    metricas_b = metricas_classificacao(y_te, (proba_b >= 0.5).astype(int), proba_b)

    resultado = ResultadoModelo(
        nome="XGBClassifier",
        alvo=ALVO_CLASSIFICACAO,
        features=list(X.columns),
        n_treino=len(X_tr),
        n_teste=len(X_te),
        metricas_cv=_resumir_cv(cv),
        metricas_teste=metricas_classificacao(y_te, pred, proba),
        baseline_teste=metricas_b,
        importancia_features=_importancia(pipeline),
        scale_pos_weight=peso_positivo,
    )
    if anos is not None:
        resultado.metricas_temporal = _validacao_temporal(
            construir_classificador(scale_pos_weight=peso_positivo), X, y, anos, classificacao=True
        )
    logger.info("Classificação - teste: %s", {k: v for k, v in resultado.metricas_teste.items() if k != "matriz_confusao"})
    return pipeline, resultado


def treinar_regressor(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, ResultadoModelo]:
    """Treina, valida e avalia o regressor de lentidão.

    Args:
        X: Atributos (saída de :func:`preparar_dados_regressao`).
        y: Alvo contínuo (km).

    Returns:
        Pipeline treinado em *todo* o treino e o :class:`ResultadoModelo`.
    """
    anos = X.attrs.get("ano")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    pipeline = construir_regressor()
    cv = cross_validate(
        pipeline, X_tr, y_tr,
        cv=KFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"),
        n_jobs=1,
    )
    pipeline.fit(X_tr, y_tr)
    pred = np.clip(pipeline.predict(X_te), 0, None)  # lentidão nunca é negativa

    baseline = construir_regressor(baseline=True).fit(X_tr, y_tr)
    metricas_b = metricas_regressao(y_te, np.clip(baseline.predict(X_te), 0, None))

    resumo_cv = _resumir_cv(cv)
    # Converte scores negativos do scikit-learn em valores positivos legíveis.
    resumo_cv = {k.replace("neg_", "").replace("root_mean_squared_error", "rmse").replace("mean_absolute_error", "mae"):
                 ({"media": -v["media"], "desvio": v["desvio"]} if k.startswith("neg_") else v)
                 for k, v in resumo_cv.items()}

    resultado = ResultadoModelo(
        nome="XGBRegressor",
        alvo=ALVO_REGRESSAO,
        features=list(X.columns),
        n_treino=len(X_tr),
        n_teste=len(X_te),
        metricas_cv=resumo_cv,
        metricas_teste=metricas_regressao(y_te, pred),
        baseline_teste=metricas_b,
        importancia_features=_importancia(pipeline),
    )
    if anos is not None:
        resultado.metricas_temporal = _validacao_temporal(construir_regressor(), X, y, anos, classificacao=False)
    logger.info("Regressão - teste: %s", resultado.metricas_teste)
    return pipeline, resultado


def _validacao_temporal(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, anos: np.ndarray, classificacao: bool
) -> dict[str, Any]:
    """Treina no primeiro ano e testa no último para medir deriva temporal."""
    ano_min, ano_max = int(anos.min()), int(anos.max())
    if ano_min == ano_max:
        return {}
    treino, teste = anos == ano_min, anos == ano_max
    pipeline.fit(X[treino], y[treino])
    if classificacao:
        proba = pipeline.predict_proba(X[teste])[:, 1]
        metricas = metricas_classificacao(y[teste], (proba >= 0.5).astype(int), proba)
    else:
        metricas = metricas_regressao(y[teste], np.clip(pipeline.predict(X[teste]), 0, None))
    return {"treino_ano": ano_min, "teste_ano": ano_max, "n_teste": int(teste.sum()), **metricas}


def ajustar_probabilidade_prior(probabilidade: float | np.ndarray, scale_pos_weight: float) -> float | np.ndarray:
    """Desfaz o efeito do balanceamento para obter a probabilidade na escala real.

    Com ``scale_pos_weight = w`` o XGBoost aprende odds inflacionadas por
    ``w`` em relação à prevalência real. A correção ``p / (p + (1 - p) * w)``
    devolve a probabilidade comparável à taxa histórica (~65 % com vítima),
    o que é necessário no simulador para o usuário interpretar o número.
    A decisão de classe (limiar 0,5 na escala balanceada) não muda.

    Args:
        probabilidade: Saída de ``predict_proba[:, 1]`` do modelo balanceado.
        scale_pos_weight: Peso usado no treino (``n_negativos / n_positivos``).

    Returns:
        Probabilidade corrigida, no mesmo formato da entrada.
    """
    p = np.asarray(probabilidade, dtype=float)
    ajustada = p / (p + (1.0 - p) * scale_pos_weight)
    return float(ajustada) if np.ndim(probabilidade) == 0 else ajustada


# --------------------------------------------------------------------------- #
# Persistência
# --------------------------------------------------------------------------- #
def salvar_modelo(pipeline: Pipeline, caminho: Path) -> None:
    """Serializa o pipeline treinado com ``joblib``.

    Raises:
        OSError: Se a gravação falhar.
    """
    try:
        joblib.dump(pipeline, caminho)
    except OSError as exc:
        raise OSError(f"Falha ao gravar modelo em {caminho}: {exc}") from exc
    logger.info("Modelo gravado em %s", caminho)


def carregar_modelo(caminho: Path) -> Pipeline:
    """Carrega um pipeline ``.pkl``.

    Raises:
        FileNotFoundError: Se o arquivo não existir (execute ``python -m src.modeling``).
    """
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {caminho}. Execute `python -m src.modeling`.")
    return joblib.load(caminho)


def carregar_metadata(caminho: Path = MODELOS_METADATA_JSON) -> dict[str, Any]:
    """Lê o JSON de métricas/metadados gerado pelo treinamento.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not caminho.exists():
        raise FileNotFoundError(f"Metadata não encontrada: {caminho}. Execute `python -m src.modeling`.")
    return json.loads(caminho.read_text(encoding="utf-8"))


def executar_treinamento(consolidado: pd.DataFrame, base_horaria: pd.DataFrame, salvar: bool = True) -> dict[str, ResultadoModelo]:
    """Executa a Etapa 3 completa: dois modelos, avaliação e persistência.

    Args:
        consolidado: Base consolidada (sinistros).
        base_horaria: Base horária.
        salvar: Se ``True``, grava os ``.pkl`` e o JSON de metadata.

    Returns:
        Dicionário ``{"classificacao": ResultadoModelo, "regressao": ResultadoModelo}``.
    """
    X_c, y_c = preparar_dados_classificacao(consolidado)
    clf, res_c = treinar_classificador(X_c, y_c)

    X_r, y_r = preparar_dados_regressao(base_horaria)
    reg, res_r = treinar_regressor(X_r, y_r)

    if salvar:
        salvar_modelo(clf, MODELO_CLASSIFICACAO_PKL)
        salvar_modelo(reg, MODELO_REGRESSAO_PKL)
        metadata = {
            "classificacao": asdict(res_c),
            "regressao": asdict(res_r),
            "categorias_via": {col: sorted(X_c[col].unique().tolist()) for col in FEATURES_VIA},
            "faixas_numericas": {
                col: {"min": float(X_r[col].min()), "max": float(X_r[col].max()), "mediana": float(X_r[col].median())}
                for col in FEATURES_CLIMATICAS
            },
        }
        MODELOS_METADATA_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Metadata gravada em %s", MODELOS_METADATA_JSON)
    return {"classificacao": res_c, "regressao": res_r}


def main() -> None:
    """Ponto de entrada de linha de comando."""
    from src.preprocessing import carregar_consolidado, carregar_horario

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    resultados = executar_treinamento(carregar_consolidado(), carregar_horario(), salvar=True)
    for nome, res in resultados.items():
        print(f"\n== {nome} ({res.nome}) | treino={res.n_treino} teste={res.n_teste}")
        print("CV (treino):", json.dumps(res.metricas_cv, indent=1, ensure_ascii=False))
        print("Teste:", json.dumps(res.metricas_teste, ensure_ascii=False))
        print("Baseline:", json.dumps(res.baseline_teste, ensure_ascii=False))
        print("Temporal:", json.dumps(res.metricas_temporal, ensure_ascii=False))
        print("Top features:", list(res.importancia_features.items())[:8])


if __name__ == "__main__":
    main()
