# Global Solution 2026-2 — Citizen Intelligence & Predictive Analytics

Impacto de fatores **climáticos** e **temporais** na severidade de sinistros de
trânsito e na extensão de congestionamentos no município de **São Paulo**
(2022–2023). Python local (macOS), sem Docker e sem ORM: toda a comunicação
com dados é via Pandas e arquivos locais.

## Estrutura

```
.
├── app/main.py                  # Dashboard Streamlit (3 abas)
├── data/
│   ├── raw/                     # Fontes originais (INMET, Infosiga, CET)
│   └── processed/               # Parquets consolidados e tabelas de KPI
├── docs/
│   ├── arquitetura.dbml         # Modelo relacional dos dados processados
│   └── dicionario_de_dados.md   # Variáveis finais, tipos e unidades
├── notebooks/
│   └── 01_relatorio_analise_mobilidade.ipynb   # EDA + análise crítica dos modelos
├── src/
│   ├── config.py                # Caminhos e constantes
│   ├── preprocessing.py         # Etapa 1: limpeza, merge temporal, parquet consolidado
│   ├── features.py              # Etapa 2: features e KPIs (ISR, FRT, TSC)
│   ├── modeling.py              # Etapa 3: classificação + regressão, CV, .pkl
│   └── viz.py                   # Paleta e tema dos gráficos
├── modelo_classificacao_severidade.pkl   # gerado por src/modeling.py
├── modelo_regressao_lentidao.pkl         # gerado por src/modeling.py
├── modelos_metadata.json                 # métricas e features dos modelos
├── Makefile
└── requirements.txt
```

## Fontes de dados (`data/raw/`)

| Arquivo | Origem | Uso |
|---|---|---|
| `automaticas_processed.parquet` | INMET, estações automáticas | Filtrada a estação **A701** (Mirante de Santana); UTC → horário local |
| `sinistros_2022-2024.csv` | Infosiga-SP | Sinistros com e sem vítima; filtrado o município de São Paulo |
| `lentidaotrechos2022.csv`, `lentidao_trechos_2023.csv` | CET | Trechos em lentidão a cada 30 min, agregados por hora |

As três bases são alinhadas pela chave de tempo (hora) com `pd.merge_asof`.

## Como executar

```bash
brew install libomp          # runtime OpenMP exigido pelo XGBoost no macOS
make setup                   # cria venv e instala requirements.txt
make pipeline                # Etapas 1 → 2 → 3 (gera parquets, KPIs e .pkl)
make notebook                # executa o notebook de ponta a ponta
make run-app                 # abre o dashboard Streamlit
```

Ou módulo a módulo: `venv/bin/python -m src.preprocessing`, `-m src.features`, `-m src.modeling`.

## Indicadores

* **ISR — Índice de Severidade Relativa:** proporção de sinistros com vítima por categoria de via × faixa de chuva.
* **FRT — Fator de Retenção de Tráfego:** km médios de lentidão (hora do sinistro e seguinte) por tipo de sinistro, comparados ao típico do mesmo dia/hora.
* **TSC — Taxa de Sinistros por Condição climática:** sinistros por hora observada em cada faixa de chuva.

## Modelos

| Modelo | Alvo | Algoritmo | Teste (holdout 20 %) | Temporal (2022 → 2023) |
|---|---|---|---|---|
| Classificação | sinistro com vítima | XGBClassifier (balanceado) | F1 0,66 · ROC-AUC 0,60 · acurácia balanceada 0,57 | ROC-AUC 0,58 |
| Regressão | km de lentidão na hora | XGBRegressor | RMSE 9,6 km · R² 0,73 | R² 0,50 |

Validação cruzada 5-fold apenas no treino; métricas reportadas no teste e em
um ano futuro. Detalhes e análise crítica no notebook e na aba
*Recomendações* do dashboard.
