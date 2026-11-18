# Global Solution 2026-2 — Citizen Intelligence & Predictive Analytics

Impacto de fatores **climáticos** e **temporais** na severidade de sinistros de
trânsito e na extensão de congestionamentos no município de **São Paulo**
(2022–2023). Python local (macOS), sem Docker e sem ORM: toda a comunicação
com dados é via Pandas e arquivos locais.

## Dashboard online

**Acesse o painel publicado:** [INSERIR_LINK_DO_SITE_AQUI]

O mesmo dashboard pode ser executado localmente com `make run-app` (ver
[Como executar](#como-executar)).

## Estrutura

```
.
├── app/main.py                  # Dashboard Streamlit (painel analítico com 3 abas + documentação)
├── data/
│   ├── raw/                     # Fontes originais (INMET, Infosiga, CET); baixar à parte, ver abaixo
│   └── processed/               # Parquets consolidados, KPIs e rastreabilidade (versionados)
├── docs/
│   ├── arquitetura.dbml         # Modelo relacional dos dados processados
│   └── dicionario_de_dados.md   # Variáveis finais, tipos e unidades
├── notebooks/
│   └── 01_relatorio_analise_mobilidade.ipynb   # Preparação, KPIs, EDA com testes, modelos, vieses
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

## Obtenção dos dados (reprodução)

O repositório **já contém a pasta `data/processed/`** (cerca de 5 MB), com os
parquets consolidados, as tabelas de KPI e a tabela de rastreabilidade
(`rastreabilidade.csv`). Com ela, o notebook (`make notebook`) e o dashboard
(`make run-app`) rodam diretamente, sem os dados brutos, e os modelos `.pkl` já
treinados estão na raiz.

Para reproduzir o pipeline completo a partir das fontes originais
(`make pipeline`), baixe o arquivo ZIP com a pasta `data/raw/` exata utilizada
no projeto (cerca de 2,5 GB):

**Download de `data/raw/`:** [https://drive.google.com/drive/folders/1IjJ-x1J8dizx00CqjbglwTKg-12E3WbB?usp=sharing]

Descompacte o arquivo na raiz do repositório, de modo que os arquivos fiquem em
`data/raw/`. Os arquivos lidos pelo pipeline são:

| Arquivo | Tamanho aproximado |
|---|---|
| `data/raw/automaticas_processed.parquet` | 2,0 GB |
| `data/raw/sinistros_2022-2024.csv` | 158 MB |
| `data/raw/lentidaotrechos2022.csv` | 17 MB |
| `data/raw/lentidao_trechos_2023.csv` | 18 MB |

Em seguida, execute `make pipeline`. A Etapa 1 regenera `data/processed/`
(inclusive a tabela de rastreabilidade, com as contagens de linhas brutas,
filtradas e finais de cada fonte) e a Etapa 3 retreina os modelos com a mesma
semente (`RANDOM_STATE = 42`), reproduzindo as métricas de `modelos_metadata.json`.

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

A métrica de referência é a **validação Out-of-Time**: treino em 2022 e teste
em 2023, um ano inteiro não visto. O holdout aleatório de 20 % é reportado como
métrica secundária e otimista, porque horas vizinhas de uma série temporal caem
uma no treino e outra no teste.

| Modelo | Alvo | Algoritmo | Out-of-Time 2022 → 2023 (referência) | Baseline Out-of-Time | Holdout 20 % (otimista) |
|---|---|---|---|---|---|
| Classificação | sinistro com vítima | XGBClassifier (balanceado) | ROC-AUC 0,58 · acurácia balanceada 0,55 | Regressão logística: ROC-AUC 0,58 | ROC-AUC 0,60 · F1 0,66 |
| Regressão | km de lentidão na hora | XGBRegressor | R² 0,50 · RMSE 12,5 km | Ridge: R² 0,42 · RMSE 13,5 km | R² 0,73 · RMSE 9,6 km |

O classificador empata com o baseline logístico (diferença menor que 0,01 de
ROC-AUC): o limite preditivo está nas variáveis disponíveis antes do sinistro,
não no algoritmo. O regressor supera o Ridge com folga. Validação cruzada
5-fold apenas no treino. Detalhes, testes de hipótese e análise crítica estão
no notebook e na aba *Recomendações* do dashboard.
