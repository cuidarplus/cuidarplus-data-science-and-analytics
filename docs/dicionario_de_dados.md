# Dicionário de Dados — Mobilidade Urbana SP (2022–2023)

Este documento descreve as variáveis finais produzidas pelo pipeline
(`src/preprocessing.py` → `src/features.py` → `src/modeling.py`).
O modelo relacional correspondente está em [`arquitetura.dbml`](arquitetura.dbml).

Convenções:

* **Horário**: todas as colunas de tempo estão em horário local de São Paulo
  (`America/Sao_Paulo`, UTC-3, sem horário de verão no período). As observações
  do INMET, publicadas em UTC, foram convertidas.
* **Nulos**: colunas de clima e lentidão podem ser `NaN` quando a fonte não tem
  registro; as flags `*_disponivel` indicam a cobertura. Contagens (`qtd_*`,
  `n_*`) nunca são nulas (ausência no Infosiga significa zero).
* **Período**: 2022-01-01 00:00 a 2023-12-31 23:00 (interseção das três fontes;
  os dados de lentidão da CET só existem para 2022–2023).

---

## 1. `data/processed/mobilidade_sp_horario.parquet`

Grão: **uma linha por hora** (17.520 linhas). Base do modelo de regressão.

### 1.1 Chave

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `datetime` | datetime64 | — | Início da hora (local). Chave primária. |

### 1.2 Clima — INMET, estação A701 (Mirante de Santana)

Uma observação do INMET carimbada em `H` resume o intervalo `(H-1, H]`. Por isso,
a hora `H` da grade recebe o registro `H+1` (ou `H`, se `H+1` faltar, via
`merge_asof` com tolerância de 1 h).

| Coluna | Tipo | Unidade | Descrição | Tratamento |
|---|---|---|---|---|
| `precipitacao_mm` | float | mm | Chuva acumulada na hora | Lacunas ≤ 3 h → 0 mm; maiores → NaN |
| `precipitacao_3h_mm` | float | mm | Soma móvel das últimas 3 horas | Derivada |
| `temperatura_c` | float | °C | Temperatura do ar (bulbo seco) | Interpolação linear ≤ 6 h |
| `ponto_orvalho_c` | float | °C | Temperatura do ponto de orvalho | Interpolação linear ≤ 6 h |
| `umidade_pct` | float | % | Umidade relativa | Interpolação linear ≤ 6 h |
| `pressao_mb` | float | mB (hPa) | Pressão atmosférica na estação | Interpolação linear ≤ 6 h |
| `vento_m_s` | float | m/s | Velocidade média do vento | Interpolação linear ≤ 6 h |
| `rajada_m_s` | float | m/s | Rajada máxima na hora | Interpolação linear ≤ 6 h |
| `direcao_vento_deg` | float | graus | Direção do vento (0–360) | Interpolação linear ≤ 6 h |
| `radiacao_kj_m2` | float | kJ/m² | Radiação global | Valores negativos (ruído) → 0 |
| `clima_disponivel` | bool | — | `True` se precipitação, temperatura e umidade estão presentes | Derivada |
| `clima_imputado` | bool | — | `True` se ao menos uma variável foi imputada | Derivada |
| `defasagem_clima_min` | float | min | 0 (registro exato) ou 60 (fallback para a hora anterior) | Derivada |

### 1.3 Lentidão — CET (trechos em lentidão a cada 30 min)

Para cada instantâneo somamos o comprimento dos trechos (m → km); a hora recebe a
**média** dos instantâneos. Horas sem registro em dias com coleta = 0 km; dias
inteiros sem coleta (58 dias) = NaN.

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `km_lentidao_total` | float | km | Extensão média de lentidão na cidade. **Alvo da regressão.** |
| `km_lentidao_centro` | float | km | Lentidão na região CENTRO da CET |
| `km_lentidao_leste` | float | km | Região LESTE |
| `km_lentidao_norte` | float | km | Região NORTE |
| `km_lentidao_oeste` | float | km | Região OESTE |
| `km_lentidao_sul` | float | km | Região SUL |
| `n_trechos_lentidao` | float | unidades | Nº médio de trechos em lentidão |
| `lentidao_disponivel` | bool | — | `False` em dias sem nenhuma medição |

### 1.4 Sinistros agregados — Infosiga

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `n_sinistros` | int | ocorrências | Sinistros registrados na hora (município de SP) |
| `n_sinistros_com_vitima` | int | ocorrências | Sinistros com ao menos uma vítima |
| `n_vitimas` | int | pessoas | Vítimas fatais + graves + leves |

---

## 2. `data/processed/mobilidade_sp_consolidado.parquet`

Grão: **uma linha por sinistro** (90.682 linhas). Base da classificação e dos KPIs.
Além das colunas de clima e lentidão da seção 1 (copiadas da hora do sinistro),
contém:

### 2.1 Identificação e tempo

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `id_sinistro` | int | — | Identificador do Infosiga (chave primária) |
| `datetime` | datetime64 | — | Data e hora do sinistro. Registros sem hora (0,08 %) foram removidos |
| `datetime_hora` | datetime64 | — | Hora truncada; chave estrangeira para a tabela horária |
| `dia_da_semana` | str | — | Nome do dia (original do Infosiga) |
| `turno` | str | — | MADRUGADA, MANHA, TARDE, NOITE |

### 2.2 Severidade (alvo)

| Coluna | Tipo | Descrição |
|---|---|---|
| `tipo_registro` | str | `NOTIFICACAO` (sem vítima), `SINISTRO NAO FATAL`, `SINISTRO FATAL` |
| `severidade` | str | `SEM_VITIMA`, `COM_VITIMA`, `FATAL` |
| `com_vitima` | int8 | **Alvo da classificação**: 1 se `tipo_registro ≠ NOTIFICACAO` |
| `qtd_gravidade_fatal` / `_grave` / `_leve` / `_ileso` / `_nao_disponivel` | int16 | Pessoas por gravidade (NaN → 0) |
| `qtd_vitimas` | int | fatal + grave + leve |

### 2.3 Via e localização

| Coluna | Tipo | Descrição |
|---|---|---|
| `logradouro` | str | Nome da via (original) |
| `tipo_via` | str | `VIAS URBANAS`, `ESTRADAS E RODOVIAS`, `NAO DISPONIVEL` |
| `categoria_via` | str | Derivada do prefixo do logradouro: `AVENIDA`, `RUA`, `RODOVIA`, `ESTRADA`, `MARGINAL`, `ACESSO`, `PONTE_VIADUTO_TUNEL`, `PRACA_LARGO`, `OUTROS`, `NAO DISPONIVEL` |
| `zona` | str | `CENTRO`, `NORTE`, `SUL`, `LESTE`, `OESTE` ou `NAO DISPONIVEL`; regra geométrica em torno da Praça da Sé (quadrado de ~4,4 km = centro) |
| `administracao` | str | Órgão responsável pela via (`PREFEITURA`, `DER`, `CONCESSIONÁRIA`, ...) |
| `latitude`, `longitude` | float | Graus decimais (vírgula convertida em ponto); 0,3 % nulos |

### 2.4 Tipo de sinistro e veículos

| Coluna | Tipo | Descrição |
|---|---|---|
| `tp_sinistro_primario` | str | `COLISAO`, `ATROPELAMENTO`, `CHOQUE`, `OUTROS`, `NAO DISPONIVEL` |
| `qtd_pedestre`, `qtd_bicicleta`, `qtd_motocicleta`, `qtd_automovel`, `qtd_onibus`, `qtd_caminhao`, `qtd_veic_outros`, `qtd_veic_nao_disponivel` | int16 | Envolvidos por modal (NaN → 0) |
| `qtd_veiculos` | int | Soma das colunas acima |

### 2.5 Lentidão associada ao sinistro

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `km_lentidao_total` | float | km | Lentidão na hora do sinistro |
| `km_lentidao_h1` | float | km | Lentidão na hora seguinte |

---

## 3. Atributos derivados em tempo de análise (`src/features.py`)

Não são gravados no parquet; são recalculados por `preparar_features()`.

| Coluna | Tipo | Descrição |
|---|---|---|
| `hora` | int8 | Hora do dia (0–23) |
| `dia_semana_num` | int8 | 0 = segunda … 6 = domingo |
| `dia_semana` | str | Nome em português |
| `mes`, `ano` | int | Mês e ano |
| `feriado` | int8 | 1 em feriados nacionais/estaduais/municipais e pontos facultativos (lista em `FERIADOS_SP`) |
| `fim_de_semana` | int8 | 1 aos sábados e domingos |
| `hora_pico` | int8 | 1 em dias úteis das 07–09 h e 17–19 h |
| `periodo_dia` | str | MADRUGADA (0–5), MANHA (6–11), TARDE (12–17), NOITE (18–23) |
| `faixa_chuva` | categoria | `SEM_CHUVA` (0), `FRACA` (≤ 2,5 mm/h), `MODERADA` (≤ 10), `FORTE` (> 10) |
| `chovendo` | Int8 | 1 se `precipitacao_mm > 0` |
| `chuva_3h` | Int8 | 1 se choveu nas últimas 3 h |

---

## 4. Indicadores (KPIs)

| Indicador | Fórmula | Arquivo |
|---|---|---|
| **ISR** — Índice de Severidade Relativa | `sinistros com vítima / total`, por `categoria_via × faixa_chuva` (≥ 30 sinistros). `isr_relativo = isr / isr_geral` | `kpi_isr.csv` |
| **FRT** — Fator de Retenção de Tráfego | Média de `(km_lentidao_total + km_lentidao_h1) / 2` por categoria de sinistro; `excedente_km` compara com a mediana do mesmo dia da semana e hora | `kpi_frt_tipo_sinistro.csv`, `kpi_frt_severidade.csv` |
| **TSC** — Taxa de Sinistros por Condição climática | `sinistros / horas observadas` em cada `faixa_chuva` | `kpi_taxa_sinistros_clima.csv` |

---

## 5. Modelos (`modelos_metadata.json`)

| Modelo | Arquivo | Alvo | Atributos |
|---|---|---|---|
| Classificação de severidade (XGBClassifier, classes balanceadas) | `modelo_classificacao_severidade.pkl` | `com_vitima` | temporais + climáticas + `tipo_via`, `categoria_via`, `zona` |
| Regressão de lentidão (XGBRegressor) | `modelo_regressao_lentidao.pkl` | `km_lentidao_total` | temporais + climáticas |

Os `.pkl` contêm o `Pipeline` completo (one-hot + modelo) e recebem um
`DataFrame` com as colunas listadas em `modelos_metadata.json → features`.
