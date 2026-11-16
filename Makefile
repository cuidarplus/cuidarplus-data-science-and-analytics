VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Ordem de execução do pipeline (Etapas 1 -> 2 -> 3)
PIPELINE_MODULES := src.preprocessing src.features src.modeling

.PHONY: help setup pipeline notebook run-lab run-app clean

## Alvo padrão: lista os comandos disponíveis
help:
	@echo "Comandos disponíveis:"
	@echo "  make setup      - cria o venv e instala as dependências"
	@echo "  make pipeline   - executa src.preprocessing -> src.features -> src.modeling"
	@echo "  make notebook   - executa o notebook de ponta a ponta"
	@echo "  make run-lab    - abre o JupyterLab"
	@echo "  make run-app    - abre o dashboard Streamlit"
	@echo "  make clean      - remove bases processadas, modelos e metadata"

## Cria o ambiente virtual e instala as dependências (macOS: requer `brew install libomp` para o XGBoost)
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## Executa pré-processamento, indicadores e modelagem
pipeline:
	@for module in $(PIPELINE_MODULES); do \
		echo "==> Executando $$module"; \
		$(PYTHON) -m $$module || exit 1; \
	done

## Executa o notebook de ponta a ponta (gera as saídas embutidas)
notebook:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=600 notebooks/01_relatorio_analise_mobilidade.ipynb

run-lab:
	$(PYTHON) -m jupyter lab

run-app:
	$(PYTHON) -m streamlit run app/main.py

## Remove artefatos gerados (bases processadas, modelos e metadata)
clean:
	rm -f data/processed/*.parquet data/processed/*.csv *.pkl modelos_metadata.json
