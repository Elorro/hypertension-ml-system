.PHONY: help install install-dev data train setup api dashboard audit test lint format clean

PYTHON := python
HOST   := 127.0.0.1
PORT   := 8000

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependencias de ejecución
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Instala dependencias de ejecución y desarrollo
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

data:  ## Genera el dataset sintético (50k filas, seed=42)
	$(PYTHON) src/generate_dataset.py

train:  ## Entrena los 5 modelos y selecciona el mejor por macro-F1
	$(PYTHON) src/train_classical_models.py

setup: data train  ## Prepara todo lo necesario para levantar el servicio

api:  ## Levanta el servicio de inferencia (requiere `make setup`)
	uvicorn api.main:app --reload --host $(HOST) --port $(PORT)

dashboard:  ## Levanta el dashboard Streamlit (requiere `make api` corriendo)
	streamlit run app/dashboard.py

audit:  ## Ejecuta la auditoría de target leakage
	$(PYTHON) scripts/verify_leakage.py

test:  ## Ejecuta la suite de tests
	pytest -v --cov=src --cov=api --cov-report=term-missing

lint:  ## Verifica estilo y tipos
	ruff check .
	ruff format --check .

format:  ## Aplica formateo automático
	ruff format .
	ruff check --fix .

clean:  ## Elimina artefactos generados y caches
	rm -rf models/*.pkl models/mejor_modelo.txt
	rm -rf data/raw data/processed reports
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
