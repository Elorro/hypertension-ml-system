.PHONY: help venv install install-dev data train train-real setup api dashboard serve-api serve-dashboard audit audit-real verify-env test lint format clean

PYTHON   := python
# Intérprete base de la corrida de referencia de DT-1 (Python 3.14.6).
PYBASE   := /usr/bin/python3.14
HOST     := 127.0.0.1
PORT     := 8000

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Crea .venv desde cero con el Python de referencia e instala el lock
	rm -rf .venv
	$(PYBASE) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip setuptools wheel
	.venv/bin/python -m pip install -r requirements.lock.txt
	.venv/bin/python -m scripts.verify_env

install:  ## Instala el entorno exacto (lock) en el venv activo
	$(PYTHON) -m pip install -r requirements.lock.txt

install-dev:  ## Instala desde los rangos declarados (para regenerar el lock)
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

data:  ## Genera el dataset sintético (50k filas, seed=42)
	$(PYTHON) src/generate_dataset.py

train:  ## Entrena los 5 modelos y selecciona el mejor por macro-F1
	$(PYTHON) src/train_classical_models.py

train-real:  ## DT-1: entrena sobre el dataset real de Kaggle (~51 min)
	$(PYTHON) -m src.train_cardio_real

setup: data train  ## Pipeline sintético (evidencia del leakage; ya no alimenta al servicio)

serve-api:  ## Levanta la API con los modelos de DT-1 (requiere models/dt1_*.pkl)
	uvicorn api.main:app --host $(HOST) --port $(PORT)

serve-dashboard:  ## Levanta el dashboard (cliente de la API; API_URL, por defecto :8000)
	API_URL=$${API_URL:-http://$(HOST):$(PORT)} streamlit run app/dashboard.py

api: serve-api  ## Alias de serve-api

dashboard: serve-dashboard  ## Alias de serve-dashboard

audit:  ## Auditoría de target leakage del dataset sintético
	$(PYTHON) scripts/verify_leakage.py

audit-real:  ## Criterio de aceptación de DT-1 sobre el dataset real
	$(PYTHON) scripts/audit_cardio_leakage.py

verify-env:  ## Compuerta del entorno: versiones + sha256 + carga y predicción de los .pkl de DT-1
	$(PYTHON) -m scripts.verify_env

test:  ## Ejecuta la suite; los tests que necesitan .pkl o el CSV real se saltan con motivo visible
	$(PYTHON) -m pytest --cov=src --cov=api --cov-report=term-missing

lint:  ## Verifica estilo y tipos
	ruff check .
	ruff format --check .

format:  ## Aplica formateo automático
	ruff format .
	ruff check --fix .

clean:  ## Elimina artefactos generados y caches (conserva models/dt1_manifest.json)
	rm -rf models/*.pkl models/mejor_modelo.txt
	rm -rf data/raw data/processed reports
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
