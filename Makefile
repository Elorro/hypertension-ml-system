.PHONY: help venv install install-dev data train train-real setup api dashboard audit audit-real verify-env test lint format clean

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
	.venv/bin/python scripts/verify_env.py

install:  ## Instala el entorno exacto (lock) en el venv activo
	$(PYTHON) -m pip install -r requirements.lock.txt

install-dev:  ## Instala desde los rangos declarados (para regenerar el lock)
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

data:  ## Genera el dataset sintético (50k filas, seed=42)
	$(PYTHON) src/generate_dataset.py

train:  ## Entrena los 5 modelos y selecciona el mejor por macro-F1
	$(PYTHON) src/train_classical_models.py

train-real:  ## DT-1: entrena sobre el dataset real de Kaggle (~51 min)
	$(PYTHON) src/train_cardio_real.py

setup: data train  ## Prepara todo lo necesario para levantar el servicio

api:  ## Levanta el servicio de inferencia (requiere `make setup`)
	uvicorn api.main:app --reload --host $(HOST) --port $(PORT)

dashboard:  ## Levanta el dashboard Streamlit (requiere `make api` corriendo)
	streamlit run app/dashboard.py

audit:  ## Auditoría de target leakage del dataset sintético
	$(PYTHON) scripts/verify_leakage.py

audit-real:  ## Criterio de aceptación de DT-1 sobre el dataset real
	$(PYTHON) scripts/audit_cardio_leakage.py

verify-env:  ## Compuerta del entorno: versiones + sha256 + carga y predicción de los .pkl de DT-1
	$(PYTHON) scripts/verify_env.py

test:  ## Ejecuta la suite de tests
	pytest -v --cov=src --cov=api --cov-report=term-missing

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
