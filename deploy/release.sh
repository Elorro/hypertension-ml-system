#!/usr/bin/env bash
# Publica los .pkl servidos de DT-4 como assets de un GitHub Release.
# El tag apunta al commit del manifiesto (41572ed). Ejecutar desde la raíz, con los
# dt4_*.pkl presentes y verificados (make verify-env).
set -euo pipefail
make verify-env
gh release create models-dt4-41572ed \
  models/dt4_riesgo_cv_con_pa__modelo.pkl models/dt4_riesgo_cv_con_pa__scaler.pkl \
  models/dt4_hta_b1__modelo.pkl models/dt4_hta_b1__scaler.pkl \
  --target "$(git rev-parse 41572ed)" \
  --title "Modelos servidos DT-4 (manifiesto 41572ed)" \
  --notes "Assets: modelo y scaler de A′ (riesgo_cv_con_pa) y B1 (hta_b1). sha256 de cada uno en models/dt4_manifest.json (commit 41572ed). Descarga y verificación: MODELS_BASE_URL=https://github.com/Elorro/hypertension-ml-system/releases/download/models-dt4-41572ed make fetch-models"
