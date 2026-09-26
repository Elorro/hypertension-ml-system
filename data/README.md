# data/

Los datos **no se versionan**. Este directorio solo contiene este README; lo demás se
genera o se descarga localmente. Diccionario de variables y limitaciones:
[docs/DATA.md](../docs/DATA.md).

| Ruta | Qué es | Cómo se obtiene |
|------|--------|-----------------|
| `data/real/cardio/cardio_train.csv` | Dataset real, entrena los modelos servidos | Se descarga de la fuente (abajo) |
| `data/raw/dataset_hipertension_sintetico.csv` | Dataset sintético, evidencia del target leakage | `make data` (`seed=42`) |

## Dataset real

- **Fuente:** *Cardiovascular Disease dataset*, publicado en Kaggle por Svetlana
  Ulianova (usuario `sulianova`) — **verificar** el nombre de la autora en la página.
  <https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>
- **Versión usada:** el ZIP descargado contiene un único archivo, `cardio_train.csv`,
  con fecha 2019-10-17 en el ZIP (**verificar** que coincide con la versión publicada).
- **Formato:** separador `;`, 70.000 filas y 13 columnas (`id;age;gender;height;weight;
  ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio`); `age` en días.
- **Licencia:** la fuente no declara una licencia (**verificar** en la página de
  Kaggle). Por eso **no se redistribuye**: el CSV estuvo versionado, al detectarse que su
  licencia es desconocida se purgó de toda la historia con `git filter-repo`, y
  `.gitignore` ignora `data/real/` completo desde `4b3b5d4`. Hay que descargarlo de la
  fuente.

### Descarga

```bash
# Requiere credenciales de Kaggle en ~/.kaggle/kaggle.json
kaggle datasets download -d sulianova/cardiovascular-disease-dataset -p data/real/
unzip data/real/cardiovascular-disease-dataset.zip -d data/real/cardio/
```

Ruta esperada: `data/real/cardio/cardio_train.csv`.

### Verificación

sha256 esperado, el registrado en `models/dt1_manifest.json` y en
`models/dt4_manifest.json` (`dataset.sha256`):

```
21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c
```

```bash
sha256sum data/real/cardio/cardio_train.csv
python -m pytest tests/test_equivalencia_features.py::test_csv_presente_es_el_del_manifiesto
```

Si el sha256 no coincide, las cifras de los manifiestos no son reproducibles con ese
archivo, y `make train-dt4` aborta antes de entrenar.

### Qué lo usa

| Comando | Para qué |
|---------|----------|
| `make train-dt4` | Entrena y selecciona los modelos servidos (DT-4) |
| `make train-real` | Registro de DT-1 |
| `make audit-real` | Criterio de aceptación de DT-1 (auditoría de leakage con control positivo) |
| `make test` | Tests `requires_data` (equivalencia de features, partición, folds); se saltan si falta |
| `notebooks/EDA_cardiovascular_real.ipynb` | Análisis exploratorio |

La API y el dashboard **no** lo necesitan: la API sirve los `.pkl` ya entrenados. Está
previsto distribuirlos como assets de un GitHub Release, descargables con
`make fetch-models` (`MODELS_BASE_URL`); ese release todavía no está publicado.
