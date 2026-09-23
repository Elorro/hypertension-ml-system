# Guía de contribución

Gracias por el interés en el proyecto. Este documento describe cómo levantar el
entorno, qué estándares sigue el código y cómo proponer cambios.

---

## Antes de empezar

Lee **[docs/LEAKAGE_ANALYSIS.md](docs/LEAKAGE_ANALYSIS.md)** y
**[docs/DT1_RESULTS.md](docs/DT1_RESULTS.md)**. El proyecto nació con un defecto
metodológico en su núcleo (target leakage), que DT-1 corrigió migrando a datos reales;
el pipeline sintético se conserva como evidencia. Las métricas actuales siguen
sesgadas al alza mientras DT-4 esté abierto. Contribuir sin ese contexto lleva a
optimizar métricas que no significan nada.

Las tareas abiertas están priorizadas en **[docs/ROADMAP.md](docs/ROADMAP.md)** con
identificadores `DT-N`. Referéncialos en tus commits y PRs.

## Entorno de desarrollo

```bash
git clone https://github.com/Elorro/hypertension-ml-system.git
cd hypertension-ml-system

python -m venv .venv
source .venv/bin/activate

make install         # entorno exacto desde requirements.lock.txt
make verify-env      # compuerta: versiones + sha256 + carga de los .pkl de DT-1
make audit           # verifica que la auditoría de leakage corre
```

## Estándares de código

**Python 3.14, estilo idiomático.**

- **Type hints obligatorios** en toda función pública. El código existente en la
  raíz no los tiene de forma consistente; el código nuevo sí debe tenerlos.
- **Formateo con `ruff`.** Ejecuta `make format` antes de commitear.
- **Docstrings** en módulos y funciones no triviales. En español, igual que el resto
  del proyecto.
- **Sin `except:` desnudo.** Captura excepciones específicas. Ver DT-8.
- **Rutas con `pathlib.Path`**, no concatenación de strings ni `os.path.join`.

Comprobación antes de abrir un PR:

```bash
make lint
make test
```

## Convenciones del dominio

**Nombres de columnas.** Los modelos servidos usan los nombres del dataset real
(`age_years`, `gender`, `height`…). El pipeline sintético de `src/` usa `Estres`
**sin tilde** y `Diagnostico` como target; el heredado de la raíz usa `Estrés` y
`HTA_Nivel`. No los mezcles: son esquemas incompatibles y unificarlos es la tarea DT-5.

**Orden de features.** El de los modelos servidos vive en un solo sitio,
`src/cardio_features.py`, que importan el entrenamiento y la API; el dashboard lo
toma de `/v1/modelos`. Cambiarlo invalida los `.pkl` de DT-1 y exige reentrenar. Los
tests lo vigilan: `tests/test_equivalencia_features.py` (matrices idénticas a la
corrida de referencia) y `tests/api/test_equivalencia.py` (la API construye la misma
matriz, bit a bit).

**Reproducibilidad.** Toda operación aleatoria lleva semilla explícita
(`random_state=42` / `seed=42`). Un cambio que rompa la reproducibilidad debe
justificarse en el PR.

## Contribuciones a los modelos

Al añadir un algoritmo nuevo:

1. Entrénalo en `src/train_classical_models.py` con la misma interfaz que los demás.
2. Regístralo en el diccionario `resultados` para que compita por macro-F1.
3. Documéntalo en la tabla de modelos del README.

En el pipeline de DT-1 (`src/train_cardio_real.py`), el ganador se registra en el
manifiesto y la API lo carga desde ahí; si fuera XGBoost, el perfil `servicio` de
`src/artefactos.py` exigiría además `xgboost` exacto en `requirements-serve.txt`.

Al reportar métricas, incluye siempre la comparación contra el baseline pertinente
(regla clínica en el dataset sintético, clase mayoritaria en datos reales). Una
métrica sin baseline no se acepta.

## Commits

Formato [Conventional Commits](https://www.conventionalcommits.org/):

```
<tipo>(<alcance>): <descripción en imperativo>
```

Tipos: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.

Ejemplos:

```
feat(api): añadir endpoint batch para inferencia masiva (DT-9)
fix(train): ajustar scaler solo sobre el conjunto de entrenamiento (DT-2)
docs(model-card): documentar ausencia de análisis de equidad
test(api): cubrir validación de rangos fisiológicos
```

## Pull requests

1. Rama desde `main` con nombre descriptivo: `fix/scaler-leakage`, `feat/batch-endpoint`.
2. Un PR resuelve una cosa. Si tocas cinco `DT-N`, abre cinco PRs.
3. `make lint` y `make test` en verde.
4. Actualiza la documentación afectada en el mismo PR.
5. Describe **qué** cambia, **por qué**, y **cómo verificarlo**.

Si el cambio afecta métricas o metodología, incluye los números antes y después.

## Qué no se acepta

- Cambios que presenten el modelo como clínicamente válido, o que retiren los avisos
  médicos de la interfaz, el servicio o la documentación.
- Métricas reportadas sin su baseline.
- Artefactos binarios (`.pkl`, `.h5`, CSV generados) ni el dataset de Kaggle en el control de versiones.
- Credenciales, tokens o rutas absolutas de una máquina concreta.

## Reportar problemas

Al abrir un issue incluye: versión de Python, sistema operativo, comando ejecutado,
traza completa del error y comportamiento esperado. Si es un problema metodológico,
adjunta la evidencia numérica reproducible.
