## Descripción

<!-- Qué cambia y por qué. Si resuelve una tarea del roadmap, referénciala: DT-N -->

Resuelve: DT-

## Tipo de cambio

- [ ] `fix` — corrección de un defecto
- [ ] `feat` — funcionalidad nueva
- [ ] `docs` — solo documentación
- [ ] `refactor` — reestructuración sin cambio de comportamiento
- [ ] `test` — añade o corrige tests
- [ ] `chore` — mantenimiento, dependencias, tooling

## Cómo verificarlo

<!-- Comandos concretos para reproducir el resultado -->

```bash

```

## Impacto metodológico

<!-- Obligatorio si el PR toca datos, entrenamiento o evaluación -->

- [ ] Este PR **no** afecta datos, entrenamiento ni evaluación.
- [ ] Este PR sí los afecta. Métricas antes y después, **con su baseline**:

| Métrica | Baseline | Antes | Después |
|---------|----------|-------|---------|
|         |          |       |         |

## Checklist

- [ ] `make lint` en verde
- [ ] `make test` en verde
- [ ] Documentación afectada actualizada en este mismo PR
- [ ] Si cambié el conjunto u orden de features, lo hice en `src/cardio_features.py`, reentrené y los tests de equivalencia pasan (ver `CONTRIBUTING.md`)
- [ ] No añadí artefactos binarios, datos generados ni credenciales
- [ ] Los avisos médicos siguen presentes en servicio, dashboard y docs
