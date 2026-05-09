# Tablas de Verdad - Lógica de Predicados

## Tabla 1: Operadores Lógicos Básicos

| P | Q | P ∧ Q | P ∨ Q | P → Q | P ↔ Q | ¬P |
|---|---|-------|-------|-------|-------|-----|
| V | V | V | V | V | V | F |
| V | F | F | V | F | F | F |
| F | V | F | V | V | F | V |
| F | F | F | F | V | V | V |

## Tabla 2: Leyes de De Morgan

| Expresión | Equivalente |
|-----------|-------------|
| ¬(P ∧ Q) | ¬P ∨ ¬Q |
| ¬(P ∨ Q) | ¬P ∧ ¬Q |

## Tabla 3: Cláusulas de Horn

| Tipo | Forma | ¿Satisfacible? |
|------|-------|----------------|
| Cláusula definida | H ← B1, B2, ..., Bn | Sí |
| Cláusula de objetivo | ⊥ ← B1, B2, ..., Bn | Puede ser insatisfactible |
| Cláusula unitaria | H ← | Siempre satisfactible |

## Tabla 4: Unificación

| Expresión 1 | Expresión 2 | Unificador Más General (MGU) |
|------------|------------|-------------------------------|
| P(x, y) | P(a, b) | {x/a, y/b} |
| P(x, x) | P(a, b) | No unificable |
| P(x, y) | P(y, x) | {x/y, y/x} (cíclico) |

## Tabla 5: Algoritmo de Unificación

| Paso | Acción | Ejemplo |
|------|--------|---------|
| 1 | Iniciar con conjunto vacío | {} |
| 2 | Unificar términos | x/a → {x/a} |
| 3 | Verificar consistencia | ¿x aparece en a? |
| 4 | Componer sustituciones | {x/a} ○ {y/b} = {x/a, y/b} |

## Tabla 6: Modus Ponens

| Premisa 1 | Premisa 2 | Conclusión |
|-----------|-----------|------------|
| P → Q | P | Q |
| padre(X) → progenitor(X, Y) | padre(juan) | progenitor(juan, Y) |
