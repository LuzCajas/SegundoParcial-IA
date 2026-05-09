# Guía Didáctica de Inteligencia Artificial - Semanas 7-11

## Semana 7: Búsqueda Informada

### Algoritmo A*

A* es un algoritmo de búsqueda informada que combina el costo real del camino con una heurística admisible.

**Función de evaluación:**
```
f(n) = g(n) + h(n)
```
Donde:
- `g(n)`: costo real del camino desde el inicio hasta el nodo n
- `h(n)`: heurística que estima el costo desde n hasta el objetivo

**Propiedades:**
- **Admisible**: h(n) nunca sobreestima el costo real al objetivo
- **Optimo**: Encuentra el camino de costo mínimo si h(n) es admisible

### Comparación de Algoritmos

| Algoritmo | Complejidad Temporal | Compl. Espacial | Optimalidad |
|-----------|---------------------|-----------------|-------------|
| BFS | O(b^d) | O(b^d) | Siempre óptimo |
| DFS | O(b^m) | O(b*m) | No óptimo |
| UCS | O(b^d) | O(b^d) | Siempre óptimo |
| A* | O(b^d) | O(b^d) | Si h es admisible |

### Heurísticas Comunes

1. **Distancia euclidiana**: Para espacios continuos
   ```
   h(n) = √((x₁-x₂)² + (y₁-y₂)²)
   ```

2. **Distancia de Manhattan**: Para grafos en cuadrícula
   ```
   h(n) = |x₁-x₂| + |y₁-y₂|
   ```

3. **Heurística nula**: h(n) = 0 (convierte A* en UCS)

---

## Semana 8: Lógica de Predicados

### Sintaxis

**Términos:**
- Constantes: `juan`, `maria`, `5`
- Variables: `X`, `Y`, `_Variable`
- Funciones: ` padre(juan)`, ` hermano(X)`

**Fórmulas atómicas:**
- `progenitor(juan, pedro)`
- `¬ madre(X)`
- `X = Y`

### Cláusulas de Horn

Una cláusula de Horn es una cláusula con a lo sumo un literal positivo.

**Forma normal:**
```
H ← B₁, B₂, ..., Bₙ
```
Se lee: "H es verdadero si todos los Bi son verdaderos"

### Algoritmo de Unificación

```
UNIFICAR(t1, t2, θ):
    PARA cada par (s1, s2) en zip(APLICAR(θ, t1), APLICAR(θ, t2)):
        SI s1 es variable:
            SI s2 contiene s1: FALLA
            θ = θ ∪ {s1/s2}
        SI s2 es variable:
            θ = θ ∪ {s2/s1}
        SI s1 y s2 son constantes diferentes:
            FALLA
        SI ambos son funciones:
            SI aridad diferente: FALLA
            UNIFICAR(args(s1), args(s2), θ)
    RETORNAR θ
```

### Inferencia en Prolog

1. **Modus Ponens**: Si P→Q y P, entonces Q
2. **Encadenamiento hacia adelante**: De hechos a conclusiones
3. **Encadenamiento hacia atrás**: De goals a subgoals (usado por Prolog)

---

## Semana 9: Aprendizaje Automático

### Tipos de Aprendizaje

| Tipo | Características | Ejemplos |
|------|-----------------|----------|
| Supervisado | Datos etiquetados, función objetivo conocida | Clasificación, Regresión |
| No supervisado | Datos sin etiquetas, descubre patrones | Clustering, Reducción dimensional |
| Por refuerzo | Aprender de recompensas/penalizaciones | Juegos, Robótica |

### Algoritmos de Clasificación

**Naive Bayes:**
```
P(clase | caracteristicas) ∝ P(caracteristicas | clase) × P(clase)
```

**Árboles de Decisión:**
- Ganancia de información: IG(S, A) = H(S) - Σ(|Sᵥ|/|S|) × H(Sᵥ)
- Entropía: H(S) = -Σ pᵢ log₂(pᵢ)

### Métricas de Evaluación

| Métrica | Fórmula | Uso |
|---------|---------|-----|
| Accuracy | (TP + TN) / Total | Balanceado |
| Precision | TP / (TP + FP) | Relevancia |
| Recall | TP / (TP + FN) | Completitud |
| F1-Score | 2 × (P × R) / (P + R) | Balanceado |

### Matriz de Confusión

```
                    Predicción
                  Positiva | Negativa
Real  Positiva    [  TP    |   FN   ]
      Negativa    [  FP    |   TN   ]
```

---

## Semana 10: Redes Neuronales

### Perceptrón

**Estructura:**
```
entrada₁ → w₁ → Σ → activación → salida
entrada₂ → w₂ →       ↑
                   sesgo (b)
```

**Función de activación:**
- Step: f(x) = 1 si x ≥ 0, sino 0
- Sigmoid: f(x) = 1 / (1 + e^(-x))
- ReLU: f(x) = max(0, x)
- Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

### Entrenamiento (Regla Delta)

```
wᵢ ← wᵢ + η × (y_deseada - y_actual) × xᵢ
```

Donde η es la tasa de aprendizaje (0 < η ≤ 1).

### Limitaciones del Perceptrón

- **XOR Problem**: El perceptrón simple no puede resolver XOR
- Solución: Redes multicapa con propagación hacia atrás

### Backpropagation

**Propagación hacia adelante:**
1. Inicializar pesos aleatoriamente
2. Para cada capa: y = σ(W × x + b)
3. Calcular error en capa de salida

**Propagación hacia atrás:**
```
δₙ = (y_deseada - y_actual) × σ'(net)
δₗ = Wₗ₊₁ᵀ × δₗ₊₁ ⊙ σ'(netₗ)
Δwᵢⱼ = η × δⱼ × xᵢ
```

### Arquitecturas Comunes

| Arquitectura | Uso | Capas |
|--------------|-----|-------|
| MLP | Clasificación general | Input → Hidden(s) → Output |
| CNN | Visión por computadora | Conv → Pool → FC |
| RNN | Secuencias, NLP | Recurrentes |
| Transformer | NLP avanzado | Attention → FFN |

---

## Semana 11: Deep Learning y Tendencias

### Regularización

- **Dropout**: Desactivar neuronas aleatoriamente durante entrenamiento
- **L1/L2**: Penalizar pesos grandes
- **Early Stopping**: Detener cuando validación empeora

### Optimizadores

| Optimizador | Actualización | Ventaja |
|-------------|---------------|---------|
| SGD | w ← w - η∇L | Simple |
| Momentum | w ← w - η∇L + γΔw | Escapa mínimos locales |
| Adam | Adaptativo por parámetro | Rápido convergencia |

### Frameworks Populares

```
TensorFlow / Keras:
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
```

```
PyTorch:
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            return torch.relu(self.fc2(torch.relu(self.fc1(x))))
```

---

## Apéndice: Glosario de Términos

| Término | Definición |
|---------|-----------|
| Agente | Entidad que percibe y actúa en un entorno |
| Estado | Configuración del entorno |
| Acción | Operación que puede realizar el agente |
| Función de evaluación | Métrica heurística en A* |
| Unificación | Proceso de encontrar sustituciones que hacen términos iguales |
| Overfitting | Cuando el modelo memoriza datos de entrenamiento |
| Epoch | Una pasada completa sobre el dataset |
| Batch | Subconjunto de ejemplos procesados juntos |
