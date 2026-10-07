# 🌳 Árbol de Decisión: Empresa de Telecomunicaciones

Trabajo práctico de **Procesamiento de Aprendizaje Automático**: construcción e interpretación de un árbol de decisión para predecir si un cliente de telecomunicaciones aceptará una oferta de plan de datos móviles.

| | |
|---|---|
| **Autor** | Facundo Lugo |
| **Carrera** | Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial |
| **Materia** | Procesamiento de Aprendizaje Automático |
| **Profesora** | Yanina Scudero |
| **Fecha** | 23 de septiembre de 2026 |

---

## 📌 Descripción

El trabajo calcula **a mano** la entropía del conjunto de datos original y la **ganancia de información (Information Gain, IG)** de tres atributos candidatos. El atributo con mayor IG se elige como **nodo raíz** del árbol.

**Pregunta que responde:** dado un cliente nuevo, ¿aceptará la oferta de plan de datos móviles?

## 📊 Dataset

10 registros de clientes, con una variable objetivo binaria.

| Variable | Tipo | Descripción |
|---|---|---|
| `Edad` | Numérica (años) | Edad del cliente |
| `Uso de Datos` | Numérica (GB) | Consumo mensual de datos |
| `Tiene Línea Fija` | Categórica (Sí/No) | Si el cliente tiene línea fija contratada |
| `Aceptó Oferta` | Categórica (Sí/No) | **Variable objetivo** |

Distribución de clases: **5 Sí / 5 No** (balanceado).

### Discretización de atributos numéricos

| Atributo | Categorías |
|---|---|
| Uso de Datos | Bajo (≤ 3 GB) · Medio (3.1–6 GB) · Alto (> 6 GB) |
| Edad | Joven (≤ 30) · Adulto (31–50) · Mayor (> 50) |

## 🧮 Metodología

**Entropía de Shannon**

```
H(S) = − Σ pᵢ · log₂(pᵢ)
```

**Ganancia de información**

```
IG(S, A) = H(S) − H_pond(S | A)

H_pond(S | A) = Σ (|Sᵥ| / |S|) · H(Sᵥ)
```

**Criterio de selección del nodo raíz:** maximizar IG (equivalente a minimizar la entropía ponderada).

## 📈 Resultados

Entropía del conjunto original: **H(S) = 1.0 bit** (incertidumbre máxima, 50% / 50%).

| Atributo | Entropía ponderada | Ganancia de información | Resultado |
|---|:---:|:---:|---|
| **Tiene línea fija** | 0.0000 | **1.0000** | ✅ Óptimo (raíz) |
| Uso de datos | 0.4000 | 0.6000 | Secundario |
| Edad | 0.5510 | 0.4490 | Tercero |

## 🌲 Árbol resultante

```
        [ ¿Tiene línea fija? ]
              /        \
            No          Sí
            /            \
   [ Aceptó: NO ]    [ Aceptó: SÍ ]
      (5 casos)         (5 casos)
```

Ambas hojas son **puras** (H = 0), por lo que el algoritmo se detiene en el primer nivel.

## 📏 Reglas de predicción

1. `SI Tiene línea fija = "No"` ⟹ `Aceptó Oferta = "No"`
2. `SI Tiene línea fija = "Sí"` ⟹ `Aceptó Oferta = "Sí"`

### Ejemplos de inferencia

| Cliente | Línea fija | Predicción |
|---|:---:|---|
| 40 años, 5 GB | Sí | ✅ Acepta la oferta |
| 25 años, 2 GB | No | ❌ No acepta la oferta |

## ⚠️ Limitaciones

- **Dataset muy pequeño (N = 10):** un solo atributo separa perfectamente las clases, algo poco frecuente en datos reales.
- **Sin validación:** no hay conjunto de prueba ni validación cruzada, por lo que no se puede estimar la capacidad de generalización. El modelo podría estar sobreajustado.
- **Umbrales de discretización:** los cortes de Edad y Uso de Datos son definidos manualmente; otros cortes podrían cambiar el IG de esos atributos.

## 💻 Reproducir los cálculos (opcional)

```python
from math import log2
from collections import Counter

# (edad, uso_gb, linea_fija, acepto)
datos = [
    (24, 2.5, "No", "No"), (38, 6.0, "Sí", "Sí"), (29, 3.0, "No", "No"),
    (45, 8.0, "Sí", "Sí"), (52, 7.5, "Sí", "Sí"), (33, 4.0, "No", "No"),
    (41, 5.5, "Sí", "Sí"), (27, 2.0, "No", "No"), (36, 6.5, "Sí", "Sí"),
    (31, 3.5, "No", "No"),
]

def entropia(clases):
    n = len(clases)
    return -sum((c / n) * log2(c / n) for c in Counter(clases).values())

def ganancia(grupos, clases):
    n = len(clases)
    h_pond = sum(len(g) / n * entropia(g) for g in grupos.values())
    return entropia(clases) - h_pond

def agrupar(f):
    grupos = {}
    for fila in datos:
        grupos.setdefault(f(fila), []).append(fila[3])
    return grupos

clases = [d[3] for d in datos]

uso = lambda d: "Bajo" if d[1] <= 3 else ("Medio" if d[1] <= 6 else "Alto")
edad = lambda d: "Joven" if d[0] <= 30 else ("Adulto" if d[0] <= 50 else "Mayor")

print("H(S)            =", round(entropia(clases), 4))
print("IG Línea fija   =", round(ganancia(agrupar(lambda d: d[2]), clases), 4))
print("IG Uso de datos =", round(ganancia(agrupar(uso), clases), 4))
print("IG Edad         =", round(ganancia(agrupar(edad), clases), 4))
```

Salida esperada: `1.0`, `1.0`, `0.6`, `0.449`.

## 📁 Contenido del repositorio

```
.
├── Trabajo_Práctico_Árbol_de_Decisión.pdf   # Informe completo
└── README.md
```

## 📚 Conceptos clave

`Árbol de decisión` · `Entropía de Shannon` · `Ganancia de información` · `Clasificación binaria` · `Nodo raíz` · `Nodos puros`
