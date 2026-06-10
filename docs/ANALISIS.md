# Análisis de `transfermarkt-api` y metodología de scoring

## 1. Qué es y cómo funciona

[felipeall/transfermarkt-api](https://github.com/felipeall/transfermarkt-api) es una API
REST construida con **FastAPI** que **no usa ninguna API oficial**: scrapea las páginas
HTML de transfermarkt.com en el momento de cada request.

Flujo interno de cada endpoint:

1. `app/services/base.py` (`TransfermarktBase`) hace un `GET` a la URL de Transfermarkt
   con un User-Agent de navegador, parsea el HTML con BeautifulSoup y lo convierte a un
   árbol `lxml`.
2. Cada servicio (ej. `app/services/clubs/players.py`) extrae los datos con expresiones
   **XPath** centralizadas en `app/utils/xpath.py`.
3. Los schemas Pydantic de `app/schemas/` validan y tipan la respuesta (fechas, enteros,
   valores de mercado parseados de "€250k" a `250000`).
4. `app/main.py` arma la app FastAPI con rate limiting opcional (slowapi,
   `RATE_LIMITING_ENABLE` / `RATE_LIMITING_FREQUENCY`).

Implicancias prácticas:

- **Sin API key ni base de datos**: cada request scrapea en vivo. Conviene cachear
  resultados localmente (el script de `predicciones/` cachea en memoria por ejecución).
- **Frágil ante cambios de HTML** de Transfermarkt: si algo se rompe, revisar
  `app/utils/xpath.py`.
- **Rate limiting**: la instancia pública limita a ~2 requests / 3 segundos. Para
  análisis de varios equipos, correrla local.

## 2. Endpoints útiles para predicciones

| Endpoint | Qué devuelve | Para qué nos sirve |
|---|---|---|
| `GET /clubs/search/{nombre}` | IDs de clubes/selecciones por nombre | Encontrar el ID de "Iraq", "Uzbekistan", etc. **Las selecciones nacionales aparecen como clubes** en Transfermarkt. |
| `GET /clubs/{id}/players` | Plantel completo: nombre, posición, edad, nacionalidad, club actual, altura, pie, **valor de mercado** | La base del score de equipo: con un solo request tenemos todo el plantel valorizado. |
| `GET /clubs/{id}/profile` | Datos del club: valor total del plantel, edad promedio, estadio | Validación cruzada del score. |
| `GET /players/{id}/profile` | Perfil completo del jugador | Detalle de jugadores puntuales. |
| `GET /players/{id}/stats` | Goles, asistencias, minutos **por competición y temporada** | Fase 2: ajustar el score por rendimiento real y nivel de liga. |
| `GET /players/{id}/market_value` | Histórico de valor de mercado | Detectar jugadores en alza/baja. |
| `GET /players/{id}/injuries` | Historial de lesiones | Descontar jugadores no disponibles. |
| `GET /competitions/search/{nombre}` | Competiciones | Mapear nivel de liga (fase 2). |

## 3. Metodología de scoring (v1 — implementada en `predicciones/predictor.py`)

El problema: para partidos tipo **Irak vs. Uzbekistán** no hay xG público ni métricas
avanzadas. Lo que sí existe, para *todos* los futbolistas del planeta, es el valor de
mercado de Transfermarkt, que la literatura muestra como muy buen predictor de
resultados entre selecciones.

### Score de jugador

```
score_jugador = log10(valor_mercado) × factor_edad
```

- **Logaritmo**: la diferencia entre €100k y €1M importa más que entre €50M y €60M.
  Sin log, un solo crack distorsiona todo el equipo.
- **Factor edad**: curva con pico en 24–29 años, descuento para juveniles (proyección
  aún no realizada) y veteranos (valor de mercado sobreestima rendimiento futuro).

### Score de equipo

```
score_equipo = 0.8 × promedio(11 mejores) + 0.2 × promedio(resto del plantel)
```

El equipo titular pesa mucho más, pero la profundidad del banco cuenta (torneos largos,
lesiones, rotación).

### Probabilidades del partido

Modelo tipo Elo sobre la diferencia de scores:

```
esperanza_A = 1 / (1 + 10^(-(score_A - score_B) / escala))
p_empate    = 0.30 × e^(-|score_A - score_B|)   (los partidos parejos empatan más)
p_gana_A    = esperanza_A × (1 - p_empate)
```

La `escala` (default 0.5) calibra cuánto pesa la diferencia: con scores en escala
log10 del valor de mercado, ~0.5 puntos de diferencia ≈ 76% de esperanza.

### Limitaciones conocidas (honestidad ante todo)

- El valor de mercado mide *calidad de plantel*, no forma actual, localía, ni táctica.
- Selecciones con plantel "viejo conocido" pueden estar sobrevaloradas.
- No considera convocatoria real: usa el plantel registrado en Transfermarkt.

## 4. Ideas para la v2

1. **Nivel de liga del club actual**: ponderar el score si el jugador compite en una
   liga top (usar `/players/{id}/stats` + `/competitions/search`). Un delantero iraquí
   titular en la Bundesliga ≠ uno en la liga local.
2. **Forma reciente**: goles/minutos de la última temporada desde `/players/{id}/stats`.
3. **Disponibilidad**: cruzar con `/players/{id}/injuries`.
4. **Calibración con resultados históricos**: bajar resultados de eliminatorias AFC/CAF
   y ajustar `escala` y `p_empate` por regresión logística.
5. **Localía**: bonus fijo (~0.1–0.15 de score) para el equipo local; en eliminatorias
   asiáticas la localía pesa muchísimo.
