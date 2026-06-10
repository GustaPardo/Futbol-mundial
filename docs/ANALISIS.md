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

### Probabilidades del partido: modelo de goles Poisson

En vez de estimar directamente "quién gana", el modelo estima **cuántos goles se
espera que haga cada equipo** (λ) y de ahí deriva las probabilidades de cada
resultado. Es el enfoque estándar en modelos de predicción de fútbol (Maher 1982,
Dixon-Coles 1997) porque captura naturalmente el empate y permite incorporar localía.

```
log(λ_equipo) = a + b × (diff_elo / 400) + c × es_local
```

donde `diff_elo` mezcla 50/50 la diferencia de rating Elo histórico con la diferencia
de valor de mercado del plantel (convertida a escala Elo: 1 punto de score ≈ 250 Elo).

Con λ_A y λ_B se calcula la probabilidad de cada marcador posible (Poisson
independiente) y se suman: P(gana A), P(empate), P(gana B). El script también
reporta el **marcador más probable**.

### Calibración con 49.000 partidos reales (`calibrar.py`)

Las constantes **no son inventadas**: `calibrar.py` las ajusta por máxima
verosimilitud usando el histórico de partidos internacionales de
[martj42/international_results](https://github.com/martj42/international_results)
(49.378 partidos jugados, 1872–hoy, 336 selecciones):

1. **Elo propio**: recorre todo el histórico y calcula el rating de cada selección
   con la fórmula de eloratings.net (K según torneo, multiplicador por goleada,
   +100 de localía). Resultado en `data/elo_ratings.csv` (233 selecciones activas).
2. **Ajuste Poisson**: con ~11.600 partidos (2010–2022) ajusta `a`, `b`, `c`.
   Valores obtenidos: goles base 1.04, localía ×1.32, +100 Elo de diferencia → ×1.21
   goles. Resultado en `data/calibracion.json`.
3. **Validación honesta**: sobre 3.517 partidos posteriores (2023+) que el modelo
   nunca vio: **60,7% de acierto** en 1X2 (baseline "siempre gana el local": 47%)
   y log-loss 0.86 (azar uniforme: 1.10). Para referencia, los mejores modelos
   públicos de fútbol internacional rondan 55–62%.

El predictor carga el Elo automáticamente por nombre del equipo; `--elo-a/--elo-b`
permiten pisarlo a mano. Para regenerar todo: `python calibrar.py`.

### Limitaciones conocidas (honestidad ante todo)

- El valor de mercado mide *calidad de plantel*, no forma actual ni táctica.
- Selecciones con plantel "viejo conocido" pueden estar sobrevaloradas.
- No considera convocatoria real: usa el plantel registrado en Transfermarkt.
- La conversión valor-de-mercado → escala Elo (×250) y la mezcla 50/50 son
  heurísticas razonables; el componente Elo sí está calibrado con datos.

## 4. Ideas para la v4

1. **Disponibilidad fina**: cruzar con `/players/{id}/injuries` para distinguir
   lesión larga de molestia menor (hoy el descuento es por el status del plantel).
2. **Optimizar la mezcla plantel/Elo**: ajustar el peso 50/50 y la conversión ×250
   con datos (requiere valores de mercado históricos).
3. **Dixon-Coles**: corregir la correlación de marcadores bajos (0-0, 1-1), que el
   Poisson independiente subestima levemente.
4. **Calibrar los pesos del ajuste por stats** (nivel/rodaje/producción) contra
   resultados reales en vez de usar constantes razonables.

> Ya implementado: localía (`--local`), Elo automático desde el histórico,
> calibración con resultados reales (`calibrar.py`), simulador del Mundial con
> la llave oficial FIFA (partidos 73–104), Elo dinámico durante el torneo,
> localía aproximada de los anfitriones en eliminación directa, descuento de
> lesionados, y **ajuste por jugador según nivel de competencia (Champions/ligas
> top), minutos y producción de la última temporada**
> (`generar_scores.py --con-stats`, con caché reanudable en `data/cache_stats/`).
