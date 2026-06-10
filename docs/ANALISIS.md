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
delta = score_A - score_B                    (diferencia de calidad de plantel)
λ_A   = 1.30 × 10^(+0.4 × delta)             (goles esperados de A)
λ_B   = 1.30 × 10^(-0.4 × delta)             (goles esperados de B)
λ_local ×= 1.25                              (localía ≈ +0.3 goles, si aplica)
```

Constantes: 1.30 = goles por equipo en un partido parejo en cancha neutral (promedio
histórico ~2.6 goles totales); 0.8 (`PESO_DELTA`) convierte la escala log10 del valor
de mercado en ventaja de goles; 1.25 (`FACTOR_LOCALIA`) es la ventaja de localía
típica. Las λ se acotan a [0.2, 4.5] para partidos muy desparejos.

Con λ_A y λ_B se calcula la probabilidad de cada marcador posible (Poisson
independiente) y se suman: P(gana A), P(empate), P(gana B). El script también
reporta el **marcador más probable**.

### Mezcla opcional con ratings Elo

El valor de mercado mide *calidad de plantel*; el rating Elo de
[eloratings.net](https://www.eloratings.net/) mide *resultados reales históricos*.
La literatura muestra que la combinación supera a cualquiera de los dos por separado.
Si se pasan `--elo-a` y `--elo-b`, el modelo mezcla 50/50:

```
delta = 0.5 × delta_valor_mercado + 0.5 × (elo_A - elo_B) / 250
```

El divisor 250 pone la diferencia Elo en una escala comparable al score (heurística
razonable pendiente de calibración con resultados reales).

### Limitaciones conocidas (honestidad ante todo)

- El valor de mercado mide *calidad de plantel*, no forma actual ni táctica.
- Selecciones con plantel "viejo conocido" pueden estar sobrevaloradas.
- No considera convocatoria real: usa el plantel registrado en Transfermarkt.
- Las constantes del modelo son razonables pero no están calibradas contra un
  histórico de partidos (ver v2, punto 4).

## 4. Ideas para la v2

1. **Nivel de liga del club actual**: ponderar el score si el jugador compite en una
   liga top (usar `/players/{id}/stats` + `/competitions/search`). Un delantero iraquí
   titular en la Bundesliga ≠ uno en la liga local.
2. **Forma reciente**: goles/minutos de la última temporada desde `/players/{id}/stats`.
3. **Disponibilidad**: cruzar con `/players/{id}/injuries`.
4. **Calibración con resultados históricos**: bajar resultados de eliminatorias AFC/CAF
   y ajustar `PESO_DELTA`, `GOLES_BASE` y la mezcla Elo por máxima verosimilitud.
5. **Elo automático**: scrapear eloratings.net para no pasar los ratings a mano.

> Localía y mezcla con Elo ya están implementadas (`--local`, `--elo-a/--elo-b`).
