# ⚽ Futbol Mundial — Predicciones

Proyecto para generar **scores de equipos y jugadores** usando datos de
[Transfermarkt](https://www.transfermarkt.com/), con el objetivo de estimar quién podría
ganar en partidos de selecciones "poco conocidas" (Irak, Uzbekistán, Jordania, etc.),
donde las estadísticas tradicionales escasean.

## Estructura del repo

| Carpeta | Qué es |
|---|---|
| [`transfermarkt-api/`](transfermarkt-api/) | Copia del repo [felipeall/transfermarkt-api](https://github.com/felipeall/transfermarkt-api) (licencia MIT). API REST en FastAPI que scrapea Transfermarkt. |
| [`docs/ANALISIS.md`](docs/ANALISIS.md) | Estudio del código de la API: cómo funciona, qué endpoints sirven para predicciones y metodología de scoring. |
| [`predicciones/`](predicciones/) | Script propio que usa la API para puntuar equipos y estimar probabilidades de victoria. |

## Cómo usarlo (rápido)

### 1. Levantar la API localmente

```bash
cd transfermarkt-api
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
python app/main.py
# La API queda en http://localhost:8000 (Swagger en /docs)
```

También existe una instancia pública de prueba en `https://transfermarkt-api.fly.dev`
(con rate limiting; para uso intensivo conviene la local).

### 2. Comparar dos selecciones

```bash
cd predicciones
pip install -r requirements.txt
python predictor.py "Iraq" "Uzbekistan"
```

Salida: score de cada equipo (basado en valor de mercado, edad y profundidad del plantel),
los jugadores más valiosos de cada uno, y una probabilidad estimada de
victoria/empate/derrota.

Si querés usar la instancia pública en vez de la local:

```bash
TM_API_URL=https://transfermarkt-api.fly.dev python predictor.py "Iraq" "Jordan"
```

## La idea detrás del scoring

Para selecciones chicas casi no hay xG ni estadísticas avanzadas públicas, pero
Transfermarkt **sí** tiene para todos los jugadores del mundo:

- **Valor de mercado** → mejor proxy único de calidad disponible globalmente.
- **Edad** → para ajustar por curva de rendimiento (pico ~24–29 años).
- **Club actual** → un iraquí jugando en Europa vale más señal que uno en liga local.
- **Stats por competición** (goles, minutos) → para refinar el score (fase 2).

Ver la metodología completa en [`docs/ANALISIS.md`](docs/ANALISIS.md).

## Créditos

El código de `transfermarkt-api/` es de [Felipe Allegretti](https://github.com/felipeall),
bajo licencia MIT. Los datos pertenecen a Transfermarkt; usar con respeto a sus términos
(rate limiting, uso personal/investigación).
