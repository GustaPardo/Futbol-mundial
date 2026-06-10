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

## ¿Dónde lo corro?

### Opción 1: Google Colab (sin instalar nada) ← la más fácil

Abrí este link en el navegador y ejecutá las celdas en orden:

**[▶ Abrir el predictor en Google Colab](https://colab.research.google.com/github/GustaPardo/Futbol-mundial/blob/claude/laughing-ritchie-f4b202/predicciones/predictor_colab.ipynb)**

Cambiás los nombres de los equipos en la última celda y listo.

### Opción 2: En tu computadora (usando la API pública)

Necesitás solo Python instalado:

```bash
git clone https://github.com/GustaPardo/Futbol-mundial.git
cd Futbol-mundial/predicciones
pip install requests
TM_API_URL=https://transfermarkt-api.fly.dev python predictor.py "Iraq" "Uzbekistan"
```

### Opción 3: En tu computadora con API local (más rápida, sin rate limit)

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
python predictor.py "Iraq" "Uzbekistan"              # cancha neutral
python predictor.py "Iraq" "Uzbekistan" --local A    # Irak de local
python predictor.py "Iraq" "Uzbekistan" --solo-elo   # sin internet: solo Elo histórico
```

Salida: score de cada equipo (valor de mercado, edad y profundidad del plantel), los
jugadores más valiosos, goles esperados, el marcador más probable y las probabilidades
de victoria/empate/derrota.

El modelo combina automáticamente el **valor del plantel** (Transfermarkt) con el
**rating Elo histórico** de cada selección, calculado desde 49.000 partidos
internacionales (1872–hoy). Las constantes están calibradas con partidos reales:
**60,7% de acierto** validado sobre 3.500 partidos de 2023+ que el modelo nunca vio
(baseline: 47%). Para recalibrar con datos frescos: `python calibrar.py`.

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
