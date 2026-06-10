#!/usr/bin/env python3
"""Simulación Monte Carlo del Mundial 2026 completo.

Usa el fixture real de la fase de grupos (data/results.csv), los ratings Elo
calculados por calibrar.py y el modelo Poisson calibrado para simular el torneo
entero miles de veces:

- Fase de grupos partido por partido, con la localía real de los anfitriones.
- Los 8 mejores terceros, asignados a la llave según las restricciones de FIFA.
- La llave oficial de FIFA (partidos 73 a 104), no una llave sembrada.
- Elo dinámico: el rating se actualiza partido a partido dentro de cada
  simulación (K=60, multiplicador por goleada), capturando al equipo en racha.
- Localía aproximada de los anfitriones en eliminación directa (México y Canadá
  hasta octavos, Estados Unidos en todas las rondas).
- Si existe data/scores_plantel.csv (generar_scores.py), la fuerza de cada
  equipo mezcla 50/50 plantel de Transfermarkt + Elo histórico.

Uso:
    python simular_mundial.py            # 10.000 simulaciones
    python simular_mundial.py -n 50000
"""

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DESDE_FIXTURE = "2026-06-01"
ELO_POR_SCORE = 250  # misma conversión score de plantel → escala Elo que usa predictor.py
K_MUNDIAL = 60       # peso Elo de un partido de Mundial (fórmula eloratings.net)

# Equipo ancla para identificar la letra de cada grupo en el fixture
ANCLAS_GRUPO = {
    "A": "Mexico", "B": "Canada", "C": "Brazil", "D": "United States",
    "E": "Germany", "F": "Netherlands", "G": "Belgium", "H": "Spain",
    "I": "France", "J": "Argentina", "K": "Portugal", "L": "England",
}

# Llave oficial FIFA 2026. W=ganador de grupo, RU=segundo, T=tercero (grupos permitidos)
R32_FIFA = [
    (73, ("RU", "A"), ("RU", "B")),
    (74, ("W", "E"), ("T", "ABCDF")),
    (75, ("W", "F"), ("RU", "C")),
    (76, ("W", "C"), ("RU", "F")),
    (77, ("W", "I"), ("T", "CDFGH")),
    (78, ("RU", "E"), ("RU", "I")),
    (79, ("W", "A"), ("T", "CEFHI")),
    (80, ("W", "L"), ("T", "EHIJK")),
    (81, ("W", "D"), ("T", "BEFIJ")),
    (82, ("W", "G"), ("T", "AEHIJ")),
    (83, ("RU", "K"), ("RU", "L")),
    (84, ("W", "H"), ("RU", "J")),
    (85, ("W", "B"), ("T", "EFGIJ")),
    (86, ("W", "J"), ("RU", "H")),
    (87, ("W", "K"), ("T", "DEIJL")),
    (88, ("RU", "D"), ("RU", "G")),
]
R16_FIFA = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
            93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF_FIFA = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF_FIFA = {101: (97, 98), 102: (99, 100)}

ANFITRIONES_KO = {"United States": ("R32", "R16", "QF", "SF", "F"),
                  "Mexico": ("R32", "R16"),
                  "Canada": ("R32", "R16")}


URL_RESULTADOS = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def cargar() -> tuple[list[dict], dict[str, float], dict]:
    ruta_resultados = os.path.join(DATA_DIR, "results.csv")
    if not os.path.exists(ruta_resultados):
        import urllib.request
        print("Descargando el histórico de partidos (primera vez, ~4 MB)...")
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(URL_RESULTADOS, ruta_resultados)
    with open(ruta_resultados, newline="", encoding="utf-8") as f:
        fixture = [
            p for p in csv.DictReader(f)
            if p["tournament"] == "FIFA World Cup" and p["date"] >= DESDE_FIXTURE
        ]
    with open(os.path.join(DATA_DIR, "elo_ratings.csv"), newline="", encoding="utf-8") as f:
        elo = {fila["team"]: float(fila["elo"]) for fila in csv.DictReader(f)}
    with open(os.path.join(DATA_DIR, "calibracion.json"), encoding="utf-8") as f:
        calib = json.load(f)
    return fixture, elo, calib


def cargar_scores_plantel() -> dict[str, float]:
    """Scores de plantel de Transfermarkt generados por generar_scores.py (puede no existir)."""
    ruta = os.path.join(DATA_DIR, "scores_plantel.csv")
    if not os.path.exists(ruta):
        return {}
    with open(ruta, newline="", encoding="utf-8") as f:
        return {fila["team"]: float(fila["score"]) for fila in csv.DictReader(f)}


def mezclar_ratings(elo: dict[str, float], scores: dict[str, float], equipos: list[str]) -> dict[str, float]:
    """Rating mixto 50% Elo + 50% plantel (en escala Elo), como el predictor de partidos.

    A los equipos sin score de plantel se les imputa uno con una regresión lineal
    score ~ elo ajustada sobre los equipos que sí tienen.
    """
    con_ambos = [e for e in equipos if e in scores]
    if len(con_ambos) < len(equipos):
        xs = [elo[e] for e in con_ambos]
        ys = [scores[e] for e in con_ambos]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        pendiente = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        ordenada = my - pendiente * mx
        faltantes = [e for e in equipos if e not in scores]
        print(f"(Imputando score de plantel desde el Elo para: {', '.join(faltantes)})")
        for e in faltantes:
            scores[e] = pendiente * elo[e] + ordenada
    return {e: 0.5 * elo[e] + 0.5 * scores[e] * ELO_POR_SCORE for e in equipos}


VENTANA_CLASIFICACION = ("2023-01-01", "2026-04-01")  # ciclo de eliminatorias del Mundial 2026


def forma_clasificacion(equipos: list[str]) -> dict[str, dict]:
    """Forma reciente de cada mundialista: rendimiento real vs esperado por Elo,
    con más peso a los partidos más recientes (inercia/momentum).

    Recorre el histórico recalculando el Elo partido a partido. En la ventana del
    ciclo 2026 toma los partidos OFICIALES de cada equipo (eliminatorias, Nations
    League, Copa América/Euro, etc., sin amistosos) y, para cada uno, mide la
    diferencia entre el resultado real (1/0.5/0) y el esperado por Elo. El promedio
    pondera más los últimos partidos (decaimiento 0.88 por antigüedad), así un
    equipo "caliente" pesa más que uno que arrancó bien y se apagó. Se convierte
    en un bonus de rating acotado a ±60 Elo. Además guarda aparte el registro de
    eliminatorias (cómo clasificó) para mostrarlo en la tabla.
    """
    from calibrar import ELO_INICIAL, k_torneo

    decay = 0.88        # peso de cada partido respecto al siguiente más reciente
    escala_bonus = 240  # convierte el delta ponderado en puntos de Elo
    tope_bonus = 60.0
    max_partidos = 20   # cuántos partidos recientes mirar como mucho

    ruta = os.path.join(DATA_DIR, "results.csv")
    with open(ruta, newline="", encoding="utf-8") as f:
        partidos = [p for p in csv.DictReader(f) if p["home_score"] not in ("", "NA")]
    partidos.sort(key=lambda p: p["date"])

    objetivo = set(equipos)
    elo: dict[str, float] = {}
    oficiales: dict[str, list] = {e: [] for e in equipos}    # todos los oficiales (para el bonus)
    eliminat: dict[str, list] = {e: [] for e in equipos}     # solo eliminatorias (para mostrar)
    for p in partidos:
        local, visita = p["home_team"], p["away_team"]
        gl, gv = int(p["home_score"]), int(p["away_score"])
        el, ev = elo.get(local, ELO_INICIAL), elo.get(visita, ELO_INICIAL)
        ventaja = 0 if p["neutral"] == "TRUE" else 100
        esperado_local = 1 / (1 + 10 ** (-(el + ventaja - ev) / 400))
        resultado_local = 1.0 if gl > gv else 0.5 if gl == gv else 0.0

        if VENTANA_CLASIFICACION[0] <= p["date"] < VENTANA_CLASIFICACION[1]:
            torneo = p["tournament"].lower()
            es_eliminatoria = "world cup" in torneo and "qualification" in torneo
            es_oficial = "friendly" not in torneo
            for equipo, resultado, esperado, gf, gc in (
                (local, resultado_local, esperado_local, gl, gv),
                (visita, 1 - resultado_local, 1 - esperado_local, gv, gl),
            ):
                if equipo in objetivo:
                    if es_oficial:
                        oficiales[equipo].append((p["date"], resultado, esperado, gf, gc))
                    if es_eliminatoria:
                        eliminat[equipo].append((resultado, gf, gc))

        cambio = k_torneo(p["tournament"]) * multiplicador_goles(abs(gl - gv)) * (resultado_local - esperado_local)
        elo[local] = el + cambio
        elo[visita] = ev - cambio

    forma: dict[str, dict] = {}
    for e in equipos:
        recientes = sorted(oficiales[e], reverse=True)[:max_partidos]  # más nuevos primero
        elim = eliminat[e]
        if len(recientes) < 4:
            forma[e] = {"bonus": 0.0, "pj": 0, "g": 0, "emp": 0, "p": 0,
                        "gf": 0, "gc": 0, "forma": 0.0, "fuente": "sin datos"}
            continue
        peso_total = sum(decay**i for i in range(len(recientes)))
        delta = sum(decay**i * (r - esp) for i, (_, r, esp, _, _) in enumerate(recientes)) / peso_total
        # Para mostrar: si jugó eliminatorias, su registro; si no (anfitrión), los oficiales
        muestra = elim if elim else [(r, gf, gc) for _, r, _, gf, gc in recientes]
        forma[e] = {
            "bonus": max(min(escala_bonus * delta, tope_bonus), -tope_bonus),
            "pj": len(muestra),
            "g": sum(1 for r, _, _ in muestra if r == 1.0),
            "emp": sum(1 for r, _, _ in muestra if r == 0.5),
            "p": sum(1 for r, _, _ in muestra if r == 0.0),
            "gf": sum(gf for _, gf, _ in muestra),
            "gc": sum(gc for _, _, gc in muestra),
            "forma": delta,
            "fuente": "eliminatorias" if elim else "otros oficiales",
        }
    return forma


def aplicar_forma(ratings: dict[str, float], equipos: list[str], verboso: bool = True) -> dict[str, float]:
    """Suma el bonus de forma reciente (con peso a los últimos partidos) al rating."""
    forma = forma_clasificacion(equipos)
    if verboso:
        orden = sorted(equipos, key=lambda e: forma[e]["bonus"], reverse=True)
        arriba = ", ".join(f"{e} {forma[e]['bonus']:+.0f}" for e in orden[:3])
        abajo = ", ".join(f"{e} {forma[e]['bonus']:+.0f}" for e in orden[-3:])
        print(f"Forma reciente aplicada (±60 Elo máx, más peso a los últimos partidos). "
              f"En racha: {arriba}. En baja: {abajo}.")
    return {e: ratings[e] + forma[e]["bonus"] for e in equipos}


def detectar_grupos(fixture: list[dict]) -> list[list[str]]:
    """Los grupos son las componentes conexas del grafo de partidos de la fase de grupos."""
    adj = defaultdict(set)
    for p in fixture:
        adj[p["home_team"]].add(p["away_team"])
        adj[p["away_team"]].add(p["home_team"])
    vistos: set[str] = set()
    grupos = []
    for equipo in adj:
        if equipo in vistos:
            continue
        comp: set[str] = set()
        pila = [equipo]
        while pila:
            x = pila.pop()
            if x in comp:
                continue
            comp.add(x)
            pila.extend(adj[x] - comp)
        vistos |= comp
        grupos.append(sorted(comp))
    assert len(grupos) == 12 and all(len(g) == 4 for g in grupos), "fixture de grupos inesperado"
    return grupos


def grupos_por_letra(grupos: list[list[str]]) -> dict[str, list[str]]:
    letras = {}
    for letra, ancla in ANCLAS_GRUPO.items():
        for g in grupos:
            if ancla in g:
                letras[letra] = g
                break
        else:
            raise ValueError(f"No encontré el grupo {letra} (ancla '{ancla}') en el fixture")
    return letras


def multiplicador_goles(diff: int) -> float:
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return 1.75 + (diff - 3) / 8


def muestra_poisson(lam: float) -> int:
    """Sampler de Knuth, suficiente para lam <= 4.5."""
    limite = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= random.random()
        if p <= limite:
            return k
        k += 1


def asignar_terceros(slots: list[tuple[int, str]], letras: list[str]) -> dict[int, str] | None:
    """Asigna las letras de los 8 mejores terceros a los slots de la llave (backtracking)."""
    if not slots:
        return {}
    # Heurística: resolver primero el slot con menos opciones
    num, permitidas = min(slots, key=lambda s: sum(l in s[1] for l in letras))
    resto = [s for s in slots if s[0] != num]
    opciones = [l for l in letras if l in permitidas]
    random.shuffle(opciones)
    for letra in opciones:
        sub = asignar_terceros(resto, [l for l in letras if l != letra])
        if sub is not None:
            sub[num] = letra
            return sub
    return None


class Simulador:
    def __init__(self, ratings: dict[str, float], calib: dict):
        self.base = ratings
        self.a, self.b, self.c = calib["a"], calib["b"], calib["c"]
        self.r: dict[str, float] = {}

    def lambdas(self, eq_a: str, eq_b: str, local_a: bool, local_b: bool = False) -> tuple[float, float]:
        x = (self.r[eq_a] - self.r[eq_b]) / 400
        lam_a = math.exp(self.a + self.b * x + self.c * local_a)
        lam_b = math.exp(self.a - self.b * x + self.c * local_b)
        acotar = lambda v: min(max(v, 0.15), 4.5)  # noqa: E731
        return acotar(lam_a), acotar(lam_b)

    def actualizar_elo(self, eq_a: str, eq_b: str, g_a: int, g_b: int, local_a: bool, local_b: bool) -> None:
        """Elo dinámico durante el torneo: el equipo en racha llega más fuerte al cruce."""
        ventaja = 100 * (local_a - local_b)
        esperado_a = 1 / (1 + 10 ** (-(self.r[eq_a] + ventaja - self.r[eq_b]) / 400))
        resultado = 1.0 if g_a > g_b else 0.5 if g_a == g_b else 0.0
        cambio = K_MUNDIAL * multiplicador_goles(abs(g_a - g_b)) * (resultado - esperado_a)
        self.r[eq_a] += cambio
        self.r[eq_b] -= cambio

    def simular_partido(self, eq_a: str, eq_b: str, local_a: bool = False, local_b: bool = False) -> tuple[int, int]:
        lam_a, lam_b = self.lambdas(eq_a, eq_b, local_a, local_b)
        g_a, g_b = muestra_poisson(lam_a), muestra_poisson(lam_b)
        self.actualizar_elo(eq_a, eq_b, g_a, g_b, local_a, local_b)
        return g_a, g_b

    def eliminatoria(self, eq_a: str, eq_b: str, ronda: str, goles: Counter) -> str:
        """Partido de eliminación directa con alargue y penales. Devuelve el ganador."""
        local_a = ronda in ANFITRIONES_KO.get(eq_a, ())
        local_b = ronda in ANFITRIONES_KO.get(eq_b, ())
        if local_a and local_b:
            local_a = local_b = False
        lam_a, lam_b = self.lambdas(eq_a, eq_b, local_a, local_b)
        g_a, g_b = muestra_poisson(lam_a), muestra_poisson(lam_b)
        if g_a == g_b:
            # Alargue: 30 minutos ≈ un tercio de las tasas de gol
            g_a += muestra_poisson(lam_a / 3)
            g_b += muestra_poisson(lam_b / 3)
        self.actualizar_elo(eq_a, eq_b, g_a, g_b, local_a, local_b)
        goles[eq_a] += g_a
        goles[eq_b] += g_b
        if g_a == g_b:  # penales
            return eq_a if random.random() < 0.5 else eq_b
        return eq_a if g_a > g_b else eq_b

    def fase_grupos(self, fixture: list[dict], letras: dict[str, list[str]], goles: Counter):
        """Simula los 72 partidos. Devuelve posiciones por letra y los 8 mejores terceros."""
        pts: Counter = Counter()
        dif: Counter = Counter()
        favor: Counter = Counter()
        for p in fixture:
            a, b = p["home_team"], p["away_team"]
            g_a, g_b = self.simular_partido(a, b, local_a=p["neutral"] == "FALSE")
            goles[a] += g_a
            goles[b] += g_b
            favor[a] += g_a
            favor[b] += g_b
            dif[a] += g_a - g_b
            dif[b] += g_b - g_a
            if g_a > g_b:
                pts[a] += 3
            elif g_b > g_a:
                pts[b] += 3
            else:
                pts[a] += 1
                pts[b] += 1
        clave = lambda e: (pts[e], dif[e], favor[e], random.random())  # noqa: E731
        posiciones = {letra: sorted(grupo, key=clave, reverse=True) for letra, grupo in letras.items()}
        terceros = sorted(posiciones, key=lambda letra: clave(posiciones[letra][2]), reverse=True)[:8]
        return posiciones, terceros

    def torneo(self, fixture: list[dict], letras: dict[str, list[str]], goles: Counter) -> tuple[str, str]:
        """Simula un Mundial completo con la llave oficial. Devuelve (campeón, subcampeón)."""
        self.r = dict(self.base)
        posiciones, terceros = self.fase_grupos(fixture, letras, goles)

        slots = [(num, esp[1]) for num, _, esp in R32_FIFA if esp[0] == "T"]
        asignacion = asignar_terceros(slots, terceros)
        if asignacion is None:  # combinación sin asignación válida: relajar restricciones
            asignacion = {num: letra for (num, _), letra in zip(slots, terceros)}

        def resolver(spec: tuple[str, str], num: int) -> str:
            tipo, dato = spec
            if tipo == "W":
                return posiciones[dato][0]
            if tipo == "RU":
                return posiciones[dato][1]
            return posiciones[asignacion[num]][2]

        ganadores: dict[int, str] = {}
        for num, esp_a, esp_b in R32_FIFA:
            ganadores[num] = self.eliminatoria(resolver(esp_a, num), resolver(esp_b, num), "R32", goles)
        for num, (m1, m2) in R16_FIFA.items():
            ganadores[num] = self.eliminatoria(ganadores[m1], ganadores[m2], "R16", goles)
        for num, (m1, m2) in QF_FIFA.items():
            ganadores[num] = self.eliminatoria(ganadores[m1], ganadores[m2], "QF", goles)
        for num, (m1, m2) in SF_FIFA.items():
            ganadores[num] = self.eliminatoria(ganadores[m1], ganadores[m2], "SF", goles)
        campeon = self.eliminatoria(ganadores[101], ganadores[102], "F", goles)
        subcampeon = ganadores[102] if campeon == ganadores[101] else ganadores[101]
        return campeon, subcampeon


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula el Mundial 2026 completo N veces.")
    parser.add_argument("-n", type=int, default=10_000, help="cantidad de simulaciones (default 10000)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sin-forma", action="store_true",
                        help="no aplicar el bonus por rendimiento en las eliminatorias")
    args = parser.parse_args()
    random.seed(args.seed)

    fixture, elo, calib = cargar()
    grupos = detectar_grupos(fixture)
    letras = grupos_por_letra(grupos)
    equipos = [e for g in grupos for e in g]
    scores = cargar_scores_plantel()
    if scores:
        ratings = mezclar_ratings(elo, scores, equipos)
        print(f"Fuerza de equipos: 50% Elo histórico + 50% plantel Transfermarkt ({len(scores)} con score)")
    else:
        ratings = elo
        print("Fuerza de equipos: solo Elo histórico (corré generar_scores.py para sumar Transfermarkt)")
    if not args.sin_forma:
        ratings = aplicar_forma(ratings, equipos)
    sim = Simulador(ratings, calib)

    titulos: Counter = Counter()
    finales: Counter = Counter()
    parejas_final: Counter = Counter()
    goles_totales: Counter = Counter()

    print(f"Simulando {args.n} mundiales (llave oficial FIFA, Elo dinámico)...")
    for _ in range(args.n):
        campeon, subcampeon = sim.torneo(fixture, letras, goles_totales)
        titulos[campeon] += 1
        finales[campeon] += 1
        finales[subcampeon] += 1
        parejas_final[tuple(sorted((campeon, subcampeon)))] += 1

    n = args.n
    print(f"\n{'Equipo':<16} {'Campeón':>9} {'Finalista':>10} {'Goles esp.':>11}")
    print("-" * 50)
    for equipo, c in titulos.most_common(12):
        print(f"{equipo:<16} {c / n:>8.1%} {finales[equipo] / n:>9.1%} {goles_totales[equipo] / n:>10.2f}")

    print("\nFinales más probables:")
    for (a, b), c in parejas_final.most_common(5):
        print(f"   {a} vs {b}: {c / n:.1%}")

    print("\nGoles esperados por equipo en todo el torneo (top 8):")
    for equipo, g in goles_totales.most_common(8):
        print(f"   {equipo:<16} {g / n:5.2f}")


if __name__ == "__main__":
    main()
