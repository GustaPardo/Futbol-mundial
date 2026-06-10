#!/usr/bin/env python3
"""Simulación Monte Carlo del Mundial 2026 completo.

Usa el fixture real de la fase de grupos (data/results.csv), los ratings Elo
calculados por calibrar.py y el modelo Poisson calibrado para simular el torneo
entero miles de veces: fase de grupos partido por partido, los 8 mejores
terceros, llave de 32 sembrada por rendimiento, y eliminación directa hasta
la final (con alargue y penales).

Uso:
    python simular_mundial.py            # 10.000 simulaciones
    python simular_mundial.py -n 50000

Simplificaciones (ver docs/ANALISIS.md):
- La llave de 32 se siembra por rendimiento de grupo en vez del template FIFA.
- En eliminación directa todos los cruces se asumen en cancha neutral.
- Elo fijo durante el torneo (no se actualiza partido a partido).
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


def cargar() -> tuple[list[dict], dict[str, float], dict]:
    with open(os.path.join(DATA_DIR, "results.csv"), newline="", encoding="utf-8") as f:
        fixture = [
            p for p in csv.DictReader(f)
            if p["tournament"] == "FIFA World Cup" and p["date"] >= DESDE_FIXTURE
        ]
    with open(os.path.join(DATA_DIR, "elo_ratings.csv"), newline="", encoding="utf-8") as f:
        elo = {fila["team"]: float(fila["elo"]) for fila in csv.DictReader(f)}
    with open(os.path.join(DATA_DIR, "calibracion.json"), encoding="utf-8") as f:
        calib = json.load(f)
    return fixture, elo, calib


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


def muestra_poisson(lam: float) -> int:
    """Sampler de Knuth, suficiente para lam <= 4.5."""
    limite = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= random.random()
        if p <= limite:
            return k
        k += 1


class Simulador:
    def __init__(self, elo: dict[str, float], calib: dict):
        self.elo = elo
        self.a, self.b, self.c = calib["a"], calib["b"], calib["c"]

    def lambdas(self, eq_a: str, eq_b: str, local_a: bool) -> tuple[float, float]:
        x = (self.elo[eq_a] - self.elo[eq_b]) / 400
        lam_a = math.exp(self.a + self.b * x + self.c * local_a)
        lam_b = math.exp(self.a - self.b * x)
        acotar = lambda v: min(max(v, 0.15), 4.5)  # noqa: E731
        return acotar(lam_a), acotar(lam_b)

    def simular_partido(self, eq_a: str, eq_b: str, local_a: bool = False) -> tuple[int, int]:
        lam_a, lam_b = self.lambdas(eq_a, eq_b, local_a)
        return muestra_poisson(lam_a), muestra_poisson(lam_b)

    def ganador_eliminatoria(self, eq_a: str, eq_b: str) -> tuple[str, int]:
        """Devuelve (ganador, goles del ganador en el partido) con alargue y penales."""
        g_a, g_b = self.simular_partido(eq_a, eq_b)
        if g_a == g_b:
            # Alargue: 30 minutos ≈ un tercio de las tasas de gol
            lam_a, lam_b = self.lambdas(eq_a, eq_b, False)
            g_a += muestra_poisson(lam_a / 3)
            g_b += muestra_poisson(lam_b / 3)
        if g_a == g_b:  # penales
            return (eq_a, g_a) if random.random() < 0.5 else (eq_b, g_b)
        return (eq_a, g_a) if g_a > g_b else (eq_b, g_b)

    def fase_grupos(self, fixture: list[dict], goles: Counter) -> tuple[list[str], list[str], list[str]]:
        """Simula los 72 partidos y devuelve (primeros, segundos, 8 mejores terceros)."""
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
        primeros, segundos, terceros = [], [], []
        for grupo in self.grupos:
            orden = sorted(grupo, key=clave, reverse=True)
            primeros.append(orden[0])
            segundos.append(orden[1])
            terceros.append(orden[2])
        mejores_terceros = sorted(terceros, key=clave, reverse=True)[:8]
        return primeros, segundos, mejores_terceros

    def torneo(self, fixture: list[dict], goles: Counter) -> tuple[str, str, list[str]]:
        """Simula un Mundial completo. Devuelve (campeón, subcampeón, semifinalistas)."""
        primeros, segundos, terceros = self.fase_grupos(fixture, goles)
        clave = lambda e: (self.elo[e], random.random())  # noqa: E731
        # Sembrado: primeros por Elo, luego segundos, luego terceros; 1 vs 32, 2 vs 31...
        sembrados = (
            sorted(primeros, key=clave, reverse=True)
            + sorted(segundos, key=clave, reverse=True)
            + sorted(terceros, key=clave, reverse=True)
        )
        ronda = [(sembrados[i], sembrados[31 - i]) for i in range(16)]
        while len(ronda) > 1:
            ganadores = []
            for a, b in ronda:
                ganador, g = self.ganador_eliminatoria(a, b)
                goles[ganador] += g
                ganadores.append(ganador)
            ronda = [(ganadores[i], ganadores[i + 1]) for i in range(0, len(ganadores), 2)]
        semifinalistas = [e for par in ronda for e in par]  # los que llegaron a la final
        a, b = ronda[0]
        campeon, g = self.ganador_eliminatoria(a, b)
        goles[campeon] += g
        subcampeon = b if campeon == a else a
        return campeon, subcampeon, semifinalistas


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula el Mundial 2026 completo N veces.")
    parser.add_argument("-n", type=int, default=10_000, help="cantidad de simulaciones (default 10000)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    random.seed(args.seed)

    fixture, elo, calib = cargar()
    sim = Simulador(elo, calib)
    sim.grupos = detectar_grupos(fixture)

    titulos: Counter = Counter()
    finales: Counter = Counter()
    parejas_final: Counter = Counter()
    goles_totales: Counter = Counter()

    print(f"Simulando {args.n} mundiales...")
    for _ in range(args.n):
        campeon, subcampeon, _ = sim.torneo(fixture, goles_totales)
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
