#!/usr/bin/env python3
"""Análisis del Mundial 2026 sobre el simulador: grupos, equipos y goleador.

Subcomandos:
    python analizar.py grupos                 # P(1°, 2°, clasificar) por grupo
    python analizar.py equipo "Portugal"      # hasta dónde llega y quién lo elimina
    python analizar.py goleador               # goles esperados jugador por jugador

Todos usan la fuerza mixta Elo + plantel si existe data/scores_plantel.csv.
El análisis de goleador necesita data/planteles.json (lo genera generar_scores.py)
y mejora mucho con el caché de stats del modo --con-stats.
"""

import argparse
import json
import math
import os
import random
from collections import Counter

from simular_mundial import (
    DATA_DIR,
    Simulador,
    aplicar_forma,
    cargar,
    cargar_scores_plantel,
    detectar_grupos,
    forma_clasificacion,
    grupos_por_letra,
    mezclar_ratings,
)

RONDAS = ["eliminado en grupos", "pierde en R32", "pierde en R16",
          "pierde en QF", "pierde en SF", "pierde en F", "CAMPEÓN"]

# Peso ofensivo por posición para repartir goles cuando no hay stats del jugador
PESO_POSICION = (
    ("centre-forward", 1.0), ("striker", 1.0), ("winger", 0.75),
    ("attacking midfield", 0.55), ("central midfield", 0.30),
    ("defensive midfield", 0.18), ("midfield", 0.30), ("back", 0.10),
    ("defender", 0.10), ("goalkeeper", 0.01),
)


def preparar():
    fixture, elo, calib = cargar()
    grupos = detectar_grupos(fixture)
    letras = grupos_por_letra(grupos)
    equipos = [e for g in grupos for e in g]
    scores = cargar_scores_plantel()
    if scores:
        ratings = mezclar_ratings(elo, scores, equipos)
        print(f"Fuerza de equipos: 50% Elo + 50% plantel Transfermarkt ({len(scores)} con score)")
    else:
        ratings = elo
        print("Fuerza de equipos: solo Elo histórico (corré generar_scores.py para sumar Transfermarkt)")
    ratings = aplicar_forma(ratings, equipos)
    print()
    return fixture, letras, ratings, calib


def analizar_grupos(n: int) -> None:
    """Probabilidad de salir 1°, 2° y de clasificar a 16avos, grupo por grupo."""
    fixture, letras, ratings, calib = preparar()
    sim = Simulador(ratings, calib)
    primero: Counter = Counter()
    segundo: Counter = Counter()
    tercero_ok: Counter = Counter()
    print(f"Simulando {n} fases de grupos...\n")
    for _ in range(n):
        sim.r = dict(sim.base)
        posiciones, terceros = sim.fase_grupos(fixture, letras, Counter())
        for orden in posiciones.values():
            primero[orden[0]] += 1
            segundo[orden[1]] += 1
        for letra in terceros:
            tercero_ok[posiciones[letra][2]] += 1
    for letra in sorted(letras):
        print(f"Grupo {letra}")
        equipos = sorted(letras[letra],
                         key=lambda e: primero[e] + segundo[e] + tercero_ok[e], reverse=True)
        for e in equipos:
            clasifica = (primero[e] + segundo[e] + tercero_ok[e]) / n
            print(f"   {e:<22} 1°: {primero[e]/n:6.1%}   2°: {segundo[e]/n:6.1%}"
                  f"   clasifica: {clasifica:6.1%}")
        print()


def analizar_clasificacion() -> None:
    """Campaña clasificatoria de cada mundialista y su rendimiento vs lo esperado por Elo."""
    fixture, _, _ = cargar()
    equipos = sorted({e for g in detectar_grupos(fixture) for e in g})
    forma = forma_clasificacion(equipos)
    print("Cómo llegaron al Mundial: campaña de eliminatorias y forma reciente vs lo esperado por Elo.")
    print("El bonus (±60 Elo máx., con más peso a los últimos partidos) entra al modelo.\n")
    print(f"{'Equipo':<22} {'PJ':>3} {'G-E-P':>8} {'GF:GC':>7} {'Forma':>7} {'Bonus':>6}  Fuente")
    print("-" * 72)
    for e in sorted(equipos, key=lambda x: forma[x]["bonus"], reverse=True):
        f = forma[e]
        if f["fuente"] == "sin datos":
            print(f"{e:<22} {'—':>3} {'—':>8} {'—':>7} {'—':>7} {f['bonus']:>+6.0f}  sin datos")
            continue
        print(f"{e:<22} {f['pj']:>3} {f['g']:>2}-{f['emp']}-{f['p']:<2} "
              f"{f['gf']:>3}:{f['gc']:<3} {f['forma']:>+7.3f} {f['bonus']:>+6.0f}  {f['fuente']}")


class SimuladorConRastro(Simulador):
    """Simulador que registra los cruces de eliminación directa de un equipo."""

    def __init__(self, ratings, calib, objetivo: str):
        super().__init__(ratings, calib)
        self.objetivo = objetivo
        self.ko: list[tuple[str, str]] = []

    def torneo(self, fixture, letras, goles):
        self.ko = []
        return super().torneo(fixture, letras, goles)

    def eliminatoria(self, eq_a, eq_b, ronda, goles):
        ganador = super().eliminatoria(eq_a, eq_b, ronda, goles)
        if self.objetivo in (eq_a, eq_b):
            self.ko.append((ronda, ganador))
        return ganador


def analizar_equipo(equipo: str, n: int) -> None:
    """Radiografía: hasta qué ronda llega, quién lo elimina y sensibilidad al plantel."""
    fixture, letras, ratings, calib = preparar()
    if equipo not in ratings:
        candidatos = [e for g in letras.values() for e in g if equipo.lower() in e.lower()]
        if not candidatos:
            raise SystemExit(f"'{equipo}' no está en el Mundial. Usá el nombre en inglés (ej. 'Germany').")
        equipo = candidatos[0]
    letra_grupo = next(l for l, g in letras.items() if equipo in g)
    print(f"{equipo} — rating {ratings[equipo]:.0f}, grupo {letra_grupo}: "
          + ", ".join(f"{e} ({ratings[e]:.0f})" for e in letras[letra_grupo]))
    f = forma_clasificacion([equipo])[equipo]
    if f["fuente"] != "sin datos":
        print(f"Clasificación: {f['g']}G-{f['emp']}E-{f['p']}P, {f['gf']}:{f['gc']} en {f['pj']} partidos "
              f"({f['fuente']}) → forma {f['forma']:+.3f}, bonus {f['bonus']:+.0f} Elo\n")
    else:
        print("Clasificación: sin datos suficientes → sin bonus de forma\n")

    sim = SimuladorConRastro(ratings, calib, equipo)
    etapa: Counter = Counter()
    titulos: Counter = Counter()
    goles: Counter = Counter()
    verdugos = {r: Counter() for r in ("R32", "R16", "QF", "SF", "F")}
    print(f"Simulando {n} mundiales...\n")
    for _ in range(n):
        campeon, _ = sim.torneo(fixture, letras, goles)
        titulos[campeon] += 1
        if not sim.ko:
            etapa["eliminado en grupos"] += 1
        else:
            ronda, ganador = sim.ko[-1]
            if ganador == equipo:
                etapa["CAMPEÓN"] += 1
            else:
                etapa[f"pierde en {ronda}"] += 1
                verdugos[ronda][ganador] += 1

    puesto = [e for e, _ in titulos.most_common()].index(equipo) + 1 if titulos[equipo] else "-"
    print(f"P(campeón): {titulos[equipo]/n:.2%} (puesto {puesto} en títulos)"
          f"  |  goles esperados en el torneo: {goles[equipo]/n:.2f}\n")
    for k in RONDAS:
        if etapa[k]:
            barra = "█" * round(etapa[k] / n * 40)
            print(f"   {k:<22} {etapa[k]/n:6.1%} {barra}")
    print("\nVerdugos más frecuentes por ronda:")
    for r in ("R32", "R16", "QF", "SF", "F"):
        if verdugos[r]:
            print(f"   {r:<4} " + ", ".join(f"{e} {c/n:.1%}" for e, c in verdugos[r].most_common(3)))


def _peso_posicion(posicion: str) -> float:
    p = (posicion or "").lower()
    for clave, peso in PESO_POSICION:
        if clave in p:
            return peso
    return 0.30


def _tasa_gol(jugador: dict) -> tuple[float, str]:
    """Tasa relativa de gol del jugador: stats reales si hay caché, posición+valor si no."""
    ruta = os.path.join(DATA_DIR, "cache_stats", f"{jugador['id']}.json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            stats = json.load(f)
        temporadas = [s.get("seasonId") for s in stats if s.get("seasonId")]
        if temporadas:
            ultima = max(temporadas, key=lambda t: int("".join(c for c in str(t) if c.isdigit()) or 0))
            filas = [s for s in stats if s.get("seasonId") == ultima]
            def entero(v):
                return int("".join(c for c in str(v) if c.isdigit()) or 0) if v is not None else 0
            minutos = sum(entero(f.get("minutesPlayed")) for f in filas)
            goles = sum(entero(f.get("goals")) for f in filas)
            if minutos >= 450:
                g90 = goles / (minutos / 90)
                rodaje = 0.5 + 0.5 * min(minutos / 2500, 1.0)
                return g90 * rodaje, "stats"
    valor = jugador.get("marketValue") or 10_000
    return _peso_posicion(jugador.get("position")) * math.log10(valor) / 8, "posición"


def analizar_goleador(n: int, top: int = 15) -> None:
    """Goles esperados por jugador: goles del equipo en la simulación × cuota del jugador."""
    ruta = os.path.join(DATA_DIR, "planteles.json")
    if not os.path.exists(ruta):
        raise SystemExit("Falta data/planteles.json: corré primero generar_scores.py "
                         "(idealmente con --con-stats para usar goles reales por jugador).")
    with open(ruta, encoding="utf-8") as f:
        planteles = json.load(f)

    fixture, letras, ratings, calib = preparar()
    sim = Simulador(ratings, calib)
    goles: Counter = Counter()
    print(f"Simulando {n} mundiales para repartir goles...\n")
    for _ in range(n):
        sim.torneo(fixture, letras, goles)

    candidatos = []
    con_stats = 0
    for equipo, jugadores in planteles.items():
        aptos = [j for j in jugadores if not j.get("lesionado")]
        tasas = [_tasa_gol(j) for j in aptos]
        con_stats += sum(1 for _, fuente in tasas if fuente == "stats")
        total = sum(t for t, _ in tasas) or 1.0
        goles_equipo = goles.get(equipo, 0) / n
        for j, (tasa, fuente) in zip(aptos, tasas):
            candidatos.append((goles_equipo * tasa / total, j["name"], equipo, fuente))

    total_jugadores = sum(len(js) for js in planteles.values())
    fuente = (f"goles reales de la última temporada ({con_stats}/{total_jugadores} jugadores con stats)"
              if con_stats else "posición + valor de mercado (corré generar_scores.py --con-stats para usar goles reales)")
    print(f"Cuotas de gol por: {fuente}\n")
    print(f"{'Jugador':<28} {'Selección':<16} {'Goles esperados':>15}")
    print("-" * 62)
    for esperados, nombre, equipo, _ in sorted(candidatos, reverse=True)[:top]:
        print(f"{nombre:<28} {equipo:<16} {esperados:>15.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="comando", required=True)
    p_g = sub.add_parser("grupos", help="probabilidades de clasificación por grupo")
    p_g.add_argument("-n", type=int, default=20_000)
    p_e = sub.add_parser("equipo", help="radiografía de un equipo")
    p_e.add_argument("nombre")
    p_e.add_argument("-n", type=int, default=20_000)
    p_t = sub.add_parser("goleador", help="goles esperados por jugador")
    p_t.add_argument("-n", type=int, default=20_000)
    p_t.add_argument("--top", type=int, default=15)
    sub.add_parser("clasificacion", help="campaña clasificatoria y forma de cada mundialista")
    args = parser.parse_args()
    random.seed(2026)

    if args.comando == "grupos":
        analizar_grupos(args.n)
    elif args.comando == "equipo":
        analizar_equipo(args.nombre, args.n)
    elif args.comando == "clasificacion":
        analizar_clasificacion()
    else:
        analizar_goleador(args.n, args.top)


if __name__ == "__main__":
    main()
