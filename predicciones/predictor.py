#!/usr/bin/env python3
"""Predictor de partidos entre selecciones usando datos de Transfermarkt.

Usa la API de transfermarkt-api (carpeta ../transfermarkt-api de este repo)
para obtener el plantel de cada equipo, calcular un score basado en valor
de mercado y edad, y estimar probabilidades de victoria/empate/derrota con
un modelo de goles tipo Poisson.

Uso:
    python predictor.py "Iraq" "Uzbekistan"
    python predictor.py "Iraq" "Uzbekistan" --local A
    python predictor.py "Iraq" "Brazil" --elo-a 1480 --elo-b 2100
    TM_API_URL=https://transfermarkt-api.fly.dev python predictor.py "Iraq" "Jordan"

La metodología está documentada en docs/ANALISIS.md.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import requests

API_BASE = os.environ.get("TM_API_URL", "http://localhost:8000").rstrip("/")
# Pausa entre requests para no golpear el rate limit de la instancia pública
PAUSA_SEGUNDOS = float(os.environ.get("TM_API_PAUSA", "1.5" if "fly.dev" in API_BASE else "0"))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Constantes por defecto si no existe data/calibracion.json (regenerable con calibrar.py)
CALIBRACION_DEFAULT = {"a": 0.0438, "b": 0.7580, "c": 0.2746}
ELO_POR_SCORE = 250  # convierte la diferencia de score (log10 valor mercado) a escala Elo
MAX_GOLES = 12       # tope de goles por equipo en la grilla Poisson


def cargar_calibracion() -> dict:
    """Constantes del modelo Poisson ajustadas con ~12k partidos reales (ver calibrar.py)."""
    ruta = os.path.join(DATA_DIR, "calibracion.json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return CALIBRACION_DEFAULT


def cargar_elo() -> dict[str, float]:
    """Ratings Elo por selección, generados por calibrar.py desde el histórico de partidos."""
    ruta = os.path.join(DATA_DIR, "elo_ratings.csv")
    if not os.path.exists(ruta):
        return {}
    with open(ruta, newline="", encoding="utf-8") as f:
        return {fila["team"].lower(): float(fila["elo"]) for fila in csv.DictReader(f)}


def buscar_elo(tabla: dict[str, float], *nombres: str) -> float | None:
    for nombre in nombres:
        rating = tabla.get(nombre.lower().strip())
        if rating is not None:
            return rating
    return None


def _get(path: str) -> dict:
    if PAUSA_SEGUNDOS:
        time.sleep(PAUSA_SEGUNDOS)
    resp = requests.get(f"{API_BASE}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def buscar_equipo(nombre: str) -> dict:
    """Busca un club/selección por nombre y devuelve el primer resultado."""
    data = _get(f"/clubs/search/{nombre}")
    resultados = data.get("results", [])
    if not resultados:
        sys.exit(f"No se encontró ningún equipo para '{nombre}'. Probá con el nombre en inglés (ej. 'Iraq').")
    equipo = resultados[0]
    if len(resultados) > 1:
        otros = ", ".join(r["name"] for r in resultados[1:4])
        print(f"  (usando '{equipo['name']}' id={equipo['id']}; otros resultados: {otros})")
    return equipo


def obtener_plantel(club_id: str) -> list[dict]:
    data = _get(f"/clubs/{club_id}/players")
    return data.get("players", [])


def factor_edad(edad: int | None) -> float:
    """Curva de rendimiento por edad, con pico entre 24 y 29 años."""
    if edad is None:
        return 0.90
    if edad < 18:
        return 0.75
    if edad < 21:
        return 0.85
    if edad < 24:
        return 0.95
    if edad <= 29:
        return 1.00
    if edad <= 32:
        return 0.92
    if edad <= 35:
        return 0.80
    return 0.65


def score_jugador(jugador: dict) -> float:
    valor = jugador.get("marketValue") or jugador.get("market_value") or 0
    # Piso de €10k para jugadores sin valor publicado
    return math.log10(max(valor, 10_000)) * factor_edad(jugador.get("age"))


def score_equipo(plantel: list[dict]) -> float:
    """80% el equipo titular estimado (los 11 mejores), 20% la profundidad del banco."""
    scores = sorted((score_jugador(j) for j in plantel), reverse=True)
    if not scores:
        return 0.0
    titulares = scores[:11]
    banco = scores[11:]
    score = 0.8 * (sum(titulares) / len(titulares))
    if banco:
        score += 0.2 * (sum(banco) / len(banco))
    else:
        score /= 0.8
    return score


def lambdas_esperados(
    score_a: float | None,
    score_b: float | None,
    local: str | None = None,
    elo_a: float | None = None,
    elo_b: float | None = None,
) -> tuple[float, float]:
    """Convierte la diferencia de calidad en goles esperados, con el modelo calibrado.

    log(goles) = a + b·(diff_elo/400) + c·es_local, donde diff_elo mezcla 50/50 el
    rating Elo real (resultados históricos) con la diferencia de valor de mercado
    del plantel convertida a escala Elo. Si falta una de las dos fuentes, usa la otra.
    """
    calib = cargar_calibracion()
    a, b, c = calib["a"], calib["b"], calib["c"]
    hay_plantel = score_a is not None and score_b is not None
    hay_elo = elo_a is not None and elo_b is not None
    if not hay_plantel and not hay_elo:
        raise ValueError("se necesita al menos una fuente: scores de plantel o ratings Elo")
    if hay_plantel and hay_elo:
        delta = 0.5 * (score_a - score_b) * ELO_POR_SCORE + 0.5 * (elo_a - elo_b)
    elif hay_plantel:
        delta = (score_a - score_b) * ELO_POR_SCORE
    else:
        delta = elo_a - elo_b
    x = delta / 400
    lam_a = math.exp(a + b * x + c * (local == "A"))
    lam_b = math.exp(a - b * x + c * (local == "B"))
    acotar = lambda v: min(max(v, 0.15), 4.5)  # noqa: E731
    return acotar(lam_a), acotar(lam_b)


def poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def probabilidades(lam_a: float, lam_b: float) -> tuple[float, float, float, tuple[int, int]]:
    """Devuelve (p_gana_a, p_empate, p_gana_b, resultado_mas_probable) sumando la grilla Poisson."""
    p_a = p_emp = p_b = 0.0
    mejor_resultado = (0, 0)
    p_mejor = 0.0
    for goles_a in range(MAX_GOLES + 1):
        for goles_b in range(MAX_GOLES + 1):
            p = poisson(goles_a, lam_a) * poisson(goles_b, lam_b)
            if goles_a > goles_b:
                p_a += p
            elif goles_a == goles_b:
                p_emp += p
            else:
                p_b += p
            if p > p_mejor:
                p_mejor, mejor_resultado = p, (goles_a, goles_b)
    # Normalizar la masa de probabilidad que queda fuera de la grilla
    total = p_a + p_emp + p_b
    return p_a / total, p_emp / total, p_b / total, mejor_resultado


def formato_valor(valor: int | None) -> str:
    if not valor:
        return "—"
    if valor >= 1_000_000:
        return f"€{valor / 1_000_000:.1f}M"
    return f"€{valor / 1_000:.0f}k"


def mostrar_equipo(nombre: str, plantel: list[dict], score: float, top: int = 5) -> None:
    valor_total = sum(j.get("marketValue") or 0 for j in plantel)
    print(f"\n{nombre}  —  score: {score:.3f}  |  plantel: {len(plantel)} jugadores  |  valor total: {formato_valor(valor_total)}")
    destacados = sorted(plantel, key=score_jugador, reverse=True)[:top]
    for j in destacados:
        edad = j.get("age")
        print(
            f"   {j['name']:<28} {j.get('position', '?'):<18} "
            f"{edad if edad is not None else '?':>3}  {formato_valor(j.get('marketValue')):>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara dos selecciones/equipos con datos de Transfermarkt.")
    parser.add_argument("equipo_a", help="Nombre del primer equipo (ej. 'Iraq')")
    parser.add_argument("equipo_b", help="Nombre del segundo equipo (ej. 'Uzbekistan')")
    parser.add_argument("--local", choices=["A", "B"], help="Qué equipo juega de local (omitir si es cancha neutral)")
    parser.add_argument("--elo-a", type=float, help="Rating Elo del equipo A (si se omite, se busca en data/elo_ratings.csv)")
    parser.add_argument("--elo-b", type=float, help="Rating Elo del equipo B (si se omite, se busca en data/elo_ratings.csv)")
    parser.add_argument("--solo-elo", action="store_true",
                        help="No consultar Transfermarkt: predecir solo con el Elo histórico (funciona sin internet)")
    args = parser.parse_args()

    tabla_elo = cargar_elo()
    if args.solo_elo:
        equipo_a, equipo_b = {"name": args.equipo_a}, {"name": args.equipo_b}
        sa = sb = None
    else:
        print(f"API: {API_BASE}")
        print(f"Buscando '{args.equipo_a}'...")
        equipo_a = buscar_equipo(args.equipo_a)
        print(f"Buscando '{args.equipo_b}'...")
        equipo_b = buscar_equipo(args.equipo_b)
        plantel_a = obtener_plantel(equipo_a["id"])
        plantel_b = obtener_plantel(equipo_b["id"])
        sa = score_equipo(plantel_a)
        sb = score_equipo(plantel_b)

    elo_a = args.elo_a if args.elo_a is not None else buscar_elo(tabla_elo, equipo_a["name"], args.equipo_a)
    elo_b = args.elo_b if args.elo_b is not None else buscar_elo(tabla_elo, equipo_b["name"], args.equipo_b)
    if elo_a is None or elo_b is None:
        faltante = args.equipo_a if elo_a is None else args.equipo_b
        if args.solo_elo:
            sys.exit(f"No encontré rating Elo para '{faltante}' en data/elo_ratings.csv "
                     "(probá el nombre en inglés, ej. 'Iraq') y --solo-elo no tiene otra fuente.")
        print(f"\n(No encontré rating Elo para '{faltante}'; uso solo valor de mercado del plantel)")
        elo_a = elo_b = None

    if args.solo_elo:
        print(f"\n{equipo_a['name']}: Elo {elo_a:.0f}  |  {equipo_b['name']}: Elo {elo_b:.0f}")
    else:
        mostrar_equipo(equipo_a["name"], plantel_a, sa)
        mostrar_equipo(equipo_b["name"], plantel_b, sb)

    lam_a, lam_b = lambdas_esperados(sa, sb, local=args.local, elo_a=elo_a, elo_b=elo_b)
    p_a, p_emp, p_b, (g_a, g_b) = probabilidades(lam_a, lam_b)

    print(f"\n{'=' * 62}")
    if args.local:
        print(f"  Localía: equipo {args.local} juega de local")
    if elo_a is not None:
        mezcla = "solo Elo, sin datos de plantel" if args.solo_elo else "mezclado 50/50 con valor de plantel"
        print(f"  Elo histórico: {elo_a:.0f} vs {elo_b:.0f} ({mezcla})")
    print(f"  Goles esperados: {equipo_a['name']} {lam_a:.2f}  —  {lam_b:.2f} {equipo_b['name']}")
    print(f"  Resultado más probable: {g_a}-{g_b}")
    print(f"  {equipo_a['name']} {p_a:6.1%}  |  Empate {p_emp:6.1%}  |  {equipo_b['name']} {p_b:6.1%}")
    print(f"{'=' * 62}")
    if args.solo_elo:
        print("\nNota: predicción solo con Elo histórico (sin plantel de Transfermarkt).")
    else:
        print("\nNota: el score se basa en valor de mercado + edad del plantel.")
    print("No considera forma reciente ni convocatoria real. Ver docs/ANALISIS.md.")


if __name__ == "__main__":
    main()
