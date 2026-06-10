#!/usr/bin/env python3
"""Predictor de partidos entre selecciones usando datos de Transfermarkt.

Usa la API de transfermarkt-api (carpeta ../transfermarkt-api de este repo)
para obtener el plantel de cada equipo y calcular un score basado en valor
de mercado, edad y profundidad del plantel.

Uso:
    python predictor.py "Iraq" "Uzbekistan"
    TM_API_URL=https://transfermarkt-api.fly.dev python predictor.py "Iraq" "Jordan"

La metodología está documentada en docs/ANALISIS.md.
"""

import argparse
import math
import os
import sys
import time

import requests

API_BASE = os.environ.get("TM_API_URL", "http://localhost:8000").rstrip("/")
# Pausa entre requests para no golpear el rate limit de la instancia pública
PAUSA_SEGUNDOS = float(os.environ.get("TM_API_PAUSA", "1.5" if "fly.dev" in API_BASE else "0"))
ESCALA_ELO = 0.5


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


def probabilidades(score_a: float, score_b: float) -> tuple[float, float, float]:
    """Devuelve (p_gana_a, p_empate, p_gana_b) con un modelo tipo Elo."""
    diff = score_a - score_b
    esperanza_a = 1 / (1 + 10 ** (-diff / ESCALA_ELO))
    p_empate = 0.30 * math.exp(-abs(diff))
    p_a = esperanza_a * (1 - p_empate)
    p_b = (1 - esperanza_a) * (1 - p_empate)
    return p_a, p_empate, p_b


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
    args = parser.parse_args()

    print(f"API: {API_BASE}")
    print(f"Buscando '{args.equipo_a}'...")
    equipo_a = buscar_equipo(args.equipo_a)
    print(f"Buscando '{args.equipo_b}'...")
    equipo_b = buscar_equipo(args.equipo_b)

    plantel_a = obtener_plantel(equipo_a["id"])
    plantel_b = obtener_plantel(equipo_b["id"])

    sa = score_equipo(plantel_a)
    sb = score_equipo(plantel_b)

    mostrar_equipo(equipo_a["name"], plantel_a, sa)
    mostrar_equipo(equipo_b["name"], plantel_b, sb)

    p_a, p_emp, p_b = probabilidades(sa, sb)
    print(f"\n{'=' * 60}")
    print(f"  {equipo_a['name']} {p_a:6.1%}  |  Empate {p_emp:6.1%}  |  {equipo_b['name']} {p_b:6.1%}")
    print(f"{'=' * 60}")
    print("\nNota: score basado en valor de mercado + edad. No considera forma")
    print("reciente, localía ni convocatoria real. Ver docs/ANALISIS.md.")


if __name__ == "__main__":
    main()
