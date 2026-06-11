#!/usr/bin/env python3
"""Genera data/scores_plantel.csv: score de plantel (Transfermarkt) de los 48 mundialistas.

Consulta la API de transfermarkt-api para cada selección clasificada al Mundial 2026
(la lista sale del fixture en data/results.csv) y calcula su score de plantel con la
misma fórmula del predictor (valor de mercado + edad + profundidad), descontando
jugadores lesionados.

Modo profundo (--con-stats): además baja las estadísticas de cada jugador y ajusta
su score según el nivel de las competencias donde jugó la última temporada
(Champions League y ligas top valen más), sus minutos reales y su producción
ofensiva. Las respuestas se cachean en data/cache_stats/, así que si se corta se
puede relanzar y retoma donde quedó.

Con el archivo generado, simular_mundial.py mezcla automáticamente 50/50 el valor
de plantel con el Elo histórico, igual que el predictor de partidos.

Uso (necesita acceso a Transfermarkt):
    TM_API_URL=https://transfermarkt-api.fly.dev python generar_scores.py              # rápido (~3 min)
    TM_API_URL=https://transfermarkt-api.fly.dev python generar_scores.py --con-stats  # profundo (~30-40 min)

Con la API local (sin rate limit) el modo profundo baja a ~5 minutos.
"""

import argparse
import csv
import json
import math
import os
import sys

from predictor import API_BASE, _get, obtener_plantel, score_jugador
from simular_mundial import DATA_DIR, cargar, detectar_grupos

CACHE_STATS = os.path.join(DATA_DIR, "cache_stats")

# Nombres del dataset histórico → término que entiende el buscador de Transfermarkt
ALIAS = {
    "Ivory Coast": "Cote d'Ivoire",
    "Curaçao": "Curacao",
    "United States": "United States",
    "South Korea": "South Korea",
    "Bosnia and Herzegovina": "Bosnia",
    "DR Congo": "DR Congo",
    "Cape Verde": "Cape Verde",
}

# Selecciones cuyo buscador falla o devuelve un club equivocado: ID de Transfermarkt directo
CLUB_ID_OVERRIDE = {
    "Bosnia and Herzegovina": "3446",
    "Iran": "3582",
}


def elegir_club(nombre: str, consulta: str) -> dict:
    """Busca el club/selección y elige el resultado cuyo nombre coincide con lo buscado.

    El buscador de Transfermarkt a veces devuelve otro equipo primero (ej. 'Iran'
    devolvía Francia), así que nunca tomamos el primer resultado a ciegas.
    """
    if nombre in CLUB_ID_OVERRIDE:
        return {"id": CLUB_ID_OVERRIDE[nombre], "name": nombre}
    resultados = _get(f"/clubs/search/{consulta}").get("results", [])
    if not resultados:
        raise ValueError("sin resultados en la búsqueda")
    q = consulta.lower()
    for r in resultados:
        rn = (r.get("name") or "").lower()
        if q in rn or rn in q or rn in nombre.lower():
            return r
    print(f"     ⚠ ningún resultado de '{consulta}' coincide; opciones: "
          + ", ".join(x.get("name", "?") for x in resultados[:3]))
    raise ValueError(f"búsqueda ambigua: el primer resultado era '{resultados[0].get('name')}'")

# Palabras en el campo `status` del plantel de Transfermarkt que indican lesión
PALABRAS_LESION = ("injur", "tear", "ruptur", "surgery", "cruciate", "torn",
                   "fracture", "broken", "strain", "rehab", "ill", "problems")

# Nivel de competencia (0 a 1) por ID de Transfermarkt. La Champions y las 5
# grandes ligas valen 1; el resto escalona hacia abajo. Default conservador.
NIVEL_POR_ID = {
    "CL": 1.0, "GB1": 1.0, "ES1": 1.0, "L1": 1.0, "IT1": 1.0, "FR1": 1.0,
    "EL": 0.85, "NL1": 0.85, "PO1": 0.85, "GB2": 0.80, "BRA1": 0.80,
    "TR1": 0.75, "BE1": 0.75, "MLS1": 0.70, "SA1": 0.70, "MX1": 0.70,
    "AR1N": 0.70, "UKR1": 0.70, "RU1": 0.70, "GR1": 0.70, "A1": 0.70,
    "DK1": 0.70, "SC1": 0.70, "C1": 0.85, "CLI": 0.85,  # C1/CLI: Libertadores
    "UCOL": 0.60, "KL1": 0.55, "UAE1": 0.60, "QSL": 0.60,
}
NIVEL_POR_NOMBRE = {  # respaldo por palabra clave si el ID no está mapeado
    "champions league": 1.0, "premier league": 1.0, "laliga": 1.0,
    "serie a": 0.85, "bundesliga": 0.90, "ligue 1": 1.0,
    "europa league": 0.85, "libertadores": 0.85, "world cup": 1.0,
    "euro": 0.95, "copa américa": 0.90, "copa america": 0.90,
}
NIVEL_DEFAULT = 0.55
MINUTOS_TEMPORADA_COMPLETA = 2500  # un titular indiscutido juega ~2500-3500 min/temporada


def descontar_lesionados(plantel: list[dict]) -> tuple[list[dict], list[str]]:
    """Excluye del score a los jugadores cuyo estado en Transfermarkt indica lesión."""
    aptos, lesionados = [], []
    for j in plantel:
        estado = (j.get("status") or "").lower()
        if any(palabra in estado for palabra in PALABRAS_LESION):
            lesionados.append(j["name"])
        else:
            aptos.append(j)
    return aptos, lesionados


def _a_entero(valor) -> int:
    if valor is None:
        return 0
    if isinstance(valor, int):
        return valor
    digitos = "".join(c for c in str(valor) if c.isdigit())
    return int(digitos) if digitos else 0


def obtener_stats(player_id: str) -> list[dict]:
    """Stats por competición del jugador, con caché en disco para poder retomar."""
    ruta = os.path.join(CACHE_STATS, f"{player_id}.json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    try:
        stats = _get(f"/players/{player_id}/stats").get("stats", [])
    except Exception:  # noqa: BLE001 — no cachear errores: reintentar en la próxima corrida
        return []
    os.makedirs(CACHE_STATS, exist_ok=True)
    if stats:  # no cachear respuestas vacías: pueden ser un fallo transitorio del parser
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(stats, f)
    return stats


def nivel_competencia(fila: dict) -> float:
    comp_id = (fila.get("competitionId") or "").upper()
    if comp_id in NIVEL_POR_ID:
        return NIVEL_POR_ID[comp_id]
    nombre = (fila.get("competitionName") or "").lower()
    for clave, nivel in NIVEL_POR_NOMBRE.items():
        if clave in nombre:
            return nivel
    return NIVEL_DEFAULT


def ajuste_por_stats(stats: list[dict]) -> float:
    """Multiplicador [0.7, 1.4] según nivel de competencia, minutos y producción.

    Usa la última temporada con datos. Un titular productivo en Champions/liga top
    ronda 1.3-1.4; un suplente en una liga débil ronda 0.75. Sin datos: 1.0 neutro.
    """
    temporadas = [s.get("seasonId") for s in stats if s.get("seasonId")]
    if not temporadas:
        return 1.0
    ultima = max(temporadas, key=_a_entero)
    filas = [s for s in stats if s.get("seasonId") == ultima]
    minutos_total = sum(_a_entero(f.get("minutesPlayed")) for f in filas)
    if minutos_total == 0:
        return 0.85  # tiene página de stats pero no jugó: descuento suave
    nivel = sum(nivel_competencia(f) * _a_entero(f.get("minutesPlayed")) for f in filas) / minutos_total
    factor_nivel = 0.80 + 0.35 * nivel
    factor_rodaje = 0.85 + 0.30 * min(minutos_total / MINUTOS_TEMPORADA_COMPLETA, 1.0)
    ga = sum(_a_entero(f.get("goals")) + _a_entero(f.get("assists")) for f in filas)
    ga_por_90 = ga / (minutos_total / 90) if minutos_total >= 450 else 0.0
    factor_produccion = 1.0 + 0.10 * min(ga_por_90, 1.2)
    return min(max(factor_nivel * factor_rodaje * factor_produccion, 0.7), 1.4)


def score_equipo_desde_scores(scores: list[float]) -> float:
    """Misma fórmula 80/20 del predictor, sobre scores de jugador ya calculados."""
    scores = sorted(scores, reverse=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera scores de plantel de los 48 mundialistas.")
    parser.add_argument("--con-stats", action="store_true",
                        help="ajustar cada jugador por nivel de liga, minutos y producción (lento)")
    args = parser.parse_args()

    fixture, _, _ = cargar()
    equipos = sorted({e for g in detectar_grupos(fixture) for e in g})
    print(f"API: {API_BASE}")
    modo = "profundo (con stats por jugador)" if args.con_stats else "rápido (valor de mercado + edad)"
    print(f"Modo {modo}. Generando scores para {len(equipos)} selecciones...\n")

    if args.con_stats and not obtener_stats("28003"):  # autochequeo con Messi antes de la corrida larga
        sys.exit(
            "✘ ABORTADO: el endpoint /players/{id}/stats no devuelve datos (probado con Messi).\n"
            "  Correr la corrida larga así sería tirar 20 minutos: el ajuste daría ×1.00 en todo.\n"
            "  Diagnóstico: python diagnostico.py  (pegá la salida completa en el chat)\n"
            "  Alternativa mientras tanto: MODO = \"rapido\" (sin ajuste por nivel de liga)."
        )

    filas = []
    errores = []
    planteles: dict[str, list[dict]] = {}
    total_jugadores = total_con_stats = 0
    for nombre in equipos:
        consulta = ALIAS.get(nombre, nombre)
        try:
            club = elegir_club(nombre, consulta)
            plantel = obtener_plantel(club["id"])
            if not plantel:
                raise ValueError(f"plantel vacío para id={club['id']}")
            aptos, lesionados = descontar_lesionados(plantel)
            planteles[nombre] = [
                {"id": j["id"], "name": j["name"], "position": j.get("position"),
                 "age": j.get("age"), "marketValue": j.get("marketValue"),
                 "lesionado": j["name"] in lesionados}
                for j in plantel
            ]

            if args.con_stats:
                scores_jugadores = []
                ajustes = []
                con_stats = 0
                for j in aptos:
                    stats = obtener_stats(j["id"])
                    con_stats += bool(stats)
                    ajuste = ajuste_por_stats(stats)
                    ajustes.append(ajuste)
                    scores_jugadores.append(score_jugador(j) * ajuste)
                score = score_equipo_desde_scores(scores_jugadores)
                ajuste_medio = sum(ajustes) / len(ajustes)
                total_jugadores += len(aptos)
                total_con_stats += con_stats
                detalle = f"stats {con_stats}/{len(aptos)}, ajuste ×{ajuste_medio:.2f}"
            else:
                score = score_equipo_desde_scores([score_jugador(j) for j in aptos])
                detalle = "sin stats"

            valor_total = sum(j.get("marketValue") or 0 for j in aptos)
            filas.append([nombre, round(score, 4), valor_total, len(aptos), club["name"], club["id"]])
            nota_lesion = f", {len(lesionados)} lesionados afuera" if lesionados else ""
            print(f"  ✔ {nombre:<22} score {score:.3f}  valor €{valor_total/1e6:5.0f}M  "
                  f"({club['name']}, {len(aptos)} aptos, {detalle}{nota_lesion})")
        except Exception as e:  # noqa: BLE001 — seguir con el resto y reportar al final
            errores.append((nombre, str(e)))
            print(f"  ✘ {nombre:<22} ERROR: {e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    ruta = os.path.join(DATA_DIR, "scores_plantel.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team", "score", "market_value_total", "players", "tm_name", "tm_id"])
        writer.writerows(filas)
    with open(os.path.join(DATA_DIR, "planteles.json"), "w", encoding="utf-8") as f:
        json.dump(planteles, f, ensure_ascii=False)

    print(f"\nGuardado {len(filas)}/{len(equipos)} en {ruta} (+ planteles.json para analizar.py)")
    if args.con_stats and total_jugadores:
        pct = total_con_stats / total_jugadores
        print(f"Stats por jugador: {total_con_stats}/{total_jugadores} ({pct:.0%})")
        if pct < 0.5:
            print("⚠⚠ MENOS DE LA MITAD de los jugadores devolvió stats: el endpoint")
            print("   /players/{id}/stats está fallando. El ajuste por nivel de liga quedó")
            print("   casi neutro. Diagnóstico: corré la celda de auto-test o !tail -30 /tmp/api.log")
    if errores:
        print("Equipos sin score (el simulador les imputa uno desde su Elo):")
        for nombre, e in errores:
            print(f"   {nombre}: {e}")
        print("Si fue por rate limit o corte de red, volvé a correr: el caché retoma solo.")
    print("\nRevisá la columna tm_name: si la búsqueda eligió un club equivocado,")
    print("corregí el alias en generar_scores.py y volvé a correr.")
    print("Siguiente paso: python simular_mundial.py -n 50000")


if __name__ == "__main__":
    main()
