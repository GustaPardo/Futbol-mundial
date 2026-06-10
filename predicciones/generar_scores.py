#!/usr/bin/env python3
"""Genera data/scores_plantel.csv: score de plantel (Transfermarkt) de los 48 mundialistas.

Consulta la API de transfermarkt-api para cada selección clasificada al Mundial 2026
(la lista sale del fixture en data/results.csv) y calcula su score de plantel con la
misma fórmula del predictor (valor de mercado + edad + profundidad).

Con ese archivo presente, simular_mundial.py mezcla automáticamente 50/50 el valor
de plantel con el Elo histórico, igual que el predictor de partidos.

Uso (necesita acceso a Transfermarkt; tarda ~3 min por el rate limit):
    TM_API_URL=https://transfermarkt-api.fly.dev python generar_scores.py
"""

import csv
import os

from predictor import API_BASE, _get, obtener_plantel, score_equipo
from simular_mundial import DATA_DIR, cargar, detectar_grupos

# Nombres del dataset histórico → nombre que entiende el buscador de Transfermarkt
ALIAS = {
    "Ivory Coast": "Cote d'Ivoire",
    "Curaçao": "Curacao",
    "United States": "United States",
    "South Korea": "South Korea",
}

# Palabras en el campo `status` del plantel de Transfermarkt que indican lesión
PALABRAS_LESION = ("injur", "tear", "ruptur", "surgery", "cruciate", "torn",
                   "fracture", "broken", "strain", "rehab", "ill", "problems")


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


def main() -> None:
    fixture, _, _ = cargar()
    equipos = sorted({e for g in detectar_grupos(fixture) for e in g})
    print(f"API: {API_BASE}")
    print(f"Generando scores de plantel para {len(equipos)} selecciones...\n")

    filas = []
    errores = []
    for nombre in equipos:
        consulta = ALIAS.get(nombre, nombre)
        try:
            resultados = _get(f"/clubs/search/{consulta}").get("results", [])
            if not resultados:
                raise ValueError("sin resultados en la búsqueda")
            club = resultados[0]
            plantel = obtener_plantel(club["id"])
            if not plantel:
                raise ValueError(f"plantel vacío para id={club['id']}")
            aptos, lesionados = descontar_lesionados(plantel)
            score = score_equipo(aptos)
            valor_total = sum(j.get("marketValue") or 0 for j in aptos)
            filas.append([nombre, round(score, 4), valor_total, len(aptos), club["name"], club["id"]])
            nota_lesion = f", {len(lesionados)} lesionados afuera: {', '.join(lesionados)}" if lesionados else ""
            print(f"  ✔ {nombre:<22} score {score:.3f}  ({club['name']}, {len(aptos)} aptos{nota_lesion})")
        except Exception as e:  # noqa: BLE001 — seguir con el resto y reportar al final
            errores.append((nombre, str(e)))
            print(f"  ✘ {nombre:<22} ERROR: {e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    ruta = os.path.join(DATA_DIR, "scores_plantel.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team", "score", "market_value_total", "players", "tm_name", "tm_id"])
        writer.writerows(filas)

    print(f"\nGuardado {len(filas)}/{len(equipos)} en {ruta}")
    if errores:
        print("Equipos sin score (el simulador les imputa uno desde su Elo):")
        for nombre, e in errores:
            print(f"   {nombre}: {e}")
    print("\nRevisá la columna tm_name: si la búsqueda eligió un club equivocado,")
    print("corregí el alias en generar_scores.py y volvé a correr.")


if __name__ == "__main__":
    main()
