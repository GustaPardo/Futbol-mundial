#!/usr/bin/env python3
"""Calibra el modelo de predicción con resultados históricos reales.

Hace tres cosas:
1. Descarga el histórico de partidos internacionales (martj42/international_results,
   ~49k partidos desde 1872).
2. Calcula el rating Elo de cada selección con la fórmula de eloratings.net
   (K según importancia del torneo, multiplicador por diferencia de goles,
   +100 de localía).
3. Ajusta por máxima verosimilitud un modelo Poisson de goles:
       log(goles_esperados) = a + b·(diff_elo/400) + c·es_local
   y valida sobre los partidos más recientes.

Genera:
    data/elo_ratings.csv   — rating Elo actual de cada selección
    data/calibracion.json  — constantes a, b, c para predictor.py

Uso:
    python calibrar.py
"""

import csv
import json
import math
import os
import urllib.request
from datetime import date

URL_RESULTADOS = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DESDE_AJUSTE = "2010-01-01"   # partidos usados para ajustar el modelo de goles
DESDE_VALIDACION = "2023-01-01"  # partidos reservados para validar
ELO_INICIAL = 1500


def k_torneo(torneo: str) -> int:
    """Peso del partido según la importancia del torneo (fórmula eloratings.net)."""
    t = torneo.lower()
    if "fifa world cup" in t and "qualification" not in t:
        return 60
    if any(c in t for c in ("euro", "copa américa", "copa america", "asian cup", "african cup",
                            "africa cup", "gold cup", "confederations")):
        return 50 if "qualification" not in t else 40
    if "qualification" in t or "nations league" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30


def multiplicador_goles(diff: int) -> float:
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return 1.75 + (diff - 3) / 8


def cargar_partidos() -> list[dict]:
    destino = os.path.join(DATA_DIR, "results.csv")
    if not os.path.exists(destino):
        print(f"Descargando histórico de partidos de {URL_RESULTADOS}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(URL_RESULTADOS, destino)
    with open(destino, newline="", encoding="utf-8") as f:
        partidos = [p for p in csv.DictReader(f) if p["home_score"] not in ("", "NA")]
    print(f"Partidos con resultado: {len(partidos)}")
    return partidos


def calcular_elo(partidos: list[dict]) -> tuple[dict[str, float], dict[str, int]]:
    """Recorre el histórico en orden y devuelve (rating por equipo, partidos jugados)."""
    elo: dict[str, float] = {}
    jugados: dict[str, int] = {}
    for p in partidos:
        local, visita = p["home_team"], p["away_team"]
        gl, gv = int(p["home_score"]), int(p["away_score"])
        el = elo.get(local, ELO_INICIAL)
        ev = elo.get(visita, ELO_INICIAL)
        ventaja = 0 if p["neutral"] == "TRUE" else 100
        esperado_local = 1 / (1 + 10 ** (-(el + ventaja - ev) / 400))
        resultado = 1.0 if gl > gv else 0.5 if gl == gv else 0.0
        cambio = k_torneo(p["tournament"]) * multiplicador_goles(abs(gl - gv)) * (resultado - esperado_local)
        elo[local] = el + cambio
        elo[visita] = ev - cambio
        jugados[local] = jugados.get(local, 0) + 1
        jugados[visita] = jugados.get(visita, 0) + 1
    return elo, jugados


def filas_entrenamiento(partidos: list[dict], desde: str, hasta: str) -> list[tuple[int, float, int]]:
    """Genera (goles, diff_elo/400, es_local) por equipo y partido, recalculando Elo en línea."""
    elo: dict[str, float] = {}
    jugados: dict[str, int] = {}
    filas = []
    for p in partidos:
        local, visita = p["home_team"], p["away_team"]
        gl, gv = int(p["home_score"]), int(p["away_score"])
        el = elo.get(local, ELO_INICIAL)
        ev = elo.get(visita, ELO_INICIAL)
        # Usar el partido para entrenar solo si ambos equipos tienen Elo estable
        if desde <= p["date"] < hasta and jugados.get(local, 0) >= 30 and jugados.get(visita, 0) >= 30:
            es_neutral = p["neutral"] == "TRUE"
            filas.append((gl, (el - ev) / 400, 0 if es_neutral else 1))
            filas.append((gv, (ev - el) / 400, 0))
        ventaja = 0 if p["neutral"] == "TRUE" else 100
        esperado_local = 1 / (1 + 10 ** (-(el + ventaja - ev) / 400))
        resultado = 1.0 if gl > gv else 0.5 if gl == gv else 0.0
        cambio = k_torneo(p["tournament"]) * multiplicador_goles(abs(gl - gv)) * (resultado - esperado_local)
        elo[local] = el + cambio
        elo[visita] = ev - cambio
        jugados[local] = jugados.get(local, 0) + 1
        jugados[visita] = jugados.get(visita, 0) + 1
    return filas


def ajustar_poisson(filas: list[tuple[int, float, int]]) -> tuple[float, float, float]:
    """Máxima verosimilitud por ascenso de gradiente: log λ = a + b·x + c·local."""
    a, b, c = math.log(1.3), 1.0, 0.2
    n = len(filas)
    lr = 0.3
    for _ in range(3000):
        ga = gb = gc = 0.0
        for goles, x, es_local in filas:
            lam = math.exp(min(a + b * x + c * es_local, 2.5))
            resto = goles - lam
            ga += resto
            gb += resto * x
            gc += resto * es_local
        a += lr * ga / n
        b += lr * gb / n
        c += lr * gc / n
    return a, b, c


def poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def validar(filas_partido: list[tuple[int, int, float, int]], a: float, b: float, c: float) -> None:
    """Reporta acierto y log-loss del modelo sobre los partidos de validación."""
    aciertos = 0
    log_loss = 0.0
    aciertos_naive = 0
    for gl, gv, x, es_local in filas_partido:
        lam_l = math.exp(a + b * x + c * es_local)
        lam_v = math.exp(a - b * x)
        p_l = p_e = p_v = 0.0
        for i in range(11):
            for j in range(11):
                p = poisson(i, lam_l) * poisson(j, lam_v)
                if i > j:
                    p_l += p
                elif i == j:
                    p_e += p
                else:
                    p_v += p
        total = p_l + p_e + p_v
        p_l, p_e, p_v = p_l / total, p_e / total, p_v / total
        real = "L" if gl > gv else "E" if gl == gv else "V"
        pred = max((p_l, "L"), (p_e, "E"), (p_v, "V"))[1]
        aciertos += pred == real
        aciertos_naive += real == "L"  # baseline: siempre gana el local/equipo A
        log_loss -= math.log(max({"L": p_l, "E": p_e, "V": p_v}[real], 1e-10))
    n = len(filas_partido)
    print(f"\nValidación sobre {n} partidos desde {DESDE_VALIDACION}:")
    print(f"  Acierto del modelo:    {aciertos / n:.1%}")
    print(f"  Acierto baseline (siempre local): {aciertos_naive / n:.1%}")
    print(f"  Log-loss promedio:     {log_loss / n:.4f} (azar uniforme = {math.log(3):.4f})")


def filas_validacion(partidos: list[dict]) -> list[tuple[int, int, float, int]]:
    elo: dict[str, float] = {}
    jugados: dict[str, int] = {}
    filas = []
    for p in partidos:
        local, visita = p["home_team"], p["away_team"]
        gl, gv = int(p["home_score"]), int(p["away_score"])
        el = elo.get(local, ELO_INICIAL)
        ev = elo.get(visita, ELO_INICIAL)
        if p["date"] >= DESDE_VALIDACION and jugados.get(local, 0) >= 30 and jugados.get(visita, 0) >= 30:
            es_local = 0 if p["neutral"] == "TRUE" else 1
            filas.append((gl, gv, (el - ev) / 400, es_local))
        ventaja = 0 if p["neutral"] == "TRUE" else 100
        esperado_local = 1 / (1 + 10 ** (-(el + ventaja - ev) / 400))
        resultado = 1.0 if gl > gv else 0.5 if gl == gv else 0.0
        cambio = k_torneo(p["tournament"]) * multiplicador_goles(abs(gl - gv)) * (resultado - esperado_local)
        elo[local] = el + cambio
        elo[visita] = ev - cambio
        jugados[local] = jugados.get(local, 0) + 1
        jugados[visita] = jugados.get(visita, 0) + 1
    return filas


def main() -> None:
    partidos = cargar_partidos()
    partidos.sort(key=lambda p: p["date"])

    print("Calculando ratings Elo...")
    elo, jugados = calcular_elo(partidos)

    # Guardar solo selecciones activas (con partidos recientes)
    activos_recientes = {p[equipo] for p in partidos if p["date"] >= "2023-01-01" for equipo in ("home_team", "away_team")}
    os.makedirs(DATA_DIR, exist_ok=True)
    ruta_elo = os.path.join(DATA_DIR, "elo_ratings.csv")
    with open(ruta_elo, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team", "elo", "matches"])
        for equipo in sorted(activos_recientes, key=lambda e: -elo.get(e, 0)):
            if jugados.get(equipo, 0) >= 30:
                writer.writerow([equipo, round(elo[equipo], 1), jugados[equipo]])
    print(f"Elo guardado en {ruta_elo}")
    top = sorted(((elo[e], e) for e in activos_recientes if jugados.get(e, 0) >= 30), reverse=True)[:8]
    for rating, equipo in top:
        print(f"   {equipo:<16} {rating:7.1f}")

    print(f"\nAjustando modelo Poisson (partidos {DESDE_AJUSTE} → {DESDE_VALIDACION})...")
    filas = filas_entrenamiento(partidos, DESDE_AJUSTE, DESDE_VALIDACION)
    print(f"   {len(filas) // 2} partidos de entrenamiento")
    a, b, c = ajustar_poisson(filas)
    print(f"   a={a:.4f}  b={b:.4f}  c={c:.4f}")
    print(f"   → goles base (neutral, parejo): {math.exp(a):.2f}")
    print(f"   → localía: ×{math.exp(c):.2f}")
    print(f"   → 100 puntos Elo de diff: ×{math.exp(b * 0.25):.2f} de goles")

    validar(filas_validacion(partidos), a, b, c)

    ruta_calib = os.path.join(DATA_DIR, "calibracion.json")
    with open(ruta_calib, "w", encoding="utf-8") as f:
        json.dump(
            {
                "a": round(a, 4),
                "b": round(b, 4),
                "c": round(c, 4),
                "entrenado_desde": DESDE_AJUSTE,
                "entrenado_hasta": DESDE_VALIDACION,
                "n_filas": len(filas),
                "generado": date.today().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"\nCalibración guardada en {ruta_calib}")


if __name__ == "__main__":
    main()
