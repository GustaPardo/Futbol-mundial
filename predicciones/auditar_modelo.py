#!/usr/bin/env python3
"""Auditoría matemática del modelo: calibración out-of-sample y peso óptimo de la forma.

Responde dos preguntas con datos, no con intuición:

1. ¿Cuánto debe pesar la "forma reciente" (momentum)? Se ajusta la escala y el
   decaimiento del bonus de forma en una ventana de TUNING (2016-2022) y se
   evalúa UNA sola vez en una ventana de TEST (2023+) que el modelo nunca vio.
   Si la forma no mejora el log-loss, el peso correcto es ~0, nos guste o no.

2. ¿El modelo es sobreconfiado? Tabla de confiabilidad por deciles: cuando el
   modelo dice "el local gana con probabilidad p", ¿gana de verdad una fracción
   p de las veces? También se prueba un factor de nitidez t sobre el coeficiente
   b (t<1 = achatar probabilidades si hay sobreconfianza).

Escribe los parámetros elegidos en data/calibracion.json (forma_escala,
forma_decay, nitidez) para que simular_mundial.py y analizar.py los usen.

Uso:
    python auditar_modelo.py
"""

import csv
import json
import math
import os

from calibrar import ELO_INICIAL, k_torneo, multiplicador_goles

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TUNING = ("2016-01-01", "2023-01-01")   # ventana para elegir hiperparámetros
TEST = ("2023-01-01", "2026-07-01")     # ventana de evaluación final (nunca usada para elegir)
MIN_PARTIDOS_ELO = 30                    # partidos previos para considerar el Elo estable
MAX_FORMA = 20                           # cuántos partidos recientes guarda la forma
FACT = [math.factorial(k) for k in range(11)]


def poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / FACT[k]


def prob_1x2(x: float, es_local: int, a: float, b: float, c: float) -> tuple[float, float, float]:
    lam_l = math.exp(a + b * x + c * es_local)
    lam_v = math.exp(a - b * x)
    lam_l, lam_v = min(max(lam_l, 0.1), 5.0), min(max(lam_v, 0.1), 5.0)
    pl = pe = pv = 0.0
    pml = [poisson(i, lam_l) for i in range(11)]
    pmv = [poisson(j, lam_v) for j in range(11)]
    for i in range(11):
        for j in range(11):
            p = pml[i] * pmv[j]
            if i > j:
                pl += p
            elif i == j:
                pe += p
            else:
                pv += p
    t = pl + pe + pv
    return pl / t, pe / t, pv / t


def construir_dataset() -> tuple[list, list]:
    """Replay del histórico. Para cada partido de tuning/test guarda el estado
    PREVIO al partido (Elo e historial de forma de ambos equipos): sin fuga de datos."""
    with open(os.path.join(DATA_DIR, "results.csv"), newline="", encoding="utf-8") as f:
        partidos = [p for p in csv.DictReader(f) if p["home_score"] not in ("", "NA")]
    partidos.sort(key=lambda p: p["date"])

    elo: dict[str, float] = {}
    jugados: dict[str, int] = {}
    historial: dict[str, list] = {}  # por equipo: [(res-esp), ...] más reciente primero, oficiales
    tuning, test = [], []
    for p in partidos:
        local, visita = p["home_team"], p["away_team"]
        gl, gv = int(p["home_score"]), int(p["away_score"])
        el, ev = elo.get(local, ELO_INICIAL), elo.get(visita, ELO_INICIAL)
        es_neutral = p["neutral"] == "TRUE"
        ventaja = 0 if es_neutral else 100
        esperado_l = 1 / (1 + 10 ** (-(el + ventaja - ev) / 400))
        resultado_l = 1.0 if gl > gv else 0.5 if gl == gv else 0.0

        if jugados.get(local, 0) >= MIN_PARTIDOS_ELO and jugados.get(visita, 0) >= MIN_PARTIDOS_ELO:
            fila = (gl, gv, el, ev, 0 if es_neutral else 1,
                    list(historial.get(local, []))[:MAX_FORMA],
                    list(historial.get(visita, []))[:MAX_FORMA])
            if TUNING[0] <= p["date"] < TUNING[1]:
                tuning.append(fila)
            elif TEST[0] <= p["date"] < TEST[1]:
                test.append(fila)

        if "friendly" not in p["tournament"].lower():
            historial.setdefault(local, []).insert(0, resultado_l - esperado_l)
            historial.setdefault(visita, []).insert(0, (1 - resultado_l) - (1 - esperado_l))
            historial[local] = historial[local][:MAX_FORMA]
            historial[visita] = historial[visita][:MAX_FORMA]

        cambio = k_torneo(p["tournament"]) * multiplicador_goles(abs(gl - gv)) * (resultado_l - esperado_l)
        elo[local] = el + cambio
        elo[visita] = ev - cambio
        jugados[local] = jugados.get(local, 0) + 1
        jugados[visita] = jugados.get(visita, 0) + 1
    return tuning, test


def forma_ponderada(hist: list, decay: float) -> float:
    if len(hist) < 4:
        return 0.0
    pesos = [decay**i for i in range(len(hist))]
    return sum(w * v for w, v in zip(pesos, hist)) / sum(pesos)


def evaluar(dataset: list, decay: float, escala: float, nitidez: float, calib: dict) -> dict:
    a, b, c = calib["a"], calib["b"] * nitidez, calib["c"]
    log_loss = brier = aciertos = 0.0
    for gl, gv, el, ev, es_local, hl, hv in dataset:
        delta = (el + escala * forma_ponderada(hl, decay)) - (ev + escala * forma_ponderada(hv, decay))
        pl, pe, pv = prob_1x2(delta / 400, es_local, a, b, c)
        real = "L" if gl > gv else "E" if gl == gv else "V"
        probs = {"L": pl, "E": pe, "V": pv}
        log_loss -= math.log(max(probs[real], 1e-12))
        brier += sum((probs[r] - (r == real)) ** 2 for r in "LEV")
        aciertos += max(probs, key=probs.get) == real
    n = len(dataset)
    return {"log_loss": log_loss / n, "brier": brier / n, "acierto": aciertos / n}


def confiabilidad(dataset: list, decay: float, escala: float, nitidez: float, calib: dict) -> None:
    """¿Cuando el modelo dice p, pasa una fracción p de las veces? (prob. de victoria del equipo A)"""
    a, b, c = calib["a"], calib["b"] * nitidez, calib["c"]
    cubos = [[0.0, 0, 0] for _ in range(10)]  # [suma_prob, victorias, n]
    for gl, gv, el, ev, es_local, hl, hv in dataset:
        delta = (el + escala * forma_ponderada(hl, decay)) - (ev + escala * forma_ponderada(hv, decay))
        pl, _, _ = prob_1x2(delta / 400, es_local, a, b, c)
        i = min(int(pl * 10), 9)
        cubos[i][0] += pl
        cubos[i][1] += gl > gv
        cubos[i][2] += 1
    print(f"   {'Predicho':>9} {'Observado':>10} {'N':>6}")
    for suma, wins, n in cubos:
        if n >= 30:
            print(f"   {suma/n:>8.1%} {wins/n:>9.1%} {n:>6}")


def main() -> None:
    with open(os.path.join(DATA_DIR, "calibracion.json"), encoding="utf-8") as f:
        calib = json.load(f)

    print("Construyendo dataset sin fuga de datos (replay cronológico)...")
    tuning, test = construir_dataset()
    print(f"   tuning: {len(tuning)} partidos ({TUNING[0]}–{TUNING[1]})")
    print(f"   test:   {len(test)} partidos ({TEST[0]}+), nunca usados para elegir parámetros\n")

    base_tun = evaluar(tuning, 1.0, 0.0, 1.0, calib)
    print(f"Baseline (sin forma) en tuning: log-loss {base_tun['log_loss']:.4f}\n")

    print("Grid search del peso de la forma (en tuning):")
    print(f"   {'decay':>6} {'escala':>7} {'log-loss':>9} {'Δ vs base':>10}")
    mejor = (base_tun["log_loss"], 1.0, 0.0)
    for decay in (0.80, 0.88, 0.95, 1.00):
        for escala in (40, 80, 120, 160, 240, 320):
            r = evaluar(tuning, decay, escala, 1.0, calib)
            marca = " <-- mejor" if r["log_loss"] < mejor[0] else ""
            if r["log_loss"] < mejor[0]:
                mejor = (r["log_loss"], decay, escala)
            print(f"   {decay:>6.2f} {escala:>7} {r['log_loss']:>9.4f} {r['log_loss']-base_tun['log_loss']:>+10.4f}{marca}")
    _, decay_opt, escala_opt = mejor
    print(f"\nElegido en tuning: decay={decay_opt}, escala={escala_opt}")

    print("\nFactor de nitidez t sobre b (¿sobreconfianza?), en tuning con la forma elegida:")
    mejor_t = (math.inf, 1.0)
    for t in (0.85, 0.90, 0.95, 1.00, 1.05, 1.10):
        r = evaluar(tuning, decay_opt, escala_opt, t, calib)
        marca = ""
        if r["log_loss"] < mejor_t[0]:
            mejor_t = (r["log_loss"], t)
            marca = " <-- mejor"
        print(f"   t={t:.2f}  log-loss {r['log_loss']:.4f}{marca}")
    t_opt = mejor_t[1]

    print(f"\n{'='*64}\nEVALUACIÓN FINAL EN TEST (2023+, una sola vez):")
    base = evaluar(test, 1.0, 0.0, 1.0, calib)
    final = evaluar(test, decay_opt, escala_opt, t_opt, calib)
    print(f"   {'':<26} {'log-loss':>9} {'Brier':>8} {'acierto':>8}")
    print(f"   {'Baseline (sin forma)':<26} {base['log_loss']:>9.4f} {base['brier']:>8.4f} {base['acierto']:>8.1%}")
    print(f"   {'Con parámetros elegidos':<26} {final['log_loss']:>9.4f} {final['brier']:>8.4f} {final['acierto']:>8.1%}")

    print("\nConfiabilidad en test (modelo final): cuando dice p, ¿pasa p?")
    confiabilidad(test, decay_opt, escala_opt, t_opt, calib)

    calib.update({"forma_decay": decay_opt, "forma_escala": escala_opt, "nitidez": t_opt})
    with open(os.path.join(DATA_DIR, "calibracion.json"), "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    print(f"\nParámetros guardados en data/calibracion.json: "
          f"forma_escala={escala_opt}, forma_decay={decay_opt}, nitidez={t_opt}")


if __name__ == "__main__":
    main()
