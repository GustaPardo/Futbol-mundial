#!/usr/bin/env python3
"""Diagnóstico integral del pipeline de datos: API, log del servidor y XPaths en vivo.

Corre las cuatro pruebas y muestra TODO lo necesario para arreglar cualquier
problema de scraping en una sola pasada:

1. Endpoints de la API local (planteles y stats).
2. Últimos errores del servidor leyendo /tmp/api.log automáticamente.
3. Los XPath del parser de stats contra la página real de Transfermarkt
   (cuántas filas matchea cada uno, título de la página por si hay redirect).
4. El plantel de Algeria (el que da 500) parseado en proceso, marcando qué
   campo obligatorio viene vacío.

Uso:  python diagnostico.py
"""

import json
import os
import sys
import traceback

import requests

API = os.environ.get("TM_API_URL", "http://localhost:8000").rstrip("/")
RAIZ_API = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "transfermarkt-api"))
sys.path.insert(0, RAIZ_API)


def seccion(titulo: str) -> None:
    print(f"\n{'=' * 64}\n{titulo}\n{'=' * 64}")


def prueba_endpoints() -> None:
    seccion("1. Endpoints de la API")
    casos = [
        ("Plantel Argentina", "/clubs/3437/players", "players"),
        ("Plantel Algeria", "/clubs/3614/players", "players"),
        ("Stats Messi", "/players/28003/stats", "stats"),
        ("Stats Haaland", "/players/418560/stats", "stats"),
    ]
    for nombre, path, clave in casos:
        try:
            r = requests.get(API + path, timeout=60)
            extra = ""
            if r.status_code == 200:
                items = r.json().get(clave) or []
                extra = f" | {len(items)} {clave}"
                if items:
                    extra += f" | claves: {sorted(items[0].keys())[:8]}"
            print(f"  {nombre:<18} HTTP {r.status_code}{extra}")
            if r.status_code != 200:
                print(f"     cuerpo: {r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            print(f"  {nombre:<18} EXCEPCIÓN: {type(e).__name__}: {e}")


def leer_log_servidor() -> None:
    seccion("2. Últimos errores del servidor (/tmp/api.log)")
    try:
        lineas = open("/tmp/api.log", encoding="utf-8", errors="replace").read().splitlines()
    except FileNotFoundError:
        print("   (no existe /tmp/api.log — ¿la API corre en esta máquina?)")
        return
    interesantes = [
        ln for ln in lineas
        if any(k in ln for k in ("Error", "Traceback", "validation error", "raise ", 'File "', "Exception"))
    ]
    if not interesantes:
        print("   (sin errores en el log)")
    for ln in interesantes[-30:]:
        print("  ", ln[:170])


def prueba_xpaths_stats() -> None:
    seccion("3. XPaths del parser de stats vs la página real (Messi, id 28003)")
    try:
        from app.services.base import TransfermarktBase
        from app.utils.xpath import Players

        base = TransfermarktBase(URL="https://www.transfermarkt.com/-/leistungsdatendetails/spieler/28003")
        base.page = base.request_url_page()
        titulo = base.page.xpath("//title/text()")
        print("  título de la página:", (titulo[0][:90] if titulo else "(SIN TÍTULO: posible redirect/captcha)"))
        pruebas = {
            "Profile.URL (validación)": Players.Profile.URL,
            "Stats.ROWS": Players.Stats.ROWS,
            "Stats.HEADERS": Players.Stats.HEADERS,
            "Stats.COMPETITIONS_URLS": Players.Stats.COMPETITIONS_URLS,
            "Stats.CLUBS_URLS": Players.Stats.CLUBS_URLS,
        }
        for nombre, xp in pruebas.items():
            try:
                res = base.page.xpath(xp)
                muestra = ""
                if res and isinstance(res[0], str):
                    muestra = f" | ej: {res[0][:60]!r}"
                print(f"  {nombre:<26} {len(res)} matches{muestra}")
            except Exception as e:  # noqa: BLE001
                print(f"  {nombre:<26} ERROR: {e}")

        from app.services.players.stats import TransfermarktPlayerStats
        filas = TransfermarktPlayerStats(player_id="28003").get_player_stats().get("stats", [])
        print(f"  parser completo: {len(filas)} filas")
        if filas:
            print("  primera fila:", json.dumps(filas[0], ensure_ascii=False)[:250])
    except Exception:
        traceback.print_exc()


def prueba_plantel_algeria() -> None:
    seccion("4. Plantel de Algeria (id 3614) parseado en proceso")
    try:
        from app.services.clubs.players import TransfermarktClubPlayers

        jugadores = TransfermarktClubPlayers(club_id="3614").get_club_players().get("players", [])
        print(f"  jugadores parseados: {len(jugadores)}")
        problemas = [
            (i, {k: j.get(k) for k in ("id", "name", "marketValue")})
            for i, j in enumerate(jugadores)
            if not j.get("id") or not j.get("name")
        ]
        if problemas:
            print("  filas con id/name vacío (esto rompe la validación del schema):")
            for i, datos in problemas[:8]:
                print(f"     fila {i}: {datos}")
        elif jugadores:
            print("  primera fila:", json.dumps(jugadores[0], ensure_ascii=False, default=str)[:250])
            print("  (el parseo anda; si la API igual da 500, el detalle está en la sección 2)")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    print(f"API: {API}")
    prueba_endpoints()
    leer_log_servidor()
    prueba_xpaths_stats()
    prueba_plantel_algeria()
    print("\nFIN — pegá TODA esta salida en el chat para el arreglo definitivo.")
