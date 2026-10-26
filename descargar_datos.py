#Manuel Alejandro Heredia Nogales
"""Descarga del dataset plant_leaves desde Mendeley Data

Descargamos las 4502 imágenes directamente de la fuente original:

    Chouhan, S. S., Kaul, A., Singh, U. P., Jain, S. (2019).
    A Database of Leaf Images: Practice towards Plant Conservation with
    Plant Pathology. Mendeley Data, V1. https://doi.org/10.17632/hb74ynkjcn.1
    Licencia CC BY 4.0

Las imágenes quedan en  datos/plant_leaves/<Especie>/<healthy|diseased>/XXXX_YYYY.JPG.
El prefijo XXXX del archivo es la clase (0001 ... 0022), igual que en tfds.

Se puede volver a ejecutar sin problema: las imágenes que ya están
descargadas y cuyo sha256 coincide se saltan.

Uso:
    python descargar_datos.py                     # dataset completo (6.8 GB)
    python descargar_datos.py --max-por-clase 10  # muestra pequeña para pruebas
"""

# =============================================================================
# IMPORTACIONES

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIGURACIÓN

API      = "https://data.mendeley.com/public-api/datasets/hb74ynkjcn"
VERSION  = 1   # la versión que usa tfds
# Mendeley rechaza (403) el User-Agent por defecto de urllib
HEADERS  = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) leafs_plants-TL/1.0"}
DIR_BASE = os.path.dirname(os.path.abspath(__file__))


def pedir_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def listar_imagenes():
    """Regresa la lista de imágenes: ruta relativa, url, sha256 y tamaño."""
    carpetas = pedir_json(f"{API}/folders/{VERSION}")
    por_id = {c["id"]: c for c in carpetas}

    # Las carpetas de clase son <Especie>/healthy y <Especie>/diseased
    carpetas_clase = [c for c in carpetas
                      if c["name"] in ("healthy", "diseased") and c.get("parent_id") in por_id]

    imagenes = []
    for c in carpetas_clase:
        especie = por_id[c["parent_id"]]["name"]
        archivos = pedir_json(f"{API}/files?folder_id={c['id']}&version={VERSION}")
        for a in archivos:
            if not a["filename"].endswith(".JPG"):
                continue
            imagenes.append({
                "ruta":   os.path.join(especie, c["name"], a["filename"]),
                "url":    a["content_details"]["download_url"],
                "sha256": a["content_details"]["sha256_hash"],
                "size":   a["size"],
            })
    return sorted(imagenes, key=lambda img: img["ruta"])


def sha256_de(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(img, destino, intentos=5):
    """Descarga una imagen (con reintentos) y verifica su sha256."""
    ruta = os.path.join(destino, img["ruta"])
    if os.path.exists(ruta) and os.path.getsize(ruta) == img["size"] and sha256_de(ruta) == img["sha256"]:
        return "existente"

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    temporal = ruta + ".part"
    for intento in range(1, intentos + 1):
        try:
            req = urllib.request.Request(img["url"], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r, open(temporal, "wb") as f:
                while bloque := r.read(1 << 20):
                    f.write(bloque)
            if sha256_de(temporal) != img["sha256"]:
                raise IOError("sha256 no coincide")
            os.replace(temporal, ruta)
            return "descargada"
        except Exception as e:
            if intento == intentos:
                raise RuntimeError(f"{img['ruta']}: {e}") from e
            time.sleep(2 ** intento)


# =============================================================================
# PROGRAMA PRINCIPAL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga plant_leaves desde Mendeley Data")
    parser.add_argument("--destino", default=os.path.join(DIR_BASE, "datos", "plant_leaves"))
    parser.add_argument("--hilos", type=int, default=16, help="descargas en paralelo")
    parser.add_argument("--max-por-clase", type=int, default=None,
                        help="solo N imágenes por clase (para pruebas rápidas)")
    args = parser.parse_args()

    t = time.time()
    imagenes = listar_imagenes()
    print(f"Mendeley reporta {len(imagenes)} imágenes "
          f"({sum(i['size'] for i in imagenes) / 2**30:.2f} GiB)")

    if args.max_por_clase:
        por_clase = {}
        for img in imagenes:
            por_clase.setdefault(os.path.dirname(img["ruta"]), []).append(img)
        imagenes = [img for lista in por_clase.values() for img in lista[:args.max_por_clase]]
        print(f"Modo prueba: {len(imagenes)} imágenes ({args.max_por_clase} por clase)")

    conteo = {"descargada": 0, "existente": 0}
    errores = []
    with ThreadPoolExecutor(args.hilos) as ex:
        futuros = [ex.submit(descargar, img, args.destino) for img in imagenes]
        for n, futuro in enumerate(as_completed(futuros), 1):
            try:
                conteo[futuro.result()] += 1
            except RuntimeError as e:
                errores.append(str(e))
            if n % 250 == 0 or n == len(futuros):
                print(f"  {n}/{len(futuros)}  ({(time.time() - t)/60:.1f} min)", flush=True)

    print(f"\nDescargadas: {conteo['descargada']}, ya existentes: {conteo['existente']}, "
          f"errores: {len(errores)}")
    for e in errores:
        print("  ERROR", e)
    print(f"Tiempo de descarga: {(time.time() - t)/60:.2f} mins")

    if errores:
        raise SystemExit("Hubo errores; vuelve a ejecutar el script para reintentar.")
