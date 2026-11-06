"""
Carga los datos iniciales en MongoDB Atlas.

    python cargar_mongodb.py              # carga data/atenciones.csv y metricas.json
    python cargar_mongodb.py --reset      # borra atenciones y metricas antes de cargar

La conexión se toma de .streamlit/secrets.toml ([mongodb] uri y database)
o de las variables de entorno MONGODB_URI y MONGODB_DB.
"""
import argparse
import json
import os
import time
import tomllib

import pandas as pd
from pymongo import ASCENDING, MongoClient

LOTE = 1000


def leer_config():
    uri, bd = os.environ.get("MONGODB_URI"), os.environ.get("MONGODB_DB", "turnos_salud")
    ruta = ".streamlit/secrets.toml"
    if not uri and os.path.exists(ruta):
        with open(ruta, "rb") as f:
            cfg = tomllib.load(f).get("mongodb", {})
        uri, bd = cfg.get("uri"), cfg.get("database", bd)
    if not uri:
        raise SystemExit("Falta la conexión: define .streamlit/secrets.toml o la variable MONGODB_URI.")
    return uri, bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    uri, nombre = leer_config()
    bd = MongoClient(uri, serverSelectionTimeoutMS=8000)[nombre]
    bd.client.admin.command("ping")
    print("Conectado a MongoDB Atlas, base:", nombre)

    if a.reset:
        bd["atenciones"].delete_many({}); bd["metricas"].delete_many({})
        print("Colecciones atenciones y metricas vaciadas.")

    df = pd.read_csv("data/atenciones.csv")
    t0 = time.perf_counter()
    for i in range(0, len(df), LOTE):
        bd["atenciones"].insert_many(df.iloc[i:i + LOTE].to_dict("records"))
    print(f"{len(df)} atenciones cargadas en {time.perf_counter() - t0:.2f} s (lotes de {LOTE}).")

    with open("metricas.json", encoding="utf-8") as f:
        bd["metricas"].insert_one(json.load(f))
    print("Métricas del modelo cargadas.")

    bd["turnos"].create_index([("codigo", ASCENDING)], unique=True)
    bd["turnos"].create_index([("estado", ASCENDING), ("especialidad", ASCENDING)])
    print("Índices creados en la colección turnos.")


if __name__ == "__main__":
    main()
