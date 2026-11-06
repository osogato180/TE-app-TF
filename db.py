"""
Capa de datos de la aplicación.

* Si existe la configuración de MongoDB Atlas (secreto `mongodb.uri` en Streamlit o
  variable de entorno `MONGODB_URI`) usa la base de datos en la nube.
* Si no existe, trabaja en modo local (memoria de la sesión + data/atenciones.csv),
  lo que permite probar la aplicación sin conexión.

Colecciones: atenciones, turnos, predicciones, metricas.
"""
import json
import os
import time
import uuid

import pandas as pd
import streamlit as st

RUTA_ATENCIONES = "data/atenciones.csv"
RUTA_METRICAS = "metricas.json"
NOMBRE_BD_DEFECTO = "turnos_salud"


# ------------------------------------------------------------------ configuración
def obtener_config():
    """Devuelve (uri, nombre_bd) o (None, None) si no hay configuración."""
    uri, bd = None, NOMBRE_BD_DEFECTO
    try:
        if "mongodb" in st.secrets:
            uri = st.secrets["mongodb"].get("uri")
            bd = st.secrets["mongodb"].get("database", bd)
    except Exception:
        pass  # no hay archivo de secretos (modo local)
    if not uri:
        uri = os.environ.get("MONGODB_URI")
        bd = os.environ.get("MONGODB_DB", bd)
    return uri, bd


def crear_cliente(uri):
    from pymongo import MongoClient
    return MongoClient(uri, serverSelectionTimeoutMS=5000, appname="turnos-salud")


@st.cache_resource(show_spinner=False)
def _conectar():
    """Cliente y base de datos de Atlas (o None si no se puede conectar)."""
    uri, nombre = obtener_config()
    if not uri:
        return None, "Sin configuración de MongoDB"
    try:
        cliente = crear_cliente(uri)
        cliente.admin.command("ping")
        return cliente[nombre], "Conectado"
    except Exception as exc:
        return None, f"Error de conexión: {str(exc)[:120]}"


def base():
    return _conectar()[0]


def modo():
    """Texto que describe dónde se guardan los datos."""
    bd, msg = _conectar()
    return ("MongoDB Atlas", msg) if bd is not None else ("Modo local (memoria)", msg)


def ping_ms():
    """Latencia del ping a Atlas en milisegundos (None si no hay conexión)."""
    bd = base()
    if bd is None:
        return None
    t0 = time.perf_counter()
    try:
        bd.client.admin.command("ping")
    except Exception:
        return None
    return (time.perf_counter() - t0) * 1000


# ------------------------------------------------------------------ atenciones y métricas
@st.cache_data(ttl=300, show_spinner=False)
def leer_atenciones():
    bd = base()
    if bd is not None:
        docs = list(bd["atenciones"].find({}, {"_id": 0}))
        if docs:
            return pd.DataFrame(docs)
    return pd.read_csv(RUTA_ATENCIONES)


def leer_metricas():
    bd = base()
    if bd is not None:
        doc = bd["metricas"].find_one({}, {"_id": 0}, sort=[("generado", -1)])
        if doc:
            return doc
    if os.path.exists(RUTA_METRICAS):
        with open(RUTA_METRICAS, encoding="utf-8") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------ turnos y predicciones
def _mem(clave):
    if clave not in st.session_state:
        st.session_state[clave] = []
    return st.session_state[clave]


def insertar_turno(doc):
    bd = base()
    if bd is not None:
        bd["turnos"].insert_one(dict(doc))
    else:
        d = dict(doc); d["_id"] = uuid.uuid4().hex
        _mem("turnos").append(d)


def insertar_prediccion(doc):
    bd = base()
    if bd is not None:
        bd["predicciones"].insert_one(dict(doc))
    else:
        _mem("predicciones").append(dict(doc))


def listar_turnos():
    bd = base()
    docs = list(bd["turnos"].find({}).sort("registro", -1).limit(2000)) if bd is not None else list(reversed(_mem("turnos")))
    df = pd.DataFrame(docs)
    if not df.empty:
        df["_id"] = df["_id"].astype(str)
    return df


def listar_predicciones():
    bd = base()
    docs = list(bd["predicciones"].find({}, {"_id": 0}).limit(2000)) if bd is not None else list(_mem("predicciones"))
    return pd.DataFrame(docs)


def actualizar_turno(codigo, campos):
    bd = base()
    if bd is not None:
        bd["turnos"].update_one({"codigo": codigo}, {"$set": campos})
    else:
        for t in _mem("turnos"):
            if t["codigo"] == codigo:
                t.update(campos)
