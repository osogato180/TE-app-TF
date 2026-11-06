"""
Scripts de medición para el procedimiento experimental (usar en la rama pruebas-finales).

    python pruebas.py robustez     [--dias 30] [--seed 7]
    python pruebas.py rendimiento  [--repeticiones 30]
    python pruebas.py disponibilidad --url https://TU-APP.streamlit.app [--minutos 30] [--intervalo 60]

La conexión a MongoDB Atlas se toma de .streamlit/secrets.toml o de la variable MONGODB_URI.
Si no hay configuración, rendimiento y disponibilidad se miden solo en modo local.
"""
import argparse
import time
import urllib.request
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import db  # noqa: E402
from entrenar_modelo import RUTA_MODELO  # noqa: E402
from generar_datos import COLUMNAS_MODELO, OBJETIVO, generar_atenciones  # noqa: E402

ESCENARIOS = [("E1", "Demanda normal", 1.0, 10), ("E2", "Alta demanda", 1.5, 15), ("E3", "Demanda crítica", 2.0, 15)]


def _bd():
    uri, nombre = db.obtener_config()
    if not uri:
        return None
    try:
        cliente = db.crear_cliente(uri)
        cliente.admin.command("ping")
        return cliente[nombre]
    except Exception as exc:
        print("No se pudo conectar a MongoDB Atlas:", exc)
        return None


def robustez(a):
    modelo = joblib.load(RUTA_MODELO)
    filas = []
    for cod, nombre, factor, meta in ESCENARIOS:
        d = generar_atenciones(dias=a.dias, factor_demanda=factor, seed=a.seed)
        pred = np.clip(modelo.predict(d[COLUMNAS_MODELO]), 0, None)
        err = d[OBJETIVO].to_numpy() - pred
        mae = float(np.mean(np.abs(err)))
        filas.append(dict(escenario=cod, nombre=nombre, flujo=f"{factor}x", atenciones=len(d),
                          espera_real_promedio=round(d[OBJETIVO].mean(), 1), mae=round(mae, 2),
                          rmse=round(float(np.sqrt(np.mean(err ** 2))), 2),
                          mae_promedio_movil=round(float(np.mean(np.abs(d[OBJETIVO] - d["promedio_movil_servicio"]))), 2),
                          meta_mae=meta, resultado="Cumple" if mae <= meta else "No cumple"))
    df = pd.DataFrame(filas)
    print(df.to_string(index=False))
    df.to_csv("resultados_robustez.csv", index=False)


def rendimiento(a):
    bd = _bd()
    modelo = joblib.load(RUTA_MODELO)
    t_dash, t_reg = [], []
    ejemplo = pd.DataFrame([dict(hora_llegada=10.5, dia_semana="Lunes", especialidad="Pediatría",
                                 pacientes_en_cola=5, consultorios_activos=2, duracion_prom_5=18.0)])
    for _ in range(a.repeticiones):
        t0 = time.perf_counter()  # PR-01: lectura de atenciones + agregaciones del dashboard
        df = pd.DataFrame(list(bd["atenciones"].find({}, {"_id": 0}))) if bd is not None else pd.read_csv(db.RUTA_ATENCIONES)
        df.groupby("especialidad")[OBJETIVO].median(); df.groupby(df["hora_llegada"].astype(int))[OBJETIVO].mean()
        t_dash.append(time.perf_counter() - t0)

        t0 = time.perf_counter()  # PR-02: predicción + registro del turno
        est = float(modelo.predict(ejemplo[COLUMNAS_MODELO])[0])
        if bd is not None:
            r = bd["turnos_prueba"].insert_one(dict(especialidad="Pediatría", espera_estimada_min=est, prueba=True))
            bd["turnos_prueba"].delete_one({"_id": r.inserted_id})
        t_reg.append(time.perf_counter() - t0)
    modo = "MongoDB Atlas" if bd is not None else "modo local (sin Atlas)"
    res = pd.DataFrame([
        dict(codigo="PR-01", metrica="Actualización del dashboard", promedio_s=np.mean(t_dash), maximo_s=np.max(t_dash)),
        dict(codigo="PR-02", metrica="Registro de turno con predicción", promedio_s=np.mean(t_reg), maximo_s=np.max(t_reg))]).round(4)
    res["criterio"] = "< 3 s"; res["resultado"] = np.where(res["maximo_s"] < 3, "Cumple", "No cumple")
    print(f"Modo: {modo} | repeticiones: {a.repeticiones}"); print(res.to_string(index=False))
    res.to_csv("resultados_rendimiento.csv", index=False)


def disponibilidad(a):
    bd = _bd()
    registros, fin = [], time.time() + a.minutos * 60
    while True:
        ahora = datetime.now().isoformat(timespec="seconds")
        try:
            with urllib.request.urlopen(a.url, timeout=20) as r:
                app_ok = r.status == 200
        except Exception:
            app_ok = False
        try:
            bd_ok = bool(bd is not None and bd.client.admin.command("ping").get("ok") == 1)
        except Exception:
            bd_ok = False
        registros.append(dict(hora=ahora, app=app_ok, mongodb=bd_ok if bd is not None else None))
        print(registros[-1])
        if time.time() + a.intervalo > fin:
            break
        time.sleep(a.intervalo)
    df = pd.DataFrame(registros)
    print(f"\nDisponibilidad de la aplicación: {df['app'].mean() * 100:.1f} %")
    if bd is not None:
        print(f"Disponibilidad de MongoDB Atlas: {df['mongodb'].mean() * 100:.1f} %")
    df.to_csv("resultados_disponibilidad.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("robustez"); r.add_argument("--dias", type=int, default=30); r.add_argument("--seed", type=int, default=7)
    p = sp.add_parser("rendimiento"); p.add_argument("--repeticiones", type=int, default=30)
    d = sp.add_parser("disponibilidad"); d.add_argument("--url", required=True)
    d.add_argument("--minutos", type=int, default=30); d.add_argument("--intervalo", type=int, default=60)
    a = ap.parse_args()
    {"robustez": robustez, "rendimiento": rendimiento, "disponibilidad": disponibilidad}[a.cmd](a)
