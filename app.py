"""
Sistema inteligente para la gestión de turnos y predicción de tiempos de espera
en establecimientos de salud públicos (prototipo con datos simulados).

Aplicación Streamlit desplegada desde GitHub en Streamlit Community Cloud.
Datos en MongoDB Atlas (o modo local si no hay credenciales configuradas).
"""
import time
import warnings
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import db
from entrenar_modelo import RUTA_MODELO, entrenar
from generar_datos import (COLUMNAS_MODELO, DIAS_SEMANA, ESPECIALIDADES, OBJETIVO,
                           generar_atenciones)

st.set_page_config(page_title="Gestión de turnos y tiempos de espera", page_icon="🏥", layout="wide")

LIMA = timezone(timedelta(hours=-5))          # Perú no usa horario de verano
META_DASHBOARD_S = 3.0                         # el dashboard debe actualizarse en < 3 s
ESTADOS = ["En espera", "En atención", "Atendido"]


def ahora():
    return datetime.now(LIMA)


# ------------------------------------------------------------------ modelo
@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    """Carga modelo.joblib; si no existe o es incompatible con la versión de scikit-learn, reentrena."""
    try:
        from sklearn.exceptions import InconsistentVersionWarning
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            return joblib.load(RUTA_MODELO)
    except Exception:
        modelo, _ = entrenar(guardar=False)
        return modelo


def predecir(modelo, filas: pd.DataFrame):
    return np.clip(modelo.predict(filas[COLUMNAS_MODELO]), 0, None)


# ------------------------------------------------------------------ utilidades de turnos
def contexto_servicio(esp, turnos: pd.DataFrame, atenciones: pd.DataFrame):
    """Pacientes en cola y duración promedio de las últimas cinco atenciones del servicio."""
    en_cola, dur5 = 0, None
    if not turnos.empty:
        t = turnos[turnos["especialidad"] == esp]
        en_cola = int((t["estado"] == "En espera").sum())
        atendidos = t[t["estado"] == "Atendido"].sort_values("registro")
        if "duracion_atencion_min" in atendidos and len(atendidos) > 0:
            vals = atendidos["duracion_atencion_min"].dropna().tail(5)
            if len(vals) > 0:
                dur5 = float(vals.mean())
    if dur5 is None:  # sin historial en la app: usa el promedio histórico simulado del servicio
        dur5 = float(atenciones.loc[atenciones["especialidad"] == esp, "duracion_prom_5"].mean())
    return en_cola, round(dur5, 2)


def minutos_entre(inicio_iso, fin_dt):
    return (fin_dt - datetime.fromisoformat(inicio_iso)).total_seconds() / 60


# ------------------------------------------------------------------ páginas
def pagina_registro(modelo):
    st.header("Registro de turnos")
    st.caption("Registra un paciente y obtén el tiempo de espera estimado por el modelo Random Forest.")
    atenciones, turnos = db.leer_atenciones(), db.listar_turnos()

    col1, col2 = st.columns([1, 1])
    with col1:
        esp = st.selectbox("Especialidad", ESPECIALIDADES)
        consultorios = st.slider("Consultorios activos", 1, 4, 2)
        en_cola, dur5 = contexto_servicio(esp, turnos, atenciones)
    with col2:
        st.metric("Pacientes en cola (calculado)", en_cola)
        st.metric("Duración promedio de las últimas 5 atenciones", f"{dur5:.1f} min")

    if st.button("Registrar turno y estimar espera", type="primary"):
        t0 = time.perf_counter()
        ahora_lima = ahora()
        hora = ahora_lima.hour + ahora_lima.minute / 60
        dia = DIAS_SEMANA[min(ahora_lima.weekday(), 5)]  # domingo se trata como sábado
        fila = pd.DataFrame([dict(hora_llegada=round(hora, 2), dia_semana=dia, especialidad=esp,
                                  pacientes_en_cola=en_cola, consultorios_activos=consultorios,
                                  duracion_prom_5=dur5)])
        estimado = float(predecir(modelo, fila)[0])
        codigo = f"T-{ahora_lima:%Y%m%d}-{len(turnos) + 1:04d}"
        db.insertar_turno(dict(codigo=codigo, registro=ahora_lima.isoformat(), especialidad=esp,
                               hora_llegada=fila.loc[0, "hora_llegada"], dia_semana=dia,
                               pacientes_en_cola=en_cola, consultorios_activos=consultorios,
                               duracion_prom_5=dur5, espera_estimada_min=round(estimado, 1),
                               estado="En espera"))
        db.insertar_prediccion(dict(codigo=codigo, registro=ahora_lima.isoformat(),
                                    entrada=fila.iloc[0].to_dict(), espera_estimada_min=round(estimado, 1)))
        seg = time.perf_counter() - t0
        st.success(f"Turno **{codigo}** registrado. Tiempo de espera estimado: **{estimado:.0f} minutos**.")
        st.caption(f"Tiempo de registro y predicción: {seg:.2f} s")

    st.subheader("Últimos turnos")
    if turnos.empty:
        st.info("Aún no hay turnos registrados.")
    else:
        cols = ["codigo", "especialidad", "estado", "pacientes_en_cola", "espera_estimada_min", "registro"]
        st.dataframe(turnos[cols].head(15), hide_index=True)


def pagina_gestion():
    st.header("Gestión de turnos")
    turnos = db.listar_turnos()
    if turnos.empty:
        st.info("No hay turnos registrados todavía.")
        return
    activos = turnos[turnos["estado"] != "Atendido"].sort_values("registro")
    st.dataframe(activos[["codigo", "especialidad", "estado", "espera_estimada_min", "registro"]], hide_index=True)
    if activos.empty:
        st.success("No hay pacientes pendientes.")
        return
    codigo = st.selectbox("Selecciona un turno", activos["codigo"])
    turno = activos[activos["codigo"] == codigo].iloc[0]
    c1, c2 = st.columns(2)
    if turno["estado"] == "En espera" and c1.button("Iniciar atención"):
        ahora_lima = ahora()
        db.actualizar_turno(codigo, dict(estado="En atención", inicio_atencion=ahora_lima.isoformat(),
                                         espera_real_min=round(minutos_entre(turno["registro"], ahora_lima), 1)))
        st.rerun()
    if turno["estado"] == "En atención" and c2.button("Finalizar atención"):
        ahora_lima = ahora()
        db.actualizar_turno(codigo, dict(estado="Atendido", fin_atencion=ahora_lima.isoformat(),
                                         duracion_atencion_min=round(minutos_entre(turno["inicio_atencion"], ahora_lima), 1)))
        st.rerun()


def pagina_dashboard():
    st.header("Dashboard de turnos")
    t0 = time.perf_counter()
    atenciones, turnos = db.leer_atenciones(), db.listar_turnos()

    total = len(turnos)
    en_espera = int((turnos["estado"] == "En espera").sum()) if total else 0
    est_prom = float(turnos.loc[turnos["estado"] == "En espera", "espera_estimada_min"].mean()) if en_espera else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Turnos registrados", total)
    c2.metric("Pacientes en espera", en_espera)
    c3.metric("Espera estimada promedio (en espera)", f"{est_prom:.0f} min")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Turnos por especialidad")
        if total:
            st.bar_chart(turnos.groupby("especialidad").size().rename("turnos"))
        else:
            st.caption("Sin turnos registrados.")
    with g2:
        st.subheader("Espera estimada por especialidad (turnos en espera)")
        if en_espera:
            st.bar_chart(turnos[turnos["estado"] == "En espera"].groupby("especialidad")["espera_estimada_min"].mean())
        else:
            st.caption("Sin pacientes en espera.")

    st.subheader("Histórico simulado")
    h1, h2 = st.columns(2)
    with h1:
        st.caption("Espera promedio por hora de llegada (min)")
        st.bar_chart(atenciones.groupby(atenciones["hora_llegada"].astype(int))[OBJETIVO].mean())
    with h2:
        st.caption("Espera mediana por especialidad (min)")
        st.bar_chart(atenciones.groupby("especialidad")[OBJETIVO].median())

    seg = time.perf_counter() - t0
    (st.success if seg < META_DASHBOARD_S else st.warning)(
        f"Tiempo de actualización del dashboard: {seg:.2f} s (meta: menos de {META_DASHBOARD_S:.0f} s)")

    st.subheader("Exportar datos (CSV)")
    d1, d2 = st.columns(2)
    if total:
        d1.download_button("Descargar turnos", turnos.drop(columns=["_id"], errors="ignore").to_csv(index=False),
                           "turnos.csv", "text/csv")
    preds = db.listar_predicciones()
    if not preds.empty:
        d2.download_button("Descargar predicciones", preds.astype(str).to_csv(index=False),
                           "predicciones.csv", "text/csv")


def pagina_metricas():
    st.header("Métricas del modelo")
    inf = db.leer_metricas()
    if not inf:
        st.warning("No se encontró metricas.json. Ejecuta `python entrenar_modelo.py`.")
        return
    st.caption(f"{inf['n_entrenamiento']} registros de entrenamiento (70 %) y {inf['n_prueba']} de prueba (30 %).")
    tabla = pd.DataFrame(inf["modelos"]).T.rename(columns={"mae": "MAE (min)", "rmse": "RMSE (min)", "r2": "R²"})
    st.dataframe(tabla)
    metas, cumple, rf = inf["metas"], inf["cumple"], inf["modelos"]["Random Forest"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE Random Forest", f"{rf['mae']:.2f} min", f"meta ≤ {metas['mae']:.0f}  {'✔' if cumple['mae'] else '✘'}", delta_color="off")
    c2.metric("RMSE Random Forest", f"{rf['rmse']:.2f} min", f"meta ≤ {metas['rmse']:.0f}  {'✔' if cumple['rmse'] else '✘'}", delta_color="off")
    c3.metric("R² Random Forest", f"{rf['r2']:.3f}", f"meta ≥ {metas['r2']}  {'✔' if cumple['r2'] else '✘'}", delta_color="off")
    c4.metric("Mejora vs promedio móvil", f"{inf['mejora_vs_base_pct']:.1f} %", f"meta ≥ {metas['mejora_pct']:.0f} %  {'✔' if cumple['mejora'] else '✘'}", delta_color="off")
    st.subheader("Importancia de las variables")
    st.bar_chart(pd.Series(inf["importancias"], name="importancia"))


def pagina_simulacion(modelo):
    st.header("Simulación de demanda")
    st.caption("Genera días nuevos con el flujo de pacientes multiplicado y mide el error del modelo "
               "entrenado con demanda normal. Meta: MAE ≤ 10 min (normal) y ≤ 15 min (1,5x y 2x).")
    dias = st.slider("Días simulados por escenario", 10, 60, 20)
    seed = st.number_input("Semilla", 1, 9999, 7)
    escenarios = [("E1", "Demanda normal", 1.0, 10), ("E2", "Alta demanda", 1.5, 15), ("E3", "Demanda crítica", 2.0, 15)]
    if st.button("Ejecutar los tres escenarios", type="primary"):
        filas = []
        with st.spinner("Simulando..."):
            for cod, nombre, factor, meta in escenarios:
                t0 = time.perf_counter()
                d = generar_atenciones(dias=dias, factor_demanda=factor, seed=int(seed))
                pred = predecir(modelo, d)
                err = d[OBJETIVO].to_numpy() - pred
                mae = float(np.mean(np.abs(err)))
                mae_base = float(np.mean(np.abs(d[OBJETIVO] - d["promedio_movil_servicio"])))
                filas.append({"Escenario": cod, "Nombre": nombre, "Flujo": f"{factor}x", "Atenciones": len(d),
                              "Espera real promedio (min)": round(d[OBJETIVO].mean(), 1),
                              "MAE modelo (min)": round(mae, 2), "RMSE modelo (min)": round(float(np.sqrt(np.mean(err**2))), 2),
                              "MAE promedio móvil (min)": round(mae_base, 2), "Meta MAE (min)": meta,
                              "Resultado": "Cumple" if mae <= meta else "No cumple",
                              "Tiempo (s)": round(time.perf_counter() - t0, 2)})
        st.session_state["sim"] = pd.DataFrame(filas)
    if "sim" in st.session_state:
        st.dataframe(st.session_state["sim"], hide_index=True)


def pagina_servicios():
    st.header("Estado de los servicios")
    nombre, msg = db.modo()
    st.write(f"**Almacenamiento:** {nombre} — {msg}")
    if st.button("Verificar disponibilidad"):
        filas = [{"Servicio": "Aplicación (Streamlit Community Cloud)", "Estado": "Disponible", "Detalle": "Aplicación cargada correctamente"}]
        ms = db.ping_ms()
        filas.append({"Servicio": "MongoDB Atlas", "Estado": "Disponible" if ms is not None else "No disponible",
                      "Detalle": f"Ping en {ms:.0f} ms" if ms is not None else msg})
        st.dataframe(pd.DataFrame(filas), hide_index=True)


# ------------------------------------------------------------------ navegación
def main():
    modelo = cargar_modelo()
    st.sidebar.title("🏥 Turnos y espera")
    st.sidebar.caption("Prototipo con datos simulados")
    pagina = st.sidebar.radio("Módulos", ["Registro de turnos", "Gestión de turnos", "Dashboard",
                                          "Métricas del modelo", "Simulación de demanda", "Estado de servicios"])
    nombre, _ = db.modo()
    st.sidebar.info(f"Datos: {nombre}")
    {"Registro de turnos": lambda: pagina_registro(modelo), "Gestión de turnos": pagina_gestion,
     "Dashboard": pagina_dashboard, "Métricas del modelo": pagina_metricas,
     "Simulación de demanda": lambda: pagina_simulacion(modelo), "Estado de servicios": pagina_servicios}[pagina]()


main()
