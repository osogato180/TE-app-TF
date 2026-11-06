"""
Generador de datos sintéticos de consulta externa.

Simula la atención de cuatro servicios (Medicina General, Pediatría, Ginecología
y Cardiología) de un hospital público:

* Llegadas: proceso de Poisson con tasa distinta por franja horaria.
* Duración de la atención: distribución log-normal por especialidad.
* Atención: cola FIFO con varios consultorios activos por servicio.

La mediana del tiempo de espera se calibra cerca de 60 minutos con alta
dispersión, según lo reportado por ENSUSALUD.

Uso:
    python generar_datos.py                      # genera data/atenciones.csv (~10 000 filas)
    python generar_datos.py --factor 1.5         # escenario de 1,5 veces el flujo normal
"""
import argparse
import heapq
import os
from collections import deque

import numpy as np
import pandas as pd

SEED = 42
DIAS_SIMULADOS = 82      # ~3 meses de lunes a sábado -> ~10 000 atenciones
ESPECIALIDADES = ["Medicina General", "Pediatría", "Ginecología", "Cardiología"]

# duración media de la atención (min), llegadas base por hora y dispersión log-normal
PARAMS = {
    "Medicina General": dict(media=14, sigma=0.45, tasa=3.84),
    "Pediatría":        dict(media=18, sigma=0.45, tasa=3.36),
    "Ginecología":      dict(media=22, sigma=0.50, tasa=2.88),
    "Cardiología":      dict(media=28, sigma=0.50, tasa=2.28),
}
# multiplicador de llegadas por hora del día (08:00 a 16:59)
PERFIL_HORARIO = {8: 0.9, 9: 1.3, 10: 1.5, 11: 1.4, 12: 1.0, 13: 0.8, 14: 0.9, 15: 0.8, 16: 0.5}
HORA_APERTURA = 8
# refuerzo operativo: si la cola crece, el hospital habilita consultorios adicionales
UMBRAL_REFUERZO = 8      # pacientes en cola por consultorio activo
MAX_CONSULTORIOS = 4
RETRASO_REFUERZO = 20    # minutos que tarda en abrir un consultorio adicional
DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]
COLUMNAS_MODELO = ["hora_llegada", "dia_semana", "especialidad", "pacientes_en_cola",
                   "consultorios_activos", "duracion_prom_5"]
OBJETIVO = "tiempo_espera"


def _mu_lognormal(media, sigma):
    return np.log(media) - sigma ** 2 / 2


def generar_atenciones(dias=DIAS_SIMULADOS, factor_demanda=1.0, seed=SEED):
    """Devuelve un DataFrame con una fila por atención simulada."""
    rng = np.random.default_rng(seed)
    filas = []
    historial_espera = {e: deque(maxlen=10) for e in ESPECIALIDADES}  # para el promedio móvil

    for dia in range(dias):
        dia_semana = DIAS_SEMANA[dia % 6]
        # algunos días tienen más demanda (campañas, temporada de frío, etc.)
        mult_dia = float(np.clip(rng.lognormal(0.0, 0.35), 0.5, 2.2))

        for esp in ESPECIALIDADES:
            p = PARAMS[esp]
            mu = _mu_lognormal(p["media"], p["sigma"])
            n_cons = int(rng.choice([1, 2, 3], p=[0.75, 0.20, 0.05]))

            # llegadas del día (minutos desde las 00:00)
            llegadas = []
            for hora, mult in PERFIL_HORARIO.items():
                lam = p["tasa"] * mult * mult_dia * factor_demanda
                n = rng.poisson(lam)
                llegadas.extend(hora * 60 + rng.uniform(0, 60, n))
            llegadas.sort()

            libres = [HORA_APERTURA * 60.0] * n_cons
            heapq.heapify(libres)
            inicios, fines, duraciones = [], [], []

            n_actual = n_cons
            for t in llegadas:
                dur = float(rng.lognormal(mu, p["sigma"]))
                en_cola = sum(1 for s in inicios if s > t)
                n_registro = n_actual            # consultorios activos conocidos al registrar el turno
                if en_cola >= UMBRAL_REFUERZO * n_actual and n_actual < MAX_CONSULTORIOS:
                    heapq.heappush(libres, t + RETRASO_REFUERZO)
                    n_actual += 1
                libre = heapq.heappop(libres)
                inicio = max(t, libre)
                heapq.heappush(libres, inicio + dur)

                # variables conocidas en el momento del registro
                terminadas = [d for s, d in zip(inicios, duraciones) if s + d <= t]
                ult5 = terminadas[-5:]
                dur_prom_5 = float(np.mean(ult5)) if ult5 else float(p["media"])
                prom_movil = float(np.mean(historial_espera[esp])) if historial_espera[esp] else 60.0

                espera = inicio - t
                filas.append(dict(
                    dia=dia + 1,
                    hora_llegada=round(t / 60.0, 2),
                    dia_semana=dia_semana,
                    especialidad=esp,
                    pacientes_en_cola=en_cola,
                    consultorios_activos=n_registro,
                    duracion_prom_5=round(dur_prom_5, 2),
                    promedio_movil_servicio=round(prom_movil, 2),   # solo línea base, no es variable del modelo
                    tiempo_espera=round(espera, 2),
                ))
                inicios.append(inicio); duraciones.append(dur); fines.append(inicio + dur)
                # el promedio móvil solo conoce esperas de pacientes que ya entraron a consulta
                historial_espera[esp].append(espera)

    df = pd.DataFrame(filas)
    # orden cronológico real para que el promedio móvil sea una estimación operativa razonable
    return df.reset_index(drop=True)


def resumen(df):
    q = df[OBJETIVO].quantile([0.25, 0.5, 0.75]).round(1).tolist()
    return (f"{len(df)} atenciones | espera mediana {q[1]} min "
            f"(P25 {q[0]}, P75 {q[2]}) | media {df[OBJETIVO].mean():.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=DIAS_SIMULADOS)
    ap.add_argument("--factor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--salida", default="data/atenciones.csv")
    a = ap.parse_args()
    datos = generar_atenciones(a.dias, a.factor, a.seed)
    os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
    datos.to_csv(a.salida, index=False)
    print(resumen(datos)); print("Guardado en", a.salida)
