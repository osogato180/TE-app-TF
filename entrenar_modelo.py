"""
Entrenamiento y evaluación de los modelos de predicción del tiempo de espera.

* Modelo principal: Random Forest de regresión (scikit-learn).
* Modelos de comparación: promedio móvil por servicio (línea base),
  regresión lineal y XGBoost.
* División 70 % entrenamiento / 30 % prueba.

Genera:
    modelo.joblib    -> pipeline entrenado (preprocesamiento + Random Forest)
    metricas.json    -> MAE, RMSE, R², mejora frente a la línea base e importancia de variables

Uso:
    python entrenar_modelo.py            # entrena con demanda normal (por defecto)
    python entrenar_modelo.py --mixto    # agrega días de 1,5x y 2x de demanda al entrenamiento
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from generar_datos import COLUMNAS_MODELO, OBJETIVO, SEED, generar_atenciones

RUTA_DATOS = "data/atenciones.csv"
RUTA_MODELO = "modelo.joblib"
RUTA_METRICAS = "metricas.json"
CATEGORICAS = ["dia_semana", "especialidad"]

# metas definidas en el proyecto
METAS = {"mae": 10.0, "rmse": 15.0, "r2": 0.70, "mejora_pct": 20.0}


def _preprocesador():
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)],
        remainder="passthrough",
    )


def construir_pipeline(modelo=None):
    modelo = modelo or RandomForestRegressor(
        n_estimators=150, max_depth=16, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    return Pipeline([("prep", _preprocesador()), ("modelo", modelo)])


def metricas(y_real, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_real, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_real, y_pred))), 3),
        "r2": round(float(r2_score(y_real, y_pred)), 4),
    }


def cargar_datos(ruta=RUTA_DATOS):
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    return generar_atenciones()


def importancias(pipe):
    """Suma la importancia de las columnas dummy para devolver una por variable original."""
    nombres = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["modelo"].feature_importances_
    agregada = {}
    for nombre, valor in zip(nombres, imp):
        base = nombre.split("__", 1)[1]
        for c in CATEGORICAS:
            if base.startswith(c):
                base = c
        agregada[base] = agregada.get(base, 0.0) + float(valor)
    return {k: round(v, 4) for k, v in sorted(agregada.items(), key=lambda x: -x[1])}


def entrenar(df=None, guardar=True, mixto=False):
    """mixto=True agrega al entrenamiento días simulados con 1,5x y 2x de demanda
    (la prueba sigue usando solo el 30 % de datos con demanda normal)."""
    df = cargar_datos() if df is None else df
    X, y = df[COLUMNAS_MODELO], df[OBJETIVO]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.30, random_state=SEED
    )
    if mixto:
        extra = pd.concat([generar_atenciones(41, 1.5, seed=SEED + 1),
                           generar_atenciones(41, 2.0, seed=SEED + 2)])
        X_tr = pd.concat([X_tr, extra[COLUMNAS_MODELO]])
        y_tr = pd.concat([y_tr, extra[OBJETIVO]])

    resultados = {}

    # línea base: promedio móvil de las últimas esperas del servicio
    resultados["Promedio móvil (base)"] = metricas(y_te, df.loc[idx_te, "promedio_movil_servicio"])

    # regresión lineal
    lineal = construir_pipeline(LinearRegression()).fit(X_tr, y_tr)
    resultados["Regresión lineal"] = metricas(y_te, lineal.predict(X_te))

    # XGBoost (opcional)
    try:
        from xgboost import XGBRegressor
        xgb = construir_pipeline(XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                              subsample=0.9, random_state=SEED, n_jobs=-1)).fit(X_tr, y_tr)
        resultados["XGBoost"] = metricas(y_te, xgb.predict(X_te))
    except Exception as exc:  # xgboost no instalado
        print("XGBoost omitido:", exc)

    # Random Forest (modelo principal)
    rf = construir_pipeline().fit(X_tr, y_tr)
    resultados["Random Forest"] = metricas(y_te, rf.predict(X_te))

    base = resultados["Promedio móvil (base)"]["mae"]
    mejora = round((base - resultados["Random Forest"]["mae"]) / base * 100, 2)
    rfm = resultados["Random Forest"]
    informe = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "entrenamiento": "mixto (1x, 1,5x y 2x)" if mixto else "demanda normal",
        "n_total": int(len(df)), "n_entrenamiento": int(len(X_tr)), "n_prueba": int(len(X_te)),
        "modelos": resultados,
        "mejora_vs_base_pct": mejora,
        "importancias": importancias(rf),
        "metas": METAS,
        "cumple": {
            "mae": rfm["mae"] <= METAS["mae"], "rmse": rfm["rmse"] <= METAS["rmse"],
            "r2": rfm["r2"] >= METAS["r2"], "mejora": mejora >= METAS["mejora_pct"],
        },
    }
    if guardar:
        joblib.dump(rf, RUTA_MODELO, compress=3)
        with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
            json.dump(informe, f, ensure_ascii=False, indent=2)
    return rf, informe


if __name__ == "__main__":
    import sys
    _, inf = entrenar(mixto="--mixto" in sys.argv)
    print(json.dumps(inf, ensure_ascii=False, indent=2))
    print("\nModelo guardado en", RUTA_MODELO, "| métricas en", RUTA_METRICAS)
