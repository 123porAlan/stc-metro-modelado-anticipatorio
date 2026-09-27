import re
import argparse
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuración visual para las gráficas de la tesis
sns.set_theme(style="whitegrid")

ARCHIVO_DATASET = "datos_procesados/dataset_features_entrenamiento.csv"
RUTA_MODELO = "modelos/modelo_anticipatorio.pkl"
ARCHIVO_COMPARACION = "modelos/comparacion_modelos.csv"
ARCHIVO_GRAFICA = "importancia_variables.png"

TARGET = 'target_congestibilidad_t_plus_1'
FEATURES_BASE = [
    'hora',
    'tiempo_ideal',
    'congestibilidad_t',
    'congestibilidad_t_minus_1',
    'hay_evento',
    'severidad_evento',
]

# Suavizado del target encoding de 'tramo': un tramo con pocas observaciones se acerca a
# la media global en vez de confiar ciegamente en su propio promedio.
SUAVIZADO_TRAMO = 20

# Número de días finales usados como pliegues de validación "rolling origin":
# para cada uno se entrena con TODOS los días anteriores y se evalúa solo en ese día.
DIAS_VALIDACION = 5

CANDIDATOS = {
    'RandomForest': lambda: RandomForestRegressor(
        n_estimators=150, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1),
    'GradientBoosting': lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42),
    'HistGradientBoosting': lambda: HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6, random_state=42),
}

def cargar_y_preparar_datos(ruta_archivo):
    """Carga el dataset tabular y asegura el orden cronológico."""
    print(f"1. Cargando dataset desde: {ruta_archivo}...")
    df = pd.read_csv(ruta_archivo)

    # ORDEN CRONOLÓGICO: Vital para series temporales.
    # Aseguramos que el modelo aprenda del pasado para predecir el futuro.
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(by=['fecha', 'hora']).reset_index(drop=True)

    return df

def linea_de_nodo(nodo_id):
    """Misma convención que simulador_congestion.py: el ID GTFS del andén trae 'L<línea>-'."""
    m = re.search(r"L([0-9A-Za-z]+)[-_]", nodo_id)
    return m.group(1) if m else None

def asignar_linea(df):
    """Línea del tramo; las aristas que unen andenes de líneas distintas son 'transbordo'."""
    linea_origen = df['nodo_origen'].map(linea_de_nodo)
    linea_destino = df['nodo_destino'].map(linea_de_nodo)
    return pd.Series(np.where(linea_origen == linea_destino, linea_origen, 'transbordo'), index=df.index)

def ajustar_codificacion(df_train):
    """
    Aprende, SOLO con datos de entrenamiento, cómo convertir las variables categóricas a
    numéricas: categorías de tipo_evento y línea (para dummies con columnas fijas) y el
    target encoding de 'tramo' (congestión futura media por tramo, suavizada). Calcularlo
    con el set de prueba filtraría el futuro al modelo.
    """
    media_global = df_train[TARGET].mean()
    stats = df_train.groupby('tramo')[TARGET].agg(['mean', 'count'])
    media_suavizada = (stats['mean'] * stats['count'] + media_global * SUAVIZADO_TRAMO) / (stats['count'] + SUAVIZADO_TRAMO)
    return {
        'tipos_evento': sorted(df_train['tipo_evento'].unique()),
        'lineas': sorted(asignar_linea(df_train).unique()),
        'tramo_media': media_suavizada.to_dict(),
        'media_global': media_global,
    }

def construir_features(df, codificacion, usar_geo=True):
    """
    Matriz de features numéricas. tipo_evento se expande a dummies porque los modelos de
    sklearn no aceptan texto. Con usar_geo=True se agrega la ubicación en la red: dummies
    de línea + congestión media histórica del tramo (target encoding).
    """
    X = df[FEATURES_BASE].copy()
    for tipo in codificacion['tipos_evento']:
        X[f'evento_{tipo}'] = (df['tipo_evento'] == tipo).astype(int)
    if usar_geo:
        linea = asignar_linea(df)
        for l in codificacion['lineas']:
            X[f'linea_{l}'] = (linea == l).astype(int)
        X['tramo_congestion_media'] = df['tramo'].map(codificacion['tramo_media']).fillna(codificacion['media_global'])
    return X

def predecir(modelo, X):
    """La congestibilidad (retraso sobre el tiempo ideal) no puede ser negativa."""
    return np.clip(modelo.predict(X), 0, None)

def calcular_metricas(y, y_pred, hay_evento):
    """MAE/RMSE global y, por separado, en filas con evento activo (<2% del dataset)."""
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    evento = np.asarray(hay_evento) == 1
    return {
        'MAE': mean_absolute_error(y, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE_evento': mean_absolute_error(y[evento], y_pred[evento]),
        'RMSE_evento': np.sqrt(mean_squared_error(y[evento], y_pred[evento])),
        'n_evento': int(evento.sum()),
    }

def entrenar_configuracion(df_train, nombre_modelo, usar_geo):
    codificacion = ajustar_codificacion(df_train)
    X_train = construir_features(df_train, codificacion, usar_geo)
    modelo = CANDIDATOS[nombre_modelo]()
    modelo.fit(X_train, df_train[TARGET])
    return modelo, codificacion

def configuraciones():
    return [(nombre, usar_geo) for usar_geo in (False, True) for nombre in CANDIDATOS]

def nombre_configuracion(nombre_modelo, usar_geo):
    return f"{nombre_modelo} {'+ línea/tramo' if usar_geo else '(sin ubicación)'}"

def separar_train_test(df):
    """Separación Temporal (80% Entrenamiento, 20% Prueba).
    No usamos train_test_split aleatorio para evitar "Data Leakage" (hacer trampa viendo el futuro)."""
    split_idx = int(len(df) * 0.8)
    return df.iloc[:split_idx], df.iloc[split_idx:]

def evaluar_split_80_20(df_train, df_test):
    resultados = []
    for nombre_modelo, usar_geo in configuraciones():
        modelo, codificacion = entrenar_configuracion(df_train, nombre_modelo, usar_geo)
        y_pred = predecir(modelo, construir_features(df_test, codificacion, usar_geo))
        metricas = calcular_metricas(df_test[TARGET], y_pred, df_test['hay_evento'])
        resultados.append({'evaluacion': 'split_80_20', 'modelo': nombre_modelo, 'usar_geo': usar_geo, **metricas})
    return resultados

def evaluar_validacion_por_dias(df):
    """
    Validación "rolling origin" por días: el split 80/20 deja solo ~3 días (y ~140 filas
    con evento) en prueba, demasiado poco para comparar modelos de forma fiable. Aquí cada
    uno de los últimos DIAS_VALIDACION días se predice con un modelo entrenado con todos
    los días previos, y los errores de todos los pliegues se agregan.
    """
    dias = sorted(df['fecha'].unique())
    y_real, y_pred, eventos = {}, {}, {}
    for dia in dias[-DIAS_VALIDACION:]:
        df_train = df[df['fecha'] < dia]
        df_dia = df[df['fecha'] == dia]
        print(f"   - Pliegue {pd.Timestamp(dia).date()} (entrena con {df_train['fecha'].nunique()} días)")
        for config in configuraciones():
            modelo, codificacion = entrenar_configuracion(df_train, *config)
            y_pred.setdefault(config, []).append(predecir(modelo, construir_features(df_dia, codificacion, config[1])))
            y_real.setdefault(config, []).append(df_dia[TARGET].values)
            eventos.setdefault(config, []).append(df_dia['hay_evento'].values)

    resultados = []
    for nombre_modelo, usar_geo in configuraciones():
        config = (nombre_modelo, usar_geo)
        metricas = calcular_metricas(np.concatenate(y_real[config]), np.concatenate(y_pred[config]), np.concatenate(eventos[config]))
        resultados.append({'evaluacion': f'validacion_{DIAS_VALIDACION}_dias', 'modelo': nombre_modelo, 'usar_geo': usar_geo, **metricas})
    return resultados

def imprimir_tabla(resultados, titulo):
    df_res = pd.DataFrame(resultados)
    df_res['configuracion'] = [nombre_configuracion(m, g) for m, g in zip(df_res['modelo'], df_res['usar_geo'])]
    print(f"\n--- {titulo} ---")
    print(df_res[['configuracion', 'MAE', 'RMSE', 'MAE_evento', 'RMSE_evento', 'n_evento']]
          .sort_values('RMSE_evento').to_string(index=False, float_format=lambda x: f"{x:.4f}"))

def comparar_modelos(df, archivo_comparacion=ARCHIVO_COMPARACION):
    """Compara RandomForest vs. GradientBoosting vs. HistGradientBoosting, con y sin línea/tramo."""
    print("\n2. Comparando modelos candidatos...")
    df_train, df_test = separar_train_test(df)
    print(f"   - Datos de Entrenamiento (Pasado): {len(df_train)} registros")
    print(f"   - Datos de Prueba (Futuro): {len(df_test)} registros ({df_test['hay_evento'].sum()} con evento activo)")

    resultados_split = evaluar_split_80_20(df_train, df_test)
    imprimir_tabla(resultados_split, "Split cronológico 80/20")

    print(f"\n   Validación por días (últimos {DIAS_VALIDACION} días, rolling origin):")
    resultados_validacion = evaluar_validacion_por_dias(df)
    imprimir_tabla(resultados_validacion, f"Validación por días ({DIAS_VALIDACION} pliegues)")

    os.makedirs(os.path.dirname(archivo_comparacion) or ".", exist_ok=True)
    pd.DataFrame(resultados_split + resultados_validacion).to_csv(archivo_comparacion, index=False)
    print(f"\n   [OK] Comparación guardada en: {archivo_comparacion}")

    # Criterio de selección: menor RMSE en filas con evento en la validación por días.
    # Es el segmento que el ruteo anticipatorio necesita predecir bien (los picos de
    # retraso), y la validación por días es la estimación con más muestras de evento.
    mejor = min(resultados_validacion, key=lambda r: r['RMSE_evento'])
    print(f"   [OK] Seleccionado: {nombre_configuracion(mejor['modelo'], mejor['usar_geo'])}")
    return mejor['modelo'], mejor['usar_geo'], df_train, df_test

def entrenar_modelo_final(df_train, df_test, nombre_modelo, usar_geo):
    """Entrena el modelo elegido con el 80% inicial y lo evalúa en el 20% final. No se
    reentrena con todo el dataset para que el ruteo pueda evaluarse sobre días no vistos."""
    print("\n3. Entrenando el Modelo de Inteligencia Anticipatoria...")
    modelo, codificacion = entrenar_configuracion(df_train, nombre_modelo, usar_geo)
    print("   [OK] Entrenamiento completado.")

    print("\n4. Evaluando KPIs Predictivos...")
    X_test = construir_features(df_test, codificacion, usar_geo)
    metricas = calcular_metricas(df_test[TARGET], predecir(modelo, X_test), df_test['hay_evento'])

    print("--- RESULTADOS DEL MODELO ---")
    print(f"MAE  (Error Absoluto Medio): {metricas['MAE']:.4f} minutos.")
    print(f"RMSE (Raíz Error Cuadrático): {metricas['RMSE']:.4f} minutos.")
    print(f"MAE  en filas con evento ({metricas['n_evento']}): {metricas['MAE_evento']:.4f} minutos.")
    print(f"RMSE en filas con evento ({metricas['n_evento']}): {metricas['RMSE_evento']:.4f} minutos.")
    print("-----------------------------")

    return modelo, codificacion, X_test, metricas

def importancia_por_permutacion(modelo, X, y, repeticiones=5, semilla=42):
    """
    Cuánto empeora el RMSE al barajar cada variable. Es agnóstica al modelo (sirve igual
    para RandomForest que para boosting) y agrupa las dummies de una misma variable
    categórica (tipo_evento, línea) para que se lean como una sola variable.
    """
    grupos = {}
    for columna in X.columns:
        grupo = 'tipo_evento' if columna.startswith('evento_') else 'linea' if columna.startswith('linea_') else columna
        grupos.setdefault(grupo, []).append(columna)

    rng = np.random.default_rng(semilla)
    rmse_base = np.sqrt(mean_squared_error(y, predecir(modelo, X)))
    importancias = {}
    for grupo, columnas in grupos.items():
        incrementos = []
        for _ in range(repeticiones):
            X_perm = X.copy()
            X_perm[columnas] = X[columnas].values[rng.permutation(len(X))]
            incrementos.append(np.sqrt(mean_squared_error(y, predecir(modelo, X_perm))) - rmse_base)
        importancias[grupo] = np.mean(incrementos)
    return importancias

def graficar_explicabilidad(modelo, X_test, y_test, nombre_modelo, ruta_grafica=ARCHIVO_GRAFICA):
    """Genera la gráfica de importancia de características para la tesis."""
    print("\n5. Generando análisis de transparencia algorítmica...")
    importancias = importancia_por_permutacion(modelo, X_test, y_test)
    df_imp = pd.DataFrame({
        'Característica': list(importancias.keys()),
        'Importancia': list(importancias.values())
    }).sort_values(by='Importancia', ascending=False)
    print(df_imp.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importancia', y='Característica', data=df_imp, hue='Característica', palette='viridis', legend=False)

    plt.title(f'Explicabilidad: Peso Predictivo de las Variables ({nombre_modelo})', fontsize=14)
    plt.xlabel('Aumento del RMSE al permutar la variable (minutos)', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.tight_layout()

    plt.savefig(ruta_grafica)
    print(f"   [OK] Gráfica guardada como: '{ruta_grafica}' (Lista para tu tesis).")
    # plt.show() # Descomenta esto si quieres que la gráfica se abra en una ventana

def guardar_modelo(paquete, ruta_salida=RUTA_MODELO):
    """Exporta el modelo junto con su codificación para usarlo en el algoritmo de búsqueda de rutas."""
    print("\n6. Exportando modelo...")
    os.makedirs(os.path.dirname(ruta_salida) or ".", exist_ok=True)
    joblib.dump(paquete, ruta_salida)
    print(f"   [OK] Modelo exportado en: {ruta_salida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara modelos candidatos, entrena el elegido y lo exporta.")
    parser.add_argument("--dataset", default=ARCHIVO_DATASET)
    parser.add_argument("--modelo", default=RUTA_MODELO)
    parser.add_argument("--comparacion", default=ARCHIVO_COMPARACION)
    parser.add_argument("--grafica", default=ARCHIVO_GRAFICA)
    args = parser.parse_args()

    try:
        # 1. Cargar datos
        df_features = cargar_y_preparar_datos(args.dataset)

        # 2. Comparar candidatos y elegir
        nombre_modelo, usar_geo, df_train, df_test = comparar_modelos(df_features, args.comparacion)

        # 3 y 4. Entrenar y Evaluar el modelo elegido
        modelo, codificacion, X_test, metricas = entrenar_modelo_final(df_train, df_test, nombre_modelo, usar_geo)

        # 5. Generar gráfica de Explicabilidad (Objetivo de la tesis)
        graficar_explicabilidad(modelo, X_test, df_test[TARGET], nombre_modelo, args.grafica)

        # 6. Guardar modelo + todo lo necesario para reconstruir sus features
        primera_fila_test = df_test.iloc[0]
        guardar_modelo({
            'nombre_modelo': nombre_modelo,
            'usar_geo': usar_geo,
            'modelo': modelo,
            'codificacion': codificacion,
            'metricas_test': metricas,
            # El corte 80/20 puede caer a mitad de una hora: el ruteo solo evalúa horas
            # posteriores a esta, que el modelo nunca vio.
            'inicio_test': (primera_fila_test['fecha'], primera_fila_test['hora']),
        }, args.modelo)

        print("\n🚀 ¡Fase de Inteligencia Anticipatoria completada con éxito!")

    except Exception as e:
        print(f"\n❌ Error en la ejecución: {e}")
        raise SystemExit(1)
