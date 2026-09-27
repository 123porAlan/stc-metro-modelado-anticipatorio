"""
Repite el pipeline simulador -> entrenador -> ruteo con varias semillas de eventos
estocásticos y reporta la media y el intervalo de confianza de los resultados.

Con las tasas calibradas los eventos son raros (unas pocas decenas de filas con evento
en el set de prueba por semilla), así que una sola corrida no basta para concluir qué
sistema de ruteo o qué modelo es mejor. La demanda (matrices O-D) no depende de la
semilla: solo cambian los eventos.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Las 5 semillas originales + 25 nuevas: con eventos raros, 5 semillas dejaban solo ~12
# horas con incidentes de impacto y la comparación de sistemas no era concluyente.
SEMILLAS = [42, 7, 13, 101, 2026] + list(range(1001, 1026))
CORRIDAS_EN_PARALELO = 3
DIRECTORIO_SALIDA = "datos_procesados/semillas"
ARCHIVO_RESULTADOS_RUTEO = "modelos/resultados_semillas_ruteo.csv"
ARCHIVO_RESULTADOS_MODELOS = "modelos/resultados_semillas_modelos.csv"
ARCHIVO_BOOTSTRAP = "modelos/resultados_semillas_bootstrap.csv"

def rutas_semilla(semilla):
    base = os.path.join(DIRECTORIO_SALIDA, f"semilla_{semilla}")
    return {
        'dir': base,
        'dataset': os.path.join(base, "dataset.csv"),
        'modelo': os.path.join(base, "modelo.pkl"),
        'comparacion': os.path.join(base, "comparacion_modelos.csv"),
        'grafica': os.path.join(base, "importancia_variables.png"),
        'evaluacion': os.path.join(base, "evaluacion_ruteo.csv"),
        'resumen': os.path.join(base, "resumen_ruteo.csv"),
        'log': os.path.join(base, "log.txt"),
    }

def correr_semilla(semilla):
    r = rutas_semilla(semilla)
    os.makedirs(r['dir'], exist_ok=True)
    pasos = [
        ["simulador_congestion.py", "--semilla", str(semilla), "--salida", r['dataset']],
        ["entrenador_anticipatorio.py", "--dataset", r['dataset'], "--modelo", r['modelo'],
         "--comparacion", r['comparacion'], "--grafica", r['grafica']],
        ["ruteo_anticipatorio.py", "--dataset", r['dataset'], "--modelo", r['modelo'],
         "--evaluacion", r['evaluacion'], "--resumen", r['resumen']],
    ]
    with open(r['log'], "w", encoding="utf-8") as log:
        for paso in pasos:
            print(f"[semilla {semilla}] {paso[0]}...", flush=True)
            subprocess.run([sys.executable] + paso, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "MPLBACKEND": "Agg"})
    print(f"[semilla {semilla}] listo", flush=True)
    return semilla

SISTEMAS = ['reactivo', 'anticipatorio', 'oraculo']
REPETICIONES_BOOTSTRAP = 5000

def agregar_ruteo():
    """
    El % del ahorro posible capturado NO se promedia por semilla: en semillas donde el
    oráculo casi no puede ahorrar (ningún evento que abra una ruta alternativa) el
    cociente se dispara o queda indefinido. En su lugar se juntan los casos de todas las
    semillas y el intervalo de confianza se obtiene con un bootstrap por conglomerados
    que remuestrea HORAS con evento (semilla, fecha, hora) completas: los pares O-D de
    una misma hora comparten los mismos tramos afectados y no son independientes.
    """
    filas = []
    for semilla in SEMILLAS:
        df = pd.read_csv(rutas_semilla(semilla)['evaluacion'])
        df.insert(0, 'semilla', semilla)
        filas.append(df)
    df_eval = pd.concat(filas, ignore_index=True)
    for sistema in SISTEMAS:
        df_eval[f'ahorro_{sistema}'] = df_eval['tiempo_real_estatico'] - df_eval[f'tiempo_real_{sistema}']
        df_eval[f'perdida_{sistema}'] = df_eval[f'ahorro_{sistema}'].clip(upper=0)

    columnas = [f'ahorro_{s}' for s in SISTEMAS] + [f'perdida_{s}' for s in SISTEMAS] + \
               [f'cambio_ruta_{s}' for s in SISTEMAS]
    por_hora = df_eval.groupby(['semilla', 'fecha', 'hora'])[columnas].sum()
    por_hora['casos'] = df_eval.groupby(['semilla', 'fecha', 'hora']).size()
    por_hora.to_csv(ARCHIVO_RESULTADOS_RUTEO)

    print("\n=== Ruteo en horas con evento: ahorro total (min) por semilla ===")
    por_semilla = por_hora.groupby('semilla')[[f'ahorro_{s}' for s in SISTEMAS] + ['casos']].sum()
    por_semilla.insert(0, 'horas_con_evento', por_hora.groupby('semilla').size())
    print(por_semilla.round(1).to_string())

    def estadisticos(muestra):
        total = muestra.sum()
        res = {}
        for s in SISTEMAS:
            res[f'ahorro_{s}'] = total[f'ahorro_{s}']
            res[f'perdida_{s}'] = total[f'perdida_{s}']
        for s in ['reactivo', 'anticipatorio']:
            res[f'pct_capturado_{s}'] = 100 * total[f'ahorro_{s}'] / total['ahorro_oraculo']
        res['diferencia_min'] = total['ahorro_anticipatorio'] - total['ahorro_reactivo']
        res['diferencia_perdida_min'] = total['perdida_anticipatorio'] - total['perdida_reactivo']
        return res

    puntual = estadisticos(por_hora)
    rng = np.random.default_rng(0)
    muestras = []
    for _ in range(REPETICIONES_BOOTSTRAP):
        idx = rng.integers(len(por_hora), size=len(por_hora))
        muestra = por_hora.iloc[idx]
        if muestra['ahorro_oraculo'].sum() > 0:
            muestras.append(estadisticos(muestra))
    df_boot = pd.DataFrame(muestras)

    horas_con_ahorro = int((por_hora['ahorro_oraculo'] > 0.5).sum())
    print(f"\n=== Ruteo agregado ({len(por_hora)} horas con evento, {int(por_hora['casos'].sum())} casos; "
          f"solo {horas_con_ahorro} horas con ahorro posible > 0.5 min) ===")
    for s in SISTEMAS:
        print(f"{s}: ahorro total {por_hora[f'ahorro_{s}'].sum():.1f} min, pérdidas {por_hora[f'perdida_{s}'].sum():.1f} min, "
              f"cambios de ruta {int(por_hora[f'cambio_ruta_{s}'].sum())}")
    filas_ic = []
    for clave, valor in puntual.items():
        bajo, alto = df_boot[clave].quantile([0.025, 0.975])
        filas_ic.append({'estadistico': clave, 'puntual': valor, 'ic95_bajo': bajo, 'ic95_alto': alto})
        print(f"{clave}: {valor:.1f} [IC 95% bootstrap: {bajo:.1f}, {alto:.1f}]")
    prob_ahorro = 100 * (df_boot['diferencia_min'] > 0).mean()
    prob_perdida = 100 * (df_boot['diferencia_perdida_min'] > 0).mean()
    print(f"Probabilidad bootstrap de que el anticipatorio ahorre más que el reactivo: {prob_ahorro:.0f}%")
    print(f"Probabilidad bootstrap de que el anticipatorio pierda menos que el reactivo: {prob_perdida:.0f}%")
    filas_ic.append({'estadistico': 'prob_anticipatorio_ahorra_mas', 'puntual': prob_ahorro})
    filas_ic.append({'estadistico': 'prob_anticipatorio_pierde_menos', 'puntual': prob_perdida})
    pd.DataFrame(filas_ic).to_csv(ARCHIVO_BOOTSTRAP, index=False)
    return por_hora

def agregar_modelos():
    filas = []
    for semilla in SEMILLAS:
        df = pd.read_csv(rutas_semilla(semilla)['comparacion'])
        df.insert(0, 'semilla', semilla)
        filas.append(df)
    df_modelos = pd.concat(filas, ignore_index=True)
    df_modelos.to_csv(ARCHIVO_RESULTADOS_MODELOS, index=False)

    validacion = df_modelos[df_modelos['evaluacion'].str.startswith('validacion')].copy()
    validacion['configuracion'] = validacion['modelo'] + np.where(validacion['usar_geo'], ' + línea/tramo', ' (sin ubicación)')
    # Métricas agregadas sobre todas las semillas, ponderando por número de filas con evento
    validacion['se_evento'] = validacion['RMSE_evento'] ** 2 * validacion['n_evento']
    agregado = validacion.groupby('configuracion').agg(
        RMSE=('RMSE', 'mean'), MAE=('MAE', 'mean'),
        se_evento=('se_evento', 'sum'), n_evento=('n_evento', 'sum'),
        MAE_evento=('MAE_evento', 'mean'))
    agregado['RMSE_evento_agregado'] = np.sqrt(agregado['se_evento'] / agregado['n_evento'])
    print("\n=== Modelos: validación por días, agregada sobre todas las semillas ===")
    print(agregado.drop(columns='se_evento').sort_values('RMSE_evento_agregado').round(4).to_string())

    # Estabilidad de los criterios de selección por semilla (el modelo exportado es fijo)
    for criterio in ('RMSE_evento', 'RMSE'):
        ganadores = validacion.loc[validacion.groupby('semilla')[criterio].idxmin(), 'configuracion']
        print(f"\nGanador por semilla con menor {criterio}:")
        print(ganadores.value_counts().to_string())
    return df_modelos

if __name__ == "__main__":
    # --solo-agregar: recalcula los resúmenes con las corridas ya existentes
    if "--solo-agregar" not in sys.argv:
        with ThreadPoolExecutor(max_workers=CORRIDAS_EN_PARALELO) as ejecutor:
            list(ejecutor.map(correr_semilla, SEMILLAS))
    agregar_modelos()
    agregar_ruteo()
