"""
Repite el pipeline simulador -> entrenador -> ruteo con varias semillas de eventos
estocásticos y reporta la media y el intervalo de confianza de los resultados.

Con las tasas calibradas los eventos son raros (unas pocas decenas de filas con evento
en el set de prueba por semilla), así que una sola corrida no basta para concluir qué
sistema de ruteo o qué modelo es mejor. La demanda (matrices O-D) no depende de la
semilla: solo cambian los eventos.
"""
import argparse
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

# Variantes del experimento (ver avances.md, Sección 7.5). Cada una escribe en
# DIRECTORIO_SALIDA/<variante>/semilla_<s>/ para no sobrescribir a las demás.
# - simulador / entrenador: argumentos extra de cada paso.
# - dataset_de: variante cuyo dataset se reutiliza (las que solo cambian el modelo no
#   necesitan resimular: con la misma semilla el dataset sería idéntico).
VARIANTES = {
    'base': {'simulador': [], 'entrenador': [], 'dataset_de': None},
    'hgb': {'simulador': [], 'entrenador': ['--configuracion', 'HistGradientBoosting+geo'], 'dataset_de': 'base'},
    'aviso_perfecto': {'simulador': ['--ruido-aviso', '0'], 'entrenador': [], 'dataset_de': None},
    'aviso_ruidoso': {'simulador': ['--ruido-aviso', '0.5'], 'entrenador': [], 'dataset_de': None},
}

def archivo_resultados(nombre, variante):
    # 'base' conserva los nombres de la Sección 7
    sufijo = "" if variante == 'base' else f"_{variante}"
    return f"modelos/resultados_semillas_{nombre}{sufijo}.csv"

def rutas_semilla(semilla, variante='base'):
    base = os.path.join(DIRECTORIO_SALIDA, variante, f"semilla_{semilla}")
    origen_dataset = VARIANTES[variante]['dataset_de'] or variante
    return {
        'dir': base,
        'dataset': os.path.join(DIRECTORIO_SALIDA, origen_dataset, f"semilla_{semilla}", "dataset.csv"),
        'modelo': os.path.join(base, "modelo.pkl"),
        'comparacion': os.path.join(base, "comparacion_modelos.csv"),
        'grafica': os.path.join(base, "importancia_variables.png"),
        'evaluacion': os.path.join(base, "evaluacion_ruteo.csv"),
        'resumen': os.path.join(base, "resumen_ruteo.csv"),
        'log': os.path.join(base, "log.txt"),
    }

def correr_semilla(semilla, variante='base', reusar_dataset=False):
    r = rutas_semilla(semilla, variante)
    config = VARIANTES[variante]
    os.makedirs(r['dir'], exist_ok=True)
    pasos = []
    if reusar_dataset and os.path.exists(r['dataset']):
        print(f"[{variante} {semilla}] reusa {r['dataset']}", flush=True)
    else:
        os.makedirs(os.path.dirname(r['dataset']), exist_ok=True)
        pasos.append(["simulador_congestion.py", "--semilla", str(semilla), "--salida", r['dataset']] + config['simulador'])
    pasos += [
        ["entrenador_anticipatorio.py", "--dataset", r['dataset'], "--modelo", r['modelo'],
         "--comparacion", r['comparacion'], "--grafica", r['grafica']] + config['entrenador'],
        ["ruteo_anticipatorio.py", "--dataset", r['dataset'], "--modelo", r['modelo'],
         "--evaluacion", r['evaluacion'], "--resumen", r['resumen']],
    ]
    with open(r['log'], "w", encoding="utf-8") as log:
        for paso in pasos:
            print(f"[{variante} {semilla}] {paso[0]}...", flush=True)
            subprocess.run([sys.executable] + paso, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "MPLBACKEND": "Agg"})
    print(f"[{variante} {semilla}] listo", flush=True)
    return semilla

SISTEMAS = ['reactivo', 'anticipatorio', 'oraculo']
REPETICIONES_BOOTSTRAP = 5000

def cargar_por_hora(variante):
    """Casos O-D de todas las semillas, sumados por hora con evento (semilla, fecha, hora)."""
    filas = []
    for semilla in SEMILLAS:
        df = pd.read_csv(rutas_semilla(semilla, variante)['evaluacion'])
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
    return por_hora

def agregar_ruteo(variante='base'):
    """
    El % del ahorro posible capturado NO se promedia por semilla: en semillas donde el
    oráculo casi no puede ahorrar (ningún evento que abra una ruta alternativa) el
    cociente se dispara o queda indefinido. En su lugar se juntan los casos de todas las
    semillas y el intervalo de confianza se obtiene con un bootstrap por conglomerados
    que remuestrea HORAS con evento (semilla, fecha, hora) completas: los pares O-D de
    una misma hora comparten los mismos tramos afectados y no son independientes.
    """
    por_hora = cargar_por_hora(variante)
    por_hora.to_csv(archivo_resultados('ruteo', variante))

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
    print(f"\n=== [{variante}] Ruteo agregado ({len(por_hora)} horas con evento, {int(por_hora['casos'].sum())} casos; "
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
    pd.DataFrame(filas_ic).to_csv(archivo_resultados('bootstrap', variante), index=False)
    return por_hora

def comparar_variantes(variante_a, variante_b):
    """
    Comparación PAREADA del sistema anticipatorio entre dos variantes. Con las mismas
    semillas las horas con evento son las mismas (el dataset solo difiere en columnas que
    no deciden qué horas se evalúan), así que cada remuestreo toma horas completas y suma,
    para esas mismas horas, la diferencia b − a. La variación entre horas se cancela y el
    IC es más estrecho que comparar los IC de cada variante por separado.
    """
    a, b = cargar_por_hora(variante_a), cargar_por_hora(variante_b)
    union = a.index.union(b.index)
    comunes = a.index.intersection(b.index)
    if len(comunes) != len(union):
        print(f"AVISO: {len(union) - len(comunes)} horas no están en ambas variantes; se comparan solo las {len(comunes)} comunes")
    a, b = a.loc[comunes], b.loc[comunes]
    for s in ('reactivo', 'oraculo'):
        desajuste = (a[f'ahorro_{s}'] - b[f'ahorro_{s}']).abs().max()
        if desajuste > 1e-6:
            print(f"AVISO: el ahorro del {s} difiere entre variantes (máx. {desajuste:.3f} min por hora)")

    pareado = pd.DataFrame({
        'ahorro_a': a['ahorro_anticipatorio'], 'ahorro_b': b['ahorro_anticipatorio'],
        'perdida_a': a['perdida_anticipatorio'], 'perdida_b': b['perdida_anticipatorio'],
        'ahorro_reactivo': b['ahorro_reactivo'], 'perdida_reactivo': b['perdida_reactivo'],
        'ahorro_oraculo': b['ahorro_oraculo'],
    })

    def estadisticos(muestra):
        t = muestra.sum()
        return {
            'pct_capturado_a': 100 * t['ahorro_a'] / t['ahorro_oraculo'],
            'pct_capturado_b': 100 * t['ahorro_b'] / t['ahorro_oraculo'],
            'diferencia_ahorro_b_menos_a': t['ahorro_b'] - t['ahorro_a'],
            'diferencia_perdida_b_menos_a': t['perdida_b'] - t['perdida_a'],
            'diferencia_ahorro_b_menos_reactivo': t['ahorro_b'] - t['ahorro_reactivo'],
            'diferencia_perdida_b_menos_reactivo': t['perdida_b'] - t['perdida_reactivo'],
        }

    puntual = estadisticos(pareado)
    rng = np.random.default_rng(0)
    muestras = []
    for _ in range(REPETICIONES_BOOTSTRAP):
        muestra = pareado.iloc[rng.integers(len(pareado), size=len(pareado))]
        if muestra['ahorro_oraculo'].sum() > 0:
            muestras.append(estadisticos(muestra))
    df_boot = pd.DataFrame(muestras)

    print(f"\n=== Comparación pareada anticipatorio: {variante_b} − {variante_a} ({len(pareado)} horas) ===")
    print(f"{variante_a}: ahorro {pareado['ahorro_a'].sum():.1f}, pérdidas {pareado['perdida_a'].sum():.1f}")
    print(f"{variante_b}: ahorro {pareado['ahorro_b'].sum():.1f}, pérdidas {pareado['perdida_b'].sum():.1f}")
    filas = []
    for clave, valor in puntual.items():
        bajo, alto = df_boot[clave].quantile([0.025, 0.975])
        filas.append({'estadistico': clave, 'puntual': valor, 'ic95_bajo': bajo, 'ic95_alto': alto})
        print(f"{clave}: {valor:.1f} [IC 95% bootstrap: {bajo:.1f}, {alto:.1f}]")
    probabilidades = {
        f'prob_{variante_b}_ahorra_mas_que_{variante_a}': (df_boot['diferencia_ahorro_b_menos_a'] > 0).mean(),
        f'prob_{variante_b}_pierde_menos_que_{variante_a}': (df_boot['diferencia_perdida_b_menos_a'] > 0).mean(),
        f'prob_{variante_b}_ahorra_mas_que_reactivo': (df_boot['diferencia_ahorro_b_menos_reactivo'] > 0).mean(),
        f'prob_{variante_b}_pierde_menos_que_reactivo': (df_boot['diferencia_perdida_b_menos_reactivo'] > 0).mean(),
    }
    for clave, p in probabilidades.items():
        print(f"{clave}: {100 * p:.1f}%")
        filas.append({'estadistico': clave, 'puntual': 100 * p})
    por_hora_ganador = np.sign(pareado['ahorro_b'] - pareado['ahorro_a'])
    print(f"Horas en que {variante_b} ahorra más / menos / igual que {variante_a}: "
          f"{(por_hora_ganador > 0).sum()} / {(por_hora_ganador < 0).sum()} / {(por_hora_ganador == 0).sum()}")
    pd.DataFrame(filas).to_csv(f"modelos/comparacion_pareada_{variante_b}_vs_{variante_a}.csv", index=False)
    return pareado

def agregar_modelos(variante='base'):
    filas = []
    for semilla in SEMILLAS:
        df = pd.read_csv(rutas_semilla(semilla, variante)['comparacion'])
        df.insert(0, 'semilla', semilla)
        filas.append(df)
    df_modelos = pd.concat(filas, ignore_index=True)
    df_modelos.to_csv(archivo_resultados('modelos', variante), index=False)

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
    parser = argparse.ArgumentParser(description="Pipeline multi-semilla por variante y comparación pareada.")
    parser.add_argument("--variante", default='base', choices=list(VARIANTES))
    parser.add_argument("--reusar-dataset", action="store_true",
                        help="Omite el simulador si el dataset de la semilla ya existe")
    parser.add_argument("--solo-agregar", action="store_true",
                        help="Recalcula los resúmenes con las corridas ya existentes")
    parser.add_argument("--comparar", nargs=2, metavar=('A', 'B'), choices=list(VARIANTES),
                        help="Solo la comparación pareada B − A entre dos variantes ya corridas")
    parser.add_argument("--paralelo", type=int, default=CORRIDAS_EN_PARALELO)
    args = parser.parse_args()

    if args.comparar:
        comparar_variantes(*args.comparar)
        sys.exit(0)
    if not args.solo_agregar:
        with ThreadPoolExecutor(max_workers=args.paralelo) as ejecutor:
            list(ejecutor.map(lambda s: correr_semilla(s, args.variante, args.reusar_dataset), SEMILLAS))
    agregar_modelos(args.variante)
    agregar_ruteo(args.variante)
