"""
Repite el pipeline simulador -> entrenador -> ruteo con varias semillas de eventos
estocásticos y reporta la media y el intervalo de confianza de los resultados.

Con las tasas calibradas los eventos son raros (unas pocas decenas de filas con evento
en el set de prueba por semilla), así que una sola corrida no basta para concluir qué
sistema de ruteo o qué modelo es mejor. La demanda (matrices O-D) no depende de la
semilla: solo cambian los eventos.

Desde la Sección 13 de README.md el pipeline trabaja en bloques de 15 min y el ruteo se
evalúa a 15, 30, 45 y 60 min: todos los agregados se reportan por horizonte.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd

# Las 5 semillas originales + 25 nuevas: con eventos raros, 5 semillas dejaban solo ~12
# horas con incidentes de impacto y la comparación de sistemas no era concluyente.
SEMILLAS = [42, 7, 13, 101, 2026] + list(range(1001, 1026))
CORRIDAS_EN_PARALELO = 3
DIRECTORIO_SALIDA = "datos_procesados/semillas"

# Variantes del experimento (ver README.md, Sección 7.5). Cada una escribe en
# DIRECTORIO_SALIDA/<variante>/semilla_<s>/ para no sobrescribir a las demás.
# - simulador / entrenador: argumentos extra de cada paso.
# - dataset_de: variante cuyo dataset se reutiliza (las que solo cambian el modelo no
#   necesitan resimular: con la misma semilla el dataset sería idéntico).
VARIANTES = {
    'base': {'simulador': [], 'entrenador': [], 'dataset_de': None},
    'hgb': {'simulador': [], 'entrenador': ['--configuracion', 'HistGradientBoosting+geo'], 'dataset_de': 'base'},
    'aviso_perfecto': {'simulador': ['--ruido-aviso', '0'], 'entrenador': [], 'dataset_de': None},
    'aviso_ruidoso': {'simulador': ['--ruido-aviso', '0.5'], 'entrenador': [], 'dataset_de': None},
    # Error aditivo δ ∈ {−1, 0, +1} (ver README.md, Sección 9): a diferencia del ruido
    # multiplicativo, puede anunciar 0 cuando el incidente sigue.
    'aviso_aditivo': {'simulador': ['--error-aviso-aditivo', '1'], 'entrenador': [], 'dataset_de': None},
    # Sensibilidad a la probabilidad de error (ver README.md, Sección 11): el aviso es exacto
    # salvo en una fracción p de los casos, donde se equivoca en ±1 bloque (15 min).
    # aviso_aditivo equivale a p = 2/3 y aviso_perfecto a p = 0.
    **{f'aviso_error_{pct}': {'simulador': ['--prob-error-aviso', str(pct / 100)], 'entrenador': [], 'dataset_de': None}
       for pct in (10, 20, 33)},
}

# Con bloques de 15 min la comparación de 6 configuraciones tarda ~10 min por semilla. El
# modelo es fijo (RF + línea/tramo, Sección 7.1; se confirmó con la semilla 42 en la
# Sección 13), así que las corridas multi-semilla entrenan solo esa configuración.
ARGS_ENTRENADOR_COMUNES = ['--sin-comparacion']

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
        'resumen_edad': os.path.join(base, "resumen_ruteo_por_edad.csv"),
        'log': os.path.join(base, "log.txt"),
    }

def correr_semilla(semilla, variante='base', reusar_dataset=False, solo_ruteo=False):
    r = rutas_semilla(semilla, variante)
    config = VARIANTES[variante]
    os.makedirs(r['dir'], exist_ok=True)
    pasos = []
    if solo_ruteo:
        # Reevalúa el ruteo con el dataset y el modelo ya guardados (sin resimular ni reentrenar)
        pass
    elif reusar_dataset and os.path.exists(r['dataset']):
        print(f"[{variante} {semilla}] reusa {r['dataset']}", flush=True)
    else:
        os.makedirs(os.path.dirname(r['dataset']), exist_ok=True)
        pasos.append(["simulador_congestion.py", "--semilla", str(semilla), "--salida", r['dataset']] + config['simulador'])
    if not solo_ruteo:
        pasos.append(["entrenador_anticipatorio.py", "--dataset", r['dataset'], "--modelo", r['modelo'],
                      "--comparacion", r['comparacion'], "--grafica", r['grafica']]
                     + ARGS_ENTRENADOR_COMUNES + config['entrenador'])
    pasos.append(["ruteo_anticipatorio.py", "--dataset", r['dataset'], "--modelo", r['modelo'],
                  "--evaluacion", r['evaluacion'], "--resumen", r['resumen'], "--resumen-edad", r['resumen_edad']])
    with open(r['log'], "a" if solo_ruteo else "w", encoding="utf-8") as log:
        for paso in pasos:
            print(f"[{variante} {semilla}] {paso[0]}...", flush=True)
            subprocess.run([sys.executable] + paso, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "MPLBACKEND": "Agg"})
    print(f"[{variante} {semilla}] listo", flush=True)
    return semilla

# reactivo_duracion: reactivo que ignora los eventos con 2 h o más de edad (Sección 11).
# hibrido: IA en eventos jóvenes + la misma regla en los viejos (Sección 12). Los sufijos
# _1 y _3 son los umbrales alternativos de la regla (edad >= 1 h y >= 3 h, es decir 4 y 12
# bloques); sin sufijo, >= 2 h.
UMBRALES_REGLA = {1: '_1', 2: '', 3: '_3'}
SISTEMAS = (['reactivo'] + [f'reactivo_duracion{s}' for s in UMBRALES_REGLA.values()] + ['anticipatorio']
            + [f'hibrido{s}' for s in UMBRALES_REGLA.values()] + ['oraculo'])
REACTIVOS = ['reactivo', 'reactivo_duracion']
# Pares (b, a) cuya diferencia b − a se remuestrea: para cada umbral, el híbrido contra la
# regla sola, contra la IA y contra el reactivo, y la regla contra la IA.
PARES_HIBRIDO = [p for s in UMBRALES_REGLA.values() for p in (
    (f'hibrido{s}', f'reactivo_duracion{s}'), (f'hibrido{s}', 'anticipatorio'), (f'hibrido{s}', 'reactivo'),
    (f'reactivo_duracion{s}', 'anticipatorio'))]
REPETICIONES_BOOTSTRAP = 5000

HORIZONTES = [15, 30, 45, 60]
# Conglomerado del bootstrap: un bloque de 15 min con evento de una semilla. Los pares O-D
# de un mismo bloque comparten los mismos tramos afectados y no son independientes. Los
# bloques consecutivos de un mismo evento también están correlacionados, así que los IC
# pueden ser algo optimistas (lo mismo pasaba con las horas consecutivas en la versión horaria).
CONGLOMERADO = ['semilla', 'fecha', 'bloque']
NIVEL_FINO = ['horizonte_min'] + CONGLOMERADO + ['hora_evento', 'tipo_evento']
_conglomerados = {}

def cargar_conglomerados(variante):
    """
    Casos O-D de todas las semillas, sumados al nivel más fino que usa el análisis
    (horizonte, bloque con evento, hora del evento y tipo de evento). Cada
    evaluacion_ruteo.csv pesa ~100 MB, así que se agrega semilla por semilla en vez de
    concatenar los casos de todas; el resultado se guarda para las demás tablas.
    """
    if variante in _conglomerados:
        return _conglomerados[variante]
    columnas = [f'ahorro_{s}' for s in SISTEMAS] + [f'perdida_{s}' for s in SISTEMAS] + \
               [f'cambio_ruta_{s}' for s in SISTEMAS]
    usecols = NIVEL_FINO[:1] + NIVEL_FINO[2:] + ['tiempo_real_estatico'] + \
              [f'{c}_{s}' for s in SISTEMAS for c in ('tiempo_real', 'cambio_ruta')]
    filas = []
    for semilla in SEMILLAS:
        df = pd.read_csv(rutas_semilla(semilla, variante)['evaluacion'], usecols=usecols)
        df.insert(0, 'semilla', semilla)
        for sistema in SISTEMAS:
            df[f'ahorro_{sistema}'] = df['tiempo_real_estatico'] - df[f'tiempo_real_{sistema}']
            df[f'perdida_{sistema}'] = df[f'ahorro_{sistema}'].clip(upper=0)
        agregado = df.groupby(NIVEL_FINO)[columnas].sum()
        agregado['casos'] = df.groupby(NIVEL_FINO).size()
        filas.append(agregado)
    _conglomerados[variante] = pd.concat(filas)
    return _conglomerados[variante]

def cargar_por_bloque(variante, horizonte, por_edad=False, por_tipo=False):
    """
    Casos de un horizonte sumados por bloque con evento (semilla, fecha, bloque). Con
    por_edad=True se separan además por la hora del evento que cruza cada caso (0, +1,
    +2, ...), de modo que un bloque puede aportar un conglomerado a cada edad; con
    por_tipo=True, también por el tipo de ese evento.
    """
    finos = cargar_conglomerados(variante).xs(horizonte, level='horizonte_min')
    claves = CONGLOMERADO + (['hora_evento'] if por_edad else []) + (['tipo_evento'] if por_tipo else [])
    return finos.groupby(level=claves).sum()

def estadisticos_ruteo(muestra):
    total = muestra.sum()
    res = {}
    for s in SISTEMAS:
        res[f'ahorro_{s}'] = total[f'ahorro_{s}']
        res[f'perdida_{s}'] = total[f'perdida_{s}']
    for s in REACTIVOS + ['anticipatorio']:
        res[f'pct_capturado_{s}'] = 100 * total[f'ahorro_{s}'] / total['ahorro_oraculo']
    res['diferencia_min'] = total['ahorro_anticipatorio'] - total['ahorro_reactivo']
    res['diferencia_perdida_min'] = total['perdida_anticipatorio'] - total['perdida_reactivo']
    res['diferencia_vs_duracion_min'] = total['ahorro_anticipatorio'] - total['ahorro_reactivo_duracion']
    res['diferencia_perdida_vs_duracion_min'] = total['perdida_anticipatorio'] - total['perdida_reactivo_duracion']
    res['diferencia_duracion_vs_reactivo_min'] = total['ahorro_reactivo_duracion'] - total['ahorro_reactivo']
    for b, a in PARES_HIBRIDO:
        res[f'pct_capturado_{b}'] = 100 * total[f'ahorro_{b}'] / total['ahorro_oraculo']
        res[f'dif_ahorro_{b}__{a}'] = total[f'ahorro_{b}'] - total[f'ahorro_{a}']
        res[f'dif_perdida_{b}__{a}'] = total[f'perdida_{b}'] - total[f'perdida_{a}']
    return res

def probabilidades_ruteo(df_boot):
    """P bootstrap (%) de que la IA supere a cada reactivo, y de que la regla de duración mejore al reactivo."""
    return {
        'prob_anticipatorio_ahorra_mas': 100 * (df_boot['diferencia_min'] > 0).mean(),
        'prob_anticipatorio_pierde_menos': 100 * (df_boot['diferencia_perdida_min'] > 0).mean(),
        'prob_anticipatorio_ahorra_mas_que_duracion': 100 * (df_boot['diferencia_vs_duracion_min'] > 0).mean(),
        'prob_anticipatorio_pierde_menos_que_duracion': 100 * (df_boot['diferencia_perdida_vs_duracion_min'] > 0).mean(),
        'prob_duracion_ahorra_mas_que_reactivo': 100 * (df_boot['diferencia_duracion_vs_reactivo_min'] > 0).mean(),
        **{f'prob_{b}_ahorra_mas_que_{a}': 100 * (df_boot[f'dif_ahorro_{b}__{a}'] > 0).mean() for b, a in PARES_HIBRIDO},
        **{f'prob_{b}_pierde_menos_que_{a}': 100 * (df_boot[f'dif_perdida_{b}__{a}'] > 0).mean() for b, a in PARES_HIBRIDO},
    }

def bootstrap_ruteo(por_bloque):
    """Estadísticos puntuales y remuestreos por conglomerados (filas de por_bloque completas)."""
    puntual = estadisticos_ruteo(por_bloque)
    rng = np.random.default_rng(0)
    muestras = []
    for _ in range(REPETICIONES_BOOTSTRAP):
        muestra = por_bloque.iloc[rng.integers(len(por_bloque), size=len(por_bloque))]
        if muestra['ahorro_oraculo'].sum() > 0:
            muestras.append(estadisticos_ruteo(muestra))
    return puntual, pd.DataFrame(muestras)

def agregar_ruteo(variante='base'):
    """
    El % del ahorro posible capturado NO se promedia por semilla: en semillas donde el
    oráculo casi no puede ahorrar (ningún evento que abra una ruta alternativa) el
    cociente se dispara o queda indefinido. En su lugar se juntan los casos de todas las
    semillas y el intervalo de confianza se obtiene con un bootstrap por conglomerados
    que remuestrea BLOQUES con evento (semilla, fecha, bloque) completos. Cada horizonte
    se analiza por separado.
    """
    tablas, filas_ic = [], []
    for horizonte in HORIZONTES:
        por_bloque = cargar_por_bloque(variante, horizonte)
        tablas.append(por_bloque.assign(horizonte_min=horizonte))

        print(f"\n=== [{variante}, {horizonte} min] Ruteo en bloques con evento: ahorro total (min) por semilla ===")
        por_semilla = por_bloque.groupby('semilla')[[f'ahorro_{s}' for s in SISTEMAS] + ['casos']].sum()
        por_semilla.insert(0, 'bloques_con_evento', por_bloque.groupby('semilla').size())
        print(por_semilla.round(1).to_string())

        puntual, df_boot = bootstrap_ruteo(por_bloque)

        bloques_con_ahorro = int((por_bloque['ahorro_oraculo'] > 0.5).sum())
        print(f"\n=== [{variante}, {horizonte} min] Ruteo agregado ({len(por_bloque)} bloques con evento, "
              f"{int(por_bloque['casos'].sum())} casos; solo {bloques_con_ahorro} bloques con ahorro posible > 0.5 min) ===")
        for s in SISTEMAS:
            print(f"{s}: ahorro total {por_bloque[f'ahorro_{s}'].sum():.1f} min, pérdidas {por_bloque[f'perdida_{s}'].sum():.1f} min, "
                  f"cambios de ruta {int(por_bloque[f'cambio_ruta_{s}'].sum())}")
        for clave, valor in puntual.items():
            bajo, alto = df_boot[clave].quantile([0.025, 0.975])
            filas_ic.append({'horizonte_min': horizonte, 'estadistico': clave, 'puntual': valor, 'ic95_bajo': bajo, 'ic95_alto': alto})
            print(f"{clave}: {valor:.1f} [IC 95% bootstrap: {bajo:.1f}, {alto:.1f}]")
        for clave, p in probabilidades_ruteo(df_boot).items():
            print(f"{clave}: {p:.1f}%")
            filas_ic.append({'horizonte_min': horizonte, 'estadistico': clave, 'puntual': p})
    pd.concat(tablas).to_csv(archivo_resultados('ruteo', variante))
    pd.DataFrame(filas_ic).to_csv(archivo_resultados('bootstrap', variante), index=False)
    return pd.DataFrame(filas_ic)

EDADES_REPORTADAS = [0, 1, 2]  # hora del evento

def agregar_ruteo_por_edad(variante='base'):
    """
    Reactivo vs. anticipatorio separado por hora del evento: hora 0 (primera hora del
    evento), hora + 1 y hora + 2, en cada horizonte. Mismo bootstrap por conglomerados que
    agregar_ruteo, con un conglomerado por (bloque con evento, hora del evento).
    """
    filas = []
    print(f"\n=== [{variante}] Ruteo por hora del evento ===")
    for horizonte in HORIZONTES:
        por_bloque = cargar_por_bloque(variante, horizonte, por_edad=True)
        edades = por_bloque.index.get_level_values('hora_evento')
        for edad in EDADES_REPORTADAS:
            if edad not in edades:
                continue
            grupo = por_bloque.xs(edad, level='hora_evento')
            puntual, df_boot = bootstrap_ruteo(grupo)
            fila = {'horizonte_min': horizonte, 'hora_evento': edad, 'bloques': len(grupo), 'casos': int(grupo['casos'].sum()),
                    'bloques_con_ahorro_posible': int((grupo['ahorro_oraculo'] > 0.5).sum())}
            for clave, valor in puntual.items():
                fila[clave] = valor
                fila[f'{clave}_ic95_bajo'], fila[f'{clave}_ic95_alto'] = df_boot[clave].quantile([0.025, 0.975])
            fila.update(probabilidades_ruteo(df_boot))
            filas.append(fila)
            print(f"{horizonte} min, hora + {edad}: {fila['bloques']} bloques, {fila['casos']} casos | ahorro IA {puntual['ahorro_anticipatorio']:.0f} "
                  f"vs. reactivo {puntual['ahorro_reactivo']:.0f} vs. reactivo+duración {puntual['ahorro_reactivo_duracion']:.0f} "
                  f"vs. híbrido {puntual['ahorro_hibrido']:.0f} "
                  f"(oráculo {puntual['ahorro_oraculo']:.0f}) | IA − reactivo {puntual['diferencia_min']:.0f} "
                  f"[{fila['diferencia_min_ic95_bajo']:.0f}; {fila['diferencia_min_ic95_alto']:.0f}], "
                  f"P {fila['prob_anticipatorio_ahorra_mas']:.1f}% | IA − reactivo+duración {puntual['diferencia_vs_duracion_min']:.0f} "
                  f"[{fila['diferencia_vs_duracion_min_ic95_bajo']:.0f}; {fila['diferencia_vs_duracion_min_ic95_alto']:.0f}], "
                  f"P {fila['prob_anticipatorio_ahorra_mas_que_duracion']:.1f}%")
        fuera = por_bloque[~edades.isin(EDADES_REPORTADAS)]
        print(f"{horizonte} min: excluidos (hora del evento >= 3): {len(fuera)} conglomerados, {int(fuera['casos'].sum())} casos")
    pd.DataFrame(filas).to_csv(archivo_resultados('por_edad', variante), index=False)
    return pd.DataFrame(filas)

def agregar_ruteo_por_tipo(variante='base'):
    """
    Desglose de cada hora del evento por tipo del evento que cruza el caso (lluvia, falla
    mecánica, incidente de plataforma), en cada horizonte. Un conglomerado por (bloque con
    evento, hora del evento, tipo). Sirve para ver si la lluvia, que afecta líneas
    completas, domina el número de casos.
    """
    filas = []
    print(f"\n=== [{variante}] Ruteo por hora y tipo de evento ===")
    for horizonte in HORIZONTES:
        por_bloque = cargar_por_bloque(variante, horizonte, por_edad=True, por_tipo=True)
        for (edad, tipo), grupo in por_bloque.groupby(level=['hora_evento', 'tipo_evento']):
            if edad not in EDADES_REPORTADAS:
                continue
            puntual, df_boot = bootstrap_ruteo(grupo)
            fila = {'horizonte_min': horizonte, 'hora_evento': edad, 'tipo_evento': tipo, 'bloques': len(grupo),
                    'casos': int(grupo['casos'].sum()),
                    'bloques_con_ahorro_posible': int((grupo['ahorro_oraculo'] > 0.5).sum())}
            for s in SISTEMAS:
                fila[f'cambio_ruta_{s}'] = int(grupo[f'cambio_ruta_{s}'].sum())
            for clave, valor in puntual.items():
                fila[clave] = valor
                if not df_boot.empty:
                    fila[f'{clave}_ic95_bajo'], fila[f'{clave}_ic95_alto'] = df_boot[clave].quantile([0.025, 0.975])
            if not df_boot.empty:
                fila.update(probabilidades_ruteo(df_boot))
            filas.append(fila)
            print(f"{horizonte} min, hora + {edad} {tipo}: {fila['bloques']} bloques, {fila['casos']} casos | "
                  f"oráculo {puntual['ahorro_oraculo']:.0f} | IA {puntual['ahorro_anticipatorio']:.0f}, "
                  f"reactivo {puntual['ahorro_reactivo']:.0f}, reactivo+duración {puntual['ahorro_reactivo_duracion']:.0f}, "
                  f"híbrido {puntual['ahorro_hibrido']:.0f}")
    pd.DataFrame(filas).to_csv(archivo_resultados('por_tipo', variante), index=False)
    return pd.DataFrame(filas)

def comparar_variantes(variante_a, variante_b):
    """Comparación pareada de comparar_variantes_horizonte en cada horizonte; un solo CSV."""
    filas = []
    for horizonte in HORIZONTES:
        filas += comparar_variantes_horizonte(variante_a, variante_b, horizonte)
    pd.DataFrame(filas).to_csv(f"modelos/comparacion_pareada_{variante_b}_vs_{variante_a}.csv", index=False)

def comparar_variantes_horizonte(variante_a, variante_b, horizonte):
    """
    Comparación PAREADA del sistema anticipatorio entre dos variantes. Con las mismas
    semillas los bloques con evento son los mismos (el dataset solo difiere en columnas que
    no deciden qué bloques se evalúan), así que cada remuestreo toma bloques completos y
    suma, para esos mismos bloques, la diferencia b − a. La variación entre bloques se
    cancela y el IC es más estrecho que comparar los IC de cada variante por separado.
    """
    a, b = cargar_por_bloque(variante_a, horizonte), cargar_por_bloque(variante_b, horizonte)
    union = a.index.union(b.index)
    comunes = a.index.intersection(b.index)
    if len(comunes) != len(union):
        print(f"AVISO: {len(union) - len(comunes)} bloques no están en ambas variantes; se comparan solo los {len(comunes)} comunes")
    a, b = a.loc[comunes], b.loc[comunes]
    for s in ('reactivo', 'reactivo_duracion', 'oraculo'):
        desajuste = (a[f'ahorro_{s}'] - b[f'ahorro_{s}']).abs().max()
        if desajuste > 1e-6:
            print(f"AVISO: el ahorro del {s} difiere entre variantes (máx. {desajuste:.3f} min por bloque)")

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

    print(f"\n=== Comparación pareada anticipatorio, {horizonte} min: {variante_b} − {variante_a} ({len(pareado)} bloques) ===")
    print(f"{variante_a}: ahorro {pareado['ahorro_a'].sum():.1f}, pérdidas {pareado['perdida_a'].sum():.1f}")
    print(f"{variante_b}: ahorro {pareado['ahorro_b'].sum():.1f}, pérdidas {pareado['perdida_b'].sum():.1f}")
    filas = []
    for clave, valor in puntual.items():
        bajo, alto = df_boot[clave].quantile([0.025, 0.975])
        filas.append({'horizonte_min': horizonte, 'estadistico': clave, 'puntual': valor, 'ic95_bajo': bajo, 'ic95_alto': alto})
        print(f"{clave}: {valor:.1f} [IC 95% bootstrap: {bajo:.1f}, {alto:.1f}]")
    probabilidades = {
        f'prob_{variante_b}_ahorra_mas_que_{variante_a}': (df_boot['diferencia_ahorro_b_menos_a'] > 0).mean(),
        f'prob_{variante_b}_pierde_menos_que_{variante_a}': (df_boot['diferencia_perdida_b_menos_a'] > 0).mean(),
        f'prob_{variante_b}_ahorra_mas_que_reactivo': (df_boot['diferencia_ahorro_b_menos_reactivo'] > 0).mean(),
        f'prob_{variante_b}_pierde_menos_que_reactivo': (df_boot['diferencia_perdida_b_menos_reactivo'] > 0).mean(),
    }
    for clave, p in probabilidades.items():
        print(f"{clave}: {100 * p:.1f}%")
        filas.append({'horizonte_min': horizonte, 'estadistico': clave, 'puntual': 100 * p})
    ganador = np.sign(pareado['ahorro_b'] - pareado['ahorro_a'])
    print(f"Bloques en que {variante_b} ahorra más / menos / igual que {variante_a}: "
          f"{(ganador > 0).sum()} / {(ganador < 0).sum()} / {(ganador == 0).sum()}")
    return filas

# Variantes de la sensibilidad del aviso, de menor a mayor probabilidad de error (Sección 11)
SENSIBILIDAD_AVISO = [('aviso_perfecto', 0.0), ('aviso_error_10', 0.10), ('aviso_error_20', 0.20),
                      ('aviso_error_33', 0.33), ('aviso_aditivo', 2 / 3)]

def precision_aviso(variante):
    """
    Acierto del aviso en la distinción "termina en este bloque / sigue" y en el número
    exacto de bloques, sobre las filas de incidente o falla de todas las semillas. El
    restante real se toma del dataset de aviso_perfecto de la misma semilla (mismos eventos).
    """
    exacto, binario, n = 0, 0, 0
    for semilla in SEMILLAS:
        columnas = ['hay_evento', 'tipo_evento', 'bloques_restantes_anunciados']
        real = pd.read_csv(rutas_semilla(semilla, 'aviso_perfecto')['dataset'], usecols=columnas)
        aviso = pd.read_csv(rutas_semilla(semilla, variante)['dataset'], usecols=columnas)
        filas = (real['hay_evento'] == 1) & (real['tipo_evento'] != 'lluvia')
        r, a = real.loc[filas, 'bloques_restantes_anunciados'], aviso.loc[filas, 'bloques_restantes_anunciados']
        exacto += int((r == a).sum())
        binario += int(((r == 0) == (a == 0)).sum())
        n += int(filas.sum())
    return 100 * exacto / n, 100 * binario / n, n

def bootstrap_por_horizonte(variante):
    """Tabla bootstrap de agregar_ruteo, separada por horizonte e indexada por estadístico."""
    boot = pd.read_csv(archivo_resultados('bootstrap', variante))
    return {h: g.set_index('estadistico') for h, g in boot.groupby('horizonte_min')}

def fila_sensibilidad(fila, boot):
    """Agrega a la fila de sensibilidad los estadísticos de la IA de un horizonte."""
    for clave in ('ahorro_anticipatorio', 'perdida_anticipatorio', 'pct_capturado_anticipatorio',
                  'diferencia_min', 'diferencia_vs_duracion_min'):
        fila[clave] = boot.loc[clave, 'puntual']
        fila[f'{clave}_ic95_bajo'], fila[f'{clave}_ic95_alto'] = boot.loc[clave, ['ic95_bajo', 'ic95_alto']]
    for clave in ('prob_anticipatorio_ahorra_mas', 'prob_anticipatorio_ahorra_mas_que_duracion',
                  'prob_anticipatorio_pierde_menos_que_duracion'):
        fila[clave] = boot.loc[clave, 'puntual']

def resumen_sensibilidad_aviso():
    """Tabla de la sensibilidad a la probabilidad de error del aviso con las variantes ya corridas."""
    filas = []
    for variante, p in [('base', float('nan'))] + SENSIBILIDAD_AVISO:
        precision = precision_aviso(variante) if variante != 'base' else None
        for horizonte, boot in bootstrap_por_horizonte(variante).items():
            fila = {'horizonte_min': horizonte, 'variante': variante, 'prob_error': p}
            if precision:
                fila['pct_aviso_exacto'], fila['pct_termina_sigue_correcto'], fila['filas_aviso'] = precision
            fila_sensibilidad(fila, boot)
            filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv("modelos/resultados_sensibilidad_aviso.csv", index=False)
    print("\n=== Sensibilidad a la probabilidad de error del aviso ===")
    print(df[['horizonte_min', 'variante', 'prob_error', 'pct_termina_sigue_correcto', 'pct_capturado_anticipatorio',
              'prob_anticipatorio_ahorra_mas', 'prob_anticipatorio_ahorra_mas_que_duracion']].round(2).to_string(index=False))
    return df

def resumen_hibrido_umbral(variantes):
    """
    Tabla de la Sección 12: para cada variante ya corrida y cada umbral de la regla,
    ahorro, pérdidas y % capturado de la regla sola y del híbrido, y las probabilidades
    bootstrap de que el híbrido supere a la regla y a la IA.
    """
    filas = []
    for variante in variantes:
        for horizonte, boot in bootstrap_por_horizonte(variante).items():
            for umbral, s in UMBRALES_REGLA.items():
                fila = {'horizonte_min': horizonte, 'variante': variante, 'umbral_edad_h': umbral}
                for sistema, nombre in ((f'reactivo_duracion{s}', 'regla'), (f'hibrido{s}', 'hibrido'), ('anticipatorio', 'ia')):
                    for clave in (f'ahorro_{sistema}', f'perdida_{sistema}', f'pct_capturado_{sistema}'):
                        fila[clave.replace(sistema, nombre)] = boot.loc[clave, 'puntual']
                        fila[f"{clave.replace(sistema, nombre)}_ic95_bajo"], fila[f"{clave.replace(sistema, nombre)}_ic95_alto"] = \
                            boot.loc[clave, ['ic95_bajo', 'ic95_alto']]
                for b, a, nombre in ((f'hibrido{s}', f'reactivo_duracion{s}', 'hibrido_vs_regla'),
                                     (f'hibrido{s}', 'anticipatorio', 'hibrido_vs_ia'),
                                     (f'reactivo_duracion{s}', 'anticipatorio', 'regla_vs_ia')):
                    for medida in ('ahorro', 'perdida'):
                        clave = f'dif_{medida}_{b}__{a}'
                        fila[f'dif_{medida}_{nombre}'] = boot.loc[clave, 'puntual']
                        fila[f'dif_{medida}_{nombre}_ic95_bajo'], fila[f'dif_{medida}_{nombre}_ic95_alto'] = \
                            boot.loc[clave, ['ic95_bajo', 'ic95_alto']]
                    fila[f'prob_ahorra_mas_{nombre}'] = boot.loc[f'prob_{b}_ahorra_mas_que_{a}', 'puntual']
                    fila[f'prob_pierde_menos_{nombre}'] = boot.loc[f'prob_{b}_pierde_menos_que_{a}', 'puntual']
                filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv("modelos/resultados_hibrido_umbral.csv", index=False)
    print("\n=== Híbrido y sensibilidad del umbral de la regla ===")
    print(df[['horizonte_min', 'variante', 'umbral_edad_h', 'pct_capturado_regla', 'pct_capturado_ia', 'pct_capturado_hibrido',
              'prob_ahorra_mas_hibrido_vs_regla', 'prob_ahorra_mas_hibrido_vs_ia',
              'prob_pierde_menos_hibrido_vs_regla']].round(1).to_string(index=False))
    return df

def agregar_modelos(variante='base'):
    if not all(os.path.exists(rutas_semilla(s, variante)['comparacion']) for s in SEMILLAS):
        print(f"\n[{variante}] Sin comparación de candidatos en todas las semillas (--sin-comparacion); se omite")
        return None
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

def metricas_por_horizonte(variante='base'):
    """
    Métricas del modelo exportado (split 80/20) en cada horizonte, agregadas sobre las
    semillas: RMSE medio y RMSE en filas con evento ponderado por número de esas filas.
    """
    filas = []
    for semilla in SEMILLAS:
        paquete = joblib.load(rutas_semilla(semilla, variante)['modelo'])
        for horizonte, r in paquete['modelos_por_horizonte'].items():
            filas.append({'semilla': semilla, 'horizonte_min': horizonte, **r['metricas_test']})
    df = pd.DataFrame(filas)
    df['se_evento'] = df['RMSE_evento'] ** 2 * df['n_evento']
    agregado = df.groupby('horizonte_min').agg(RMSE=('RMSE', 'mean'), MAE=('MAE', 'mean'),
                                               se_evento=('se_evento', 'sum'), n_evento=('n_evento', 'sum'))
    agregado['RMSE_evento'] = np.sqrt(agregado['se_evento'] / agregado['n_evento'])
    return agregado.drop(columns='se_evento')

SISTEMAS_TABLA_HORIZONTES = ['reactivo', 'reactivo_duracion', 'anticipatorio', 'hibrido']

def resumen_horizontes(variantes):
    """
    Tabla principal de la Sección 13: por variante y horizonte, % del ahorro posible que
    captura cada sistema (con IC), probabilidad bootstrap de que la IA supere al reactivo
    y a la regla de duración, y el error del modelo en ese horizonte.
    """
    filas = []
    for variante in variantes:
        metricas = metricas_por_horizonte(variante)
        for horizonte, boot in bootstrap_por_horizonte(variante).items():
            fila = {'variante': variante, 'horizonte_min': horizonte,
                    'ahorro_oraculo': boot.loc['ahorro_oraculo', 'puntual']}
            for sistema in SISTEMAS_TABLA_HORIZONTES:
                clave = f'pct_capturado_{sistema}'
                fila[clave] = boot.loc[clave, 'puntual']
                fila[f'{clave}_ic95_bajo'], fila[f'{clave}_ic95_alto'] = boot.loc[clave, ['ic95_bajo', 'ic95_alto']]
            for clave in ('perdida_reactivo', 'perdida_anticipatorio', 'perdida_hibrido',
                          'prob_anticipatorio_ahorra_mas', 'prob_anticipatorio_ahorra_mas_que_duracion'):
                fila[clave] = boot.loc[clave, 'puntual']
            for clave in ('RMSE', 'RMSE_evento', 'n_evento'):
                fila[f'modelo_{clave}'] = metricas.loc[horizonte, clave]
            filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv("modelos/resultados_horizontes.csv", index=False)
    print("\n=== Resumen por horizonte ===")
    print(df[['variante', 'horizonte_min'] + [f'pct_capturado_{s}' for s in SISTEMAS_TABLA_HORIZONTES]
             + ['prob_anticipatorio_ahorra_mas', 'modelo_RMSE_evento']].round(2).to_string(index=False))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline multi-semilla por variante y comparación pareada.")
    parser.add_argument("--variante", nargs='+', default=['base'], choices=list(VARIANTES),
                        help="Una o más variantes; se corren y agregan en orden")
    parser.add_argument("--reusar-dataset", action="store_true",
                        help="Omite el simulador si el dataset de la semilla ya existe")
    parser.add_argument("--solo-agregar", action="store_true",
                        help="Recalcula los resúmenes con las corridas ya existentes")
    parser.add_argument("--comparar", nargs=2, metavar=('A', 'B'), choices=list(VARIANTES),
                        help="Solo la comparación pareada B − A entre dos variantes ya corridas")
    parser.add_argument("--solo-ruteo", action="store_true",
                        help="Reevalúa solo el ruteo con los datasets y modelos ya guardados")
    parser.add_argument("--sensibilidad-aviso", action="store_true",
                        help="Solo la tabla de sensibilidad a la probabilidad de error del aviso (Sección 11)")
    parser.add_argument("--hibrido-umbral", action="store_true",
                        help="Solo la tabla del híbrido y del umbral de la regla para las variantes dadas (Sección 12)")
    parser.add_argument("--tabla-horizontes", action="store_true",
                        help="Solo la tabla resumen por horizonte para las variantes dadas (Sección 13)")
    parser.add_argument("--paralelo", type=int, default=CORRIDAS_EN_PARALELO)
    args = parser.parse_args()

    if args.comparar:
        comparar_variantes(*args.comparar)
        sys.exit(0)
    if args.sensibilidad_aviso:
        resumen_sensibilidad_aviso()
        sys.exit(0)
    if args.hibrido_umbral:
        resumen_hibrido_umbral(args.variante)
        sys.exit(0)
    if args.tabla_horizontes:
        resumen_horizontes(args.variante)
        sys.exit(0)
    for variante in args.variante:
        if not args.solo_agregar:
            with ThreadPoolExecutor(max_workers=args.paralelo) as ejecutor:
                list(ejecutor.map(lambda s: correr_semilla(s, variante, args.reusar_dataset, args.solo_ruteo), SEMILLAS))
        agregar_modelos(variante)
        agregar_ruteo(variante)
        agregar_ruteo_por_edad(variante)
        agregar_ruteo_por_tipo(variante)
    resumen_horizontes(args.variante)
