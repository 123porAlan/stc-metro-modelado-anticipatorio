import os
import re
import argparse
import collections

import pandas as pd
import networkx as nx
import numpy as np

parser = argparse.ArgumentParser(description="Simula la congestión horaria de la red para los días del manifiesto.")
parser.add_argument("--semilla", type=int, default=42, help="Semilla de los eventos estocásticos")
parser.add_argument("--salida", default="datos_procesados/dataset_features_entrenamiento.csv")
args = parser.parse_args()

print("Cargando infraestructura (Grafo Base)...")
G_base = nx.read_gexf("grafo_base_metro.gexf")

print("Cargando manifiesto de días de simulación (generado por generador_sintetico_horario.py)...")
df_manifiesto = pd.read_csv("datos_procesados/manifiesto_dias_simulados.csv")

def funcion_penalizacion_bpr(tiempo_base_min, afluencia_tramo, capacidad_tramo, alpha=0.15, beta=4):
    """
    Calcula el nuevo tiempo de viaje usando la función BPR adaptada para transporte público.
    """
    if pd.isna(afluencia_tramo) or afluencia_tramo == 0:
        return tiempo_base_min

    # Relación Volumen/Capacidad (V/C) en EL TRAMO ESPECÍFICO
    saturacion = afluencia_tramo / capacidad_tramo
    tiempo_congestivo = tiempo_base_min * (1 + alpha * (saturacion ** beta))
    tiempo_maximo = tiempo_base_min * 4

    return min(tiempo_congestivo, tiempo_maximo)

horas_operacion = range(5, 24)  # El snapshot se toma cada 1 unidad de tiempo (1 hora)

def linea_de_nodo(nodo_id):
    m = re.search(r"L([0-9A-Za-z]+)[-_]", nodo_id)
    return m.group(1) if m else None

# Tramos "puros" de cada línea (excluye aristas de transbordo, que conectan dos líneas
# distintas y no pertenecen operativamente a ninguna de las dos).
edges_por_linea = collections.defaultdict(list)
for u, v in G_base.edges():
    lu, lv = linea_de_nodo(u), linea_de_nodo(v)
    if lu is not None and lu == lv:
        edges_por_linea[lu].append((u, v))

# ====================================================================================
# CAPACIDAD POR LÍNEA (reemplaza la constante única de 35,000 pasajeros/hora)
# ====================================================================================
# Capacidad por sentido = trenes por hora × carros por tren × pasajeros por carro.
# Fuentes (ver avances.md, Sección 6.1):
# - Trenes asignados por línea y capacidad por tren: STC, "Parque Vehicular"
#   (6 carros = 1,020 pasajeros, 9 carros = 1,530 -> 170 por carro).
# - Carros por tren: trenes de 6 carros neumáticos (29) = Líneas 4 y 6; férreos de
#   7 carros (30) = Línea 12; Línea A = 22 trenes de 9 carros + 11 de 6 (promedio 8).
# - Disponibilidad: 256 de 394 trenes en servicio (STC vía transparencia, nov. 2024).
# - Intervalo mínimo: 2 minutos en hora pico (STC, preguntas frecuentes).
# La frecuencia por línea no se publica, así que se deriva: los trenes en servicio de
# la línea recorren un ciclo de ida y vuelta (2 × tiempo de recorrido del GTFS; sin
# tiempo de maniobra en terminales, que no está documentado).
PASAJEROS_POR_CARRO = 170
TRENES_POR_LINEA = {'1': 50, '2': 41, '3': 54, '4': 14, '5': 25, '6': 15, '7': 32,
                    '8': 30, '9': 34, 'A': 33, 'B': 36, '12': 30}
CARROS_POR_TREN = {'4': 6, '6': 6, '12': 7, 'A': 8}  # el resto: 9 carros
DISPONIBILIDAD_TRENES = 256 / 394
INTERVALO_MINIMO_MIN = 2.0
# Los pasillos de transbordo no tienen dato de capacidad: conservan el supuesto anterior.
CAPACIDAD_TRANSBORDO_HORA = 35000

def calcular_capacidad_por_linea():
    capacidades = {}
    for linea, tramos in edges_por_linea.items():
        recorrido_min = sum(G_base[u][v]['tiempo_minutos'] for u, v in tramos)
        trenes_hora = TRENES_POR_LINEA[linea] * DISPONIBILIDAD_TRENES * 60 / (2 * recorrido_min)
        trenes_hora = min(trenes_hora, 60 / INTERVALO_MINIMO_MIN)
        capacidades[linea] = trenes_hora * CARROS_POR_TREN.get(linea, 9) * PASAJEROS_POR_CARRO
    return capacidades

CAPACIDAD_SENTIDO_POR_LINEA = calcular_capacidad_por_linea()
print("Capacidad por sentido (pasajeros/hora):")
for _linea, _cap in sorted(CAPACIDAD_SENTIDO_POR_LINEA.items(), key=lambda x: x[1]):
    print(f"   Línea {_linea}: {_cap:,.0f}")

def capacidad_tramo(u, v):
    lu, lv = linea_de_nodo(u), linea_de_nodo(v)
    if lu is not None and lu == lv:
        return CAPACIDAD_SENTIDO_POR_LINEA[lu]
    return CAPACIDAD_TRANSBORDO_HORA

# ====================================================================================
# GENERADOR DE EVENTOS ESTOCÁSTICOS
# ====================================================================================
# Tasas calibradas con fuentes públicas (ver avances.md, Sección 6.3). Siguen siendo una
# aproximación documentada, no una calibración estadística rigurosa.
TIPOS_EVENTO = ["lluvia", "falla_mecanica", "incidente_plataforma"]

# Líneas con tramos elevados/de superficie, más expuestas a lluvia.
LINEAS_SUPERFICIE = ["A", "B", "12"]

# Lluvia: número medio de días con lluvia por mes, Normal Climatológica 1981-2010 del SMN,
# estación 9048 Tacubaya Central (Obs). La probabilidad de que llueva en un día es
# días_con_lluvia / días_del_mes; un día lluvioso tiene un solo episodio que afecta a la
# ciudad (no un sorteo independiente por línea).
DIAS_CON_LLUVIA_POR_MES = {1: 2.8, 2: 2.2, 3: 3.6, 4: 7.2, 5: 11.5, 6: 17.6,
                           7: 22.0, 8: 21.0, 9: 17.9, 10: 9.9, 11: 3.0, 12: 1.5}
# Probabilidad de que un episodio de lluvia alcance a cada línea de superficie. Ilustrativa
# hasta contar con datos de la red pluviométrica del OH-IIUNAM (correlación espacial).
PROB_LLUVIA_ALCANZA_LINEA = 0.75

# Falla mecánica + incidente de plataforma: 3,708 incidentes con desalojo entre enero de
# 2018 y agosto de 2022 (STC vía Plataforma Nacional de Transparencia) ≈ 2.2 por día en
# 12 líneas -> λ ≈ 2.2 / (12 × (13 + 2.5 × 6)) ≈ 0.0065 por línea-hora valle. No hay
# desglose por causa: se reparte en la misma proporción que las tasas anteriores (5:3).
TASA_BASE_FALLA_MECANICA = 0.0041        # por línea
TASA_BASE_INCIDENTE_PLATAFORMA = 0.0024  # por línea

HORAS_PICO = {7, 8, 9, 18, 19, 20}
FACTOR_HORA_PICO = 2.5  # más carga -> más desgaste/afluencia -> más probabilidad de incidente

# Impacto de lluvia y falla mecánica: carga fantasma relativa a la capacidad del tramo
# (marcha lenta / menos trenes). El incidente de plataforma, en cambio, SUSPENDE el tramo.
FACTOR_IMPACTO_EVENTO = {
    "lluvia": 0.6,
    "falla_mecanica": 1.4,
}
# Suspensión: la severidad s es la fracción de la hora con el tramo cerrado. Un pasajero
# que llega durante el cierre espera en promedio la mitad de lo que falta (60·s/2 min) y
# una fracción s de los pasajeros llega durante el cierre -> retraso medio de la hora =
# s × 60·s/2 = 30·s² minutos (hasta 30 min si el tramo pasa toda la hora cerrado).
MINUTOS_HORA = 60

def retraso_por_suspension(severidad):
    return severidad * (MINUTOS_HORA * severidad / 2)

def tasa_evento(tipo_evento, hora):
    factor_hora = FACTOR_HORA_PICO if hora in HORAS_PICO else 1.0
    if tipo_evento == "falla_mecanica":
        return TASA_BASE_FALLA_MECANICA * factor_hora
    return TASA_BASE_INCIDENTE_PLATAFORMA * factor_hora

def sortear_severidad_duracion(rng, n_ocurrencias=1):
    severidad = float(np.clip(rng.beta(2, 5) * min(n_ocurrencias, 3), 0.0, 1.0))
    duracion = int(np.clip(rng.poisson(1) + 1, 1, 4))
    return severidad, duracion

def generar_eventos_del_dia(rng, fecha, horas_operacion):
    """
    Lluvia: se sortea si el día es lluvioso (probabilidad mensual SMN) y, si lo es, un
    episodio con hora de inicio uniforme (sin factor de hora pico: la lluvia no depende
    de la afluencia), severidad y duración, que alcanza a cada línea de superficie con
    PROB_LLUVIA_ALCANZA_LINEA (al menos a una).
    Falla mecánica / incidente de plataforma: para cada (hora, línea) se sortea un proceso
    de Poisson y, si ocurre, severidad (Beta, sesgada a eventos leves), duración y tramo.
    """
    fecha_ts = pd.Timestamp(fecha)
    eventos = []

    prob_lluvia_dia = DIAS_CON_LLUVIA_POR_MES[fecha_ts.month] / fecha_ts.days_in_month
    if rng.random() < prob_lluvia_dia:
        hora_inicio = int(rng.choice(list(horas_operacion)))
        severidad, duracion = sortear_severidad_duracion(rng)
        alcanzadas = [l for l in LINEAS_SUPERFICIE if rng.random() < PROB_LLUVIA_ALCANZA_LINEA]
        if not alcanzadas:
            alcanzadas = [LINEAS_SUPERFICIE[rng.integers(len(LINEAS_SUPERFICIE))]]
        for linea in alcanzadas:
            eventos.append({
                "fecha": fecha,
                "tipo_evento": "lluvia",
                "linea": linea,
                "hora_inicio": hora_inicio,
                "duracion": duracion,
                "severidad": severidad,
                # La lluvia afecta a toda la línea de superficie, no un solo tramo.
                "tramos": list(edges_por_linea[linea]),
            })

    for hora in horas_operacion:
        for tipo_evento in ["falla_mecanica", "incidente_plataforma"]:
            for linea, tramos_linea in edges_por_linea.items():
                n_ocurrencias = rng.poisson(tasa_evento(tipo_evento, hora))
                if n_ocurrencias <= 0:
                    continue
                severidad, duracion = sortear_severidad_duracion(rng, n_ocurrencias)
                # Localizado en un tramo.
                idx = rng.integers(len(tramos_linea))
                eventos.append({
                    "fecha": fecha,
                    "tipo_evento": tipo_evento,
                    "linea": linea,
                    "hora_inicio": hora,
                    "duracion": duracion,
                    "severidad": severidad,
                    "tramos": [tramos_linea[idx]],
                })
    return eventos

def clave_arista(u, v):
    return (u, v) if u <= v else (v, u)

# ====================================================================================
# MOTOR DE SIMULACIÓN (multi-día)
# ====================================================================================
print("Iniciando motor de simulación y captura de snapshots (multi-día)...")

# Diccionario para mapear nombres de estaciones a sus IDs en el grafo para el ruteo
nombre_a_nodos = {}
for n, data in G_base.nodes(data=True):
    nombre = data.get('nombre')
    if nombre:
        nombre_a_nodos.setdefault(nombre, []).append(n)

# --- Caché de rutas estáticas ---
# El ruteo usa 'tiempo_minutos', que es un peso ESTÁTICO del grafo (no cambia entre
# horas ni entre días). Por lo tanto, la ruta óptima entre un mismo par
# origen-destino es siempre la misma; solo cambia cuánta gente la usa. La versión de
# un solo día recalculaba nx.shortest_path por cada fila de la matriz O-D y por cada
# hora (~616,000 llamadas/día); con varios días eso ya no escala. Aquí la ruta se
# calcula una sola vez por par (origen, destino) y se reutiliza para todas las horas
# y todos los días.
cache_rutas = {}

def obtener_tramos_ruta(origen_nombre, destino_nombre):
    clave = (origen_nombre, destino_nombre)
    if clave in cache_rutas:
        return cache_rutas[clave]
    if origen_nombre not in nombre_a_nodos or destino_nombre not in nombre_a_nodos:
        cache_rutas[clave] = None
        return None
    nodo_origen = nombre_a_nodos[origen_nombre][0]
    nodo_destino = nombre_a_nodos[destino_nombre][0]
    try:
        ruta = nx.shortest_path(G_base, source=nodo_origen, target=nodo_destino, weight='tiempo_minutos')
        tramos = list(zip(ruta[:-1], ruta[1:]))
    except nx.NetworkXNoPath:
        tramos = None
    cache_rutas[clave] = tramos
    return tramos

rng = np.random.default_rng(args.semilla)
print(f"Semilla de eventos estocásticos: {args.semilla}")
datos_ml = []
total_eventos_generados = 0

for _, fila_manifiesto in df_manifiesto.iterrows():
    dia_simulacion = fila_manifiesto['fecha']
    tipo_dia = fila_manifiesto['tipo_dia']
    print(f"\n=== Simulando {dia_simulacion} ({tipo_dia}) ===")

    df_od = pd.read_csv(fila_manifiesto['archivo_od'])
    eventos_dia = generar_eventos_del_dia(rng, dia_simulacion, horas_operacion)
    total_eventos_generados += len(eventos_dia)
    print(f"   Eventos estocásticos generados: {len(eventos_dia)}")

    grafos_temporales = {}
    info_eventos_por_hora = {}

    for hora in horas_operacion:
        G_hora = G_base.copy()
        nx.set_edge_attributes(G_hora, 0, 'carga_pasajeros')
        # El grafo es no dirigido, pero la capacidad es POR SENTIDO: se lleva la carga de
        # cada sentido por separado y la congestión se calcula con el sentido más cargado.
        carga_por_sentido = collections.defaultdict(float)

        # 1. ENRUTAMIENTO (usa la caché de rutas estáticas)
        df_hora = df_od[df_od['hora'] == hora]
        for origen, destino, pasajeros in zip(df_hora['origen'], df_hora['destino'], df_hora['pasajeros_viaje']):
            tramos = obtener_tramos_ruta(origen, destino)
            if tramos is None:
                continue
            for u, v in tramos:
                G_hora[u][v]['carga_pasajeros'] += pasajeros
                carga_por_sentido[(u, v)] += pasajeros

        # 2. EVENTOS ESTOCÁSTICOS ACTIVOS ESTA HORA
        eventos_activos = [
            e for e in eventos_dia
            if e['hora_inicio'] <= hora < e['hora_inicio'] + e['duracion']
        ]
        info_tramos_hora = {}
        carga_fantasma_por_tramo = collections.defaultdict(float)
        retraso_suspension_por_tramo = collections.defaultdict(float)
        for evento in eventos_activos:
            for u, v in evento['tramos']:
                k = clave_arista(u, v)
                if evento['tipo_evento'] == 'incidente_plataforma':
                    retraso_suspension_por_tramo[k] = max(retraso_suspension_por_tramo[k],
                                                          retraso_por_suspension(evento['severidad']))
                else:
                    carga_fantasma_por_tramo[k] += evento['severidad'] * FACTOR_IMPACTO_EVENTO[evento['tipo_evento']] * capacidad_tramo(u, v)
                previo = info_tramos_hora.get(k)
                if previo is None or evento['severidad'] > previo['severidad']:
                    info_tramos_hora[k] = {
                        'tipo_evento': evento['tipo_evento'],
                        'severidad': evento['severidad'],
                        # Edad del evento: horas desde que empezó (0 en su primera hora).
                        # Solo usa hora_inicio, que sería observable en tiempo real; la
                        # duración NO se expone al modelo (no se conoce hasta que termina).
                        'edad': hora - evento['hora_inicio'],
                    }
        info_eventos_por_hora[hora] = info_tramos_hora

        # 3. CÁLCULO DE CONGESTIÓN (BPR): la carga fantasma de un evento se suma solo
        # para calcular el tiempo congestionado, NO se escribe en 'carga_pasajeros'
        # (esa columna refleja únicamente el ridership real enrutado, ambos sentidos; el
        # efecto del evento se aprende de forma explícita vía hay_evento/tipo_evento/severidad_evento).
        # La suspensión por incidente de plataforma se suma aparte, después del BPR.
        for u, v, data in G_hora.edges(data=True):
            carga_sentido_critico = max(carga_por_sentido[(u, v)], carga_por_sentido[(v, u)])
            tiempo_ideal = data.get('tiempo_minutos', 2.0)
            k = clave_arista(u, v)
            carga_tramo_efectiva = carga_sentido_critico + carga_fantasma_por_tramo.get(k, 0.0)

            nuevo_tiempo = funcion_penalizacion_bpr(tiempo_ideal, carga_tramo_efectiva, capacidad_tramo(u, v))
            nuevo_tiempo += retraso_suspension_por_tramo.get(k, 0.0)

            G_hora[u][v]['weight'] = round(nuevo_tiempo, 2)
            G_hora[u][v]['congestibilidad'] = round(nuevo_tiempo - tiempo_ideal, 2)

        grafos_temporales[hora] = G_hora

    # --- DATA PIPELINE: Extracción de snapshots para Machine Learning ---
    horas_disponibles = sorted(grafos_temporales.keys())

    for i, hora_actual in enumerate(horas_disponibles):
        if i + 1 >= len(horas_disponibles) or i < 1:
            continue

        hora_siguiente = horas_disponibles[i + 1]
        hora_pasada = horas_disponibles[i - 1]

        G_actual = grafos_temporales[hora_actual]
        G_pasado = grafos_temporales[hora_pasada]
        G_siguiente = grafos_temporales[hora_siguiente]
        info_tramos_actual = info_eventos_por_hora[hora_actual]

        for u, v, data in G_actual.edges(data=True):
            nombre_origen = G_actual.nodes[u].get('nombre', str(u))
            nombre_destino = G_actual.nodes[v].get('nombre', str(v))
            tramo_id = f"{nombre_origen}-{nombre_destino}"

            tiempo_ideal = data.get('tiempo_minutos', 0)
            carga_actual = data.get('carga_pasajeros', 0)
            congest_actual = data.get('congestibilidad', 0)
            congest_pasada = G_pasado[u][v].get('congestibilidad', 0) if G_pasado.has_edge(u, v) else 0
            congest_futura = G_siguiente[u][v].get('congestibilidad', 0) if G_siguiente.has_edge(u, v) else 0

            info_evento = info_tramos_actual.get(clave_arista(u, v))
            hay_evento = int(info_evento is not None)
            tipo_evento = info_evento['tipo_evento'] if info_evento else 'ninguno'
            severidad_evento = round(info_evento['severidad'], 4) if info_evento else 0.0
            edad_evento = info_evento['edad'] if info_evento else -1  # -1 = sin evento

            datos_ml.append({
                'fecha': dia_simulacion,
                'tipo_dia': tipo_dia,
                'hora': hora_actual,
                'nodo_origen': u,
                'nodo_destino': v,
                'tramo': tramo_id,
                'origen': nombre_origen,
                'tiempo_ideal': tiempo_ideal,
                'carga_pasajeros_red': carga_actual,
                'congestibilidad_t_minus_1': congest_pasada,
                'congestibilidad_t': congest_actual,
                'hay_evento': hay_evento,
                'tipo_evento': tipo_evento,
                'severidad_evento': severidad_evento,
                'edad_evento': edad_evento,
                'target_congestibilidad_t_plus_1': congest_futura,
            })

print(f"\nTotal de eventos estocásticos generados en {len(df_manifiesto)} días: {total_eventos_generados}")
print("Simulación completada. Procesando dataset para la IA...")

df_ml = pd.DataFrame(datos_ml)
os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
archivo_dataset = args.salida
df_ml.to_csv(archivo_dataset, index=False, encoding='utf-8-sig')

print(f"¡Dataset tabular (Snapshots) generado exitosamente con {len(df_ml)} registros de la red!")
print(f"Guardado en: {archivo_dataset}")
print(f"Filas con evento activo: {df_ml['hay_evento'].sum()} ({100*df_ml['hay_evento'].mean():.2f}%)")
print(df_ml[df_ml['hay_evento'] == 1]['tipo_evento'].value_counts())
