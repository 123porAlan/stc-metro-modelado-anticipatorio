import re
import collections

import pandas as pd
import networkx as nx
import numpy as np

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

# Capacidad base teórica de un tramo de línea por hora
CAPACIDAD_PROMEDIO_TRAMO_HORA = 35000
horas_operacion = range(5, 24)  # El snapshot se toma cada 1 unidad de tiempo (1 hora)

# ====================================================================================
# GENERADOR DE EVENTOS ESTOCÁSTICOS (reemplaza la falla determinista de Línea 9)
# ====================================================================================
# Valores ilustrativos para un prototipo con datos sintéticos (no una calibración
# estadística sobre incidentes reales del STC, para lo cual no hay feed público
# disponible — ver avances.md, sección "Análisis de factibilidad técnica"). Cada
# constante representa un supuesto de simulación explícito y documentado, en la misma
# línea que la función BPR y los perfiles origen/destino ya usados en el prototipo.
SEMILLA_ALEATORIA = 42
TIPOS_EVENTO = ["lluvia", "falla_mecanica", "incidente_plataforma"]

# Líneas con tramos elevados/de superficie, más expuestas a lluvia.
LINEAS_SUPERFICIE = {"A", "B", "12"}

# Tasas base (eventos esperados por hora, proceso de Poisson) en hora valle:
TASA_BASE_LLUVIA = 0.02              # por línea de superficie
TASA_BASE_FALLA_MECANICA = 0.01      # por línea
TASA_BASE_INCIDENTE_PLATAFORMA = 0.006  # por línea

HORAS_PICO = {7, 8, 9, 18, 19, 20}
FACTOR_HORA_PICO = 2.5  # más carga -> más desgaste/afluencia -> más probabilidad de incidente

MESES_TEMPORADA_LLUVIAS = {5, 6, 7, 8, 9, 10}
FACTOR_TEMPORADA_LLUVIAS = 6.0  # mayo-octubre vs resto del año (climatología CDMX, ilustrativo)

# Impacto relativo de cada tipo de evento sobre la carga efectiva del tramo afectado.
FACTOR_IMPACTO_EVENTO = {
    "lluvia": 0.6,
    "falla_mecanica": 1.4,
    "incidente_plataforma": 2.0,
}

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

def tasa_evento(tipo_evento, hora, mes):
    factor_hora = FACTOR_HORA_PICO if hora in HORAS_PICO else 1.0
    if tipo_evento == "lluvia":
        factor_temporada = FACTOR_TEMPORADA_LLUVIAS if mes in MESES_TEMPORADA_LLUVIAS else 1.0
        return TASA_BASE_LLUVIA * factor_temporada * factor_hora
    elif tipo_evento == "falla_mecanica":
        return TASA_BASE_FALLA_MECANICA * factor_hora
    else:
        return TASA_BASE_INCIDENTE_PLATAFORMA * factor_hora

def generar_eventos_del_dia(rng, fecha, horas_operacion):
    """
    Sortea, para cada (hora, línea, tipo de evento) del día, si ocurre un evento
    (proceso de Poisson) y, si ocurre, su severidad (Beta, sesgada hacia eventos
    leves) y duración (horas, acotada a un máximo razonable de contingencia).
    """
    mes = pd.Timestamp(fecha).month
    eventos = []
    for hora in horas_operacion:
        for tipo_evento in TIPOS_EVENTO:
            lineas_candidatas = LINEAS_SUPERFICIE if tipo_evento == "lluvia" else edges_por_linea.keys()
            for linea in lineas_candidatas:
                tramos_linea = edges_por_linea.get(linea)
                if not tramos_linea:
                    continue
                lam = tasa_evento(tipo_evento, hora, mes)
                n_ocurrencias = rng.poisson(lam)
                if n_ocurrencias <= 0:
                    continue
                severidad = float(np.clip(rng.beta(2, 5) * min(n_ocurrencias, 3), 0.0, 1.0))
                duracion = int(np.clip(rng.poisson(1) + 1, 1, 4))
                if tipo_evento == "lluvia":
                    # La lluvia afecta a toda la línea de superficie, no un solo tramo.
                    tramos_afectados = list(tramos_linea)
                else:
                    # Falla mecánica / incidente de plataforma: localizado en un tramo.
                    idx = rng.integers(len(tramos_linea))
                    tramos_afectados = [tramos_linea[idx]]
                eventos.append({
                    "fecha": fecha,
                    "tipo_evento": tipo_evento,
                    "linea": linea,
                    "hora_inicio": hora,
                    "duracion": duracion,
                    "severidad": severidad,
                    "tramos": tramos_afectados,
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

rng = np.random.default_rng(SEMILLA_ALEATORIA)
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

        # 1. ENRUTAMIENTO (usa la caché de rutas estáticas)
        df_hora = df_od[df_od['hora'] == hora]
        for _, row in df_hora.iterrows():
            tramos = obtener_tramos_ruta(row['origen'], row['destino'])
            if tramos is None:
                continue
            pasajeros = row['pasajeros_viaje']
            for u, v in tramos:
                G_hora[u][v]['carga_pasajeros'] += pasajeros

        # 2. EVENTOS ESTOCÁSTICOS ACTIVOS ESTA HORA
        eventos_activos = [
            e for e in eventos_dia
            if e['hora_inicio'] <= hora < e['hora_inicio'] + e['duracion']
        ]
        info_tramos_hora = {}
        carga_fantasma_por_tramo = collections.defaultdict(float)
        for evento in eventos_activos:
            carga_fantasma = evento['severidad'] * FACTOR_IMPACTO_EVENTO[evento['tipo_evento']] * CAPACIDAD_PROMEDIO_TRAMO_HORA
            for u, v in evento['tramos']:
                k = clave_arista(u, v)
                carga_fantasma_por_tramo[k] += carga_fantasma
                previo = info_tramos_hora.get(k)
                if previo is None or evento['severidad'] > previo['severidad']:
                    info_tramos_hora[k] = {
                        'tipo_evento': evento['tipo_evento'],
                        'severidad': evento['severidad'],
                    }
        info_eventos_por_hora[hora] = info_tramos_hora

        # 3. CÁLCULO DE CONGESTIÓN (BPR): la carga fantasma de un evento se suma solo
        # para calcular el tiempo congestionado, NO se escribe en 'carga_pasajeros'
        # (esa columna refleja únicamente el ridership real enrutado; el efecto del
        # evento se aprende de forma explícita vía hay_evento/tipo_evento/severidad_evento).
        for u, v, data in G_hora.edges(data=True):
            carga_tramo = data['carga_pasajeros']
            tiempo_ideal = data.get('tiempo_minutos', 2.0)
            carga_tramo_efectiva = carga_tramo + carga_fantasma_por_tramo.get(clave_arista(u, v), 0.0)

            nuevo_tiempo = funcion_penalizacion_bpr(tiempo_ideal, carga_tramo_efectiva, CAPACIDAD_PROMEDIO_TRAMO_HORA)

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
                'target_congestibilidad_t_plus_1': congest_futura,
            })

print(f"\nTotal de eventos estocásticos generados en {len(df_manifiesto)} días: {total_eventos_generados}")
print("Simulación completada. Procesando dataset para la IA...")

df_ml = pd.DataFrame(datos_ml)
archivo_dataset = "datos_procesados/dataset_features_entrenamiento.csv"
df_ml.to_csv(archivo_dataset, index=False, encoding='utf-8-sig')

print(f"¡Dataset tabular (Snapshots) generado exitosamente con {len(df_ml)} registros de la red!")
print(f"Guardado en: {archivo_dataset}")
print(f"Filas con evento activo: {df_ml['hay_evento'].sum()} ({100*df_ml['hay_evento'].mean():.2f}%)")
print(df_ml[df_ml['hay_evento'] == 1]['tipo_evento'].value_counts())
