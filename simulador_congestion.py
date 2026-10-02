import os
import re
import argparse
import collections

import pandas as pd
import networkx as nx
import numpy as np

parser = argparse.ArgumentParser(description="Simula la congestión de la red en bloques de 15 min para los días del manifiesto.")
parser.add_argument("--semilla", type=int, default=42, help="Semilla de los eventos estocásticos")
parser.add_argument("--salida", default="datos_procesados/dataset_features_entrenamiento.csv")
parser.add_argument("--ruido-aviso", type=float, default=None,
                    help="σ del ruido log-normal del aviso de restablecimiento (0 = aviso perfecto). "
                         "Sin este argumento no se genera la columna bloques_restantes_anunciados.")
parser.add_argument("--error-aviso-aditivo", type=int, default=None,
                    help="K: suma al aviso un error entero δ uniforme en {-K, ..., K} bloques de 15 min "
                         "(ver README.md, Sección 9). Puede anunciar 0 cuando el evento sigue. Se combina "
                         "con --ruido-aviso (σ = 0 si no se pasa).")
parser.add_argument("--prob-error-aviso", type=float, default=None,
                    help="p: con probabilidad p el aviso se equivoca en ±1 bloque de 15 min (δ = −1 o +1 con igual "
                         "probabilidad); si no, es exacto (ver README.md, Sección 11). No se combina con "
                         "--error-aviso-aditivo.")
args = parser.parse_args()
if args.prob_error_aviso is not None and args.error_aviso_aditivo is not None:
    parser.error("--prob-error-aviso y --error-aviso-aditivo son excluyentes")
# El aviso se genera si se pidió cualquiera de sus errores.
GENERAR_AVISO = any(a is not None for a in (args.ruido_aviso, args.error_aviso_aditivo, args.prob_error_aviso))
SIGMA_AVISO = args.ruido_aviso or 0.0
ERROR_ADITIVO_AVISO = args.error_aviso_aditivo or 0
PROB_ERROR_AVISO = args.prob_error_aviso or 0.0

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

# ====================================================================================
# RESOLUCIÓN TEMPORAL: BLOQUES DE 15 MINUTOS (limitación #4, ver README.md)
# ====================================================================================
# El snapshot se toma cada bloque de 15 min. La demanda O-D sigue siendo horaria; se
# reparte en bloques con interpolar_demanda_bloques(). Los bloques se numeran desde 0
# (05:00) hasta BLOQUES_POR_DIA - 1 (23:45).
MINUTOS_BLOQUE = 15
BLOQUES_POR_HORA = 60 // MINUTOS_BLOQUE
horas_operacion = range(5, 24)
BLOQUES_POR_DIA = len(horas_operacion) * BLOQUES_POR_HORA
bloques_operacion = range(BLOQUES_POR_DIA)

def hora_de_bloque(bloque):
    return horas_operacion[0] + bloque // BLOQUES_POR_HORA

def minuto_de_bloque(bloque):
    return (bloque % BLOQUES_POR_HORA) * MINUTOS_BLOQUE

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
# Fuentes (ver README.md, Sección 6.1):
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
    """Capacidad por sentido en UN BLOQUE: la horaria repartida proporcionalmente
    (los trenes de un bloque de 15 min son 1/4 de los de la hora)."""
    lu, lv = linea_de_nodo(u), linea_de_nodo(v)
    if lu is not None and lu == lv:
        capacidad_hora = CAPACIDAD_SENTIDO_POR_LINEA[lu]
    else:
        capacidad_hora = CAPACIDAD_TRANSBORDO_HORA
    return capacidad_hora / BLOQUES_POR_HORA

# ====================================================================================
# GENERADOR DE EVENTOS ESTOCÁSTICOS
# ====================================================================================
# Tasas calibradas con fuentes públicas (ver README.md, Sección 6.3). Siguen siendo una
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
# Suspensión: mientras dura el incidente, el servicio del tramo se interrumpe en cierres
# de MINUTOS_CICLO_CIERRE·s minutos por cada ciclo de MINUTOS_CICLO_CIERRE (la severidad s
# es la fracción del tiempo cerrado). Una fracción s de los pasajeros llega durante un
# cierre y espera en promedio la mitad -> retraso medio = s × MINUTOS_CICLO_CIERRE·s/2 =
# 30·s² minutos, constante mientras el evento está activo y 0 al terminar. Es la misma
# magnitud que el modelo horario de las Secciones 7-12.
# El retraso NO depende de cuánto falta para que termine el evento (ver README.md,
# Sección 14): con s²·R/2 (Sección 13) el modelo podía despejar R del retraso observado,
# porque s es una feature, y el aviso de restablecimiento quedaba redundante.
MINUTOS_CICLO_CIERRE = 60

def retraso_por_suspension(severidad):
    return severidad * (MINUTOS_CICLO_CIERRE * severidad / 2)

def tasa_evento(tipo_evento, hora):
    """Tasa por línea y BLOQUE: la horaria calibrada repartida entre los bloques de la
    hora, para conservar el número esperado de eventos por hora."""
    factor_hora = FACTOR_HORA_PICO if hora in HORAS_PICO else 1.0
    if tipo_evento == "falla_mecanica":
        return TASA_BASE_FALLA_MECANICA * factor_hora / BLOQUES_POR_HORA
    return TASA_BASE_INCIDENTE_PLATAFORMA * factor_hora / BLOQUES_POR_HORA

# Duración en bloques: 1 + Poisson(7), con tope de 4 horas. La media (~8 bloques = 2 h)
# coincide con la del sorteo horario anterior (1 + Poisson(1) horas, tope 4).
LAMBDA_DURACION_BLOQUES = 7
DURACION_MAXIMA_BLOQUES = 4 * BLOQUES_POR_HORA

def sortear_severidad_duracion(rng, n_ocurrencias=1):
    severidad = float(np.clip(rng.beta(2, 5) * min(n_ocurrencias, 3), 0.0, 1.0))
    duracion = int(np.clip(rng.poisson(LAMBDA_DURACION_BLOQUES) + 1, 1, DURACION_MAXIMA_BLOQUES))
    return severidad, duracion

def generar_eventos_del_dia(rng, fecha, bloques_operacion):
    """
    Lluvia: se sortea si el día es lluvioso (probabilidad mensual SMN) y, si lo es, un
    episodio con bloque de inicio uniforme (sin factor de hora pico: la lluvia no depende
    de la afluencia), severidad y duración, que alcanza a cada línea de superficie con
    PROB_LLUVIA_ALCANZA_LINEA (al menos a una).
    Falla mecánica / incidente de plataforma: para cada (bloque, línea) se sortea un proceso
    de Poisson y, si ocurre, severidad (Beta, sesgada a eventos leves), duración y tramo.
    Las duraciones se miden en bloques.
    """
    fecha_ts = pd.Timestamp(fecha)
    eventos = []

    prob_lluvia_dia = DIAS_CON_LLUVIA_POR_MES[fecha_ts.month] / fecha_ts.days_in_month
    if rng.random() < prob_lluvia_dia:
        bloque_inicio = int(rng.choice(list(bloques_operacion)))
        severidad, duracion = sortear_severidad_duracion(rng)
        alcanzadas = [l for l in LINEAS_SUPERFICIE if rng.random() < PROB_LLUVIA_ALCANZA_LINEA]
        if not alcanzadas:
            alcanzadas = [LINEAS_SUPERFICIE[rng.integers(len(LINEAS_SUPERFICIE))]]
        for linea in alcanzadas:
            eventos.append({
                "fecha": fecha,
                "tipo_evento": "lluvia",
                "linea": linea,
                "bloque_inicio": bloque_inicio,
                "duracion": duracion,
                "severidad": severidad,
                # La lluvia afecta a toda la línea de superficie, no un solo tramo.
                "tramos": list(edges_por_linea[linea]),
            })

    for bloque in bloques_operacion:
        for tipo_evento in ["falla_mecanica", "incidente_plataforma"]:
            for linea, tramos_linea in edges_por_linea.items():
                n_ocurrencias = rng.poisson(tasa_evento(tipo_evento, hora_de_bloque(bloque)))
                if n_ocurrencias <= 0:
                    continue
                severidad, duracion = sortear_severidad_duracion(rng, n_ocurrencias)
                # Localizado en un tramo.
                idx = rng.integers(len(tramos_linea))
                eventos.append({
                    "fecha": fecha,
                    "tipo_evento": tipo_evento,
                    "linea": linea,
                    "bloque_inicio": bloque,
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
# Generador aparte para el ruido del aviso: sortear de `rng` desplazaría la secuencia de
# eventos y el dataset dejaría de ser comparable con el de la misma semilla sin aviso.
rng_aviso = np.random.default_rng(args.semilla + 1_000_000)
if GENERAR_AVISO:
    print(f"Aviso de restablecimiento con ruido σ = {SIGMA_AVISO}, error aditivo ±{ERROR_ADITIVO_AVISO} "
          f"y probabilidad de error ±1 = {PROB_ERROR_AVISO}")

def bloques_restantes_de(evento, bloque):
    """Bloques en que el evento sigue activo, contando el actual (>= 1 si está activo)."""
    return evento['bloque_inicio'] + evento['duracion'] - bloque

def bloques_restantes_anunciados(evento, bloque):
    """
    Aviso de tiempo estimado de restablecimiento (ver README.md, Sección 7.5). El STC lo
    publica para incidentes y fallas; la lluvia no lo trae (-1). Restante real = bloques
    completos que el evento seguirá activo después de este (0 = termina al cerrar el bloque).
    El ruido multiplicativo (log-normal) nunca se equivoca con un evento que termina en este
    bloque; el error aditivo δ sí (anuncia 0 cuando sigue, o 1 cuando termina). Los errores
    se miden en bloques: δ = ±1 equivale a ±15 min. Con --prob-error-aviso, δ = ±1 solo en
    una fracción p de los avisos.
    """
    if evento['tipo_evento'] == 'lluvia':
        return -1
    restante_real = max(bloques_restantes_de(evento, bloque) - 1, 0)
    epsilon = rng_aviso.normal(0, SIGMA_AVISO) if SIGMA_AVISO > 0 else 0.0
    delta = int(rng_aviso.integers(-ERROR_ADITIVO_AVISO, ERROR_ADITIVO_AVISO + 1)) if ERROR_ADITIVO_AVISO > 0 else 0
    if PROB_ERROR_AVISO > 0 and rng_aviso.random() < PROB_ERROR_AVISO:
        delta = int(rng_aviso.choice([-1, 1]))
    return max(int(round(restante_real * np.exp(epsilon))) + delta, 0)
def interpolar_demanda_bloques(carga_por_hora):
    """
    Reparte en bloques de 15 min la carga horaria por sentido de tramo (arreglo de forma
    (n_sentidos, n_horas)). La demanda O-D solo existe por hora, así que se interpola
    linealmente entre los centros de horas vecinas (la primera y la última hora no tienen
    vecina hacia fuera: ahí el perfil es plano) y se reescala para que los bloques de cada
    hora sumen exactamente la carga de esa hora. Se interpola la carga ya enrutada y no cada
    par O-D: la ruta de un par no cambia dentro del día, así que el resultado es el mismo
    salvo por el reescalado, que aquí conserva el total por tramo en vez de por par.
    Devuelve un arreglo de forma (n_sentidos, BLOQUES_POR_DIA).
    """
    carga = np.asarray(carga_por_hora, dtype=float)
    anterior = np.concatenate([carga[:, :1], carga[:, :-1]], axis=1)
    siguiente = np.concatenate([carga[:, 1:], carga[:, -1:]], axis=1)
    # Posición del centro de cada bloque respecto al centro de su hora, en horas:
    # -0.375, -0.125, +0.125, +0.375 con bloques de 15 min.
    desfase = (np.arange(BLOQUES_POR_HORA) + 0.5) / BLOQUES_POR_HORA - 0.5
    vecina = np.where(desfase < 0, anterior[..., None], siguiente[..., None])
    perfil = carga[..., None] + np.abs(desfase) * (vecina - carga[..., None])
    suma = perfil.sum(axis=-1, keepdims=True)
    bloques = np.divide(carga[..., None] * perfil, suma, out=np.zeros_like(perfil), where=suma > 0)
    return bloques.reshape(len(carga), -1)

datos_ml = []
total_eventos_generados = 0
# Ventanas del snapshot: 4 rezagos (hasta 60 min atrás) y 4 horizontes de predicción
# (15, 30, 45 y 60 min), el rango de 10-60 min que pide el objetivo de la tesis.
N_REZAGOS = BLOQUES_POR_HORA
N_HORIZONTES = BLOQUES_POR_HORA
indice_hora = {h: i for i, h in enumerate(horas_operacion)}

for _, fila_manifiesto in df_manifiesto.iterrows():
    dia_simulacion = fila_manifiesto['fecha']
    tipo_dia = fila_manifiesto['tipo_dia']
    print(f"\n=== Simulando {dia_simulacion} ({tipo_dia}) ===")

    df_od = pd.read_csv(fila_manifiesto['archivo_od'])
    eventos_dia = generar_eventos_del_dia(rng, dia_simulacion, bloques_operacion)
    total_eventos_generados += len(eventos_dia)
    print(f"   Eventos estocásticos generados: {len(eventos_dia)}")

    # 1. ENRUTAMIENTO HORARIO (usa la caché de rutas estáticas). El grafo es no dirigido,
    # pero la capacidad es POR SENTIDO: se lleva la carga de cada sentido por separado y la
    # congestión se calcula con el sentido más cargado.
    carga_sentido_hora = collections.defaultdict(lambda: np.zeros(len(horas_operacion)))
    for hora, origen, destino, pasajeros in zip(df_od['hora'], df_od['origen'], df_od['destino'], df_od['pasajeros_viaje']):
        if hora not in indice_hora:
            continue
        tramos = obtener_tramos_ruta(origen, destino)
        if tramos is None:
            continue
        for u, v in tramos:
            carga_sentido_hora[(u, v)][indice_hora[hora]] += pasajeros

    # 2. REPARTO DE LA CARGA HORARIA EN BLOQUES DE 15 MIN
    sentidos = list(carga_sentido_hora)
    matriz_horaria = np.array([carga_sentido_hora[s] for s in sentidos]).reshape(len(sentidos), len(horas_operacion))
    carga_sentido_bloque = dict(zip(sentidos, interpolar_demanda_bloques(matriz_horaria)))
    sin_carga = np.zeros(BLOQUES_POR_DIA)

    grafos_temporales = {}
    info_eventos_por_bloque = {}

    for bloque in bloques_operacion:
        G_bloque = G_base.copy()

        # 3. EVENTOS ESTOCÁSTICOS ACTIVOS ESTE BLOQUE
        eventos_activos = [
            e for e in eventos_dia
            if e['bloque_inicio'] <= bloque < e['bloque_inicio'] + e['duracion']
        ]
        info_tramos_bloque = {}
        carga_fantasma_por_tramo = collections.defaultdict(float)
        retraso_suspension_por_tramo = collections.defaultdict(float)
        for evento in eventos_activos:
            for u, v in evento['tramos']:
                k = clave_arista(u, v)
                if evento['tipo_evento'] == 'incidente_plataforma':
                    retraso_suspension_por_tramo[k] = max(
                        retraso_suspension_por_tramo[k],
                        retraso_por_suspension(evento['severidad']))
                else:
                    carga_fantasma_por_tramo[k] += evento['severidad'] * FACTOR_IMPACTO_EVENTO[evento['tipo_evento']] * capacidad_tramo(u, v)
                previo = info_tramos_bloque.get(k)
                if previo is None or evento['severidad'] > previo['severidad']:
                    info_tramos_bloque[k] = {
                        'tipo_evento': evento['tipo_evento'],
                        'severidad': evento['severidad'],
                        # Edad del evento: bloques desde que empezó (0 en su primer bloque).
                        # Solo usa bloque_inicio, que sería observable en tiempo real; la
                        # duración NO se expone al modelo (no se conoce hasta que termina).
                        'edad': bloque - evento['bloque_inicio'],
                    }
                    if GENERAR_AVISO:
                        info_tramos_bloque[k]['aviso'] = bloques_restantes_anunciados(evento, bloque)
        info_eventos_por_bloque[bloque] = info_tramos_bloque

        # 4. CÁLCULO DE CONGESTIÓN (BPR) con carga y capacidad del bloque: la carga
        # fantasma de un evento se suma solo para calcular el tiempo congestionado, NO se
        # escribe en 'carga_pasajeros' (esa columna refleja únicamente el ridership real
        # enrutado, ambos sentidos; el efecto del evento se aprende de forma explícita vía
        # hay_evento/tipo_evento/severidad_evento). La suspensión por incidente de
        # plataforma se suma aparte, después del BPR.
        for u, v, data in G_bloque.edges(data=True):
            carga_uv = carga_sentido_bloque.get((u, v), sin_carga)[bloque]
            carga_vu = carga_sentido_bloque.get((v, u), sin_carga)[bloque]
            tiempo_ideal = data.get('tiempo_minutos', 2.0)
            k = clave_arista(u, v)
            carga_tramo_efectiva = max(carga_uv, carga_vu) + carga_fantasma_por_tramo.get(k, 0.0)

            nuevo_tiempo = funcion_penalizacion_bpr(tiempo_ideal, carga_tramo_efectiva, capacidad_tramo(u, v))
            nuevo_tiempo += retraso_suspension_por_tramo.get(k, 0.0)

            data['carga_pasajeros'] = round(carga_uv + carga_vu, 1)
            data['weight'] = round(nuevo_tiempo, 2)
            data['congestibilidad'] = round(nuevo_tiempo - tiempo_ideal, 2)

        grafos_temporales[bloque] = G_bloque

    # --- DATA PIPELINE: Extracción de snapshots para Machine Learning ---
    # Solo bloques con N_REZAGOS bloques previos y N_HORIZONTES bloques posteriores el
    # mismo día (no se cruza la noche: el servicio se interrumpe).
    for bloque in bloques_operacion[N_REZAGOS:BLOQUES_POR_DIA - N_HORIZONTES]:
        G_actual = grafos_temporales[bloque]
        info_tramos_actual = info_eventos_por_bloque[bloque]

        for u, v, data in G_actual.edges(data=True):
            nombre_origen = G_actual.nodes[u].get('nombre', str(u))
            nombre_destino = G_actual.nodes[v].get('nombre', str(v))
            tramo_id = f"{nombre_origen}-{nombre_destino}"

            info_evento = info_tramos_actual.get(clave_arista(u, v))
            hay_evento = int(info_evento is not None)
            tipo_evento = info_evento['tipo_evento'] if info_evento else 'ninguno'
            severidad_evento = round(info_evento['severidad'], 4) if info_evento else 0.0
            edad_evento = info_evento['edad'] if info_evento else -1  # -1 = sin evento

            def congestibilidad_en(desfase):
                return grafos_temporales[bloque + desfase][u][v].get('congestibilidad', 0)

            datos_ml.append({
                'fecha': dia_simulacion,
                'tipo_dia': tipo_dia,
                'hora': hora_de_bloque(bloque),
                'minuto': minuto_de_bloque(bloque),
                'bloque': bloque,
                'nodo_origen': u,
                'nodo_destino': v,
                'tramo': tramo_id,
                'origen': nombre_origen,
                'tiempo_ideal': data.get('tiempo_minutos', 0),
                'carga_pasajeros_red': data.get('carga_pasajeros', 0),
                **{f'congestibilidad_t_minus_{r}': congestibilidad_en(-r) for r in range(N_REZAGOS, 0, -1)},
                'congestibilidad_t': data.get('congestibilidad', 0),
                'hay_evento': hay_evento,
                'tipo_evento': tipo_evento,
                'severidad_evento': severidad_evento,
                'edad_evento': edad_evento,
                **({'bloques_restantes_anunciados': info_evento['aviso'] if info_evento else -1}
                   if GENERAR_AVISO else {}),
                **{f'target_congestibilidad_t_plus_{h}': congestibilidad_en(h) for h in range(1, N_HORIZONTES + 1)},
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
