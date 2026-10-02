import argparse
import networkx as nx
import pandas as pd
import joblib
from entrenador_anticipatorio import ARCHIVO_DATASET, RUTA_MODELO, HORIZONTES, MINUTOS_BLOQUE, construir_features, predecir

BLOQUES_POR_HORA = 60 // MINUTOS_BLOQUE
HORA_INICIO_SERVICIO = 5  # el bloque 0 del simulador es 05:00

parser = argparse.ArgumentParser(description="Ruteo anticipatorio: caso de prueba + evaluación en bloques de 15 min con evento.")
parser.add_argument("--dataset", default=ARCHIVO_DATASET)
parser.add_argument("--modelo", default=RUTA_MODELO)
parser.add_argument("--evaluacion", default="datos_procesados/evaluacion_ruteo_eventos.csv")
parser.add_argument("--resumen", default="datos_procesados/resumen_ruteo_eventos.csv")
parser.add_argument("--resumen-edad", default="datos_procesados/resumen_ruteo_eventos_por_edad.csv",
                    help="Resumen restringido a casos cuyo evento está en su hora 0, +1 o +2")
parser.add_argument("--horizontes", type=int, nargs='+', default=list(HORIZONTES), choices=list(HORIZONTES),
                    help="Horizontes (min) a evaluar; por defecto todos")
args = parser.parse_args()

def obtener_id_nodo(G, nombre_estacion):
    """Busca el ID real del nodo en el grafo usando el nombre común."""
    for n, data in G.nodes(data=True):
        if data.get('nombre') == nombre_estacion:
            return n
    raise ValueError(f"Estación '{nombre_estacion}' no encontrada en el grafo.")

print("1. Cargando el Grafo Base de la infraestructura...")
G_base = nx.read_gexf("grafo_base_metro.gexf")

print("2. Cargando el Cerebro Anticipatorio...")
paquete = joblib.load(args.modelo)
modelos = paquete['modelos_por_horizonte']
print(f"   Modelo: {paquete['nombre_modelo']} ({'con' if paquete['usar_geo'] else 'sin'} línea/tramo)")

print("3. Cargando contexto actual (Memoria de la red)...")
# Usaremos el dataset de entrenamiento para simular que leemos los "sensores" actuales del metro
df_contexto = pd.read_csv(args.dataset)
df_contexto['fecha'] = pd.to_datetime(df_contexto['fecha'])

def bloque_de(hora, minuto=0):
    return (hora - HORA_INICIO_SERVICIO) * BLOQUES_POR_HORA + minuto // MINUTOS_BLOQUE

def columna_predicha(horizonte):
    return f'retraso_predicho_{horizonte}'

def proyectar_bloque(fecha, bloque, horizontes):
    """
    Tramos de la red en (fecha, bloque) con tres visiones del retraso a cada horizonte k:
    - congestibilidad_t: lo que ve hoy un sistema reactivo (el tráfico de ESTE bloque).
    - retraso_predicho_<k>: lo que proyecta la IA para dentro de k minutos.
    - HORIZONTES[k]: lo que realmente ocurrió en t+k según la simulación (verdad de referencia).
    """
    # El dataset trae varios días: filtramos por fecha para no mezclar el tráfico de días distintos
    df_bloque = df_contexto[(df_contexto['fecha'] == pd.Timestamp(fecha)) & (df_contexto['bloque'] == bloque)].copy()
    if df_bloque.empty:
        raise ValueError("No hay datos de contexto para esa fecha y bloque.")
    for horizonte in horizontes:
        paquete_h = modelos[horizonte]
        X_pred = construir_features(df_bloque, paquete_h['codificacion'], paquete['usar_geo'])
        df_bloque[columna_predicha(horizonte)] = predecir(paquete_h['modelo'], X_pred)
    return df_bloque

def grafo_con_retraso(df_bloque, columna_retraso):
    """Copia del grafo base con peso = tiempo ideal + retraso de la columna indicada."""
    G = G_base.copy()
    # Tramos sin dato conservan su tiempo ideal (el atributo 'weight' del .gexf viene en segundos)
    for u, v, data in G.edges(data=True):
        data['weight'] = data['tiempo_minutos']
    for u, v, tiempo_ideal, retraso in zip(df_bloque['nodo_origen'], df_bloque['nodo_destino'],
                                            df_bloque['tiempo_ideal'], df_bloque[columna_retraso]):
        if G.has_edge(u, v):
            G[u][v]['weight'] = round(tiempo_ideal + retraso, 2)
    return G

def tiempo_ruta(G, ruta):
    return sum(G[ruta[i]][ruta[i + 1]]['weight'] for i in range(len(ruta) - 1))

def nombres_ruta(ruta):
    return ' -> '.join(G_base.nodes[n].get('nombre', str(n)) for n in ruta)

# ==========================================
# 🚀 CASO DE PRUEBA: SISTEMA REACTIVO VS ANTICIPATORIO
# ==========================================
# Vamos a simular un viaje en HORA PICO MATUTINA (7:00 AM)
fecha_viaje = "2026-01-13"  # Día laboral del caso de prueba original
hora_viaje = 7
horizonte_viaje = MINUTOS_BLOQUE  # la IA proyecta el bloque siguiente
origen = "Pantitlán"
destino = "Auditorio" # Un viaje clásico de periferia a centro laboral

print(f"\nGenerando proyecciones de tráfico para el {fecha_viaje} a las {hora_viaje}:00 hrs "
      f"(horizonte {horizonte_viaje} min)...")
df_proyeccion = proyectar_bloque(fecha_viaje, bloque_de(hora_viaje), [horizonte_viaje])
G_anticipatorio = grafo_con_retraso(df_proyeccion, columna_predicha(horizonte_viaje))
G_real = grafo_con_retraso(df_proyeccion, HORIZONTES[horizonte_viaje])

nodo_origen = obtener_id_nodo(G_base, origen)
nodo_destino = obtener_id_nodo(G_base, destino)

# 1. Ruteo Estático (El "Google Maps" tradicional, solo ve distancia física)
ruta_estatica = nx.shortest_path(G_base, source=nodo_origen, target=nodo_destino, weight='tiempo_minutos')
# 2. Ruteo Anticipatorio (Tu IA, que elige la ruta esquivando los pesos proyectados altos)
ruta_ia = nx.shortest_path(G_anticipatorio, source=nodo_origen, target=nodo_destino, weight='weight')

print(f"\n=======================================================")
print(f"🚉 VIAJE SOLICITADO: {origen} a {destino} a las {hora_viaje}:00 AM")
print(f"=======================================================")

print(f"\n[SISTEMA REACTIVO / ESTÁTICO]")
print(f"Ruta sugerida: {nombres_ruta(ruta_estatica)}")
print(f"Tiempo IDEAL (sin considerar tráfico): {nx.shortest_path_length(G_base, nodo_origen, nodo_destino, 'tiempo_minutos'):.2f} mins")
# Cuánto tardaría *realmente* esa ruta enfrentándose al tráfico simulado del horizonte
print(f"Tiempo REAL que sufrirá el usuario: {tiempo_ruta(G_real, ruta_estatica):.2f} mins")

print(f"\n[SISTEMA ANTICIPATORIO (IA)]")
print(f"Ruta sugerida: {nombres_ruta(ruta_ia)}")
print(f"Tiempo estimado proyectado: {tiempo_ruta(G_anticipatorio, ruta_ia):.2f} mins")
print(f"Tiempo REAL que sufrirá el usuario: {tiempo_ruta(G_real, ruta_ia):.2f} mins")

if ruta_estatica != ruta_ia:
    print(f"\n✨ ¡LA IA CAMBIÓ LA RUTA! Ahorro real: {(tiempo_ruta(G_real, ruta_estatica) - tiempo_ruta(G_real, ruta_ia)):.2f} minutos.")
else:
    print(f"\nℹ️ La ruta es idéntica en ambos sistemas (no hubo alternativa más rápida a pesar del tráfico).")

# ==========================================
# 📊 EVALUACIÓN SISTEMÁTICA SOBRE BLOQUES CON EVENTO (días no vistos)
# ==========================================
# Un solo caso no demuestra nada: aquí se recorren todos los bloques de 15 min del set de
# prueba con al menos un tramo en evento activo, y para cada uno todos los pares
# origen-destino cuya ruta estática cruza un tramo afectado. Cada caso se evalúa en cada
# horizonte k (15, 30, 45 y 60 min): las rutas se miden contra el tráfico REAL de t+k (no
# contra la proyección de la IA, que favorecería a la IA por construcción):
# - estático: tiempo ideal (el baseline sin información de tráfico).
# - reactivo: congestión observada ahora (sistema reactivo, índice 5.5 de la tesis). Su
#   ruta no depende de k; solo cambia contra qué tráfico real se mide.
# - anticipatorio: congestión que la IA proyecta para t+k.
# - oráculo: congestión real de t+k (cota superior: el mejor ahorro posible).
# Cada caso guarda además la edad del evento que cruza su ruta estática (en bloques; si
# cruza varios, el más reciente) y su hora_evento (0 = primera hora del evento, 1 = hora
# + 1, ...), para ver si la ventaja de anticipar se concentra al inicio del evento, cuando
# menos se sabe si va a seguir, y el tipo de ese evento (lluvia, falla o incidente).
# - reactivo_duracion: reactivo con una regla de duración (ver README.md, Sección 11). En
#   los tramos cuyo evento tiene EDAD_IGNORADA bloques o más supone que el evento ya no
#   sigue y usa el perfil histórico sin evento del tramo en lugar de la congestión observada.
# - hibrido: la IA, salvo en los tramos con evento de EDAD_IGNORADA bloques o más, donde
#   usa la misma regla de duración (Sección 12).
# Los umbrales son los de las Secciones 11-12 (1, 2 y 3 horas) convertidos a bloques, sin
# reajustarlos: sufijo _1 y _3 para 1 y 3 horas; sin sufijo, el umbral por defecto (2 horas).
EDADES_REPORTADAS = [0, 1, 2]  # hora del evento
EDAD_IGNORADA = 2 * BLOQUES_POR_HORA
UMBRALES_REGLA = [h * BLOQUES_POR_HORA for h in (1, 2, 3)]

def sufijo_umbral(umbral):
    return "" if umbral == EDAD_IGNORADA else f"_{umbral // BLOQUES_POR_HORA}"

SISTEMAS_REACTIVOS = ['reactivo'] + [f'reactivo_duracion{sufijo_umbral(u)}' for u in UMBRALES_REGLA]
SISTEMAS = SISTEMAS_REACTIVOS + ['anticipatorio'] + [f'hibrido{sufijo_umbral(u)}' for u in UMBRALES_REGLA] + ['oraculo']

def resumir(df_eval):
    ahorro_posible = (df_eval['tiempo_real_estatico'] - df_eval['tiempo_real_oraculo']).sum()
    resumen = []
    for nombre in SISTEMAS:
        ahorro = df_eval['tiempo_real_estatico'] - df_eval[f'tiempo_real_{nombre}']
        cambio = df_eval[f'cambio_ruta_{nombre}']
        resumen.append({
            'sistema': nombre,
            'casos_con_cambio_ruta': int(cambio.sum()),
            'pct_cambio_ruta': 100 * cambio.mean(),
            'ahorro_medio_min': ahorro.mean(),
            'ahorro_medio_si_cambia_min': ahorro[cambio].mean() if cambio.any() else 0.0,
            'ahorro_max_min': ahorro.max(),
            'casos_peor_que_estatico': int((ahorro < -1e-9).sum()),
            'pct_ahorro_posible_capturado': 100 * ahorro.sum() / ahorro_posible if ahorro_posible > 0 else float('nan'),
        })
    return pd.DataFrame(resumen)

def resumir_por_horizonte(df_eval):
    """resumir() por separado para cada horizonte."""
    filas = []
    for horizonte, df_h in df_eval.groupby('horizonte_min'):
        resumen = resumir(df_h)
        resumen.insert(0, 'horizonte_min', horizonte)
        resumen.insert(1, 'bloques_con_evento', df_h[['fecha', 'bloque']].drop_duplicates().shape[0])
        resumen.insert(2, 'casos_evaluados', len(df_h))
        filas.append(resumen)
    return pd.concat(filas, ignore_index=True)

def resumir_por_edad(df_eval):
    """Mismo resumen, por horizonte, solo con los casos en la hora 0, +1 y +2 del evento."""
    filas = []
    for horizonte, df_h in df_eval.groupby('horizonte_min'):
        for edad in EDADES_REPORTADAS:
            df_edad = df_h[df_h['hora_evento'] == edad]
            if df_edad.empty:
                continue
            resumen = resumir(df_edad)
            resumen.insert(0, 'horizonte_min', horizonte)
            resumen.insert(1, 'hora_evento', edad)
            resumen.insert(2, 'bloques', df_edad[['fecha', 'bloque']].drop_duplicates().shape[0])
            resumen.insert(3, 'casos', len(df_edad))
            for nombre in SISTEMAS:
                resumen.loc[resumen['sistema'] == nombre, 'ahorro_total_min'] = \
                    (df_edad['tiempo_real_estatico'] - df_edad[f'tiempo_real_{nombre}']).sum()
            filas.append(resumen)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()

def perfil_sin_evento(df_entrenamiento):
    """
    Congestión típica de cada tramo sin evento: media de congestibilidad_t en las filas de
    entrenamiento sin evento, por (tramo, tipo de día, bloque). Es lo que el reactivo con
    regla de duración supone que habrá cuando descarta un evento viejo. Solo usa días de
    entrenamiento, así que no filtra información del set de prueba.
    """
    sin_evento = df_entrenamiento[df_entrenamiento['hay_evento'] == 0]
    return sin_evento.groupby(['nodo_origen', 'nodo_destino', 'tipo_dia', 'bloque'])['congestibilidad_t'].mean()

def congestion_con_regla_duracion(df_bloque, perfil, columna='congestibilidad_t', umbral=EDAD_IGNORADA):
    """La columna indicada, salvo en tramos con evento de edad >= umbral bloques (perfil sin evento; 0 si no hay)."""
    viejo = df_bloque['edad_evento'] >= umbral
    claves = pd.MultiIndex.from_frame(df_bloque.loc[viejo, ['nodo_origen', 'nodo_destino', 'tipo_dia', 'bloque']])
    congestion = df_bloque[columna].copy()
    congestion[viejo] = perfil.reindex(claves).fillna(0.0).to_numpy()
    return congestion

def rutas_de(G, nodos_od, cache):
    """Rutas más cortas desde cada origen, calculadas bajo demanda y guardadas en cache."""
    def ruta(o, d):
        if o not in cache:
            cache[o] = nx.single_source_dijkstra_path(G, o, weight='weight')
        return cache[o][d]
    return ruta

def evaluar_ruteo_en_eventos():
    print("\n\n=======================================================")
    print("📊 EVALUACIÓN SISTEMÁTICA: bloques con evento en días no vistos")
    print("=======================================================")
    fecha_inicio, bloque_inicio = paquete['inicio_test']
    posterior_al_corte = (df_contexto['fecha'] > fecha_inicio) | (
        (df_contexto['fecha'] == fecha_inicio) & (df_contexto['bloque'] > bloque_inicio))
    df_test = df_contexto[posterior_al_corte]
    perfil = perfil_sin_evento(df_contexto[~posterior_al_corte])
    bloques_evento = df_test[df_test['hay_evento'] == 1][['fecha', 'bloque']].drop_duplicates().sort_values(['fecha', 'bloque'])
    print(f"Bloques con evento en el set de prueba: {len(bloques_evento)} | horizontes: {args.horizontes} min")

    # Mismo criterio que el simulador: primer andén de cada estación como origen/destino
    nodo_por_estacion = {}
    for n, data in G_base.nodes(data=True):
        nodo_por_estacion.setdefault(data.get('nombre'), n)
    nodos_od = list(nodo_por_estacion.values())

    # La ruta estática no depende del bloque: se calcula una sola vez
    for u, v, data in G_base.edges(data=True):
        data['peso_estatico'] = data['tiempo_minutos']
    rutas_estaticas = dict(nx.all_pairs_dijkstra_path(G_base, weight='peso_estatico'))

    # Un DataFrame por bloque: acumular ~500k diccionarios de casos por semilla pasa de 2 GB.
    tablas = []
    for fecha, bloque in bloques_evento.itertuples(index=False):
        registros = []
        df_bloque = proyectar_bloque(fecha, bloque, args.horizontes)
        con_evento = df_bloque[df_bloque['hay_evento'] == 1]
        # tramo afectado -> (edad, tipo) de su evento (el dataset guarda el evento más severo del tramo)
        afectados = {frozenset((u, v)): (edad, tipo) for u, v, edad, tipo in zip(
            con_evento['nodo_origen'], con_evento['nodo_destino'], con_evento['edad_evento'], con_evento['tipo_evento'])}
        casos = []
        for o in nodos_od:
            for d in nodos_od:
                if o == d:
                    continue
                ruta_est = rutas_estaticas[o][d]
                eventos = [afectados[frozenset(t)] for t in zip(ruta_est[:-1], ruta_est[1:]) if frozenset(t) in afectados]
                if eventos:
                    # El evento más reciente de la ruta (a igual edad, el primero en el recorrido)
                    casos.append((o, d, ruta_est, min(eventos, key=lambda e: e[0])))

        # Los sistemas reactivos ven la congestión de ahora: su ruta es la misma en todo horizonte.
        pesos = {'reactivo': 'congestibilidad_t'}
        for umbral in UMBRALES_REGLA:
            clave = f'reactivo_duracion{sufijo_umbral(umbral)}'
            df_bloque[clave] = congestion_con_regla_duracion(df_bloque, perfil, 'congestibilidad_t', umbral)
            pesos[clave] = clave
        rutas_reactivas = {nombre: rutas_de(grafo_con_retraso(df_bloque, pesos[nombre]), nodos_od, {})
                           for nombre in SISTEMAS_REACTIVOS}

        hora, minuto = HORA_INICIO_SERVICIO + bloque // BLOQUES_POR_HORA, (bloque % BLOQUES_POR_HORA) * MINUTOS_BLOQUE
        for horizonte in args.horizontes:
            predicho = columna_predicha(horizonte)
            G_real = grafo_con_retraso(df_bloque, HORIZONTES[horizonte])
            grafos = {'anticipatorio': grafo_con_retraso(df_bloque, predicho)}
            for umbral in UMBRALES_REGLA:
                clave = f'hibrido{sufijo_umbral(umbral)}'
                df_bloque[clave] = congestion_con_regla_duracion(df_bloque, perfil, predicho, umbral)
                grafos[clave] = grafo_con_retraso(df_bloque, clave)
            grafos['oraculo'] = G_real
            rutas = rutas_reactivas | {nombre: rutas_de(G, nodos_od, {}) for nombre, G in grafos.items()}

            for o, d, ruta_est, (edad, tipo) in casos:
                registro = {
                    'fecha': fecha.date(), 'hora': hora, 'minuto': minuto, 'bloque': bloque,
                    'horizonte_min': horizonte,
                    'origen': G_base.nodes[o]['nombre'], 'destino': G_base.nodes[d]['nombre'],
                    'edad_evento': edad, 'hora_evento': edad // BLOQUES_POR_HORA, 'tipo_evento': tipo,
                    'tiempo_real_estatico': tiempo_ruta(G_real, ruta_est),
                }
                for nombre in SISTEMAS:
                    ruta = rutas[nombre](o, d)
                    registro[f'tiempo_real_{nombre}'] = tiempo_ruta(G_real, ruta)
                    registro[f'cambio_ruta_{nombre}'] = ruta != ruta_est
                registros.append(registro)
        tablas.append(pd.DataFrame(registros))

    df_eval = pd.concat(tablas, ignore_index=True)
    df_eval.to_csv(args.evaluacion, index=False, encoding='utf-8-sig')
    print(f"Casos evaluados (pares O-D que cruzan un tramo con evento, por horizonte): {len(df_eval)}")
    print(f"Detalle guardado en: {args.evaluacion}")

    df_resumen = resumir_por_horizonte(df_eval)
    df_resumen.to_csv(args.resumen, index=False)
    print("\n--- Ahorro de tiempo REAL frente al ruteo estático, por horizonte ---")
    print(df_resumen.drop(columns=['bloques_con_evento', 'casos_evaluados']).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    df_edad = resumir_por_edad(df_eval)
    df_edad.to_csv(args.resumen_edad, index=False)
    fuera = int((~df_eval['hora_evento'].isin(EDADES_REPORTADAS)).sum())
    print(f"\n--- Por hora del evento (0, +1, +2; {fuera} casos con hora >= 3 excluidos) ---")
    if not df_edad.empty:
        print(df_edad[['horizonte_min', 'hora_evento', 'bloques', 'casos', 'sistema', 'ahorro_total_min',
                       'casos_peor_que_estatico', 'pct_ahorro_posible_capturado']].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return df_eval

evaluar_ruteo_en_eventos()
