import networkx as nx
import pandas as pd
import joblib
from entrenador_anticipatorio import ARCHIVO_DATASET, RUTA_MODELO, TARGET, construir_features, predecir

ARCHIVO_EVALUACION = "datos_procesados/evaluacion_ruteo_eventos.csv"

def obtener_id_nodo(G, nombre_estacion):
    """Busca el ID real del nodo en el grafo usando el nombre común."""
    for n, data in G.nodes(data=True):
        if data.get('nombre') == nombre_estacion:
            return n
    raise ValueError(f"Estación '{nombre_estacion}' no encontrada en el grafo.")

print("1. Cargando el Grafo Base de la infraestructura...")
G_base = nx.read_gexf("grafo_base_metro.gexf")

print("2. Cargando el Cerebro Anticipatorio...")
paquete = joblib.load(RUTA_MODELO)
modelo = paquete['modelo']
print(f"   Modelo: {paquete['nombre_modelo']} ({'con' if paquete['usar_geo'] else 'sin'} línea/tramo)")

print("3. Cargando contexto actual (Memoria de la red)...")
# Usaremos el dataset de entrenamiento para simular que leemos los "sensores" actuales del metro
df_contexto = pd.read_csv(ARCHIVO_DATASET)
df_contexto['fecha'] = pd.to_datetime(df_contexto['fecha'])

def proyectar_hora(fecha, hora):
    """
    Tramos de la red en (fecha, hora) con tres visiones del retraso de la hora siguiente:
    - congestibilidad_t: lo que ve hoy un sistema reactivo (el tráfico de ESTA hora).
    - retraso_predicho: lo que proyecta la IA para t+1.
    - TARGET: lo que realmente ocurrió en t+1 según la simulación (verdad de referencia).
    """
    # El dataset trae varios días: filtramos por fecha para no mezclar el tráfico de días distintos
    df_hora = df_contexto[(df_contexto['fecha'] == pd.Timestamp(fecha)) & (df_contexto['hora'] == hora)].copy()
    if df_hora.empty:
        raise ValueError("No hay datos de contexto para esa fecha y hora.")
    X_pred = construir_features(df_hora, paquete['codificacion'], paquete['usar_geo'])
    df_hora['retraso_predicho'] = predecir(modelo, X_pred)
    return df_hora

def grafo_con_retraso(df_hora, columna_retraso):
    """Copia del grafo base con peso = tiempo ideal + retraso de la columna indicada."""
    G = G_base.copy()
    # Tramos sin dato conservan su tiempo ideal (el atributo 'weight' del .gexf viene en segundos)
    for u, v, data in G.edges(data=True):
        data['weight'] = data['tiempo_minutos']
    for u, v, tiempo_ideal, retraso in zip(df_hora['nodo_origen'], df_hora['nodo_destino'],
                                            df_hora['tiempo_ideal'], df_hora[columna_retraso]):
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
origen = "Pantitlán"
destino = "Auditorio" # Un viaje clásico de periferia a centro laboral

print(f"\nGenerando proyecciones de tráfico para el {fecha_viaje} a las {hora_viaje}:00 hrs...")
df_proyeccion = proyectar_hora(fecha_viaje, hora_viaje)
G_anticipatorio = grafo_con_retraso(df_proyeccion, 'retraso_predicho')
G_real = grafo_con_retraso(df_proyeccion, TARGET)

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
# Cuánto tardaría *realmente* esa ruta enfrentándose al tráfico simulado de la hora siguiente
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
# 📊 EVALUACIÓN SISTEMÁTICA SOBRE HORAS CON EVENTO (días no vistos)
# ==========================================
# Un solo caso no demuestra nada: aquí se recorren todas las horas del set de prueba con
# al menos un tramo en evento activo, y para cada una todos los pares origen-destino cuya
# ruta estática cruza un tramo afectado. Las cuatro rutas se miden contra el tráfico REAL
# de la hora siguiente (no contra la proyección de la IA, que favorecería a la IA por
# construcción):
# - estático: tiempo ideal (el baseline sin información de tráfico).
# - reactivo: congestión observada ahora (sistema reactivo, índice 5.5 de la tesis).
# - anticipatorio: congestión que la IA proyecta para t+1.
# - oráculo: congestión real de t+1 (cota superior: el mejor ahorro posible).
def evaluar_ruteo_en_eventos():
    print("\n\n=======================================================")
    print("📊 EVALUACIÓN SISTEMÁTICA: horas con evento en días no vistos")
    print("=======================================================")
    fecha_inicio, hora_inicio = paquete['inicio_test']
    posterior_al_corte = (df_contexto['fecha'] > fecha_inicio) | (
        (df_contexto['fecha'] == fecha_inicio) & (df_contexto['hora'] > hora_inicio))
    df_test = df_contexto[posterior_al_corte]
    horas_evento = df_test[df_test['hay_evento'] == 1][['fecha', 'hora']].drop_duplicates().sort_values(['fecha', 'hora'])
    print(f"Horas con evento en el set de prueba: {len(horas_evento)}")

    # Mismo criterio que el simulador: primer andén de cada estación como origen/destino
    nodo_por_estacion = {}
    for n, data in G_base.nodes(data=True):
        nodo_por_estacion.setdefault(data.get('nombre'), n)
    nodos_od = list(nodo_por_estacion.values())

    # La ruta estática no depende de la hora: se calcula una sola vez
    for u, v, data in G_base.edges(data=True):
        data['peso_estatico'] = data['tiempo_minutos']
    rutas_estaticas = dict(nx.all_pairs_dijkstra_path(G_base, weight='peso_estatico'))

    registros = []
    for fecha, hora in horas_evento.itertuples(index=False):
        df_hora = proyectar_hora(fecha, hora)
        afectados = {frozenset(t) for t in zip(df_hora[df_hora['hay_evento'] == 1]['nodo_origen'],
                                                  df_hora[df_hora['hay_evento'] == 1]['nodo_destino'])}
        G_real = grafo_con_retraso(df_hora, TARGET)
        grafos = {
            'reactivo': grafo_con_retraso(df_hora, 'congestibilidad_t'),
            'anticipatorio': grafo_con_retraso(df_hora, 'retraso_predicho'),
            'oraculo': G_real,
        }
        rutas_desde = {nombre: {} for nombre in grafos}

        for o in nodos_od:
            for d in nodos_od:
                if o == d:
                    continue
                ruta_est = rutas_estaticas[o][d]
                if not any(frozenset(t) in afectados for t in zip(ruta_est[:-1], ruta_est[1:])):
                    continue
                registro = {
                    'fecha': fecha.date(), 'hora': hora,
                    'origen': G_base.nodes[o]['nombre'], 'destino': G_base.nodes[d]['nombre'],
                    'tiempo_real_estatico': tiempo_ruta(G_real, ruta_est),
                }
                for nombre, G in grafos.items():
                    if o not in rutas_desde[nombre]:
                        rutas_desde[nombre][o] = nx.single_source_dijkstra_path(G, o, weight='weight')
                    ruta = rutas_desde[nombre][o][d]
                    registro[f'tiempo_real_{nombre}'] = tiempo_ruta(G_real, ruta)
                    registro[f'cambio_ruta_{nombre}'] = ruta != ruta_est
                registros.append(registro)

    df_eval = pd.DataFrame(registros)
    df_eval.to_csv(ARCHIVO_EVALUACION, index=False, encoding='utf-8-sig')
    print(f"Casos evaluados (pares O-D que cruzan un tramo con evento): {len(df_eval)}")
    print(f"Detalle guardado en: {ARCHIVO_EVALUACION}")

    ahorro_posible = (df_eval['tiempo_real_estatico'] - df_eval['tiempo_real_oraculo']).sum()
    resumen = []
    for nombre in ['reactivo', 'anticipatorio', 'oraculo']:
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
    print("\n--- Ahorro de tiempo REAL frente al ruteo estático ---")
    print(pd.DataFrame(resumen).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df_eval

evaluar_ruteo_en_eventos()
