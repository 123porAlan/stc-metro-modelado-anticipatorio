import networkx as nx
import pandas as pd
import joblib

def obtener_id_nodo(G, nombre_estacion):
    """Busca el ID real del nodo en el grafo usando el nombre común."""
    for n, data in G.nodes(data=True):
        if data.get('nombre') == nombre_estacion:
            return n
    raise ValueError(f"Estación '{nombre_estacion}' no encontrada en el grafo.")

print("1. Cargando el Grafo Base de la infraestructura...")
G_base = nx.read_gexf("grafo_base_metro.gexf")

print("2. Cargando el Cerebro Anticipatorio (Modelo Random Forest)...")
modelo = joblib.load("modelos/modelo_anticipatorio_rf.pkl")

print("3. Cargando contexto actual (Memoria de la red)...")
# Usaremos el dataset de entrenamiento para simular que leemos los "sensores" actuales del metro
df_contexto = pd.read_csv("datos_procesados/dataset_features_entrenamiento.csv")

def crear_grafo_futuro(fecha_prediccion, hora_prediccion):
    """
    Usa la IA para predecir los retrasos y genera el grafo futuro con precisión absoluta.
    """
    G_futuro = G_base.copy()
    # El dataset trae varios días: filtramos por fecha para no mezclar el tráfico de días distintos
    df_hora = df_contexto[(df_contexto['fecha'] == fecha_prediccion) & (df_contexto['hora'] == hora_prediccion)].copy()
    
    if df_hora.empty:
        raise ValueError("No hay datos de contexto para esa fecha y hora.")

    # La IA predice el tráfico con las mismas features con que fue entrenada
    # (incluye dummies de tipo_evento; las que no aparezcan en esta hora se rellenan con 0)
    if 'tipo_evento' in df_hora.columns:
        df_hora = pd.concat([df_hora, pd.get_dummies(df_hora['tipo_evento'], prefix='evento')], axis=1)
    X_pred = df_hora.reindex(columns=modelo.feature_names_in_, fill_value=0)
    df_hora['retraso_futuro_predicho'] = modelo.predict(X_pred)
    
    # Inyectamos el tráfico usando los IDs Reales (nodos exactos)
    for _, row in df_hora.iterrows():
        u = row['nodo_origen']
        v = row['nodo_destino']
        
        nuevo_tiempo = row['tiempo_ideal'] + row['retraso_futuro_predicho']
        
        # Como usamos el ID único, la topología se actualiza perfectamente
        if G_futuro.has_edge(u, v):
            G_futuro[u][v]['weight'] = round(nuevo_tiempo, 2)
            
    return G_futuro

def imprimir_ruta(G, ruta, nombre_ruta):
    tiempo_total = 0
    camino_nombres = []
    
    for i in range(len(ruta) - 1):
        u = ruta[i]
        v = ruta[i+1]
        peso = G[u][v]['weight']
        tiempo_total += peso
        camino_nombres.append(G.nodes[u].get('nombre', str(u)))
    
    camino_nombres.append(G.nodes[ruta[-1]].get('nombre', str(ruta[-1])))
    
    print(f"\n--- {nombre_ruta} ---")
    print(f"Ruta: {' -> '.join(camino_nombres)}")
    print(f"Tiempo estimado: {tiempo_total:.2f} minutos")
    return tiempo_total

# ==========================================
# 🚀 CASO DE PRUEBA: SISTEMA REACTIVO VS ANTICIPATORIO
# ==========================================
# Vamos a simular un viaje en HORA PICO MATUTINA (7:00 AM)
fecha_viaje = "2026-01-13"  # Día laboral del caso de prueba original
hora_viaje = 7
origen = "Pantitlán"
destino = "Auditorio" # Un viaje clásico de periferia a centro laboral

print(f"\nGenerando proyecciones de tráfico para el {fecha_viaje} a las {hora_viaje}:00 hrs...")
G_anticipatorio = crear_grafo_futuro(fecha_viaje, hora_viaje)

nodo_origen = obtener_id_nodo(G_base, origen)
nodo_destino = obtener_id_nodo(G_base, destino)

# 1. Ruteo Estático (El "Google Maps" tradicional, solo ve distancia física)
ruta_estatica = nx.shortest_path(G_base, source=nodo_origen, target=nodo_destino, weight='tiempo_minutos')
# Calculamos cuánto tardaría *realmente* esa ruta estática enfrentándose al tráfico real
tiempo_estatico_en_trafico = sum([G_anticipatorio[ruta_estatica[i]][ruta_estatica[i+1]]['weight'] for i in range(len(ruta_estatica)-1)])

# 2. Ruteo Anticipatorio (Tu IA, que elige la ruta esquivando los pesos proyectados altos)
ruta_ia = nx.shortest_path(G_anticipatorio, source=nodo_origen, target=nodo_destino, weight='weight')
tiempo_ia = sum([G_anticipatorio[ruta_ia[i]][ruta_ia[i+1]]['weight'] for i in range(len(ruta_ia)-1)])

print(f"\n=======================================================")
print(f"🚉 VIAJE SOLICITADO: {origen} a {destino} a las {hora_viaje}:00 AM")
print(f"=======================================================")

print(f"\n[SISTEMA REACTIVO / ESTÁTICO]")
ruta_nombres_estatica = [G_base.nodes[n].get('nombre', str(n)) for n in ruta_estatica]
print(f"Ruta sugerida: {' -> '.join(ruta_nombres_estatica)}")
print(f"Tiempo IDEAL (sin considerar tráfico): {nx.shortest_path_length(G_base, nodo_origen, nodo_destino, 'tiempo_minutos'):.2f} mins")
print(f"Tiempo REAL que sufrirá el usuario: {tiempo_estatico_en_trafico:.2f} mins")

print(f"\n[SISTEMA ANTICIPATORIO (IA)]")
ruta_nombres_ia = [G_base.nodes[n].get('nombre', str(n)) for n in ruta_ia]
print(f"Ruta sugerida: {' -> '.join(ruta_nombres_ia)}")
print(f"Tiempo estimado proyectado: {tiempo_ia:.2f} mins")

if ruta_estatica != ruta_ia:
    print(f"\n✨ ¡LA IA CAMBIÓ LA RUTA! Esquivó nodos saturados y ahorró {(tiempo_estatico_en_trafico - tiempo_ia):.2f} minutos.")
else:
    print(f"\nℹ️ La ruta es idéntica en ambos sistemas (no hubo alternativa más rápida a pesar del tráfico).")
