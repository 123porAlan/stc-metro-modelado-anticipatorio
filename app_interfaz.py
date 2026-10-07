"""
Interfaz de usuario del MVP (Sección 6.2): ruta estática vs. ruta de IA anticipatoria.

Ejecutar con:  streamlit run app_interfaz.py

El modelo necesita el historial reciente de cada tramo (congestibilidad t-1..t-4), así que
la app usa un día simulado del dataset como lectura de los "sensores" de la red para el
tipo de día y bloque elegidos, y le inyecta encima el evento opcional del usuario.
"""
import networkx as nx
import pandas as pd
import joblib
import streamlit as st
from entrenador_anticipatorio import ARCHIVO_DATASET, RUTA_MODELO, HORIZONTES, MINUTOS_BLOQUE, \
    construir_features, predecir, linea_de_nodo

ARCHIVO_GRAFO = "grafo_base_metro.gexf"
BLOQUES_POR_HORA = 60 // MINUTOS_BLOQUE
HORA_INICIO_SERVICIO = 5  # el bloque 0 del simulador es 05:00

TIPOS_DIA = {"Laboral": "laboral", "Fin de semana": "fin_de_semana"}
TIPOS_EVENTO = {"Lluvia": "lluvia", "Falla mecánica": "falla_mecanica", "Incidente en plataforma": "incidente_plataforma"}
LINEAS_SUPERFICIE = ["A", "B", "12"]  # las que alcanza la lluvia en el simulador
# Mismos parámetros que simulador_congestion.py para reproducir el efecto de un evento.
MINUTOS_CICLO_CIERRE = 60
FACTOR_IMPACTO_EVENTO = {"lluvia": 0.6, "falla_mecanica": 1.4}
BPR_ALPHA, BPR_BETA, BPR_TOPE = 0.15, 4, 4

COLORES_LINEA = {
    "1": "#F04E98", "2": "#005EB8", "3": "#AF9800", "4": "#6BBBAE", "5": "#FFD100",
    "6": "#DA291C", "7": "#E87722", "8": "#009A44", "9": "#512F2E", "A": "#981D97",
    "B": "#7FA88B", "12": "#B0A32A",
}


# ==========================================
# Carga de recursos (una sola vez por sesión del servidor)
# ==========================================
@st.cache_resource
def cargar_grafo():
    return nx.read_gexf(ARCHIVO_GRAFO)

@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)

@st.cache_data
def cargar_contexto():
    df = pd.read_csv(ARCHIVO_DATASET)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df


# ==========================================
# Lógica de proyección y ruteo
# ==========================================
def hora_de_bloque(bloque):
    minutos = HORA_INICIO_SERVICIO * 60 + bloque * MINUTOS_BLOQUE
    return f"{minutos // 60:02d}:{minutos % 60:02d}"

def fecha_representativa(df, tipo_dia):
    """Último día simulado del tipo pedido: los últimos días son los del set de prueba."""
    return df.loc[df['tipo_dia'] == tipo_dia, 'fecha'].max()

def lineas_de_tramos(df):
    return pd.Series([linea_de_nodo(u) if linea_de_nodo(u) == linea_de_nodo(v) else 'transbordo'
                      for u, v in zip(df['nodo_origen'], df['nodo_destino'])], index=df.index)

def congestion_con_evento(congestion, tiempo_ideal, tipo_evento, severidad):
    """
    Congestión observada de un tramo con el evento activo, como en el simulador: el
    incidente en plataforma suspende el tramo (+30·s² min); lluvia y falla mecánica suman
    una carga fantasma de s·factor veces la capacidad a la saturación del BPR, que se
    despeja de la congestión sin evento (congestión = t0·α·saturación^β).
    """
    if tipo_evento == 'incidente_plataforma':
        return congestion + severidad * MINUTOS_CICLO_CIERRE * severidad / 2
    saturacion = (congestion / (tiempo_ideal * BPR_ALPHA)) ** (1 / BPR_BETA)
    saturacion += severidad * FACTOR_IMPACTO_EVENTO[tipo_evento]
    return (tiempo_ideal * BPR_ALPHA * saturacion ** BPR_BETA).clip(upper=tiempo_ideal * (BPR_TOPE - 1))

def inyectar_evento(df_bloque, afectados, tipo_evento, severidad, edad):
    """
    Marca los tramos afectados con el evento, como lo vería el modelo en tiempo real, y
    aplica su efecto a la congestión observada en este bloque y en los previos en que ya
    estaba activo (edad en bloques). El modelo se apoya sobre todo en esa congestión: con
    solo marcar el evento su proyección casi no cambia.
    """
    df = df_bloque.copy()
    df.loc[afectados, ['hay_evento', 'tipo_evento', 'severidad_evento', 'edad_evento']] = \
        [1, tipo_evento, severidad, edad]
    columnas = ['congestibilidad_t'] + [f'congestibilidad_t_minus_{k}' for k in range(1, min(edad, 4) + 1)]
    for columna in columnas:
        df.loc[afectados, columna] = congestion_con_evento(
            df.loc[afectados, columna], df.loc[afectados, 'tiempo_ideal'], tipo_evento, severidad)
    return df

def proyectar(df_bloque, paquete, horizonte):
    paquete_h = paquete['modelos_por_horizonte'][horizonte]
    X = construir_features(df_bloque, paquete_h['codificacion'], paquete['usar_geo'])
    return predecir(paquete_h['modelo'], X)

def grafo_proyectado(G_base, df_bloque, retraso):
    """Copia del grafo con peso = tiempo ideal + retraso proyectado (sin dato: tiempo ideal)."""
    G = G_base.copy()
    for _, _, data in G.edges(data=True):
        data['weight'] = data['tiempo_minutos']
    for u, v, tiempo_ideal, r in zip(df_bloque['nodo_origen'], df_bloque['nodo_destino'],
                                     df_bloque['tiempo_ideal'], retraso):
        if G.has_edge(u, v):
            G[u][v]['weight'] = tiempo_ideal + r
    return G

def ruta_optima(G, origen, destino, peso):
    """Ruta más corta entre estaciones (no andenes): se parte del andén más conveniente."""
    H = G.copy()
    for virtual, nombre in (('__origen__', origen), ('__destino__', destino)):
        H.add_node(virtual)
        for n, data in G.nodes(data=True):
            if data.get('nombre') == nombre:
                H.add_edge(virtual, n, **{peso: 0.0})
    return nx.shortest_path(H, '__origen__', '__destino__', weight=peso)[1:-1]

def tiempo_ruta(G, ruta, peso):
    return sum(G[a][b][peso] for a, b in zip(ruta[:-1], ruta[1:]))

def desglose(G, ruta, peso):
    """Tramos consecutivos agrupados por línea, con los transbordos entre ellos."""
    pasos = []
    for a, b in zip(ruta[:-1], ruta[1:]):
        tiempo = G[a][b][peso]
        if G[a][b].get('tipo') == 'transbordo':
            pasos.append({'linea': None, 'desde': G.nodes[a]['nombre'], 'hasta': G.nodes[b]['nombre'],
                          'estaciones': 0, 'minutos': tiempo})
        elif pasos and pasos[-1]['linea'] == linea_de_nodo(a):
            pasos[-1]['hasta'] = G.nodes[b]['nombre']
            pasos[-1]['estaciones'] += 1
            pasos[-1]['minutos'] += tiempo
        else:
            pasos.append({'linea': linea_de_nodo(a), 'desde': G.nodes[a]['nombre'], 'hasta': G.nodes[b]['nombre'],
                          'estaciones': 1, 'minutos': tiempo})
    return pasos


# ==========================================
# Componentes visuales
# ==========================================
ESTILOS = """
<style>
.paso { display:flex; align-items:center; gap:.75rem; padding:.45rem 0; border-bottom:1px solid rgba(128,128,128,.15); }
.linea { min-width:2.4rem; text-align:center; font-weight:700; color:#fff; border-radius:.4rem; padding:.15rem .4rem; }
.transbordo { min-width:2.4rem; text-align:center; font-size:1.1rem; }
.detalle { flex:1; }
.detalle small { opacity:.65; }
.minutos { font-variant-numeric:tabular-nums; opacity:.8; }
</style>
"""

def mostrar_desglose(pasos):
    filas = []
    for p in pasos:
        if p['linea'] is None:
            filas.append(f"<div class='paso'><span class='transbordo'>⇄</span>"
                         f"<span class='detalle'>Transbordo en <b>{p['desde']}</b></span>"
                         f"<span class='minutos'>{p['minutos']:.1f} min</span></div>")
        else:
            color = COLORES_LINEA.get(p['linea'], '#666')
            estaciones = f"{p['estaciones']} estación" + ("" if p['estaciones'] == 1 else "es")
            filas.append(f"<div class='paso'><span class='linea' style='background:{color}'>{p['linea']}</span>"
                         f"<span class='detalle'>{p['desde']} → {p['hasta']}<br><small>{estaciones}</small></span>"
                         f"<span class='minutos'>{p['minutos']:.1f} min</span></div>")
    st.markdown("".join(filas), unsafe_allow_html=True)

def mostrar_ruta(titulo, G, ruta, peso, tiempo_proyectado, delta=None):
    pasos = desglose(G, ruta, peso)
    transbordos = sum(p['linea'] is None for p in pasos)
    with st.container(border=True):
        st.subheader(titulo)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tiempo proyectado", f"{tiempo_proyectado:.1f} min", delta=delta, delta_color="inverse")
        c2.metric("Estaciones", sum(p['estaciones'] for p in pasos))
        c3.metric("Transbordos", transbordos)
        mostrar_desglose(pasos)


# ==========================================
# Interfaz
# ==========================================
st.set_page_config(page_title="Ruteo anticipatorio · STC Metro", layout="wide")
st.markdown(ESTILOS, unsafe_allow_html=True)

G_base = cargar_grafo()
paquete = cargar_modelo()
df_contexto = cargar_contexto()

st.title("Ruteo anticipatorio · STC Metro")
st.caption(f"Compara la ruta más corta sin tráfico con la que elige la IA a partir de la congestión que "
           f"proyecta para la red. Modelo: {paquete['nombre_modelo']}.")

# --- Contexto (barra lateral) ---
with st.sidebar:
    st.header("Contexto")
    tipo_dia = TIPOS_DIA[st.radio("Tipo de día", list(TIPOS_DIA), horizontal=True)]
    bloques = sorted(df_contexto['bloque'].unique())
    bloque = st.select_slider("Hora de salida", options=bloques, value=(7 - HORA_INICIO_SERVICIO) * BLOQUES_POR_HORA,
                              format_func=hora_de_bloque)
    horizonte = st.selectbox("Horizonte de proyección", list(HORIZONTES), format_func=lambda h: f"{h} min",
                             help="Para cuántos minutos adelante proyecta la IA la congestión de la red.")

    fecha = fecha_representativa(df_contexto, tipo_dia)
    df_bloque = df_contexto[(df_contexto['fecha'] == fecha) & (df_contexto['bloque'] == bloque)].reset_index(drop=True)
    linea_tramo = lineas_de_tramos(df_bloque)

    st.divider()
    con_evento = st.toggle("Agregar evento activo")
    if con_evento:
        tipo_evento = TIPOS_EVENTO[st.selectbox("Tipo de evento", list(TIPOS_EVENTO))]
        if tipo_evento == 'lluvia':
            linea = st.selectbox("Línea afectada", LINEAS_SUPERFICIE, format_func=lambda l: f"Línea {l}",
                                 help="La lluvia afecta a toda una línea de superficie.")
            afectados = linea_tramo == linea
        else:
            opciones = list(df_bloque.index[linea_tramo != 'transbordo'])
            tramo = st.selectbox("Tramo afectado", opciones,
                                 format_func=lambda i: f"L{linea_tramo[i]} · {df_bloque.at[i, 'tramo']}")
            afectados = df_bloque.index == tramo
        severidad = st.slider("Severidad", 0.05, 1.0, 0.4, 0.05)
        minutos_activo = st.select_slider("Activo desde hace", options=list(range(0, 121, MINUTOS_BLOQUE)),
                                          value=0, format_func=lambda m: f"{m} min")
        df_bloque = inyectar_evento(df_bloque, afectados, tipo_evento, severidad, minutos_activo // MINUTOS_BLOQUE)

    st.divider()
    st.caption(f"Lectura de la red: día simulado {fecha:%d/%m/%Y} ({tipo_dia.replace('_', ' ')}), "
               f"bloque {hora_de_bloque(bloque)}.")

# --- Viaje ---
estaciones = sorted({data['nombre'] for _, data in G_base.nodes(data=True)})
c1, c2 = st.columns(2)
origen = c1.selectbox("Origen", estaciones, index=estaciones.index("Pantitlán"))
destino = c2.selectbox("Destino", estaciones, index=estaciones.index("Auditorio"))

if st.button("Calcular ruta", type="primary", width="stretch"):
    if origen == destino:
        st.warning("El origen y el destino son la misma estación.")
        st.stop()

    retraso = proyectar(df_bloque, paquete, horizonte)
    G_ia = grafo_proyectado(G_base, df_bloque, retraso)
    ruta_estatica = ruta_optima(G_base, origen, destino, 'tiempo_minutos')
    ruta_ia = ruta_optima(G_ia, origen, destino, 'weight')

    tiempo_ideal = tiempo_ruta(G_base, ruta_estatica, 'tiempo_minutos')
    tiempo_estatica = tiempo_ruta(G_ia, ruta_estatica, 'weight')
    tiempo_ia = tiempo_ruta(G_ia, ruta_ia, 'weight')
    ahorro = tiempo_estatica - tiempo_ia

    st.write("")
    if ruta_ia != ruta_estatica:
        st.success(f"La IA cambia la ruta y ahorra **{ahorro:.1f} min** frente a la ruta estática "
                   f"según la congestión proyectada a {horizonte} min.")
    else:
        st.info("La IA confirma la ruta estática: no hay una alternativa más rápida con la congestión proyectada.")

    col_estatica, col_ia = st.columns(2)
    with col_estatica:
        mostrar_ruta("Ruta estática", G_ia, ruta_estatica, 'weight', tiempo_estatica)
        st.caption(f"Elegida solo por tiempo ideal ({tiempo_ideal:.1f} min sin tráfico); "
                   f"los tiempos mostrados incluyen la congestión proyectada.")
    with col_ia:
        mostrar_ruta("Ruta IA anticipatoria", G_ia, ruta_ia, 'weight', tiempo_ia,
                     delta=f"{-ahorro:.1f} min" if ruta_ia != ruta_estatica else None)
        st.caption("Elegida con la congestión que la IA proyecta para cada tramo.")

    with st.expander("Tramos con mayor retraso proyectado en la red"):
        top = df_bloque.assign(linea=linea_tramo, retraso_min=retraso, evento=df_bloque['tipo_evento']) \
            .nlargest(10, 'retraso_min')[['linea', 'tramo', 'tiempo_ideal', 'retraso_min', 'evento']]
        st.dataframe(top, hide_index=True, width="stretch",
                     column_config={'linea': 'Línea', 'tramo': 'Tramo', 'evento': 'Evento',
                                    'tiempo_ideal': st.column_config.NumberColumn('Tiempo ideal (min)', format="%.1f"),
                                    'retraso_min': st.column_config.NumberColumn('Retraso proyectado (min)', format="%.2f")})
