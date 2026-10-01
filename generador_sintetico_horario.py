import re
import collections

import pandas as pd
import numpy as np
import networkx as nx

print("Cargando dataset de afluencia diaria...")
# Asegúrate de tener tu archivo original en la misma ruta
df_diario = pd.read_csv("afluenciastc_desglosado_01_2026.csv")

def limpiar_texto(texto):
    if isinstance(texto, str):
        try:
            return texto.encode("latin1").decode("utf-8")
        except:
            return texto
    return texto

# Nombres del CSV que no coinciden con los del grafo GTFS (grafo_base_metro.gexf). Sin
# este mapeo el simulador no encuentra el nodo de esas estaciones y descarta en silencio
# todos sus viajes (ver avances.md, Sección 15).
ALIAS_ESTACIONES = {
    "Chapultepec": "Chapultepec ",
    "Etiopía/Plaza de la Transparencia": "Etiopía y Plaza de la Transparencia",
    "Ferrería/Arena Ciudad de México": "Ferrería y Arena Ciudad de México",
    "Garibaldi/Lagunilla": "Garibaldi y Lagunilla",
    "Gómez Farias": "Gómez Farías",
    "La Villa/Basílica": "La Villa y Basílica",
    "Niños Héroes": "Niños Héroes y  Poder Judicial CDMX",
    "Peñón Viejo": "Penón Viejo",
    "Peñón viejo": "Penón Viejo",
    "UAM-Azcapotzalco": "UAM Azcapotzalco",
    "Viveros/Derechos Humanos": "Viveros y  Derechos Humanos",
    "Zócalo/Tenochtitlan": "Zócalo",
}

df_diario["estacion"] = df_diario["estacion"].apply(limpiar_texto).replace(ALIAS_ESTACIONES)
df_diario["linea"] = df_diario["linea"].apply(limpiar_texto)
df_historico = df_diario  # el histórico completo se usa para derivar los perfiles

# ====================================================================================
# PARAMETRIZACIÓN DE LOS DÍAS DE SIMULACIÓN (multi-día, laboral + fin de semana)
# ====================================================================================
# En vez de un único día fijo, generamos una ventana de 14 días naturales (2 semanas
# completas) tomada de enero 2026, el único mes con datos reales completos en el
# archivo fuente. Esto garantiza una mezcla real de días laborales y fines de semana
# (10 laborales + 4 de fin de semana) en lugar de un solo perfil fijo.
DIAS_SIMULACION = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-01-05", "2026-01-18")]

def tipo_de_dia(fecha_str):
    """Laboral (lunes-viernes) vs fin de semana (sábado-domingo)."""
    return "fin_de_semana" if pd.Timestamp(fecha_str).weekday() >= 5 else "laboral"

# Restringimos el pipeline horario a solo los días que efectivamente vamos a simular:
# el archivo fuente cubre 2021-2026 completos y desagregar TODO ese histórico a nivel
# hora (como hacía la versión de un solo día) generaba un archivo de ~300 MB sin uso
# real aguas abajo. Al filtrar aquí, el costo de cómputo/disco escala con los días de
# simulación pedidos, no con el histórico completo.
df_diario = df_historico[df_historico["fecha"].isin(DIAS_SIMULACION)].copy()

df_total_diario = (
    df_diario.groupby(["fecha", "linea", "estacion"])["afluencia"].sum().reset_index()
)
df_total_diario["tipo_dia"] = df_total_diario["fecha"].apply(tipo_de_dia)

print(f"Días de simulación: {DIAS_SIMULACION[0]} a {DIAS_SIMULACION[-1]} "
      f"({len(DIAS_SIMULACION)} días: "
      f"{sum(tipo_de_dia(d) == 'laboral' for d in DIAS_SIMULACION)} laborales, "
      f"{sum(tipo_de_dia(d) == 'fin_de_semana' for d in DIAS_SIMULACION)} fin de semana)")

print("Definiendo perfiles espaciotemporales de demanda...")

# 1. Perfil laboral (curva bimodal clásica de hora pico AM/PM) — igual al de la versión
#    original de un solo día.
perfil_laboral = pd.DataFrame(
    {
        "hora": range(24),
        "peso_origen": [0.002, 0.002, 0.002, 0.002, 0.002, 0.04, 0.15, 0.18, 0.12, 0.06, 0.04, 0.03, 0.03, 0.03, 0.04, 0.04, 0.05, 0.06, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005],
        "peso_destino": [0.002, 0.002, 0.002, 0.002, 0.002, 0.01, 0.02, 0.04, 0.05, 0.04, 0.04, 0.04, 0.05, 0.06, 0.06, 0.06, 0.08, 0.15, 0.14, 0.08, 0.04, 0.02, 0.01, 0.005],
        "peso_mixto": [0.002, 0.002, 0.002, 0.002, 0.002, 0.02, 0.09, 0.12, 0.09, 0.05, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.10, 0.09, 0.05, 0.03, 0.01, 0.005, 0.005],
    }
)
perfil_laboral["tipo_dia"] = "laboral"

def derivar_perfil_fin_semana(df_perfil):
    """
    Deriva el perfil de fin de semana a partir del laboral en lugar de inventar una
    segunda tabla de constantes: aplana la bimodalidad de hora pico (la gente no tiene
    un horario de entrada/salida laboral fijo) y desplaza la actividad hacia el
    mediodía/tarde (inicio de actividad más tardío, ocio concentrado entre 11:00 y
    19:00). El volumen total por columna se conserva (misma suma), solo cambia su
    distribución horaria.
    """
    df = df_perfil.drop(columns=["tipo_dia"]).copy()
    columnas_peso = ["peso_origen", "peso_destino", "peso_mixto"]
    for col in columnas_peso:
        total_original = df[col].sum()
        media = df[col].mean()
        # Aplanamos: mezcla 45% curva original / 55% promedio diario -> reduce picos.
        suavizado = 0.45 * df[col] + 0.55 * media
        # Redistribuimos hacia mediodía/tarde y reducimos actividad muy temprana.
        factor_hora = np.where(
            df["hora"].between(11, 19), 1.25,
            np.where(df["hora"] < 8, 0.5, 0.9),
        )
        ajustado = suavizado * factor_hora
        df[col] = ajustado / ajustado.sum() * total_original
    df["tipo_dia"] = "fin_de_semana"
    return df

perfil_fin_semana = derivar_perfil_fin_semana(perfil_laboral)
perfiles_horarios = pd.concat([perfil_laboral, perfil_fin_semana], ignore_index=True)

# ====================================================================================
# PERFILES DE ESTACIÓN DERIVADOS DE DATOS (limitación #5, ver avances.md, Sección 15)
# ====================================================================================
# El CSV de afluencia solo trae ENTRADAS DIARIAS por estación (sin hora ni salidas), así
# que no puede decir directamente si una estación es origen o destino. Cada estación se
# describe con dos índices continuos en [0, 1]:
# - c (traslado), del CSV: las estaciones de traslado al trabajo, en cualquiera de sus
#   dos extremos, pierden mucha más afluencia en domingo que las de ocio o turismo.
# - a (origen), del grafo GTFS: estaciones periféricas (tiempo medio de viaje alto) o
#   terminales de línea, donde llega el transporte alimentador (CETRAM), son origen de
#   los viajes de la mañana.
# y se reparte entre los tres perfiles: w_origen = c·a, w_destino = c·(1 − a),
# w_mixto = 1 − c. La forma de las curvas horarias de cada perfil sigue siendo un
# supuesto (no hay datos por hora); lo que sale de los datos es la mezcla de cada estación.
PERFILES = ["origen", "destino", "mixto"]
VENTANA_ESTADISTICAS = ("2025-01-01", "2025-12-31")
# Días laborales atípicos que no entran en la mediana laboral (vacaciones escolares).
VACACIONES = [("2025-01-01", "2025-01-06"), ("2025-04-14", "2025-04-18"), ("2025-12-22", "2025-12-31")]
# Un día con menos de esta fracción de la mediana de la estación se trata como cierre.
FRACCION_CIERRE = 0.2

def linea_de_nodo(nodo_id):
    """Misma convención que simulador_congestion.py: el ID GTFS del andén trae 'L<línea>-'."""
    m = re.search(r"L([0-9A-Za-z]+)[-_]", nodo_id)
    return m.group(1) if m else None

def indice_traslado(df_afluencia):
    """
    c por estación = 1 − rango percentil de r_dom, con r_dom = mediana de entradas en
    domingo / mediana en día laboral (sin vacaciones), en VENTANA_ESTADISTICAS. Se usan
    medianas y se descartan los días de cierre para que obras o suspensiones no muevan el
    índice.
    """
    df = df_afluencia[df_afluencia["fecha"].between(*VENTANA_ESTADISTICAS)]
    total = df.groupby(["fecha", "estacion"])["afluencia"].sum().reset_index()
    fecha = pd.to_datetime(total["fecha"])
    dia_semana = fecha.dt.dayofweek
    vacaciones = pd.Series(False, index=total.index)
    for inicio, fin in VACACIONES:
        vacaciones |= fecha.between(inicio, fin)
    cierre = total["afluencia"] < FRACCION_CIERRE * total.groupby("estacion")["afluencia"].transform("median")
    laboral = total[(dia_semana < 5) & ~vacaciones & ~cierre].groupby("estacion")["afluencia"].median()
    domingo = total[(dia_semana == 6) & ~cierre].groupby("estacion")["afluencia"].median()
    r_dom = (domingo / laboral).dropna()
    return pd.DataFrame({"r_dom": r_dom, "c": 1 - r_dom.rank(pct=True)})

def indice_origen(G):
    """
    a por estación = (rango percentil del tiempo medio de viaje desde la estación a todas
    las demás + 1 si es terminal de alguna línea) / 2. El tiempo es el peso estático
    'tiempo_minutos' del grafo; una estación con varios andenes toma el andén más cercano.
    """
    nombre = nx.get_node_attributes(G, "nombre")
    andenes = collections.defaultdict(list)
    for n in G:
        andenes[nombre[n]].append(n)
    distancias = dict(nx.all_pairs_dijkstra_path_length(G, weight="tiempo_minutos"))
    tiempo_medio = {}
    for estacion, nodos in andenes.items():
        tiempos = [min(distancias[a].get(b, np.inf) for a in nodos for b in otros)
                   for otra, otros in andenes.items() if otra != estacion]
        tiempo_medio[estacion] = np.mean(tiempos)
    # Terminal: algún andén con un solo tramo de su propia línea.
    grado_linea = collections.Counter()
    for u, v in G.edges():
        if linea_de_nodo(u) is not None and linea_de_nodo(u) == linea_de_nodo(v):
            grado_linea[u] += 1
            grado_linea[v] += 1
    terminal = {estacion: int(any(grado_linea[n] == 1 for n in nodos)) for estacion, nodos in andenes.items()}
    df = pd.DataFrame({"tiempo_medio_min": tiempo_medio, "terminal": terminal})
    df["periferia"] = df["tiempo_medio_min"].rank(pct=True)
    df["a"] = (df["periferia"] + df["terminal"]) / 2
    return df

def derivar_perfiles_estaciones(df_afluencia, G):
    """
    Pesos de origen/destino/mixto de cada estación del CSV y su perfil discreto (el de
    mayor peso, solo para reportar). Una estación sin estadística en la ventana (c) o sin
    nodo en el grafo (a) recibe el valor neutro 0.5 en ese índice.
    """
    estaciones = pd.Index(sorted(df_afluencia["estacion"].unique()), name="estacion")
    df = pd.DataFrame(index=estaciones).join(indice_traslado(df_afluencia)).join(indice_origen(G))
    for indice in ("c", "a"):
        faltantes = df.index[df[indice].isna()].tolist()
        if faltantes:
            print(f"   AVISO: sin índice {indice} para {faltantes}; se usa 0.5")
        df[indice] = df[indice].fillna(0.5)
    df["w_origen"] = df["c"] * df["a"]
    df["w_destino"] = df["c"] * (1 - df["a"])
    df["w_mixto"] = 1 - df["c"]
    df["perfil"] = df[[f"w_{p}" for p in PERFILES]].idxmax(axis=1).str.removeprefix("w_")
    return df.reset_index()

G_base = nx.read_gexf("grafo_base_metro.gexf")
df_perfiles = derivar_perfiles_estaciones(df_historico, G_base)
df_perfiles.to_csv("datos_procesados/perfiles_estaciones.csv", index=False, encoding="utf-8-sig")
print(f"Perfiles derivados para {len(df_perfiles)} estaciones: "
      f"{df_perfiles['perfil'].value_counts().to_dict()} (datos_procesados/perfiles_estaciones.csv)")

df_total_diario = df_total_diario.merge(df_perfiles[["estacion", "perfil", "w_origen", "w_destino", "w_mixto"]],
                                        on="estacion", how="left")

print("Cruzando datos y calculando afluencia de ENTRADA por hora...")

# Unimos por tipo_dia (no por producto cartesiano global): cada fecha solo se cruza
# con las 24 horas de SU perfil (laboral o fin de semana).
df_horario = pd.merge(df_total_diario, perfiles_horarios, on="tipo_dia")

# Curva horaria propia de cada estación: mezcla de las curvas de los tres perfiles con
# los pesos derivados de los datos.
curva_estacion = sum(df_horario[f"w_{p}"] * df_horario[f"peso_{p}"] for p in PERFILES)
df_horario["afluencia_sintetica_hora"] = (df_horario["afluencia"] * curva_estacion).astype(int)
df_horario = df_horario.drop(columns=["afluencia"] + [f"peso_{p}" for p in PERFILES]).sort_values(by=["fecha", "linea", "estacion", "hora"])
df_horario = df_horario[df_horario["hora"] >= 5]

# Guardamos el archivo base (entradas), ya multi-día.
df_horario.to_csv("datos_procesados/entradas_sinteticas_horarias.csv", index=False, encoding="utf-8-sig")

# ====================================================================================
# PUNTO 1 DE LA TESIS - CONSTRUCCIÓN DEL AMBIENTE (MATRIZ ORIGEN-DESTINO), POR DÍA
# ====================================================================================
print("\n--- Iniciando Generación de Matrices Origen-Destino (Modelo Gravitacional) ---")

# Peso de atracción de cada perfil como destino, por periodo del día (supuesto): en la
# mañana atraen las zonas laborales ('destino'), en la tarde la gente regresa a casa
# ('origen') y en horas valle el flujo es más equilibrado.
ATRACTIVIDAD_POR_PERIODO = {
    "manana": {"origen": 0.10, "destino": 0.65, "mixto": 0.25},
    "tarde": {"origen": 0.65, "destino": 0.10, "mixto": 0.25},
    "valle": {"origen": 0.25, "destino": 0.25, "mixto": 0.50},
}

def periodo_del_dia(hora):
    if 5 <= hora <= 11:
        return "manana"
    if 16 <= hora <= 21:
        return "tarde"
    return "valle"

def calcular_atractividad_destino(df):
    """Peso de cada fila (hora, estación) como destino: la tabla del periodo mezclada con
    los pesos de perfil de la estación."""
    tabla = pd.DataFrame(ATRACTIVIDAD_POR_PERIODO).T
    por_periodo = tabla.loc[df["hora"].map(periodo_del_dia)].reset_index(drop=True)
    return sum(df[f"w_{p}"].to_numpy() * por_periodo[p].to_numpy() for p in PERFILES)

def generar_matriz_od(df_horario, dia_simulacion):
    """Construye la matriz O-D sintética de un único día a partir del dataframe
    de entradas horarias (ya multi-día) y la guarda en disco."""
    df_dia = df_horario[df_horario['fecha'] == dia_simulacion].copy()
    if df_dia.empty:
        raise ValueError(f"No hay datos de afluencia para el día {dia_simulacion}.")

    df_dia['peso_atractividad'] = calcular_atractividad_destino(df_dia)

    df_origen = df_dia[['hora', 'estacion', 'perfil', 'afluencia_sintetica_hora']].rename(
        columns={'estacion': 'origen', 'perfil': 'perfil_origen', 'afluencia_sintetica_hora': 'entradas'}
    )
    df_destino = df_dia[['hora', 'estacion', 'perfil', 'peso_atractividad']].rename(
        columns={'estacion': 'destino', 'perfil': 'perfil_destino', 'peso_atractividad': 'atractividad_destino'}
    )

    df_od = pd.merge(df_origen, df_destino, on='hora')
    df_od = df_od[df_od['origen'] != df_od['destino']]

    suma_atractividad = df_od.groupby(['hora', 'origen'])['atractividad_destino'].sum().reset_index()
    suma_atractividad.rename(columns={'atractividad_destino': 'atractividad_total'}, inplace=True)
    df_od = pd.merge(df_od, suma_atractividad, on=['hora', 'origen'])

    df_od['probabilidad'] = df_od['atractividad_destino'] / df_od['atractividad_total']
    df_od['pasajeros_viaje'] = (df_od['entradas'] * df_od['probabilidad']).round().astype(int)

    df_od_final = df_od[['hora', 'origen', 'destino', 'perfil_origen', 'perfil_destino', 'pasajeros_viaje']]
    df_od_final = df_od_final[df_od_final['pasajeros_viaje'] > 0]

    archivo_od = f"datos_procesados/matriz_od_sintetica_{dia_simulacion}.csv"
    df_od_final.to_csv(archivo_od, index=False, encoding="utf-8-sig")
    return archivo_od, len(df_od_final)

manifiesto = []
for dia in DIAS_SIMULACION:
    print(f"Generando matriz O-D para {dia} ({tipo_de_dia(dia)})...")
    archivo_od, n_filas = generar_matriz_od(df_horario, dia)
    print(f"   -> {archivo_od} ({n_filas} viajes origen-destino)")
    manifiesto.append({"fecha": dia, "tipo_dia": tipo_de_dia(dia), "archivo_od": archivo_od})

# El simulador (simulador_congestion.py) lee este manifiesto para saber qué días y
# de qué tipo simular, sin tener que duplicar la lista DIAS_SIMULACION en dos scripts.
df_manifiesto = pd.DataFrame(manifiesto)
df_manifiesto.to_csv("datos_procesados/manifiesto_dias_simulados.csv", index=False, encoding="utf-8-sig")
print(f"\n¡Manifiesto de días simulados guardado en: datos_procesados/manifiesto_dias_simulados.csv!")

print("\n--- Muestra del comportamiento simulado (último día generado) ---")
df_muestra = pd.read_csv(manifiesto[-1]["archivo_od"])
viajes_manana = df_muestra[(df_muestra['hora'] == 7) & (df_muestra['origen'] == 'Pantitlán')].sort_values(by='pasajeros_viaje', ascending=False)
print(f"Top 3 destinos desde Pantitlán a las 7:00 AM ({manifiesto[-1]['fecha']}):")
print(viajes_manana.head(3)[['destino', 'perfil_destino', 'pasajeros_viaje']])

viajes_tarde = df_muestra[(df_muestra['hora'] == 18) & (df_muestra['origen'] == 'Polanco')].sort_values(by='pasajeros_viaje', ascending=False)
print(f"\nTop 3 destinos desde Polanco a las 6:00 PM ({manifiesto[-1]['fecha']}):")
print(viajes_tarde.head(3)[['destino', 'perfil_destino', 'pasajeros_viaje']])
