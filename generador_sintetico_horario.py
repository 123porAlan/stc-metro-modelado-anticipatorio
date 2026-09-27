import pandas as pd
import numpy as np

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

df_diario["estacion"] = df_diario["estacion"].apply(limpiar_texto)
df_diario["linea"] = df_diario["linea"].apply(limpiar_texto)

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
df_diario = df_diario[df_diario["fecha"].isin(DIAS_SIMULACION)].copy()

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

estaciones_origen = ["Pantitlán", "Indios Verdes", "Ciudad Azteca", "Tláhuac", "La Paz", "El Rosario", "Martín Carrera", "Tasqueña", "Universidad", "Constitución de 1917"]
estaciones_destino = ["Polanco", "Auditorio", "Insurgentes", "Chilpancingo", "Sevilla", "Zócalo/Tenochtitlan", "Bellas Artes", "Juárez", "Coyoacán", "Zapata"]

def asignar_perfil(estacion):
    if estacion in estaciones_origen: return "origen"
    elif estacion in estaciones_destino: return "destino"
    else: return "mixto"

df_total_diario["perfil"] = df_total_diario["estacion"].apply(asignar_perfil)

print("Cruzando datos y calculando afluencia de ENTRADA por hora...")

# Unimos por tipo_dia (no por producto cartesiano global): cada fecha solo se cruza
# con las 24 horas de SU perfil (laboral o fin de semana).
df_horario = pd.merge(df_total_diario, perfiles_horarios, on="tipo_dia")

condiciones = [
    df_horario["perfil"] == "origen",
    df_horario["perfil"] == "destino",
    df_horario["perfil"] == "mixto",
]
elecciones = [
    df_horario["afluencia"] * df_horario["peso_origen"],
    df_horario["afluencia"] * df_horario["peso_destino"],
    df_horario["afluencia"] * df_horario["peso_mixto"],
]

df_horario["afluencia_sintetica_hora"] = np.select(condiciones, elecciones, default=0).astype(int)
df_horario = df_horario.drop(columns=["afluencia", "peso_origen", "peso_destino", "peso_mixto"]).sort_values(by=["fecha", "linea", "estacion", "hora"])
df_horario = df_horario[df_horario["hora"] >= 5]

# Guardamos el archivo base (entradas), ya multi-día.
df_horario.to_csv("datos_procesados/entradas_sinteticas_horarias.csv", index=False, encoding="utf-8-sig")

# ====================================================================================
# PUNTO 1 DE LA TESIS - CONSTRUCCIÓN DEL AMBIENTE (MATRIZ ORIGEN-DESTINO), POR DÍA
# ====================================================================================
print("\n--- Iniciando Generación de Matrices Origen-Destino (Modelo Gravitacional) ---")

def calcular_atractividad_destino(hora, perfil):
    """
    Asigna un peso de probabilidad para que una estación sea elegida como destino.
    """
    if 5 <= hora <= 11:
        # En la mañana, las zonas laborales ('destino') atraen a la mayoría.
        if perfil == 'destino': return 0.65
        elif perfil == 'mixto': return 0.25
        else: return 0.10
    elif 16 <= hora <= 21:
        # En la tarde, la gente regresa a casa ('origen').
        if perfil == 'origen': return 0.65
        elif perfil == 'mixto': return 0.25
        else: return 0.10
    else:
        # En horas valle, el flujo es más equilibrado.
        if perfil == 'mixto': return 0.50
        else: return 0.25

def generar_matriz_od(df_horario, dia_simulacion):
    """Construye la matriz O-D sintética de un único día a partir del dataframe
    de entradas horarias (ya multi-día) y la guarda en disco."""
    df_dia = df_horario[df_horario['fecha'] == dia_simulacion].copy()
    if df_dia.empty:
        raise ValueError(f"No hay datos de afluencia para el día {dia_simulacion}.")

    df_dia['peso_atractividad'] = df_dia.apply(
        lambda row: calcular_atractividad_destino(row['hora'], row['perfil']), axis=1
    )

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
