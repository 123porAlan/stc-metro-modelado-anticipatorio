import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuración visual para las gráficas de la tesis
sns.set_theme(style="whitegrid")

def cargar_y_preparar_datos(ruta_archivo):
    """Carga el dataset tabular y asegura el orden cronológico."""
    print(f"1. Cargando dataset desde: {ruta_archivo}...")
    df = pd.read_csv(ruta_archivo)
    
    # ORDEN CRONOLÓGICO: Vital para series temporales. 
    # Aseguramos que el modelo aprenda del pasado para predecir el futuro.
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(by=['fecha', 'hora']).reset_index(drop=True)
    
    return df

def preparar_features_evento(df):
    """
    Codifica el contexto de eventos estocásticos (hay_evento, tipo_evento,
    severidad_evento) como columnas numéricas: hay_evento y severidad_evento ya son
    numéricas; tipo_evento se expande a dummies porque RandomForestRegressor no acepta
    texto directamente.
    """
    dummies_evento = pd.get_dummies(df['tipo_evento'], prefix='evento')
    df = pd.concat([df, dummies_evento], axis=1)
    return df, list(dummies_evento.columns)

def entrenar_modelo(df):
    """Divide los datos, entrena el RandomForest y evalúa su desempeño."""
    print("\n2. Preparando variables y separando Train/Test...")

    tiene_contexto_evento = 'hay_evento' in df.columns
    if tiene_contexto_evento:
        df, columnas_evento = preparar_features_evento(df)

    target = 'target_congestibilidad_t_plus_1'
    features = [
        'hora',
        'tiempo_ideal',
        'congestibilidad_t',
        'congestibilidad_t_minus_1'
    ]
    if tiene_contexto_evento:
        features += ['hay_evento', 'severidad_evento'] + columnas_evento

    # Separación Temporal (80% Entrenamiento, 20% Prueba)
    # No usamos train_test_split aleatorio para evitar "Data Leakage" (hacer trampa viendo el futuro)
    split_idx = int(len(df) * 0.8)
    
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    X_train, y_train = df_train[features], df_train[target]
    X_test, y_test = df_test[features], df_test[target]
    
    print(f"   - Datos de Entrenamiento (Pasado): {len(X_train)} registros")
    print(f"   - Datos de Prueba (Futuro): {len(X_test)} registros")

    print("\n3. Entrenando el Modelo de Inteligencia Anticipatoria...")
    # Usamos RandomForestRegressor por su capacidad con no-linealidades y su explicabilidad
    modelo = RandomForestRegressor(
        n_estimators=150,      
        max_depth=10,          
        min_samples_split=5,
        random_state=42,
        n_jobs=-1              
    )
    
    modelo.fit(X_train, y_train)
    print("   [OK] Entrenamiento completado.")

    print("\n4. Evaluando KPIs Predictivos...")
    y_pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("--- RESULTADOS DEL MODELO ---")
    print(f"MAE  (Error Absoluto Medio): {mae:.3f} minutos.")
    print(f"RMSE (Raíz Error Cuadrático): {rmse:.3f} minutos.")
    print("-----------------------------")
    
    return modelo, features

def graficar_explicabilidad(modelo, features):
    """Genera la gráfica de importancia de características para la tesis."""
    print("\n5. Generando análisis de transparencia algorítmica...")
    importancias = modelo.feature_importances_
    df_imp = pd.DataFrame({
        'Característica': features,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importancia', y='Característica', data=df_imp, palette='viridis')
    
    plt.title('Explicabilidad: Peso Predictivo de las Variables (IA Anticipatoria)', fontsize=14)
    plt.xlabel('Importancia Relativa', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.tight_layout()
    
    ruta_grafica = "importancia_variables.png"
    plt.savefig(ruta_grafica)
    print(f"   [OK] Gráfica guardada como: '{ruta_grafica}' (Lista para tu tesis).")
    # plt.show() # Descomenta esto si quieres que la gráfica se abra en una ventana

def guardar_modelo(modelo, ruta_salida="modelos/modelo_anticipatorio_rf.pkl"):
    """Exporta el modelo para usarlo en el algoritmo de búsqueda de rutas."""
    print("\n6. Exportando modelo...")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    joblib.dump(modelo, ruta_salida)
    print(f"   [OK] Modelo exportado en: {ruta_salida}")

if __name__ == "__main__":
    ARCHIVO_DATASET = "datos_procesados/dataset_features_entrenamiento.csv"
    
    try:
        # 1. Cargar datos
        df_features = cargar_y_preparar_datos(ARCHIVO_DATASET)
        
        # 2 y 3. Entrenar y Evaluar
        modelo_entrenado, lista_features = entrenar_modelo(df_features)
        
        # 4. Generar gráfica de Explicabilidad (Objetivo de la tesis)
        graficar_explicabilidad(modelo_entrenado, lista_features)
        
        # 5. Guardar modelo
        guardar_modelo(modelo_entrenado)
        
        print("\n🚀 ¡Fase de Inteligencia Anticipatoria completada con éxito!")
        
    except Exception as e:
        print(f"\n❌ Error en la ejecución: {e}")