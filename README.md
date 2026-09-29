# STC Metro Anticipatorio 🚇

**Modelado y prototipado de un sistema de Inteligencia Artificial Anticipatoria para la estimación de estados de congestión en el STC Metro de la Ciudad de México**.

Este repositorio contiene los scripts de procesamiento de datos, modelado topológico y algoritmos de simulación desarrollados para la construcción de un sistema de Inteligencia Artificial Anticipatoria. El objetivo principal es prever estados de saturación en la red del Metro en escenarios de horizonte corto (10 - 60 minutos) mediante el uso de datos sintéticos y graficas ponderados.

## 📌 Estado Actual del Proyecto

Actualmente, el proyecto ha completado la fase de **Construcción de la Base Empírica e Infraestructura de Datos**. Se han desarrollado los motores que transforman la información estática gubernamental en un entorno dinámico listo para el entrenamiento de la IA.

### 1. Modelado Topológico y Pesos Estáticos

* **Script:** `grafo_metro.py`
* **Descripción:** Aísla la topología exclusiva del STC Metro a partir de archivos GTFS masivos del Valle de México. Construye un grafica matemático $G = (V, E, W)$ que conecta cronológicamente los 195 nodos (estaciones).
* **Logro:** Calcula el "peso estático" de cada arista (el tiempo de viaje ideal en minutos/segundos) cruzando los tiempos de llegada y salida históricos.

### 2. Motor Sintético de Demanda (Afluencia Horaria)

* **Script:** `generador_sintetico_horario.py`
* **Descripción:** Expande el consolidado de afluencia diaria hacia una granularidad de alta frecuencia (por hora). Implementa un motor de distribución bimodal que clasifica las estaciones en tres perfiles (`origen`, `destino` y `mixto`) para simular la asimetría real de las horas pico matutinas y vespertinas.
* **Logro:** Genera el dataset maestro limpio y sin errores de codificación (`afluencia_sintetica_horaria_avanzada_2026.csv`), dotando al sistema de la dimensión temporal necesaria para simular estrés.

### 3. Exploración Histórica

* **Scripts:** `exploracion_gtfs.py`, `exploracion_afluencia.py`, `comparacion_lineas.py`
* **Descripción:** Herramientas de validación de datos para la identificación de anomalías y disrupciones pasadas.

### 4. Simulación de Congestión Dinámica (Motor de Estrés)

* **Script:** `simulador_congestion.py`
* **Descripción:** Representa el puente matemático entre la infraestructura física y el flujo de pasajeros. Transforma la red estática en un entorno dinámico utilizando una adaptación de la Función de Retraso de la BPR (Bureau of Public Roads) para inyectar realismo físico a la grafica.

**Funcionalidad Detallada y Metodología Matemática:**

* **Penalización por Saturación (Función BPR):** El script implementa la relación Volumen/Capacidad (V/C). No suma minutos de forma lineal, sino que aplica una penalización exponencial a los pesos de las aristas cuando la afluencia en una estación se aproxima a su límite técnico. La penalización se calcula con la siguiente formulación:

  $$T_c = T_b \times \left(1 + \alpha \left(\frac{V}{C}\right)^\beta \right)$$

  Donde:
  * **$T_c$**: Tiempo con congestión (peso dinámico de la arista).
  * **$T_b$**: Tiempo base o ideal en minutos.
  * **$V$**: Volumen actual de pasajeros inyectado por el dataset sintético.
  * **$C$**: Capacidad teórica calibrada del andén/estación.
  * **$\alpha$ y $\beta$**: Parámetros de sensibilidad de la curva ($\alpha = 0.15$, $\beta = 4$).

* **Calibración y Fricción de Andén:** La capacidad teórica se calibró para representar el umbral crítico donde comienza la **fricción de andén** (retrasos operativos causados por el exceso de usuarios, dificultad en el cierre de puertas y aumento en los tiempos de intercambio). En tramos críticos durante la hora pico, esto inyecta un incremento realista del ~40% en tiempos de traslado inmediatos.
* **Generación de Graficas Temporales:** Transforma la grafica estático en una serie de "fotografías" de la red cada 15 minutos. La demanda horaria se reparte en bloques de 15 min por interpolación entre horas vecinas y la capacidad de los trenes se divide proporcionalmente. Esto permite que el sistema registre cómo el tiempo de viaje en un mismo tramo (ej. Pantitlán-Zaragoza) fluctúa dinámicamente a lo largo del día.
* **Logro:** Provee el entorno de simulación necesario para generar los *labels* (datos etiquetados) de entrenamiento. Con esto, el sistema ahora puede comparar estados "ideales" vs "congestivos", permitiendo que la IA aprenda a pronosticar saturaciones en horizontes de 10 a 60 minutos.

**Ejecución:**
Para correr la simulación y generar los estados dinámicos de la red:
```bash
python simulador_congestion.py
```

***

## Estado de la Simulación y Construcción del Ambiente

Actualmente, el entorno de simulación central está construido y funcional. Se ha completado la integración entre el flujo externo de usuarios y la topología interna de la red del metro:

* **Construcción del Dataset de Ambiente (Matriz O-D):** Se implementó un modelo de gravedad que distribuye probabilísticamente los viajes de acuerdo al perfil espaciotemporal de cada estación (origen, destino o mixto). Esto genera una matriz sintética de Origen-Destino que emula con alta fidelidad las entradas y salidas reales de pasajeros a través de los torniquetes a lo largo del día.
* **Snapshots Dinámicos de Congestión:** El simulador fue modificado para realizar enrutamiento de pasajeros sobre la grafica topológica (`grafo_metro.py`). Utilizando una función de penalización adaptada (BPR), el sistema calcula el impacto de la carga de pasajeros en cada tramo de vía y toma *snapshots* (fotografías de estado) cada 15 minutos. Esto genera nuestro dataset tabular de entrenamiento, el cual contiene la memoria temporal de la red ($t-4, \dots, t$) y los objetivos a 15, 30, 45 y 60 minutos ($t+1, \dots, t+4$).

### Próximos Pasos (Fase Anticipatoria)

El proyecto se encuentra listo para iniciar la fase núcleo de la Inteligencia Artificial Anticipatoria (Paso 3):

* **Bucle de Aprendizaje por Fracaso:** Se implementará un ciclo interactivo donde el modelo utilice sus propias predicciones para enrutar a los pasajeros, enfrentándose a las consecuencias de sus decisiones (colapsos y fallos en la red). El modelo iterará sobre estos errores, reevaluándose continuamente hasta converger en una estrategia predictiva que logre equilibrar las cargas de manera dinámica, cumpliendo con los principios de los sistemas anticipatorios.

## 🛠️ Tecnologías y Dependencias

* `Python 3.x`
* `Pandas` (Procesamiento vectorial de alta eficiencia y manejo de DataFrames)
* `NetworkX` (Construcción y análisis de la topología de red)
* `Matplotlib` (Visualización del esqueleto de la grafica)
* NumPy (Cálculo numérico avanzado para funciones de penalización)

---

*Autor:* Alan Bellon García


