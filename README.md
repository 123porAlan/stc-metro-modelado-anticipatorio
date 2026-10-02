# Avances del Proyecto — Sistema de IA Anticipatoria para el STC Metro

**Alumno:** Alan Bellon García\
**Asesor:** M. en Fil. C. Enrique Francisco Soto Astorga\
**Fecha de este reporte:** 2026-10-01 (actualizado — ver [Sección 4](#4-extensión-multi-día--eventos-estocásticos-2026-09-27) , [Sección 5](#5-ubicación-en-la-red-comparación-de-modelos-y-evaluación-sistemática-del-ruteo-2026-09-27) , [Sección 6](#6-capacidad-por-línea-suspensión-de-tramos-tasas-calibradas-y-experimento-multi-semilla-2026-09-27) , [Sección 7](#7-selección-fija-de-modelo-30-semillas-y-edad-del-evento-2026-09-27) , [Sección 8](#8-aviso-de-restablecimiento-y-histgb-en-el-ruteo-2026-09-27) , [Sección 9](#9-aviso-con-error-aditivo-2026-09-27) , [Sección 10](#10-ruteo-por-edad-del-evento-hora-0--1-y--2-2026-09-27) , [Sección 11](#11-sensibilidad-del-aviso-reactivo-con-regla-de-duración-y-hora--1-por-tipo-2026-09-27) , [Sección 12](#12-sistema-híbrido-y-sensibilidad-del-umbral-de-la-regla-2026-09-27) , [Sección 13](#13-horizonte-sub-horario-bloques-de-15-minutos-2026-09-29) , [Sección 14](#14-retraso-por-suspensión-sin-filtración-de-la-duración-2026-09-30) y [Sección 15](#15-perfiles-de-estación-derivados-de-datos-2026-09-30))

Este documento resume el estado técnico y metodológico del prototipo descrito en el
anexo de titulación (*"Modelado y prototipado de un sistema de Inteligencia Artificial
Anticipatoria para la estimación de estados de congestión en el STC Metro de la Ciudad
de México"*), con base en el código y los datos actualmente presentes en
`00_Programas/tesis_metro_ai/`.

---

## 1. Mapeo contra los objetivos de la tesis

| Objetivo secundario (Anexo) | Estado | Evidencia en código |
|---|---|---|
| Generar dataset sintético de afluencia/disrupciones | ✅ Completo (14 días sintéticos, laboral+fin de semana, eventos estocásticos con tasas calibradas, capacidad por línea) | `generador_sintetico_horario.py`, `simulador_congestion.py`, `datos_procesados/*.csv` |
| Representar la red como grafo con pesos | ✅ Completo | `grafo_metro.py` → `grafo_base_metro.gexf` |
| Modelo de estimación a horizonte corto (10–60 min) | ✅ Bloques de 15 min con un modelo por horizonte (15, 30, 45 y 60 min) (Secciones 13 y 14); comparación de 6 configuraciones; modelo fijo RandomForest + línea/tramo porque la selección por semilla es inestable (Secciones 6.4 y 7.1); HistGB rutea peor (Sección 8.2) | `entrenador_anticipatorio.py` → `modelos/modelo_anticipatorio.pkl` |
| Algoritmo de ruteo que integre la métrica predictiva | ✅ Prueba de concepto funcional | `ruteo_anticipatorio.py` |
| Integración estimación + ruteo en prototipo funcional | ✅ Evaluado en bloques de 15 min con 7 variantes × 30 semillas (1,164 bloques con evento y 3,071,024 casos O-D por horizonte; Sección 15.7). Sin aviso, la IA captura 86.4 / 75.0 / 62.5 / 44.2% del ahorro posible a 15 / 30 / 45 / 60 min, contra 86.6 / 68.8 / 42.2 / 0.0% del reactivo: lo supera con P ≥ 99.8% a 45 y 60 min, empata a 15 min (P = 46%) y a 30 min solo lo supera con aviso. Con aviso perfecto sube a 88.2 / 79.9 / 71.3 / 62.7% (+6,238 a +33,696 min frente a sin aviso, P ≥ 99.8%). Resultados con horizonte de 1 hora: Secciones 7–12 | `ruteo_anticipatorio.py` |
| Explicabilidad de las recomendaciones | ✅ Iniciado (importancia por permutación, agnóstica al modelo) | `entrenador_anticipatorio.py` → `importancia_variables.png` |
| Sistema reactivo de comparación (índice 5.5) | ✅ Implementado y comparado: estático vs. reactivo vs. anticipatorio vs. oráculo | `ruteo_anticipatorio.py` |

En términos del índice tentativo, el proyecto ha cubierto el **capítulo 4** completo
(modelado y generación de datos) y tiene un **primer corte funcional del capítulo 5**
(diseño del prototipo: arquitectura, modelo predictivo, ruteo, baseline reactivo). Falta
evaluación formal de KPIs (5.4), profundizar explicabilidad (6.5) y ejecutar comparativas
sistemáticas (6.4) sobre múltiples escenarios, no solo el caso de prueba único.

---

## 2. Pipeline técnico actual (orden de ejecución)

```
datos_crudos/ (GTFS)                afluenciastc_desglosado_01_2026.csv
        │                                        │
        ▼                                        ▼
 grafo_metro.py                     generador_sintetico_horario.py
        │                                        │
        ▼                                        ▼
grafo_base_metro.gexf         entradas_sinteticas_horarias.csv
   (195 nodos, aristas                       │
   con peso estático +                       ▼
   transbordos 5 min)          matriz_od_sintetica_2026-01-13.csv
        │                                        │
        └───────────────┬────────────────────────┘
                         ▼
              simulador_congestion.py
        (enrutamiento por hora + función BPR
         + eventos estocásticos, 14 días)
                         │
                         ▼
      dataset_features_entrenamiento.csv (52,360 filas)
                         │
                         ▼
            entrenador_anticipatorio.py
   (compara RF / GradientBoosting / HistGB,
    con y sin línea/tramo; split 80/20 +
    validación por días; exporta modelo fijo
    RF + línea/tramo; importancia por
    permutación; .pkl)
                         │
                         ▼
             ruteo_anticipatorio.py
   (proyecta pesos futuros del grafo; compara
    estático / reactivo / IA / oráculo contra
    el tráfico real de t+1 en horas con evento)
```

### 2.1 `grafo_metro.py` — Modelado topológico
- Filtra GTFS masivo del Valle de México por `agency_id == "METRO"` para aislar rutas,
  viajes y `stop_times`.
- Construye grafo no dirigido `G(V,E,W)` con **195 nodos** (andenes/estaciones).
- Peso estático de cada arista = promedio de `departure_sec → arrival_sec` entre
  paradas consecutivas del mismo `trip_id`, filtrando deltas ilógicos (`<0` o `>1800s`).
- Agrega aristas de **transbordo peatonal** (5 min fijos) entre andenes que comparten
  `stop_name`, habilitando cambios de línea en el grafo.
- Exporta a `grafo_base_metro.gexf` (formato estándar, reutilizable por NetworkX).

**Limitación notable:** el penalti de transbordo es una constante (5 min) igual para
todas las estaciones de correspondencia, sin distinguir estaciones con transbordos
largos conocidos (p. ej. Ciudad Azteca, La Raza, Tacubaya).

### 2.2 `generador_sintetico_horario.py` — Motor de demanda
- Limpia encoding (latin1→utf-8) de estación/línea provenientes del portal de datos
  abiertos.
- Clasifica estaciones en 3 perfiles espaciotemporales (`origen`, `destino`, `mixto`)
  mediante listas fijas (hardcoded) de 10 estaciones por extremo.
- Desagrega afluencia diaria a horaria multiplicando por curvas bimodales
  (`peso_origen`, `peso_destino`, `peso_mixto`) calibradas manualmente para 24 horas.
- **Matriz Origen-Destino sintética** vía modelo gravitacional simplificado: calcula
  `atractividad_destino` según hora y perfil, normaliza por estación de origen y reparte
  las entradas reales observadas como viajes probables a cada destino.
- **[Actualizado 2026-09-27]** Parametrizado por `DIAS_SIMULACION` (lista de fechas, hoy
  14 días de enero 2026). Cada fecha se etiqueta `laboral`/`fin_de_semana` según su día
  de la semana; el fin de semana usa un perfil horario derivado programáticamente del
  laboral (aplanado de picos + desplazamiento de actividad a mediodía/tarde), no una
  segunda tabla de constantes inventada. Genera una `matriz_od_sintetica_<fecha>.csv`
  por día y un `manifiesto_dias_simulados.csv` (fecha, tipo_dia, archivo) que
  `simulador_congestion.py` consume para no duplicar la lista de días entre scripts.

### 2.3 `simulador_congestion.py` — Motor de estrés dinámico
- Carga el grafo base y la matriz O-D del día simulado.
- Por cada hora de operación (5–23h), copia el grafo, enruta cada flujo O-D con
  `nx.shortest_path` (ponderado por `tiempo_minutos`) y acumula `carga_pasajeros`
  por arista.
- Aplica **función BPR adaptada** para convertir carga en tiempo congestionado:

  T_c = T_b × (1 + α·(V/C)^β), α=0.15, β=4, tope en 4×T_b

  **[Actualizado 2026-09-27, Sección 6]** con capacidad **por sentido y por línea**
  (trenes/hora × carros × 170 pasajeros; de 12,780 pas/h en L12 a 45,699 en L1) y V/C
  calculada con la carga del sentido más cargado. Antes: 35,000 pas/h para toda la red
  contra la carga de ambos sentidos sumada.
- **[Actualizado 2026-09-27] Generador de eventos estocásticos** (reemplaza la
  inyección determinista de Línea 9): por cada (hora, línea, tipo de evento) del día se
  sortea ocurrencia de fallas mecánicas e incidentes de plataforma con un proceso de
  Poisson por (hora, línea) modulado por hora pico, y si ocurre, severidad (Beta(2,5),
  sesgada a eventos leves), duración (1-4 horas) y tramo afectado. **[Sección 6]** Las
  tasas están calibradas con fuentes públicas (desalojos del STC; días con lluvia del
  SMN). La lluvia es un episodio por día lluvioso que alcanza a las líneas de superficie
  (A/B/12) de forma correlacionada. La lluvia y la falla mecánica actúan como carga
  fantasma sobre la BPR; el incidente de plataforma **suspende** el tramo (retraso medio
  de 30·s² min). Nada de esto se escribe en `carga_pasajeros_red`, que queda como
  ridership real limpio, y el evento se registra en el dataset vía `hay_evento`/
  `tipo_evento`/`severidad_evento`. Acepta `--semilla` y `--salida`.
- El ruteo ya no recalcula `nx.shortest_path` por cada fila de la matriz O-D y por cada
  hora: como el peso de ruteo (`tiempo_minutos`) es estático, la ruta de cada par
  origen-destino se calcula una sola vez y se cachea (necesario para que escalar a 14
  días fuera viable en tiempo razonable). Con el recorrido de la matriz O-D por `zip` en
  vez de `iterrows`, los 14 días corren en ~1.5 min.
- Construye ventanas temporales (t-1, t, t+1) de `congestibilidad` por tramo (sin fugar
  información entre días) y exporta `dataset_features_entrenamiento.csv`, ahora con
  **52,360 registros** de 14 días (antes 3,740 de un solo día).

### 2.4 `entrenador_anticipatorio.py` — Modelo predictivo
- Ordena cronológicamente y separa train/test 80/20 **sin aleatorizar** (evita fuga de
  información temporal).
- Features: `hora`, `tiempo_ideal`, `congestibilidad_t`, `congestibilidad_t_minus_1`,
  `hay_evento`, `severidad_evento` y dummies one-hot de `tipo_evento` (los modelos de
  sklearn no aceptan texto). **[2026-09-27, Sección 5]** Ubicación en la red: dummies de
  `linea` (derivada del ID GTFS del andén; las aristas entre líneas distintas son
  `transbordo`) y `tramo_congestion_media`, un target encoding suavizado del tramo
  (congestión futura media del tramo, calculada **solo con entrenamiento**). Target:
  `target_congestibilidad_t_plus_1`.
- **[2026-09-27]** Compara 6 configuraciones (`RandomForestRegressor`,
  `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, cada una con y sin
  ubicación), con dos evaluaciones: el split cronológico 80/20 y una validación
  *rolling origin* sobre los últimos 5 días. Reporta MAE/RMSE global y, por separado,
  en filas con `hay_evento=1`. Elige la configuración con menor RMSE en filas con evento
  en la validación por días (hoy: `GradientBoosting + línea/tramo`).
- Explicabilidad por **importancia por permutación** (aumento del RMSE al barajar cada
  variable; las dummies de una misma variable se barajan juntas), agnóstica al modelo,
  en `importancia_variables.png`.
- Exporta a `modelos/modelo_anticipatorio.pkl` un paquete con el modelo, su codificación
  (categorías y target encoding) y el inicio del set de prueba; `ruteo_anticipatorio.py`
  reutiliza `construir_features()` del entrenador para no desincronizarse.

### 2.5 `ruteo_anticipatorio.py` — Integración estimación + ruteo
- Carga grafo base + modelo entrenado + dataset de contexto (usado como proxy de
  "sensores en tiempo real").
- `proyectar_hora(fecha, hora)` + `grafo_con_retraso(...)`: para cada arista predice el
  retraso de la hora siguiente y lo suma al tiempo ideal, generando un grafo proyectado.
  El contexto se filtra por fecha además de hora porque el dataset contiene 14 días, y
  las features se construyen con la misma función (`construir_features`) y codificación
  con que se entrenó el modelo.
- **[2026-09-27]** Los tiempos "reales" se miden contra la congestión simulada de la hora
  siguiente (`target_congestibilidad_t_plus_1`). Antes se medían contra la propia
  proyección de la IA, lo que favorecía a la IA por construcción.
- Compara ruta estática (Dijkstra sobre tiempo ideal) vs. ruta anticipatoria (Dijkstra
  sobre pesos proyectados por IA), evaluando ambas contra el tráfico real proyectado.
- Caso de prueba demostrado: Pantitlán → Auditorio, 7:00 AM (día laboral
  2026-01-13). Con el modelo multi-día, ambos sistemas eligen la misma ruta
  (Línea 9 hasta Tacubaya, transbordo a Línea 7): tiempo ideal 37.94 min, tiempo real
  en t+1 41.22 min (con el modelo RandomForest anterior la proyección era 40.33 min). Ya no se
  reproduce el desvío que mostraba la versión anterior, porque la falla fija de Línea 9
  (7-9am) que lo provocaba fue reemplazada por eventos estocásticos, y ese día a esa
  hora no hubo un evento que congestionara esta ruta.
- **[2026-09-27]** Evaluación sistemática sobre todas las horas con evento del set de
  prueba (ver Sección 5.3).

---

## 3. Limitaciones metodológicas actuales (a atender antes de tesis final)

1. ~~**Un solo día de simulación**~~ **RESUELTO 2026-09-27** (ver [Sección 4](#4-extensión-multi-día--eventos-estocásticos-2026-09-27)):
   ahora se simulan 14 días (10 laborales + 4 de fin de semana), y el split 80/20
   cronológico separa *días* completos (12 de entrenamiento, 3 de prueba, con 1 día
   compartido en la frontera del corte), no horas dentro del mismo día.
2. ~~**Evento de disrupción hardcodeado**~~ **RESUELTO 2026-09-27**: la falla fija de
   Línea 9 fue reemplazada por un generador de eventos estocásticos (lluvia, falla
   mecánica, incidente de plataforma) con tasas tipo Poisson por hora/línea/temporada y
   severidad/duración muestreadas. Tasas calibradas con fuentes públicas en la Sección 6.3.
3. ~~**Capacidad constante para toda la red**~~ **RESUELTO 2026-09-27** (Sección 6.1):
   capacidad por sentido y por línea derivada de datos del STC. Queda como supuesto la
   frecuencia por línea (derivada, no publicada; subestima el intervalo de la Línea A)
   y la capacidad de los transbordos (35,000).
4. ~~**Horizonte de predicción discretizado a 1 hora**~~ **RESUELTO 2026-09-29** (ver
   [Sección 13](#13-horizonte-sub-horario-bloques-de-15-minutos-2026-09-29)): el simulador trabaja en bloques de 15 min y el
   modelo predice a 15, 30, 45 y 60 min. La demanda O-D sigue siendo horaria y se
   interpola dentro de cada hora (supuesto documentado en la Sección 13.2).
5. ~~**Perfiles origen/destino hardcodeados**~~ **RESUELTO 2026-09-30** (ver
   [Sección 15](#15-perfiles-de-estación-derivados-de-datos-2026-09-30)): cada estación tiene pesos continuos de origen, destino y
   mixto derivados de la afluencia diaria real (domingo / día laboral) y de la topología
   GTFS (periferia y terminales). Siguen siendo supuestos la forma de las curvas horarias
   de cada perfil y la tabla de atractividad, porque el CSV no trae datos por hora.
6. **[2026-09-27] Eventos raros → poca evidencia** (Secciones 6.4 y 6.5): con tasas
   calibradas solo 0.08–0.29% de las filas tienen evento. La selección de modelo cambia
   entre semillas y la ventaja del ruteo anticipatorio en ahorro total no es
   significativa. Hace falta más simulación (más semillas o más días). Además, lluvia y
   falla mecánica casi no generan retraso (la carga fantasma apenas mueve la BPR con
   β=4); solo la suspensión por incidente lo hace. **Actualización (Sección 7):** con 30
   semillas la selección se fijó (RF + línea/tramo) y las pérdidas del ruteo ya tienen IC,
   pero la ventaja en ahorro total sigue sin ser significativa (60%): no es un problema de
   tamaño de muestra sino de que la IA predice el valor esperado de incidentes que duran
   poco (Sección 7.4). **Actualización (Sección 8):** con el aviso de tiempo estimado de
   restablecimiento como feature, la IA captura 74% del ahorro posible y supera al
   reactivo en ahorro total con P = 96% (IC de dos colas aún incluye 0).
   **Actualización (Sección 9):** eso solo se sostiene si el aviso acierta casi siempre
   si el incidente termina en la hora en curso; con un error aditivo de ±1 h la IA vuelve
   a 50% del ahorro posible y P = 64%.
7. ~~**Lluvia independiente por línea y sobreestimada**~~ **RESUELTO PARCIALMENTE
   2026-09-27** (Sección 6.3): un episodio por día lluvioso, con probabilidad mensual
   del SMN y correlacionado entre líneas. La hora de inicio (uniforme) y la probabilidad
   de alcanzar cada línea (0.75) siguen siendo supuestos, pendientes de datos del
   OH-IIUNAM.
8. ~~**[2026-09-29] El retraso por suspensión revela la duración restante**~~ **RESUELTO
   2026-09-30** (ver [Sección 14](#14-retraso-por-suspensión-sin-filtración-de-la-duración-2026-09-30)): el retraso ahora es constante mientras dura
   el incidente y el aviso recupera su valor. Descripción original (Sección 13.6):
   con bloques de 15 min el retraso de un tramo suspendido es `s²·R/2`, con R los minutos
   de cierre que faltan. Como la severidad `s` es una feature, el modelo puede despejar R
   del estado actual. Eso vuelve redundante el aviso de restablecimiento y hace que parte
   de la ventaja de la IA a 30–60 min se deba a la construcción del simulador. Hay que
   corregirlo antes de citar esos resultados.

Ninguno de estos puntos invalida el trabajo — son exactamente el tipo de simplificación
esperable en una primera iteración de prototipo — pero deben documentarse como alcance y
quedar como candidatos directos para la siguiente fase.

---

## 4. Extensión multi-día + eventos estocásticos (2026-09-27)

Esta sección documenta la ejecución del plan de la sesión anterior (ver historial de
commits). Cambios de código:

- `generador_sintetico_horario.py`: parametrizado por `DIAS_SIMULACION` (14 fechas de
  enero 2026), perfil horario laboral/fin de semana por `tipo_dia`, una matriz O-D por
  día y un `manifiesto_dias_simulados.csv` de salida. También se restringió el
  histórico fuente a solo los días simulados (antes desagregaba las ~1,857 fechas del
  CSV completo a nivel hora sin usarlas después: un archivo intermedio de ~300 MB para
  un solo día útil).
- `simulador_congestion.py`: lee el manifiesto y simula los 14 días en un solo run;
  reemplaza la falla fija de Línea 9 por el generador de eventos estocásticos
  (Poisson por hora/línea/tipo, severidad Beta(2,5), duración 1-4h); cachea rutas
  estáticas por par origen-destino (el cuello de botella real al escalar a varios
  días); agrega `tipo_dia`, `hay_evento`, `tipo_evento`, `severidad_evento` al dataset.
- `entrenador_anticipatorio.py`: si el dataset trae columnas de evento, las agrega como
  features (`hay_evento`, `severidad_evento`, dummies de `tipo_evento`).
- `ruteo_anticipatorio.py`: **corrección detectada al verificar el pipeline completo.**
  El script seguía construyendo `X_pred` con las 4 features originales, mientras el
  modelo reentrenado espera 10, y fallaba con `ValueError: The feature names should
  match those that were passed during fit.` Además tenía un error silencioso: filtraba
  el contexto solo por `hora`, así que con 14 días cada arista quedaba con el valor del
  último día leído, mezclando días distintos. Ahora usa `modelo.feature_names_in_` y
  filtra por `fecha_viaje` + `hora_viaje` (caso de prueba: `2026-01-13`, 7:00).

### 4.1 Dataset resultante

| | Versión anterior (1 día, evento determinista) | Versión actual (14 días, eventos estocásticos) |
|---|---|---|
| Registros | 3,740 | 52,360 |
| Días distintos | 1 | 14 (10 laborales, 4 fin de semana) |
| Filas con evento activo | ~ desconocido (no etiquetado) | 870 (1.66%): 707 lluvia, 94 incidente de plataforma, 69 falla mecánica |
| Split train/test (80/20 cronológico) | Mismo día en train y test (fuga temporal de facto) | 12 días train / 3 días test (1 día compartido en la frontera del corte) |
| Tiempo de generación (`simulador_congestion.py`) | segundos | ~5.5 min (14 días, con caché de rutas) |

### 4.2 Cambio en MAE / RMSE

| Métrica | Baseline (1 día, sin eventos aleatorios) | Nuevo (14 días, eventos estocásticos) | Δ |
|---|---|---|---|
| MAE (min) | 0.0000109 | 0.0028 | ×256 |
| RMSE (min) | 0.000233 | 0.0313 | ×134 |

**Lectura correcta de este resultado: el error subió, y eso es lo esperado y lo
correcto, no una regresión del modelo.** El baseline de 1 día tenía una fuga temporal
de facto — el split 80/20 separaba *horas* del mismo día con el mismo patrón de falla
recurrente (L9, 7-9am, todos los días idéntico), así que el modelo memorizaba
trivialmente el patrón y el error caía a prácticamente cero (MAE ≈ 10⁻⁵ min, una
precisión sin sentido físico). Con 14 días y eventos verdaderamente estocásticos
(distintos tipos, tramos, horas, severidades y duraciones cada día), el modelo debe
generalizar de 12 días vistos a 3 días no vistos con eventos nuevos que nunca ocurrieron
exactamente igual en entrenamiento. Un MAE de ~0.003 min y RMSE de ~0.031 min sobre un
target cuyo rango va de 0 a 6.09 min (media 0.015, con el 75% de los tramos en 0 —
la red solo se congestiona en tramos/horas puntuales) es un desempeño razonable para un
primer modelo con features todavía limitadas (no incluye, por ejemplo, la línea/tramo
como variable categórica). Esto también resuelve la limitación #1 de la Sección 3: el
split ahora separa días reales, no horas del mismo día.

Importancia de variables (nuevo modelo): `congestibilidad_t` (50.3%) y `hora` (41.9%)
siguen dominando; el bloque de variables de evento (`severidad_evento`,
`hay_evento`, dummies de `tipo_evento`) aporta en conjunto ~1.9% — señal débil pero
presente, consistente con que solo 1.66% de las filas tienen evento activo. Ampliar esa
señal (más días, o sobremuestrear horas con evento) es candidato directo para la
siguiente iteración si se quiere que el modelo dependa más del contexto de eventos y
menos de la inercia (`congestibilidad_t`).

### 4.3 Supuestos de simulación explícitos (para citar en el capítulo de metodología)

- Ventana de 14 días naturales de enero 2026 (única franja con datos reales completos
  en el CSV fuente); no hay quincena ni variación mensual/estacional real en la muestra,
  solo la distinción laboral/fin de semana.
- Perfil horario de fin de semana derivado programáticamente del laboral (aplanado +
  desplazamiento a mediodía/tarde), no calibrado con datos reales de fin de semana.
- Tasas de eventos (Poisson), factores de hora pico/temporada de lluvias y factores de
  impacto por tipo de evento son **ilustrativos**, no una calibración estadística sobre
  incidencia real del STC (ver Sección "Análisis de factibilidad técnica" más abajo,
  que ya anticipaba este punto y sigue vigente para la siguiente iteración).

### 4.4 Siguiente paso (ejecutado — ver [Sección 5](#5-ubicación-en-la-red-comparación-de-modelos-y-evaluación-sistemática-del-ruteo-2026-09-27))

**Bitácora de ingeniería.** Punto de partida: el dataset de entrenamiento ya tenía 14
días con eventos estocásticos, pero la señal de evento (`hay_evento`/`tipo_evento`/
`severidad_evento`) solo aportaba ~1.9% de importancia porque apenas 1.66% de las filas
tenían un evento activo. Antes de calibrar tasas contra fuentes reales se buscó subir la
proporción de señal explotable. Pasos implementados:

1. Se añadió `linea`/`tramo` como variable categórica (dummies de línea + target
   encoding del tramo) para que el modelo pudiera aprender qué tramos son
   estructuralmente más propensos a congestión; hasta entonces el modelo no sabía en qué
   parte de la red estaba parado.
2. Se evaluó migrar de `RandomForestRegressor` a `GradientBoostingRegressor` y a
   `HistGradientBoostingRegressor` (que maneja mejor el desbalance de clases raras),
   reportando la predicción en las filas con `hay_evento=1` aparte del MAE/RMSE global,
   ya que ese segmento era <2% de los datos y podía quedar diluido.
3. Se documentó en `avances.md` una propuesta concreta de fuentes para calibrar
   `TASA_BASE_LLUVIA`, `TASA_BASE_FALLA_MECANICA` y `TASA_BASE_INCIDENTE_PLATAFORMA` en
   `simulador_congestion.py` (climatología SMN/Conagua para lluvia; boletines de
   contingencia, notas de prensa y transparencia del Metro para fallas/incidentes), como
   aproximación documentada y no como calibración estadística rigurosa (Sección 5.4).
4. En `ruteo_anticipatorio.py`, además del caso fijo (Pantitlán → Auditorio,
   2026-01-13 7:00, donde la ruta IA coincidía con la estática), se agregó un recorrido
   de las fechas/horas del conjunto de prueba con `hay_evento=1` que compara ruta
   estática vs. anticipatoria en los pares O-D que cruzan los tramos afectados y reporta
   en cuántos casos la IA cambia la ruta y cuántos minutos ahorra (Sección 5.3).

---

## 5. Ubicación en la red, comparación de modelos y evaluación sistemática del ruteo (2026-09-27)

Ejecución del plan de la Sección 4.4. Cambios de código:

- `entrenador_anticipatorio.py`: agrega la ubicación en la red como features (`linea`
  one-hot + `tramo_congestion_media`, target encoding suavizado calculado solo con
  entrenamiento); compara RandomForest / GradientBoosting / HistGradientBoosting con y
  sin ubicación; agrega validación *rolling origin* por días; reporta métricas en filas
  con evento aparte; cambia la explicabilidad a importancia por permutación; exporta un
  paquete (`modelos/modelo_anticipatorio.pkl`, reemplaza a `modelo_anticipatorio_rf.pkl`)
  con modelo + codificación + inicio del set de prueba.
- `ruteo_anticipatorio.py`: reutiliza `construir_features()` del entrenador; mide los
  tiempos contra la congestión real de t+1; agrega la evaluación sistemática sobre horas
  con evento (Sección 5.3), que guarda el detalle en
  `datos_procesados/evaluacion_ruteo_eventos.csv`.

### 5.1 Comparación de modelos

Métricas en minutos de retraso sobre el tiempo ideal. "Evento" = filas con
`hay_evento=1`.

**Split cronológico 80/20** (prueba: 16–18 de enero; 10,472 filas, **137 con evento**):

| Configuración | MAE | RMSE | MAE evento | RMSE evento |
|---|---|---|---|---|
| HistGradientBoosting (sin ubicación) | 0.0030 | 0.0297 | 0.0301 | 0.1236 |
| GradientBoosting (sin ubicación) | 0.0032 | 0.0303 | 0.0291 | 0.1263 |
| **GradientBoosting + línea/tramo** | 0.0039 | 0.0306 | 0.0309 | 0.1296 |
| HistGradientBoosting + línea/tramo | 0.0036 | 0.0307 | 0.0324 | 0.1346 |
| RandomForest + línea/tramo | 0.0027 | 0.0313 | 0.0364 | 0.1496 |
| RandomForest (sin ubicación) — *modelo anterior* | 0.0028 | 0.0313 | 0.0363 | 0.1541 |

**Validación por días** (5 pliegues: cada día del 14 al 18 de enero se predice con un
modelo entrenado con todos los días previos; **255 filas con evento**):

| Configuración | MAE | RMSE | MAE evento | RMSE evento |
|---|---|---|---|---|
| **GradientBoosting + línea/tramo** | 0.0042 | **0.0272** | **0.0369** | **0.1193** |
| RandomForest + línea/tramo | 0.0032 | 0.0285 | 0.0488 | 0.1459 |
| HistGradientBoosting + línea/tramo | 0.0044 | 0.0311 | 0.0562 | 0.1728 |
| HistGradientBoosting (sin ubicación) | 0.0044 | 0.0315 | 0.0591 | 0.1801 |
| RandomForest (sin ubicación) — *modelo anterior* | 0.0036 | 0.0321 | 0.0579 | 0.1844 |
| GradientBoosting (sin ubicación) | 0.0043 | 0.0385 | 0.0529 | 0.1965 |
| *Referencia: persistencia (predecir t+1 = t)* | 0.0151 | 0.0806 | 0.0491 | 0.1802 |

Lectura:

1. **Las dos evaluaciones no coinciden, y la validación por días es la más confiable.**
   En el split 80/20 la ubicación parece no ayudar, pero ese set tiene solo 3 días y 137
   filas con evento. En la validación por días (5 días, 255 filas con evento) la
   ubicación mejora a los tres modelos, y más en el segmento con evento: RandomForest
   baja su RMSE con evento de 0.184 a 0.146 (−21%) y GradientBoosting de 0.197 a 0.119
   (−39%). Por eso el criterio de selección es el RMSE con evento en la validación por
   días.
2. **Modelo elegido: GradientBoosting + línea/tramo.** Frente al modelo anterior
   (RandomForest sin ubicación), en la validación por días reduce el RMSE con evento 35%
   (0.184 → 0.119) y el RMSE global 15% (0.032 → 0.027). A cambio, su MAE global es algo
   peor (0.0042 vs. 0.0036): predice retrasos pequeños mayores a 0 en tramos que no se
   congestionan (75% del target es 0). Para el ruteo pesa más acertar los picos de
   retraso, que es lo que mide el RMSE.
3. **HistGradientBoosting no mejoró sobre GradientBoosting** a pesar de lo planteado en
   la Sección 4.4. El problema no es tanto el desbalance como la poca señal (punto 5).
4. **Todos los modelos superan a la persistencia** en RMSE global y con evento, así que
   el modelo aprende algo más que "mañana igual que hoy".
5. **La señal de evento sigue siendo débil.** En la importancia por permutación del modelo
   elegido (aumento del RMSE al barajar la variable, set de prueba 80/20), dominan
   `congestibilidad_t` (0.022 min) y `congestibilidad_t_minus_1` (0.017), seguidas de
   `hora` (0.003) y `tramo_congestion_media` (0.003). `tipo_evento` aporta 0.0005 y
   `hay_evento`/`severidad_evento` ≈ 0. La causa probable está en el simulador, no en
   el modelo: con capacidad constante y BPR, un evento típico cambia poco el retraso, y
   ese efecto ya queda reflejado en `congestibilidad_t` (el evento ya estaba activo en t).
   Se agrega como limitación #6 en la Sección 3.

### 5.2 Métricas del modelo exportado

El modelo exportado es `GradientBoosting + línea/tramo`, entrenado con el 80% inicial
(no con todo el dataset, para que el ruteo se evalúe en días que el modelo no vio). En el
20% final: MAE 0.0039 min, RMSE 0.0306 min; en las 137 filas con evento: MAE 0.0309 min,
RMSE 0.1296 min.

### 5.3 Evaluación sistemática del ruteo en horas con evento

> **Nota:** estos resultados usan las tasas sin calibrar y la carga de ambos sentidos
> sumada. Quedan reemplazados por la Sección 6.5, donde la conclusión cambia.

**Diseño.** Se recorren las **27 horas con al menos un tramo en evento** del set de
prueba (16, 17 y 18 de enero). Para cada hora se toman todos los pares origen-destino
(una estación por nombre, 163 estaciones) cuya **ruta estática cruza un tramo con
evento**: **84,650 casos**. Para cada caso se calculan cuatro rutas y todas se miden
contra el **tráfico real simulado de la hora siguiente**:

- **Estático**: tiempo ideal, sin información de tráfico (baseline).
- **Reactivo**: congestión observada en la hora actual t (sistema reactivo del índice 5.5).
- **Anticipatorio**: congestión que la IA proyecta para t+1.
- **Oráculo**: congestión real de t+1. Es la cota superior: el mayor ahorro posible.

| Sistema | Cambia ruta | Casos que ganan / pierden vs. estático | Ahorro medio si cambia | Ahorro neto total | % del ahorro posible capturado |
|---|---|---|---|---|---|
| Reactivo | 542 (0.64%) | 220 / 320 | −0.02 min | −8.8 min | **−5.0%** |
| Anticipatorio (IA) | 282 (0.33%) | 148 / 134 | +0.40 min | +114.0 min | **64.6%** |
| Oráculo | 234 (0.28%) | 234 / 0 | +0.75 min | +176.3 min | 100% |

Lectura:

1. **Resultado central: el ruteo anticipatorio captura 64.6% del ahorro máximo posible;
   el reactivo empeora al usuario (−5%).** El sistema reactivo esquiva la congestión de
   la hora actual, que en la hora siguiente ya cambió (los eventos duran 1–4 h y la
   demanda cambia cada hora). Por eso cambia de ruta más veces (542) y pierde en más
   casos de los que gana (320 vs. 220). La IA cambia menos (282) y gana más de lo que
   pierde: +1.12 min en promedio cuando acierta y −0.39 min cuando falla. Es el
   argumento cuantitativo para la tesis: anticipar es mejor que reaccionar.
2. **La magnitud absoluta es pequeña.** Solo 0.28% de los casos tiene una ruta mejor que
   la estática (oráculo), el ahorro máximo en un caso es 1.97 min y el viaje medio dura
   ~40 min. Esto viene del simulador, no del ruteo: con la capacidad constante de 35,000
   pas/h y BPR, un evento casi nunca vuelve una ruta alternativa (con transbordos de
   5 min) más rápida que la directa. Para defender el ahorro en tiempo, antes hay que
   diferenciar capacidades por tramo (limitación #3) o dar a los incidentes un efecto
   de cierre de tramo (ya previsto en el análisis de factibilidad, tabla 3.1).
3. **La IA se equivoca en casi la mitad de sus cambios de ruta** (134 de 282). Esto es
   consistente con su RMSE con evento (~0.12–0.13 min): el ahorro disponible por caso es
   del mismo orden que el error del modelo.
4. **Cautela estadística:** los 84,650 casos no son independientes. Salen de 27 horas y
   de pocos eventos, y muchos pares O-D comparten los mismos tramos afectados. La
   conclusión (anticipatorio > estático > reactivo) es consistente en esta muestra, pero
   para citarla con intervalos de confianza hay que repetir la simulación con varias
   semillas aleatorias.

### 5.4 Propuesta de fuentes para calibrar las tasas de eventos

Parámetros en `simulador_congestion.py`: `TASA_BASE_LLUVIA = 0.02`,
`TASA_BASE_FALLA_MECANICA = 0.01` y `TASA_BASE_INCIDENTE_PLATAFORMA = 0.006` (eventos
esperados por hora y por línea en hora valle), `FACTOR_HORA_PICO = 2.5` (6 horas pico)
y `FACTOR_TEMPORADA_LLUVIAS = 6.0` (mayo–octubre). La operación simulada va de las 5 a
las 23 h (19 horas: 13 valle + 6 pico), así que el número esperado de eventos por línea
y por día es `λ_base × (13 + 2.5 × 6) = λ_base × 28`.

**Chequeo de orden de magnitud con los valores actuales:**

| Tipo | Eventos esperados/día (tasas actuales) | Referencia real |
|---|---|---|
| Falla mecánica + incidente de plataforma (12 líneas) | (0.01 + 0.006) × 28 × 12 ≈ **5.4** | **3,708 incidentes con desalojo de trenes entre 2018 y agosto de 2022**, según respuesta del STC vía la Plataforma Nacional de Transparencia (El Universal): ≈ 1,704 días → **≈ 2.2 por día** en toda la red. La falla más frecuente fue el sistema de puertas (34 de cada 100). |
| Lluvia en enero (3 líneas de superficie) | 0.02 × 28 × 3 ≈ **1.7** | Enero es temporada seca en la CDMX; los días con lluvia del mes se obtienen de las Normales Climatológicas del SMN (observatorio de Tacubaya). |
| Lluvia en mayo–octubre | 0.02 × 6 × 28 ≈ **3.4 por línea** | Idem, meses de temporada de lluvias. |

Las tasas de falla/incidente están en el orden correcto (unas 2.5 veces la referencia).
Parte de la diferencia es esperable, porque los desalojos son solo los incidentes más
graves y la referencia subestima el total. La lluvia de enero parece sobreestimada.

**Fuentes propuestas y método de calibración:**

1. **Lluvia: frecuencia diaria.** Normales Climatológicas del SMN/Conagua
   (<https://smn.conagua.gob.mx/es/climatologia/informacion-climatologica/normales-climatologicas-por-estado>),
   Ciudad de México, estación Tacubaya (observatorio de referencia del SMN): número medio
   de días con precipitación por mes. Con eso, `P(lluvia en el día | mes)` sustituye al
   factor binario de temporada (`FACTOR_TEMPORADA_LLUVIAS`) por un valor para cada mes.
2. **Lluvia: distribución horaria y extensión espacial.** Red pluviométrica del
   Observatorio Hidrológico del Instituto de Ingeniería UNAM (OH-IIUNAM), con unas 55
   estaciones en la CDMX que registran la intensidad minuto a minuto
   (<https://www.iingen.unam.mx/es-mx/AlmacenDigital/Notas/Paginas/observatoriohidrologico.aspx>).
   Con las estaciones cercanas a las líneas A, B y 12 se estima (a) la probabilidad de
   lluvia por hora del día, que en la CDMX se concentra en la tarde-noche y no sigue la
   hora pico del Metro (hoy se le aplica `FACTOR_HORA_PICO` sin justificación física), y
   (b) con qué frecuencia llueve a la vez en varias líneas, para dejar de sortear cada
   línea por separado (limitación #7). Hay que pedir los datos al OH-IIUNAM; su
   publicación en tiempo real es por redes y web.
3. **Fallas mecánicas e incidentes de plataforma.**
   - Punto de partida documentado: la cifra de 3,708 desalojos 2018–2022 citada arriba
     (El Universal, <https://www.eluniversal.com.mx/metropoli/de-2018-2022-se-reportaron-3-mil-708-incidentes>),
     que da `λ_base ≈ 2.2 / (12 líneas × 28) ≈ 0.0065` por línea-hora valle para ambos
     tipos juntos, suponiendo el mismo factor de hora pico.
   - Para separarlo **por línea, tipo y hora**: solicitud de información al STC por la
     Plataforma Nacional de Transparencia o su Unidad de Transparencia
     (<https://www.metro.cdmx.gob.mx/tranparencias/transparencia-cdmx>), pidiendo el
     registro de incidentes/averías con fecha, hora, línea, estación y causa. Hay
     antecedentes de que el STC entrega este tipo de datos: la cifra anterior y el
     reporte de 136 averías en la Línea 12 tras su reapertura (El Gráfico, 2024).
   - Complemento (y validación del factor de hora pico): recopilación manual de los
     avisos de la cuenta oficial del Metro en redes sociales durante algunas semanas.
4. **Método de conversión a tasa de Poisson** (para cualquiera de las fuentes): si en
   `D` días se observaron `N` eventos de un tipo en una línea, con `F` = factor de hora
   pico, entonces `λ_base = N / (D × (13 + 6·F))`. Si la fuente trae la hora del evento,
   `F` también se estima: la tasa por hora en hora pico dividida entre la tasa por hora
   en valle.

Esto es una **aproximación documentada, no una calibración estadística rigurosa**: los
desalojos no son todos los incidentes, la cifra agrega todas las líneas y todo el
periodo 2018–2022, y la climatología de Tacubaya representa un solo punto de la ciudad.
Aun así, cada parámetro queda respaldado por una fuente citable en lugar de un valor
ilustrativo.

### 5.5 Siguiente paso (ejecutado — ver [Sección 6](#6-capacidad-por-línea-suspensión-de-tramos-tasas-calibradas-y-experimento-multi-semilla-2026-09-27))

**Bitácora de ingeniería.** Punto de partida: la evaluación sistemática (Sección 5.3)
mostraba que el ruteo anticipatorio capturaba 64.6% del ahorro posible y el reactivo
empeoraba (−5%), pero el ahorro absoluto era pequeño (el oráculo solo mejoraba 0.28% de
los casos, máximo 1.97 min) porque, con capacidad constante y BPR, los eventos casi no
cambiaban la ruta óptima; además el modelo apenas usaba las variables de evento. Se atacó
la causa en el simulador. Pasos implementados:

1. Se sustituyó `CAPACIDAD_PROMEDIO_TRAMO_HORA` por una capacidad por línea (trenes por
   hora × capacidad del tren, con valores documentados del STC: parque vehicular,
   disponibilidad e intervalo mínimo) (Sección 6.1).
2. Se hizo que `incidente_plataforma` suspenda temporalmente el tramo en lugar de solo
   agregar carga fantasma (Sección 6.2).
3. Se aplicaron las tasas de la Sección 5.4 (`λ_base ≈ 0.0065` combinado para
   falla+incidente, lluvia por mes según días con precipitación de Tacubaya, sin factor
   de hora pico para lluvia, y lluvia correlacionada entre las líneas A, B y 12)
   (Sección 6.3).
4. Se regeneró el dataset, se reentrenó (el entrenador ya comparaba modelos) y se volvió
   a correr `ruteo_anticipatorio.py`. La simulación se repitió con 5 semillas para
   reportar la media y el intervalo del % de ahorro capturado por cada sistema
   (Secciones 6.4 y 6.5), y se actualizó `avances.md`.

---

## 6. Capacidad por línea, suspensión de tramos, tasas calibradas y experimento multi-semilla (2026-09-27)

Ejecución del plan de la Sección 5.5. Cambios de código:

- `simulador_congestion.py`:
  - Capacidad por sentido y por línea en lugar de la constante de 35,000 pas/h.
  - La V/C se calcula con la carga del **sentido más cargado**.
  - `incidente_plataforma` suspende el tramo.
  - Tasas de eventos calibradas con fuentes públicas; la lluvia es un solo episodio por ciudad.
  - Argumentos `--semilla` y `--salida`.
  - Cambió `iterrows` por `zip`: 14 días en ~1.5 min en vez de ~5.3 min. La carga de pasajeros resultante es idéntica fila por fila a la de la versión anterior.
- `entrenador_anticipatorio.py` y `ruteo_anticipatorio.py`: argumentos de rutas de
  entrada/salida (los valores por defecto son los de siempre). El ruteo guarda además un
  resumen (`datos_procesados/resumen_ruteo_eventos.csv`).
- `experimento_semillas.py` (nuevo): corre simulador → entrenador → ruteo con 5 semillas
  (42, 7, 13, 101, 2026), 3 en paralelo, en `datos_procesados/semillas/semilla_<s>/`, y
  agrega los resultados (`--solo-agregar` recalcula sin volver a simular).

### 6.1 Capacidad por línea

Capacidad por sentido = trenes/hora × carros por tren × 170 pasajeros por carro.

| Dato | Valor | Fuente |
|---|---|---|
| Capacidad por tren | 6 carros = 1,020 pas.; 9 carros = 1,530 → 170 por carro | STC, [Parque Vehicular](https://www.metro.cdmx.gob.mx/parque-vehicular) |
| Trenes asignados por línea | L1 50, L2 41, L3 54, L4 14, L5 25, L6 15, L7 32, L8 30, L9 34, LA 33, LB 36, L12 30 | STC, Parque Vehicular |
| Carros por tren | 6 en L4 y L6 (29 trenes neumáticos de 6 carros = 14 + 15); 7 en L12 (30 férreos de 7 carros); LA: 22 de 9 + 11 de 6 → promedio 8; resto 9 | Deducido del desglose del Parque Vehicular |
| Disponibilidad | 256 de 394 trenes en servicio (64.9%) | STC vía transparencia, [Expansión, nov. 2024](https://politica.expansion.mx/cdmx/2024/11/23/metro-cdmx-opera-con-256-de-sus-394-trenes) |
| Intervalo mínimo | 2 min en hora pico | STC, [Preguntas frecuentes](https://www.metro.cdmx.gob.mx/acerca-del-metro/mas-informacion/preguntas-frecuentes) |

El STC no publica la frecuencia por línea, así que se deriva:
`trenes/hora = trenes de la línea × 0.649 × 60 / (2 × tiempo de recorrido GTFS)`, con tope
en 30 trenes/hora (intervalo de 2 min). No se incluye tiempo de maniobra en terminales
porque no está documentado.

| Línea | 12 | 6 | 4 | B | 8 | 5 | 2 | 7 | A | 3 | 9 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Capacidad por sentido (pas/h) | 12,780 | 16,830 | 17,539 | 28,003 | 28,973 | 30,758 | 31,876 | 34,653 | 35,707 | 41,432 | 43,370 | 45,699 |
| Intervalo derivado (min) | 5.6 | 3.6 | 3.5 | 3.3 | 3.2 | 3.0 | 2.9 | 2.7 | 2.3 | 2.2 | 2.1 | 2.0 |

Contraste con intervalos de hora pico reportados por un sitio no oficial
([metrocdmx.net](https://metrocdmx.net/linea-1/)): L1 ≈ 2 min (derivado 2.0 ✓), L3 2–3
(2.2 ✓), L12 3–5 (5.6, algo más largo), LA 4–6 (2.3, **subestimado**). Para la Línea A
la capacidad derivada probablemente es demasiado alta; la causa más probable es que su
disponibilidad real sea menor que el promedio del sistema.

**Carga por sentido.** El grafo es no dirigido y antes la carga de un tramo sumaba los
dos sentidos, que se comparaba contra una capacidad de un solo sentido. Ahora se lleva
la carga de cada sentido (las rutas del simulador son secuencias ordenadas) y la V/C usa
la del sentido más cargado. `carga_pasajeros_red` en el dataset sigue siendo la suma de
ambos sentidos. Los pasillos de transbordo conservan la capacidad anterior (35,000),
porque no hay dato.

Efecto: la congestión se concentra en las líneas de menor capacidad (12, 4, 6, B, 8). La
Línea 9 a las 7:00 deja de congestionarse, y en el caso de prueba Pantitlán → Auditorio
el tiempo real queda en 37.96 min, frente a 37.94 ideales.

### 6.2 Suspensión de tramos por incidente de plataforma

`incidente_plataforma` ya no suma carga fantasma: suspende el tramo. La severidad `s` se
interpreta como la fracción de la hora con el tramo cerrado. Un pasajero que llega
durante el cierre espera en promedio `60·s/2` min, y llega durante el cierre una
fracción `s` de los pasajeros. Por eso el retraso medio de la hora es **`30·s²` min**
(hasta 30 min si el tramo pasa la hora completa cerrado) y se suma después del BPR. La
lluvia y la falla mecánica siguen como carga fantasma, ahora proporcional a la capacidad
de la línea afectada.

Con la semilla 42, las 27 filas con incidente tienen congestión media de 3.3 min
(máximo 7.7). La falla mecánica queda en 0.18 min y la lluvia en 0.003. Con capacidad
por sentido y BPR con β=4, la carga fantasma casi no mueve el tiempo de viaje; solo la
suspensión produce retrasos grandes.

### 6.3 Tasas de eventos calibradas

| Parámetro | Antes | Ahora | Fuente / cálculo |
|---|---|---|---|
| Falla mecánica + incidente de plataforma | 0.016 por línea-hora valle (≈ 5.4 eventos/día en la red) | **0.0065** (≈ 2.2/día) | 3,708 desalojos 2018–ago 2022 (Sección 5.4). Reparto entre tipos sin fuente: se conserva la proporción 5:3 → 0.0041 / 0.0024 |
| Factor hora pico (fallas/incidentes) | 2.5 | 2.5 | Sin cambio: no hay dato horario (pendiente de solicitud PNT) |
| Lluvia: probabilidad diaria | Poisson 0.02 por línea-hora, ×6 may–oct | **días con lluvia del mes / días del mes**: enero 2.8/31 = 9% | [Normal climatológica 1981–2010, estación 9048 Tacubaya Central (SMN)](https://smn.conagua.gob.mx/tools/RESOURCES/Normales_Climatologicas/Normales8110/df/nor8110_09048.txt): días con lluvia ENE 2.8, FEB 2.2, MAR 3.6, ABR 7.2, MAY 11.5, JUN 17.6, JUL 22.0, AGO 21.0, SEP 17.9, OCT 9.9, NOV 3.0, DIC 1.5 |
| Lluvia: hora | Poisson con factor de hora pico | Un episodio por día lluvioso, hora de inicio uniforme | Sin factor de hora pico: la lluvia no depende de la afluencia. Distribución horaria real pendiente (OH-IIUNAM) |
| Lluvia: extensión | Sorteo independiente por línea | Un episodio afecta a cada línea de superficie con probabilidad 0.75 (al menos una) | 0.75 es ilustrativo; pendiente de correlación espacial OH-IIUNAM |

Efecto en el dataset (semilla 42): de 102 eventos y 870 filas con evento (1.66%) se
pasa a **40 eventos y 130 filas (0.25%)**. En las 5 semillas hay entre 25 y 40 eventos y
entre 0.08% y 0.29% de filas con evento. En enero real, los eventos que afectan al
servicio son raros.

### 6.4 Resultados multi-semilla: modelos

Con eventos raros, cada semilla deja solo 13–25 filas con evento en la validación por
días. La selección de modelo por semilla es inestable: **se eligió un modelo distinto en
4 de las 5 semillas** (42: HistGB sin ubicación; 7 y 2026: GradientBoosting + línea/tramo;
13: RandomForest + línea/tramo; 101: GradientBoosting sin ubicación).

Validación por días agregada sobre las 5 semillas (89 filas con evento en total):

| Configuración | RMSE global | MAE evento | RMSE evento (agregado) |
|---|---|---|---|
| **RandomForest + línea/tramo** | **0.0666** | 0.340 | **1.078** |
| RandomForest (sin ubicación) | 0.0670 | 0.379 | 1.088 |
| HistGradientBoosting + línea/tramo | 0.0906 | 0.417 | 1.117 |
| GradientBoosting (sin ubicación) | 0.0672 | 0.371 | 1.173 |
| GradientBoosting + línea/tramo | 0.0684 | 0.344 | 1.245 |
| HistGradientBoosting (sin ubicación) | 0.1055 | 0.497 | 1.321 |

Lectura: con el simulador recalibrado, la conclusión de la Sección 5.1 (GradientBoosting
+ línea/tramo) **no se sostiene**. Agregando semillas gana RandomForest + línea/tramo en
RMSE global y con evento, aunque las diferencias entre las dos variantes de RF son
pequeñas. El criterio de selección por semilla (RMSE con evento con ~20 filas) es ruido:
conviene fijar el modelo con el resultado agregado o elegir por RMSE global (ver 6.6).
El modelo exportado para la semilla 42 (el del pipeline por defecto) es HistGB sin
ubicación, el peor en el agregado. Su importancia por permutación da peso negativo a
`severidad_evento` (−0.013: barajarla *mejora* el error), señal de sobreajuste a los
pocos eventos.

### 6.5 Resultados multi-semilla: ruteo

Se juntan los casos de las 5 semillas: 39 horas con evento en días no vistos y 82,784
casos O-D. El % del ahorro posible **no se promedia por semilla**: cuando el oráculo casi
no puede ahorrar, el cociente explota (semilla 101: −2,880%) o queda indefinido (semilla
7: ahorro posible 0). El intervalo de confianza sale de un **bootstrap por conglomerados**
(5,000 repeticiones) que remuestrea horas con evento completas, porque los pares O-D de
una misma hora comparten tramos afectados y no son independientes.

| Sistema | Ahorro total (min) | Pérdidas totales (min) | Cambios de ruta | % del ahorro posible [IC 95%] |
|---|---|---|---|---|
| Reactivo | 5,567 | **−11,070** | 9,587 | 33.4 [−4,305; 97.0] |
| Anticipatorio (IA) | 6,400 | **−598** | 2,261 | 38.5 [−176; 72.3] |
| Oráculo | 16,644 | 0 | 5,112 | 100 |

Por semilla (ahorro total, min):

| Semilla | Horas con evento | Reactivo | Anticipatorio | Oráculo |
|---|---|---|---|---|
| 42 | 12 | 8,958 | 6,916 | 16,556 |
| 7 | 6 | −1,597 | −73 | 0 |
| 13 | 10 | −30 | 13 | 26 |
| 101 | 10 | −1,764 | −455 | 61 |
| 2026 | 1 | 0.1 | 0.1 | 0.1 |

Lectura:

1. **No se puede afirmar que la IA ahorre más que el reactivo.** Diferencia puntual:
   +833 min a favor de la IA, pero la probabilidad bootstrap de que la IA ahorre más es
   solo **54%**. Solo 12 de las 39 horas tienen ahorro posible > 0.5 min, y el total
   depende casi por completo de un incidente de la semilla 42 (13–15 h del 16 de enero).
2. **Resultado robusto: la IA es mucho más conservadora y pierde 18 veces menos.** El
   reactivo cambia de ruta 4 veces más y acumula −11,070 min en casos donde empeora al
   usuario, contra −598 de la IA. Esto se repite en las 3 semillas con pérdidas (42, 7,
   101). El mecanismo se ve en el incidente de la semilla 42:
   - 13 h: el reactivo capta el 100% del ahorro (8,026 min) y la IA 74% (5,958).
   - 14 h: el incidente continúa; el reactivo vuelve a captar el 100% (8,026) y la IA
     solo 845.
   - 15 h: **el tramo reabre**. El reactivo sigue esquivándolo y pierde 7,299 min; la IA
     pierde 36.

   Es decir, el reactivo acierta mientras la disrupción persiste y falla al terminar. La
   IA hace lo contrario: anticipa el final pero subestima la persistencia.
3. Para defender en la tesis que "anticipar es mejor que reaccionar" hacen falta muchas
   más horas con incidentes de impacto. Con 5 semillas × 3 días de prueba hay unos pocos
   incidentes relevantes. La conclusión de la Sección 5.3 (IA 64.6% vs. reactivo −5%)
   dependía de las tasas sobreestimadas y de la carga sumando ambos sentidos, y **queda
   reemplazada** por esta.

### 6.6 Siguiente paso (ejecutado — ver [Sección 7](#7-selección-fija-de-modelo-30-semillas-y-edad-del-evento-2026-09-27))

**Bitácora de ingeniería.** Punto de partida: con el simulador recalibrado (Sección 6), el
ruteo anticipatorio perdía 18 veces menos que el reactivo, pero la diferencia en ahorro
total no era significativa (probabilidad bootstrap 54%) porque solo había ~12 horas con
incidentes de impacto en 5 semillas, y la selección de modelo cambiaba de ganador en 4 de
5 semillas. Pasos implementados (resultados en la [Sección 7](#7-selección-fija-de-modelo-30-semillas-y-edad-del-evento-2026-09-27)):

1. Se cambió el criterio de selección en `entrenador_anticipatorio.py` para que no
   dependiera de ~20 filas con evento: se fijó la configuración con el resultado
   agregado multi-semilla (RandomForest + línea/tramo, constante `MODELO_ELEGIDO`) y se
   justificó en `avances.md` (Sección 7.1).
2. Se aumentó la evidencia: `SEMILLAS` en `experimento_semillas.py` pasó de 5 a 30 (las
   5 originales + 1001–1025; 36.7 min con 3 corridas en paralelo) y se volvió a reportar
   el bootstrap por horas con evento, incluida la probabilidad de que el anticipatorio
   ahorre más que el reactivo y un IC para las pérdidas de cada sistema (Sección 7.3).
3. Se agregó al modelo la variable `edad_evento` (horas desde que empezó el evento;
   el simulador conoce `hora_inicio` y `duracion`, pero solo expone la hora de inicio,
   que sería observable en tiempo real) para que la IA pudiera aprender la persistencia
   de los incidentes, que era donde perdía contra el reactivo (Sección 7.4).
4. Se actualizó `avances.md` con los resultados.

---

## 7. Selección fija de modelo, 30 semillas y edad del evento (2026-09-27)

Ejecución del plan de la Sección 6.6. Cambios de código:

- `simulador_congestion.py`: guarda `edad_evento` por tramo y hora (`hora − hora_inicio`
  del evento más severo del tramo; 0 en la primera hora, −1 sin evento). La duración no
  se expone. Los sorteos aleatorios no cambian: para una misma semilla el dataset es
  idéntico al de la Sección 6 salvo la columna nueva.
- `entrenador_anticipatorio.py`: `edad_evento` entra en `FEATURES_BASE`; el modelo
  exportado es fijo (`MODELO_ELEGIDO = ('RandomForest', True)`). La comparación de las 6
  configuraciones se sigue calculando y guardando, e imprime qué habrían elegido los
  criterios por semilla (menor RMSE con evento y menor RMSE global), solo como auditoría.
- `experimento_semillas.py`: 30 semillas; el bootstrap reporta IC 95% del ahorro y de
  las pérdidas de cada sistema, de la diferencia de ahorro y de pérdidas entre
  anticipatorio y reactivo, y las probabilidades bootstrap de que el anticipatorio ahorre
  más y pierda menos. Se guarda en `modelos/resultados_semillas_bootstrap.csv`. La
  estabilidad de la selección se reporta como conteo de ganadores por semilla.
- Se regeneró el pipeline por defecto (semilla 42): dataset, `modelos/modelo_anticipatorio.pkl`
  (ahora RandomForest + línea/tramo) e `importancia_variables.png`.

### 7.1 Criterio de selección del modelo

**Decisión:** el modelo exportado es siempre **RandomForest + línea/tramo**; ya no se elige
por semilla.

**Justificación.** El criterio anterior (menor RMSE con evento en la validación por días)
se decidía con 13–25 filas con evento por semilla, y con 5 semillas eligió 4 modelos
distintos (Sección 6.4). Con 30 semillas la inestabilidad se confirma: el ganador por RMSE
con evento se reparte entre las 6 configuraciones (HistGB sin ubicación 8, GB + línea/tramo
6, HistGB + línea/tramo 6, RF + línea/tramo 4, GB sin ubicación 4, RF sin ubicación 2). Un
criterio que elige casi al azar no sirve para el capítulo de metodología. Se fija la
configuración que ganó en el agregado de la Sección 6.4, y la comparación se sigue
reportando para auditarla.

**Contraste con 30 semillas** (validación por días agregada; 933 filas con evento, contra
89 con 5 semillas):

| Configuración | RMSE global (media por semilla) | MAE global | MAE evento | RMSE evento (agregado) |
|---|---|---|---|---|
| HistGradientBoosting + línea/tramo | 0.0963 | 0.0063 | 0.489 | **1.085** |
| HistGradientBoosting (sin ubicación) | 0.1054 | 0.0067 | 0.512 | 1.099 |
| **RandomForest + línea/tramo** (elegido) | 0.0743 | **0.0034** | 0.511 | 1.179 |
| RandomForest (sin ubicación) | **0.0735** | **0.0034** | 0.522 | 1.193 |
| GradientBoosting + línea/tramo | 0.0743 | 0.0043 | **0.488** | 1.230 |
| GradientBoosting (sin ubicación) | 0.0740 | 0.0042 | 0.515 | 1.265 |

Ganador por semilla con menor RMSE global: RF sin ubicación 14, GB sin ubicación 5, GB +
línea/tramo 5, RF + línea/tramo 3, HistGB + línea/tramo 3.

Lectura:

1. **La conclusión de la Sección 6.4 tampoco se sostiene del todo con 30 semillas.** En el
   RMSE con evento ahora gana HistGB + línea/tramo (1.085 vs. 1.179 del elegido, −8%), pero
   a cambio de un RMSE global 30% peor (0.096 vs. 0.074) y casi el doble de MAE global,
   error que viene de las filas sin evento (99.8% del dataset). El orden entre familias
   cambia según la métrica y las diferencias dentro de cada familia son pequeñas.
2. **Se mantiene RandomForest + línea/tramo** porque queda en el grupo de menor error
   global (a 1% de RF sin ubicación, diferencia menor que la variación entre semillas),
   tiene el menor MAE y es el mejor de ese grupo en RMSE con evento. Elegir por RMSE global
   puro daría RF sin ubicación, prácticamente equivalente (+1% RMSE con evento).
3. **Queda documentado como decisión abierta:** si el objetivo del prototipo es solo
   rutear en horas con evento, HistGB + línea/tramo es candidato. El experimento de
   ruteo con 30 semillas se hizo con RF + línea/tramo; no se repitió con HistGB.

### 7.2 Métricas del modelo exportado (semilla 42)

RandomForest + línea/tramo con `edad_evento`, entrenado con el 80% inicial. En el 20% final:
MAE 0.0047 min, RMSE 0.1209 min; en las 13 filas con evento: MAE 1.325 min, RMSE 2.564
min. Importancia por permutación: `congestibilidad_t` 0.0436, `tramo_congestion_media`
0.0037, `linea` 0.0014, `severidad_evento` 0.0012, `congestibilidad_t_minus_1` 0.0005,
`hora` 0.0002, `edad_evento` 0.0001; el resto ≈ 0. El caso de prueba Pantitlán →
Auditorio (2026-01-13, 7:00) no cambia: la IA mantiene la ruta estática (37.96 min reales).

### 7.3 Resultados con 30 semillas: ruteo

258 horas con evento en días no vistos, **525,446 casos O-D** (antes 39 horas y 82,784
casos). Solo **53 horas** tienen ahorro posible > 0.5 min. Bootstrap por conglomerados
(5,000 repeticiones sobre horas con evento completas):

| Sistema | Ahorro total (min) [IC 95%] | Pérdidas totales (min) [IC 95%] | Cambios de ruta | % del ahorro posible [IC 95%] |
|---|---|---|---|---|
| Reactivo | 20,154 [−16,622; 57,996] | −29,024 [−50,965; −12,320] | 33,953 | 40.9 [−66.0; 78.8] |
| Anticipatorio (IA) | 23,494 [6,434; 43,875] | **−5,483 [−9,744; −2,433]** | 16,233 | **47.7 [23.2; 67.6]** |
| Oráculo | 49,224 [21,000; 81,549] | 0 | 19,527 | 100 |

| Comparación anticipatorio − reactivo | Puntual | IC 95% | Probabilidad bootstrap |
|---|---|---|---|
| Diferencia de ahorro total | +3,340 min | [−21,618; 28,904] | **60%** de que la IA ahorre más |
| Diferencia de pérdidas | +23,542 min | [7,993; 44,024] | **100%** de que la IA pierda menos |

Por semilla, la IA ahorra más que el reactivo en 18 de 30. En las 5 semillas originales
los totales casi no cambian respecto a la Sección 6.5 (IA 6,307 vs. 6,400 min; pérdidas
−577 vs. −598), y las 25 semillas nuevas repiten el patrón (IA 17,187 vs. reactivo
14,587; pérdidas −4,906 vs. −17,955).

Lectura:

1. **Resultado robusto, ahora con IC: la IA pierde 5.3 veces menos que el reactivo**
   (−5,483 vs. −29,024 min) y el IC de la diferencia excluye el 0. Con más semillas el
   cociente baja de 18× a 5.3×: el 18× de la Sección 6.5 venía de pocas horas.
2. **La IA es el único sistema con % de ahorro capturado significativamente positivo**
   (47.7%, IC [23.2; 67.6]). El IC del reactivo incluye valores muy negativos: según qué
   horas toquen, empeora al usuario en neto.
3. **La ventaja en ahorro total sigue sin ser significativa** (probabilidad 60%, antes
   54%). Sextuplicar la muestra no la resolvió; la causa es la estructura, no el tamaño:
   - En las 53 horas con ahorro posible, el **reactivo gana en 38**, la IA en 9 (6
     empates): mientras la disrupción persiste, reaccionar capta más (37,044 vs. 28,010
     min).
   - En las 205 horas sin ahorro posible (el incidente ya terminó o no abre alternativa),
     el reactivo pierde −16,889 min y la IA −4,516.

   El balance neto depende de cuántas horas de "incidente que sigue" contra "incidente que
   acaba" caigan en la muestra, y eso varía mucho entre semillas.
4. **Para la tesis:** lo defendible es "anticipar reduce el riesgo" (menos pérdidas, %
   capturado con IC positivo), no "anticipar ahorra más minutos que reaccionar".

### 7.4 Edad del evento: efecto nulo

Ablación sobre los 30 datasets: RandomForest + línea/tramo en validación por días, con y
sin `edad_evento` (mismos pliegues):

| Segmento | Filas | RMSE sin edad | RMSE con edad |
|---|---|---|---|
| Global | 561,000 | 0.0859 | 0.0859 |
| Con evento | 933 | 1.1790 | 1.1789 |
| Incidente de plataforma | 194 | 2.581 | 2.581 |
| Incidente, primera hora (edad 0) | 104 | 2.763 | 2.757 |
| Incidente, edad ≥ 1 | 90 | 2.354 | 2.362 |

Persistencia observada de los incidentes y predicción media del modelo:

| Edad (h) | Filas | Sigue activo en t+1 | Retraso real medio en t+1 | Predicción media sin edad | Predicción media con edad |
|---|---|---|---|---|---|
| 0 | 104 | 39% | 1.85 | 1.86 | 1.87 |
| 1 | 58 | 29% | 1.33 | 0.97 | 1.03 |
| 2 | 27 | 15% | 0.35 | 0.61 | 0.61 |
| 3 | 5 | 0% | 0.00 | 0.21 | 0.23 |

("Sigue activo" = retraso real en t+1 > 1 min.)

Lectura:

1. **La variable no aporta nada** (importancia 0.0001; RMSE idéntico). La información ya
   estaba en el modelo: la predicción baja con la edad aun sin `edad_evento`, porque
   `congestibilidad_t_minus_1` distingue un incidente que acaba de empezar (t−1 sin
   retraso) de uno que ya llevaba una hora. Las gradaciones restantes (edad 1 vs. 2 vs. 3)
   tendrían que aprenderse de solo ~19 filas de incidente por semilla.
2. **La hipótesis de la Sección 6.6 era incorrecta.** La IA no pierde contra el reactivo por
   no saber cuánto lleva el incidente: su predicción media ya está calibrada (1.87 vs. 1.85
   en la primera hora). Pierde porque predice el **valor esperado**: con 39% de
   probabilidad de que el incidente siga, el retraso esperado rara vez supera el costo del
   desvío, y la IA no cambia de ruta. El reactivo supone continuidad al 100%: gana cuando
   el incidente sigue y pierde cuando acaba. Para el usuario promedio el valor esperado es
   el criterio correcto (por eso pierde menos). Capturar más ahorro requeriría información
   que el simulador no expone (duración anunciada del cierre) o un criterio de decisión
   distinto del valor esperado.
3. Se conserva `edad_evento` en el modelo: no empeora y sería relevante con duraciones más
   largas o datos reales de incidentes.

### 7.5 Siguiente paso (ejecutado — ver [Sección 8](#8-aviso-de-restablecimiento-y-histgb-en-el-ruteo-2026-09-27))

**Plan.** La Sección 7 dejó dos preguntas abiertas: (a) si la IA pierde ahorro frente al
reactivo por falta de información sobre la duración de los incidentes (Sección 7.4), y
(b) si HistGB + línea/tramo rutea mejor que el modelo fijo (Sección 7.1). Ambas se
responden con el mismo experimento de 30 semillas, así que se atienden juntas. Las
limitaciones #4 y #5 de la Sección 3 quedan para la fase siguiente (punto 5).

1. **Aviso de tiempo estimado de restablecimiento** (`simulador_congestion.py`). Cuando
   hay un incidente o una falla, el STC anuncia un tiempo estimado de restablecimiento;
   el simulador tiene la duración real pero no la expone. Se agregará la columna
   `horas_restantes_anunciadas` (−1 sin evento; la lluvia no la trae):
   - Restante real en t = `hora_inicio + duracion − hora − 1` (0 = el evento termina al
     cerrar esta hora).
   - Aviso = `round(restante_real × exp(ε))`, con `ε ~ N(0, σ)` y mínimo 0. `σ` se
     controla con un argumento `--ruido-aviso` (0 = aviso perfecto).
   - El ruido se sortea con un generador aparte (`default_rng(semilla + 1_000_000)`)
     para no alterar la secuencia de eventos: con la misma semilla, el dataset debe ser
     idéntico al de la Sección 7 salvo la columna nueva. Se verificará comparando las
     demás columnas.
2. **Feature en el entrenador** (`entrenador_anticipatorio.py`). `horas_restantes_anunciadas`
   entra en `FEATURES_BASE` solo si está en el dataset, para que los datasets de la
   Sección 7 sigan siendo entrenables. Se agregará un argumento `--configuracion` que
   sobrescriba `MODELO_ELEGIDO` (p. ej. `HistGradientBoosting+geo`), sin cambiar el valor
   por defecto.
3. **Experimento** (`experimento_semillas.py`). Se agregarán:
   - Un argumento `--variante <nombre>` que escriba en
     `datos_procesados/semillas/<variante>/semilla_<s>/` y pase al simulador y al
     entrenador los argumentos de esa variante, para no sobrescribir los resultados de la
     Sección 7.
   - Un modo `--reusar-dataset` que omita el simulador si el dataset ya existe (las
     variantes que solo cambian el modelo no necesitan resimular; ahorra ~1.5 min por
     semilla).
   - Una comparación **pareada** entre dos variantes: como las horas con evento son las
     mismas (mismas semillas), el bootstrap remuestrea horas y calcula la diferencia de
     ahorro y de pérdidas entre variantes en cada hora. Da un IC más estrecho que
     comparar los IC por separado.

   Variantes a correr (misma lista de 30 semillas):

   | Variante | Simulador | Modelo | Qué responde | Costo aprox. |
   |---|---|---|---|---|
   | `base` | Sección 7 (ya existe) | RF + línea/tramo | Referencia | 0 |
   | `hgb` | reusa `base` | HistGB + línea/tramo | Decisión abierta 7.1 | ~25 min |
   | `aviso_perfecto` | `--ruido-aviso 0` | RF + línea/tramo | Cota: cuánto vale saber la duración exacta | ~37 min |
   | `aviso_ruidoso` | `--ruido-aviso 0.5` | RF + línea/tramo | Valor de un aviso realista | ~37 min |

   `σ = 0.5` significa que, antes de redondear, el aviso queda entre 0.6 y 1.65 veces el
   restante real en ~2 de cada 3 casos. Como el ruido es multiplicativo, un evento que
   termina en esta hora (restante 0) siempre se anuncia bien. Es un supuesto ilustrativo: no hay datos del STC sobre la
   precisión de sus avisos.
4. **Qué se reporta en `avances.md`** (nueva Sección 8), para cada variante contra
   `base`: ahorro total, pérdidas, % del ahorro posible capturado (con IC), diferencia
   pareada con IC y probabilidad bootstrap; y lo mismo contra el reactivo. Criterios de
   lectura:
   - Si `aviso_perfecto` no mejora a `base`, la hipótesis de la Sección 7.4 (la IA pierde
     por predecir el valor esperado sin saber la duración) es falsa y hay que buscar la
     causa en otro lado (p. ej. el costo del desvío o el umbral de decisión).
   - Si `aviso_perfecto` mejora pero `aviso_ruidoso` no, el valor del aviso depende de
     su precisión, y así se reporta.
   - Si `hgb` no gana a `base` en ahorro pareado ni en pérdidas, se cierra la Sección
     7.1 manteniendo RandomForest + línea/tramo.
   - Solo si `aviso_ruidoso` hace que la IA supere al reactivo en ahorro total con
     probabilidad bootstrap ≥ 95%, se actualiza la afirmación de la tesis de "anticipar
     reduce el riesgo" a "anticipar ahorra más".
5. **Fase siguiente (fuera de este plan):** limitación #4 (horizonte de 10–60 min, que
   requiere pasos sub-horarios en el simulador) y #5 (perfiles O-D derivados de la matriz
   de afluencia real). Ambas cambian el dataset de raíz e invalidarían la comparación
   entre variantes, por eso van después.

Tiempo total estimado: ~1 h 40 min de cómputo (3 corridas en paralelo) más el análisis.

## 8. Aviso de restablecimiento y HistGB en el ruteo (2026-09-27)

Ejecución del plan de la Sección 7.5. Cambios de código:

- `simulador_congestion.py`: argumento `--ruido-aviso σ`. Si se pasa, agrega la columna
  `horas_restantes_anunciadas`: restante real = `hora_inicio + duracion − hora − 1`
  (mínimo 0), aviso = `round(restante × exp(ε))`, `ε ~ N(0, σ)`, mínimo 0; −1 sin evento o
  si el evento más severo del tramo es lluvia. El ruido sale de un generador aparte
  (`default_rng(semilla + 1_000_000)`) y se sortea de nuevo cada hora. Sin el argumento
  el dataset es el de la Sección 7. **Verificado:** con la semilla 42 y `σ = 0.5` las 16
  columnas previas son idénticas al dataset de la Sección 7 (`DataFrame.equals`).
- `entrenador_anticipatorio.py`: `horas_restantes_anunciadas` entra en las features solo
  si está en el dataset; la lista de features se guarda en la codificación del `.pkl`
  para que el ruteo arme las mismas columnas. Argumento `--configuracion <modelo>[+geo]`
  que sobrescribe `MODELO_ELEGIDO` (por defecto sigue RandomForest + línea/tramo).
- `experimento_semillas.py`: `--variante` (`base`, `hgb`, `aviso_perfecto`,
  `aviso_ruidoso`) con salida en `datos_procesados/semillas/<variante>/semilla_<s>/`;
  `--reusar-dataset` omite el simulador si el dataset existe (`hgb` usa el de `base`);
  `--comparar A B` hace el bootstrap pareado B − A (5,000 repeticiones sobre horas con
  evento completas) y lo guarda en `modelos/comparacion_pareada_<B>_vs_<A>.csv`. Las
  corridas de la Sección 7 se movieron a `semillas/base/`; al reagregarlas se reproducen
  exactamente las cifras de la Sección 7.3.

Las cuatro variantes evalúan las mismas 258 horas con evento y 525,446 casos O-D; el
reactivo y el oráculo son idénticos en todas (verificado hora por hora), así que toda
diferencia viene del sistema anticipatorio.

### 8.1 Resultados por variante

Bootstrap por conglomerados (horas con evento), IC 95%:

| Variante | Ahorro total IA (min) [IC 95%] | Pérdidas IA (min) [IC 95%] | Cambios de ruta | % del ahorro posible [IC 95%] |
|---|---|---|---|---|
| Reactivo (referencia) | 20,154 [−16,622; 57,996] | −29,024 [−50,965; −12,320] | 33,953 | 40.9 [−66.0; 78.8] |
| `base` (RF + línea/tramo) | 23,494 [6,434; 43,875] | −5,483 [−9,744; −2,433] | 16,233 | 47.7 [23.2; 67.6] |
| `hgb` (HistGB + línea/tramo) | 21,249 [3,056; 43,406] | −8,143 [−14,007; −3,559] | 16,889 | 43.2 [10.3; 66.9] |
| `aviso_perfecto` (σ = 0) | **36,461 [12,815; 63,490]** | −3,920 [−6,217; −2,048] | 18,376 | **74.1 [57.6; 81.9]** |
| `aviso_ruidoso` (σ = 0.5) | **36,548 [13,088; 63,434]** | **−3,334 [−5,175; −1,809]** | 17,251 | **74.2 [59.3; 82.0]** |
| Oráculo | 49,224 [21,000; 81,549] | 0 | 19,527 | 100 |

**Diferencias pareadas contra `base`** (variante − `base`; pérdidas positivas = pierde menos):

| Variante | Δ ahorro (min) [IC 95%] | P(ahorra más) | Δ pérdidas (min) [IC 95%] | P(pierde menos) | Horas mejor / peor / igual |
|---|---|---|---|---|---|
| `hgb` | −2,246 [−12,811; 8,604] | 34% | −2,660 [−5,986; 176] | 3.5% | 48 / 55 / 155 |
| `aviso_perfecto` | **+12,967 [2,198; 27,249]** | **100%** | +1,563 [−74; 4,090] | 95.9% | 31 / 11 / 216 |
| `aviso_ruidoso` | **+13,053 [2,389; 27,233]** | **100%** | **+2,149 [50; 5,613]** | **98.6%** | 27 / 10 / 221 |

**Diferencias contra el reactivo** (anticipatorio de la variante − reactivo):

| Variante | Δ ahorro (min) [IC 95%] | P(IA ahorra más) | Δ pérdidas (min) [IC 95%] | P(IA pierde menos) | Semillas con IA > reactivo |
|---|---|---|---|---|---|
| `base` | +3,340 [−21,618; 28,904] | 60% | +23,542 [7,993; 44,024] | 100% | 18 / 30 |
| `hgb` | +1,095 [−24,913; 27,713] | 54% | +20,882 [5,563; 41,431] | 100% | 18 / 30 |
| `aviso_perfecto` | +16,307 [−1,225; 38,173] | **96.3%** | +25,105 [9,319; 45,876] | 100% | 20 / 30 |
| `aviso_ruidoso` | +16,394 [−1,592; 38,426] | **96.1%** | +25,691 [9,722; 46,733] | 100% | 20 / 30 |

Comparación pareada `aviso_ruidoso` − `aviso_perfecto`: Δ ahorro +87 min [−862; 1,328]
(P = 52%), Δ pérdidas +586 min [−72; 1,669] (P de que el ruidoso pierda menos = 93%);
13 horas mejor, 17 peor, 228 iguales. **No hay diferencia.**

Desglose por tipo de hora (mismo corte que la Sección 7.3):

| Variante | 53 horas con ahorro posible: IA / reactivo (min) | Horas IA gana / reactivo gana | 205 horas sin ahorro posible: IA / reactivo (min) |
|---|---|---|---|
| `base` | 28,010 / 37,044 | 9 / 38 | −4,516 / −16,889 |
| `hgb` | 27,468 / 37,044 | 7 / 40 | −6,219 / −16,889 |
| `aviso_perfecto` | **39,230** / 37,044 | 9 / 36 | −2,769 / −16,889 |
| `aviso_ruidoso` | **38,853** / 37,044 | 9 / 36 | −2,305 / −16,889 |

Modelo (validación por días agregada, RF + línea/tramo): el RMSE con evento baja de 1.179
(`base`) a 1.027 (`aviso_perfecto`) y 1.032 (`aviso_ruidoso`), −13%. En la semilla 42,
`horas_restantes_anunciadas` es la segunda variable por importancia de permutación
(0.028, solo detrás de `congestibilidad_t` con 0.064).

Precisión del aviso ruidoso (1,569 filas de incidente o falla, 30 semillas): 81.1% se
anuncian exactos. Las 820 filas con restante 0 (52%) siempre se anuncian bien por ser
ruido multiplicativo; con restante 1, el 72% se anuncia exacto y 8% se anuncia 0.

### 8.2 Lectura (criterios de la Sección 7.5)

1. **`aviso_perfecto` sí mejora a `base`: la hipótesis de la Sección 7.4 se confirma.** La
   IA ahorra +12,967 min más (IC pareado [2,198; 27,249], P = 100%) y sube del 47.7% al
   74.1% del ahorro posible, con un IC que ya no toca valores bajos ([57.6; 81.9]). La
   mejora viene de los dos lados: en las 53 horas con ahorro posible pasa de 28,010 a
   39,230 min (el aviso le dice cuándo el incidente sigue y vale desviarse) y en las 205
   sin ahorro posible sus pérdidas bajan de −4,516 a −2,769 (el aviso "restante 0" le
   dice que el incidente termina y no conviene desviarse). La IA perdía ahorro por
   predecir el valor esperado sin saber la duración, no por el costo del desvío ni por
   el umbral de decisión.
2. **El valor del aviso no depende de su precisión, al menos con σ = 0.5.** `aviso_ruidoso`
   rinde igual que el perfecto (Δ ahorro +87 min, P = 52%) y hasta pierde un poco menos
   (−3,334 vs. −3,920; P = 93%, no significativo). La explicación está en la estructura
   del aviso: la información que decide la ruta es sobre todo si el incidente **termina
   en esta hora o no** (restante 0 vs. ≥ 1), y esa distinción sobrevive al ruido: el 0
   se anuncia siempre bien y un restante 1 solo se confunde con 0 en 8% de los casos.
   Con incidentes de 1–4 horas, errar entre "quedan 2" y "quedan 3" no cambia la
   decisión de t+1. **Limitación:** esta robustez es en parte una propiedad del supuesto
   (ruido multiplicativo, que nunca se equivoca en el 0). Un aviso real puede anunciar
   "15 minutos" para un cierre que dura una hora; ese error aditivo no se probó.
   **Actualización (Sección 9):** se probó; con `δ ∈ {−1, 0, +1}` el aviso pierde casi
   todo su valor, así que esta robustez sí era un artefacto del supuesto multiplicativo.
3. **`hgb` no gana a `base` en ahorro pareado ni en pérdidas: se cierra la Sección 7.1
   manteniendo RandomForest + línea/tramo.** HistGB ahorra −2,246 min menos (P de que
   ahorre más = 34%) y pierde −2,660 min más (P de que pierda menos = 3.5%, es decir, 96.5%
   de que pierda más). Su mejor RMSE con evento (1.085 vs. 1.179) no se traduce en mejores
   rutas: el error extra en filas sin evento (RMSE global 30% peor) mete retrasos
   espurios que desvían sin necesidad (16,889 cambios de ruta vs. 16,233).
4. **Con `aviso_ruidoso` la IA supera al reactivo en ahorro total con P = 96.1% (≥ 95%).**
   Por el criterio fijado en la Sección 7.5 se actualiza la afirmación de la tesis de
   "anticipar reduce el riesgo" a **"anticipar ahorra más, si el sistema recibe el aviso
   de restablecimiento"**, con tres matices que deben acompañarla:
   - El margen es estrecho: el IC 95% de dos colas de la diferencia ([−1,592; 38,426])
     todavía incluye el 0; P = 96% equivale a una prueba de una cola al 5%. Es la
     formulación del criterio, pero no se debe presentar como diferencia "clara".
   - En horas con ahorro posible el reactivo sigue ganando en más horas (36 vs. 9); la IA
     gana en total porque sus aciertos valen más y porque casi no pierde en las 205 horas
     sin ahorro posible (−2,305 vs. −16,889).
   - La afirmación depende de que exista el aviso con precisión similar a la simulada.
     Sin aviso (`base`), la conclusión de la Sección 7.3 sigue en pie (P = 60%).
   La ventaja en pérdidas se mantiene en todas las variantes (P = 100%, IC excluye 0).

### 8.3 Siguiente paso (punto 1 ejecutado — ver [Sección 9](#9-aviso-con-error-aditivo-2026-09-27))

1. **Probar un aviso con error aditivo** (p. ej. `aviso = max(restante + δ, 0)`,
   `δ ∈ {−1, 0, +1}`), que sí puede anunciar 0 cuando el incidente sigue, para acotar
   cuánto de la robustez del punto 2 de la Sección 8.2 viene del supuesto multiplicativo.
2. Con el aviso incorporado, retomar la fase siguiente de la Sección 7.5 (punto 5):
   limitación #4 (horizonte sub-horario de 10–60 min) y #5 (perfiles O-D de la matriz de
   afluencia real). El horizonte sub-horario es además donde el aviso debería rendir más,
   porque la granularidad de 1 hora reduce el aviso casi a una variable binaria.
3. Documentar en el capítulo de metodología las fuentes del aviso (avisos del STC en
   redes sociales o la app oficial) como insumo requerido para un despliegue real.

## 9. Aviso con error aditivo (2026-09-27)

Ejecución del punto 1 de la Sección 8.3. La Sección 8.2 (punto 2) atribuyó la robustez
del aviso a que el ruido multiplicativo nunca se equivoca cuando el incidente termina en
esta hora (restante 0). Aquí se prueba un error que sí se equivoca en ese caso.

Cambios de código:

- `simulador_congestion.py`: argumento `--error-aviso-aditivo K`. Suma al aviso un error
  entero `δ` uniforme en `{−K, …, K}`: `aviso = max(round(restante × exp(ε)) + δ, 0)`. Se
  combina con `--ruido-aviso` (σ = 0 si no se pasa) y usa el mismo generador aparte
  (`default_rng(semilla + 1_000_000)`), así que no toca los sorteos de eventos.
- `experimento_semillas.py`: variante `aviso_aditivo` = `--error-aviso-aditivo 1`
  (`δ ∈ {−1, 0, +1}` con probabilidad 1/3 cada uno, sin ruido multiplicativo).

Pruebas (semilla 42):

- Sin argumentos de aviso, el dataset es idéntico al de `base` (`DataFrame.equals`).
- Con `--ruido-aviso 0.5`, el dataset es idéntico al de `aviso_ruidoso` de la Sección 8:
  el cambio no altera la variante anterior.
- Con `--error-aviso-aditivo 1`, las 16 columnas previas son idénticas a `base`; el aviso
  difiere del restante real en −1, 0 o +1, nunca es negativo, y la lluvia y los tramos
  sin evento quedan en −1.

**Precisión del aviso** (1,569 filas de incidente o falla, 30 semillas):

| Restante real | Anunciado 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 556 | 264 | — | — | — |
| 1 | 155 | 166 | 175 | — | — |
| 2 | — | 72 | 58 | 64 | — |
| 3 | — | — | 26 | 22 | 11 |

Solo 51.1% de los avisos son exactos (81.1% con `aviso_ruidoso`). Más importante: la
distinción "termina en esta hora / sigue" se anuncia bien en 73.3% de los casos, contra
97.4% con `aviso_ruidoso` (ahí solo fallan las 41 filas con restante 1 anunciadas como
0). Con error aditivo, un tercio de los incidentes que terminan se anuncian como "queda
1 hora" y un tercio de los que tienen restante 1 se anuncian como "termina".

### 9.1 Resultados

| Variante | Ahorro total IA (min) [IC 95%] | Pérdidas IA (min) [IC 95%] | Cambios de ruta | % del ahorro posible [IC 95%] | P(IA ahorra más que reactivo) |
|---|---|---|---|---|---|
| `base` (sin aviso) | 23,494 [6,434; 43,875] | −5,483 [−9,744; −2,433] | 16,233 | 47.7 [23.2; 67.6] | 60% |
| `aviso_perfecto` | 36,461 [12,815; 63,490] | −3,920 [−6,217; −2,048] | 18,376 | 74.1 [57.6; 81.9] | 96.3% |
| `aviso_ruidoso` (σ = 0.5) | 36,548 [13,088; 63,434] | −3,334 [−5,175; −1,809] | 17,251 | 74.2 [59.3; 82.0] | 96.1% |
| `aviso_aditivo` (δ ∈ {−1, 0, +1}) | 24,606 [6,615; 46,535] | −5,869 [−10,056; −2,730] | 17,624 | 50.0 [23.7; 69.0] | **63.9%** |

**Diferencias pareadas de `aviso_aditivo`** (`aviso_aditivo` − referencia):

| Referencia | Δ ahorro (min) [IC 95%] | P(ahorra más) | Δ pérdidas (min) [IC 95%] | P(pierde menos) | Horas mejor / peor / igual |
|---|---|---|---|---|---|
| `base` | +1,112 [−809; 4,040] | 79.6% | −386 [−1,193; 179] | 14.7% | 25 / 18 / 215 |
| `aviso_perfecto` | **−11,855 [−25,089; −2,169]** | 0.0% | **−1,949 [−4,384; −243]** | 0.5% | 13 / 32 / 213 |
| `aviso_ruidoso` | **−11,942 [−25,332; −2,084]** | 0.0% | **−2,535 [−6,032; −342]** | 0.2% | 15 / 29 / 214 |
| Reactivo | +4,452 [−19,799; 29,146] | 63.9% | +23,156 [7,713; 43,342] | 100% | — |

Desglose por tipo de hora (mismo corte que las Secciones 7.3 y 8.1):

| Variante | 53 horas con ahorro posible: IA / reactivo (min) | 205 horas sin ahorro posible: IA (min) | Semillas con IA > reactivo |
|---|---|---|---|
| `base` | 28,010 / 37,044 | −4,516 | 18 / 30 |
| `aviso_perfecto` | 39,230 / 37,044 | −2,769 | 20 / 30 |
| `aviso_ruidoso` | 38,853 / 37,044 | −2,305 | 20 / 30 |
| `aviso_aditivo` | 29,486 / 37,044 | −4,880 | 18 / 30 |

Modelo (validación por días agregada, RF + línea/tramo): RMSE con evento 1.105, entre
`base` (1.179) y `aviso_perfecto` (1.027). En la semilla 42 la importancia por
permutación del aviso cae a 0.0037 (0.028 con `aviso_perfecto`), tercera variable,
apenas por encima de `severidad_evento`.

### 9.2 Lectura

1. **La robustez de la Sección 8.2 (punto 2) venía del supuesto multiplicativo.** Un error
   de solo ±1 hora, sin sesgo, borra casi todo el valor del aviso: la IA pasa de 74% a
   50% del ahorro posible, pierde −11,900 min frente a las dos variantes anteriores (IC
   pareados que excluyen el 0, P = 0%) y queda estadísticamente igual que sin aviso
   (+1,112 min frente a `base`, IC [−809; 4,040], P = 80%). En pérdidas incluso queda
   ligeramente peor que `base` (−386 min, no significativo).
2. **El valor del aviso depende de su precisión, y así se reporta** (criterio de la
   Sección 7.5). La precisión que importa no es la del número de horas sino la de la
   distinción "el incidente termina en esta hora / sigue": con 97% de acierto
   (`aviso_ruidoso`) el aviso vale lo mismo que uno perfecto; con 73% (`aviso_aditivo`)
   casi no vale. El modelo aprende a desconfiar del aviso (su importancia cae de 0.028 a
   0.004) y vuelve a apoyarse en `congestibilidad_t` y `congestibilidad_t_minus_1`, es
   decir, al comportamiento de `base`. Es consistente con el mecanismo de la Sección 7.4:
   el aviso solo ayuda si mueve la probabilidad de que el incidente siga lejos del ~39%
   que el modelo ya estima por sí solo.
3. **La afirmación de la Sección 8.2 (punto 4) se acota.** "Anticipar ahorra más" solo se
   sostiene si el aviso de restablecimiento acierta casi siempre si el incidente termina
   en la hora en curso. Con un aviso de ±1 hora, la IA vuelve a no superar al reactivo
   en ahorro total (P = 64%). Para la tesis, la formulación defendible queda:
   - Sin aviso o con aviso impreciso: **anticipar reduce el riesgo** (pérdidas 5× menores,
     P = 100% en todas las variantes).
   - Con un aviso que distingue con fiabilidad "termina / sigue": **anticipar además
     ahorra más** (P ≈ 96%, con el matiz de IC de dos colas de la Sección 8.2).
   - No hay datos del STC sobre la precisión de sus avisos, así que no se puede afirmar
     en cuál de los dos casos está la red real.
4. **Granularidad.** Un error de ±1 hora es grande respecto a incidentes que duran 1–4
   horas (media 2). Con pasos sub-horarios (limitación #4) el mismo aviso real ("15 min")
   tendría un error relativo menor; es otra razón para priorizar esa limitación.

### 9.3 Siguiente paso (punto 1 ejecutado — ver [Sección 11](#11-sensibilidad-del-aviso-reactivo-con-regla-de-duración-y-hora--1-por-tipo-2026-09-27))

1. **Sensibilidad a la probabilidad de error del aviso:** variar la fracción de avisos
   con `δ ≠ 0` (p. ej. 10%, 20%, 33%) para encontrar a partir de qué precisión de
   "termina / sigue" la IA supera al reactivo con P ≥ 95%. Da un requisito concreto de
   calidad del aviso para un despliegue real.
2. Sin cambios respecto a la Sección 8.3: limitación #4 (horizonte sub-horario) y #5
   (perfiles O-D reales), y documentar las fuentes del aviso en el capítulo de
   metodología.

## 10. Ruteo por edad del evento: hora 0, + 1 y + 2 (2026-09-27)

Análisis pedido fuera del plan de la Sección 9.3, que sigue pendiente. Pregunta: ¿la IA
supera al reactivo de forma más evidente en la **hora 0** del evento, cuando menos se
sabe si el incidente va a seguir?

Cambios de código:

- `ruteo_anticipatorio.py`: cada caso O-D guarda `edad_evento` = edad del evento que
  cruza su ruta estática (0 = primera hora del evento, 1 = hora + 1, ...; si cruza varios
  tramos afectados, el evento más reciente). Nuevo resumen restringido a las edades 0, 1
  y 2 (`--resumen-edad`, `resumen_ruteo_por_edad.csv`); los casos con edad ≥ 3 se
  excluyen y se cuentan aparte.
- `experimento_semillas.py`: `agregar_ruteo_por_edad` hace el bootstrap por
  conglomerados (5,000 repeticiones) por separado para cada edad, con un conglomerado por
  (hora con evento, edad); se ejecuta al final de cada corrida y se guarda en
  `modelos/resultados_semillas_por_edad[_<variante>].csv`. El bootstrap se factorizó
  (`estadisticos_ruteo`, `bootstrap_ruteo`) para compartirlo con `agregar_ruteo`. Nuevo
  modo `--solo-ruteo` que reevalúa el ruteo con los datasets y modelos ya guardados.

Pruebas: con la semilla 42 (`base`) la evaluación nueva es idéntica a la anterior en todas
sus columnas salvo `edad_evento`, el resumen global es idéntico y la suma por edad cuadra
con el total. Tras reevaluar las 5 variantes × 30 semillas con `--solo-ruteo`, los totales
reproducen exactamente las Secciones 7.3, 8.1 y 9.1.

Reparto de los casos (30 semillas): hora 0 = 144 conglomerados y 281,616 casos; hora + 1 =
86 y 173,334; hora + 2 = 37 y 63,230; edad ≥ 3 = 5 y 7,266 (excluidos). Ahorro posible
(oráculo): 26,591, 22,521 y 112 min.

Una aclaración de definición: en la hora 0 el evento ya está activo en `t`, así que el
reactivo **ya lo ve** en `congestibilidad_t`. Ningún sistema puede anticipar el arranque
de un evento: el dataset no trae ninguna señal previa a `hay_evento = 1`. Lo que se decide
en la hora 0 es si el evento seguirá en `t + 1`.

### 10.1 Resultados

`base` (RF + línea/tramo, sin aviso):

| Edad | Ahorro IA (min) | Ahorro reactivo | Δ IA − reactivo [IC 95%] | P(IA ahorra más) | Pérdidas IA / reactivo |
|---|---|---|---|---|---|
| Hora 0 | 17,095 | 19,174 | −2,079 [−11,429; 5,696] | 34.3% | −4,111 / −7,402 |
| Hora + 1 | 7,262 | 19,887 | −12,624 [−30,100; 247] | **2.9%** | −481 / −2,629 |
| Hora + 2 | −863 | −18,852 | **+17,989 [3,612; 37,042]** | **99.9%** | −890 / −18,938 |

P(IA ahorra más que el reactivo) por variante:

| Edad | `base` | `hgb` | `aviso_perfecto` | `aviso_ruidoso` | `aviso_aditivo` |
|---|---|---|---|---|---|
| Hora 0 | 34.3% | 11.8% | 68.0% | 70.9% | 37.0% |
| Hora + 1 | 2.9% | 4.2% | 12.1% | 10.9% | 3.0% |
| Hora + 2 | 99.9% | 99.9% | 100% | 100% | 100% |

Ahorro IA en la hora 0 y la hora + 1 (el reactivo es igual en todas: 19,174 y 19,887 min):

| Edad | `base` | `hgb` | `aviso_perfecto` | `aviso_ruidoso` | `aviso_aditivo` |
|---|---|---|---|---|---|
| Hora 0 | 17,095 | 16,111 | 20,742 | 21,124 | 17,957 |
| Hora + 1 | 7,262 | 6,158 | 16,391 | 16,057 | 7,486 |

### 10.2 Lectura

1. **No: la IA no supera al reactivo en la hora 0.** Sin aviso, el reactivo ahorra más
   (19,174 vs. 17,095 min; P de que la IA ahorre más = 34%). Ni con el aviso perfecto la
   ventaja es significativa en esa hora (+1,568 min, IC [−4,368; 8,244], P = 68%). Lo que
   sí hace la IA en la hora 0 es perder menos (−4,111 vs. −7,402 min).
2. **La hora + 1 es la peor para la IA.** El reactivo captura 88% del ahorro posible
   (19,887 de 22,521 min) y la IA sin aviso solo 32% (7,262); P = 2.9% de que la IA
   ahorre más. En esa hora hay casi tanto ahorro posible como en la hora 0: los eventos
   que ya duraron una hora siguen con frecuencia suficiente para que desviarse valga la
   pena, y el reactivo, que supone continuidad, acierta. La IA sin aviso predice el valor
   esperado y rara vez se desvía (Sección 7.4). El aviso (perfecto o ruidoso) es lo que
   más la ayuda aquí: de 7,262 a ~16,000 min, aunque sigue por debajo del reactivo.
3. **Toda la ventaja agregada de la IA viene de la hora + 2.** Ahí el ahorro posible es
   casi nulo (112 min: los eventos ya terminaron o terminan), el reactivo se sigue
   desviando por congestión que ya no estará y pierde −18,852 min; la IA casi no se
   mueve (−863). Esa sola hora aporta +17,989 min a favor de la IA (P = 99.9%) y es
   significativa en las 5 variantes. Coincide con la Sección 7.3: la IA gana por no
   perder cuando el incidente acaba, no por ganar mientras sigue.
4. **Para la tesis.** La ventaja de anticipar no está en el arranque del evento sino en su
   final: el valor de la IA es saber cuándo **dejar** de evitar un tramo. La formulación
   "anticipar reduce el riesgo" (Secciones 7.3 y 9.2) se precisa así: reduce el riesgo de
   seguir desviando a los usuarios cuando la disrupción ya terminó. En las dos primeras
   horas el reactivo es igual o mejor, y el aviso de restablecimiento solo cierra parte
   de la brecha en la hora + 1.
5. **Salvedad.** La hora + 2 descansa en 37 conglomerados y su diferencia la domina el
   error del reactivo, no un acierto de la IA. En la hora + 2 el evento sigue activo en
   `t`, así que un umbral sobre la congestión observada no cambiaría la decisión del
   reactivo; sí la cambiaría una regla simple de duración (p. ej. no desviar por eventos
   con 2 h o más de antigüedad), que captaría esta ventaja sin modelo. No se probó.
   **Actualización (Sección 11):** se probó; la regla anula las pérdidas del reactivo en la
   hora + 2 y supera a la IA en total (P de que la IA ahorre más = 2.3% sin aviso, 28% con
   aviso perfecto).

### 10.3 Siguiente paso (ejecutado — ver [Sección 11](#11-sensibilidad-del-aviso-reactivo-con-regla-de-duración-y-hora--1-por-tipo-2026-09-27))

1. Pendiente de la Sección 9.3: sensibilidad a la probabilidad de error del aviso.
2. **Reactivo más fuerte:** un reactivo con una regla de duración (ignorar eventos con
   edad ≥ 2), para ver cuánto de la ventaja de la hora + 2 sobrevive contra un baseline
   menos ingenuo.
3. Desglosar la hora + 1 por tipo de evento (lluvia, falla, incidente): la lluvia afecta
   líneas completas y puede dominar el número de casos.

## 11. Sensibilidad del aviso, reactivo con regla de duración y hora + 1 por tipo (2026-09-27)

Ejecución de los tres puntos de la Sección 10.3.

Cambios de código:

- `simulador_congestion.py`: argumento `--prob-error-aviso p`. El aviso es exacto salvo en
  una fracción `p` de los casos, en los que `δ = −1` o `+1` con igual probabilidad:
  `aviso = max(restante + δ, 0)`. Usa el mismo generador aparte que las Secciones 8 y 9;
  es excluyente con `--error-aviso-aditivo`.
- `ruteo_anticipatorio.py`:
  - Nuevo sistema `reactivo_duracion`, un reactivo con **regla de duración**. En los tramos
    cuyo evento tiene edad ≥ 2 supone que el evento ya no sigue y cambia la congestión
    observada por el **perfil histórico sin evento** del tramo: la media de
    `congestibilidad_t` en las filas de entrenamiento sin evento con el mismo tramo, tipo
    de día y hora. En el resto de los tramos es idéntico al reactivo. Solo usa días de
    entrenamiento.
  - Cada caso O-D guarda además `tipo_evento`, el tipo del evento más reciente que cruza su
    ruta estática.
- `experimento_semillas.py`:
  - Variantes `aviso_error_10`, `aviso_error_20` y `aviso_error_33` (`p` = 0.10, 0.20 y
    0.33).
  - El bootstrap agrega las diferencias IA − `reactivo_duracion` y `reactivo_duracion` −
    reactivo y sus probabilidades.
  - Nueva `agregar_ruteo_por_tipo`, con un conglomerado por (hora con evento, edad, tipo),
    que se guarda en `modelos/resultados_semillas_por_tipo[_<variante>].csv`.
  - `--variante` acepta varias variantes seguidas.
  - `--sensibilidad-aviso` arma la tabla de la Sección 11.1, con la precisión del aviso
    medida contra el restante real de `aviso_perfecto`, y la guarda en
    `modelos/resultados_sensibilidad_aviso.csv`.

Ejecución:

```
python experimento_semillas.py --solo-ruteo --variante base hgb aviso_perfecto aviso_ruidoso aviso_aditivo
python experimento_semillas.py --variante aviso_error_10 aviso_error_20 aviso_error_33
python experimento_semillas.py --sensibilidad-aviso
python experimento_semillas.py --comparar aviso_perfecto aviso_error_<p>   # p = 10, 20, 33
```

Pruebas:

- Semilla 42 con `--prob-error-aviso 0.2`: las 16 columnas de `base` son idénticas
  (`DataFrame.equals`). El aviso difiere del de `aviso_perfecto` en 17% de las filas de
  incidente o falla. No llega al 20% porque `δ = −1` sobre un restante 0 queda recortado
  a 0.
- Semilla 42 (`base`): la evaluación nueva es idéntica a la anterior en todas sus columnas;
  solo se agregan `tipo_evento` y las dos columnas de `reactivo_duracion`.
- Al reevaluar las 5 variantes previas × 30 semillas, los totales de IA, reactivo y oráculo
  reproducen exactamente las Secciones 7.3, 8.1, 9.1 y 10.1.
- `reactivo_duracion` es idéntico en todas las variantes, porque no depende del modelo. En
  las horas 0 y + 1 coincide con el reactivo, porque la regla no se activa.

### 11.1 Sensibilidad a la probabilidad de error del aviso

Precisión del aviso sobre 1,569 filas de incidente o falla (30 semillas). IA con RF +
línea/tramo, bootstrap por conglomerados con IC 95%:

| Variante | P(δ ≠ 0) | Aviso exacto | "Termina / sigue" correcto | Ahorro IA (min) [IC 95%] | % del ahorro posible [IC 95%] | P(IA ahorra más que reactivo) | RMSE con evento |
|---|---|---|---|---|---|---|---|
| `base` (sin aviso) | — | — | — | 23,494 [6,434; 43,875] | 47.7 [23.2; 67.6] | 60.4% | 1.179 |
| `aviso_perfecto` | 0 | 100% | 100% | 36,461 [12,815; 63,490] | 74.1 [57.6; 81.9] | **96.3%** | 1.027 |
| `aviso_error_10` | 0.10 | 93.6% | 95.9% | 36,416 [12,877; 63,572] | 74.0 [57.3; 82.0] | **96.1%** | 1.038 |
| `aviso_error_20` | 0.20 | 86.1% | 92.4% | 30,896 [8,535; 56,374] | 62.8 [35.2; 78.1] | 83.7% | 1.072 |
| `aviso_error_33` | 0.33 | 77.9% | 87.1% | 23,804 [5,638; 45,542] | 48.4 [20.4; 69.5] | 62.0% | 1.102 |
| `aviso_aditivo` (Sección 9) | 0.67 | 51.1% | 73.3% | 24,606 [6,615; 46,535] | 50.0 [23.7; 69.0] | 63.9% | 1.105 |

Como referencia, `aviso_ruidoso` (Sección 8, error multiplicativo) acierta "termina /
sigue" en 97.4% y rinde igual que el aviso perfecto (74.2%, P = 96.1%).

**Diferencias pareadas contra `aviso_perfecto`** (variante − `aviso_perfecto`):

| Variante | Δ ahorro (min) [IC 95%] | P(ahorra más) | Δ pérdidas (min) [IC 95%] | Horas mejor / peor / igual |
|---|---|---|---|---|
| `aviso_error_10` | −45 [−743; 550] | 47.9% | +163 [−51; 398] | 15 / 16 / 227 |
| `aviso_error_20` | **−5,566 [−15,380; −160]** | 0.6% | −1,048 [−3,159; 159] | 7 / 20 / 231 |
| `aviso_error_33` | **−12,657 [−27,676; −2,043]** | 0.0% | −1,968 [−4,634; 89] | 10 / 28 / 220 |

### 11.2 Reactivo con regla de duración

Totales sobre las 258 horas con evento y 525,446 casos. `reactivo_duracion` es el mismo en
todas las variantes:

| Sistema | Ahorro total (min) [IC 95%] | Pérdidas (min) [IC 95%] | Cambios de ruta | % del ahorro posible [IC 95%] |
|---|---|---|---|---|
| Reactivo | 20,154 [−16,622; 57,996] | −29,024 [−50,965; −12,320] | 33,953 | 40.9 [−66.0; 78.8] |
| **Reactivo + regla de duración** | **39,069 [9,850; 72,775]** | −10,031 [−17,069; −4,180] | 26,766 | **79.4 [44.2; 92.5]** |
| IA `base` | 23,494 [6,434; 43,875] | −5,483 [−9,744; −2,433] | 16,233 | 47.7 [23.2; 67.6] |
| IA `aviso_perfecto` | 36,461 [12,815; 63,490] | −3,920 [−6,217; −2,048] | 18,376 | 74.1 [57.6; 81.9] |
| Oráculo | 49,224 [21,000; 81,549] | 0 | 19,527 | 100 |

La regla sola mejora al reactivo en +18,915 min (IC [3,291; 40,303], P = 100%).

**IA − reactivo con regla de duración:**

| Variante | Δ ahorro (min) [IC 95%] | P(IA ahorra más) | Δ pérdidas (min) [IC 95%] | P(IA pierde menos) |
|---|---|---|---|---|
| `base` | **−15,575 [−35,180; −261]** | **2.3%** | +4,548 [75; 9,959] | 97.7% |
| `hgb` | **−17,821 [−39,496; −2,055]** | 0.9% | +1,888 [−2,111; 6,129] | 81.4% |
| `aviso_perfecto` | −2,608 [−11,689; 6,108] | 28.3% | +6,111 [853; 12,315] | 99.2% |
| `aviso_ruidoso` | −2,522 [−12,135; 6,800] | 30.1% | +6,697 [980; 13,430] | 99.4% |
| `aviso_error_10` | −2,654 [−11,886; 6,170] | 28.1% | +6,274 [903; 12,545] | 99.3% |
| `aviso_error_20` | −8,174 [−22,211; 3,330] | 8.7% | +5,063 [351; 10,653] | 98.7% |
| `aviso_error_33` | −15,266 [−35,683; 85] | 2.7% | +4,144 [−1,048; 10,121] | 93.9% |
| `aviso_aditivo` | **−14,463 [−33,092; −54]** | 2.4% | +4,163 [−18; 9,167] | 97.5% |

Por edad del evento (`base`):

| Edad | Ahorro posible | Reactivo | Reactivo + regla | IA `base` | IA `aviso_perfecto` | P(IA `base` > regla) |
|---|---|---|---|---|---|---|
| Hora 0 | 26,591 | 19,174 | 19,174 | 17,095 | 20,742 | 34.3% |
| Hora + 1 | 22,521 | 19,887 | 19,887 | 7,262 | 16,391 | 2.9% |
| Hora + 2 | 112 | −18,852 | **9** | −863 | −672 | **0.1%** |

En la hora + 2 la regla no pierde nada (pérdidas 0 contra −890 de la IA `base`) y supera a
la IA en todas las variantes (P entre 0.0% y 1.7% de que la IA ahorre más).

### 11.3 Hora + 1 por tipo de evento

`base`, 30 semillas. El tipo es el del evento más reciente que cruza la ruta estática del
caso. Si un tramo tiene varios eventos a la vez, el dataset guarda el más severo.

| Tipo | Horas | Casos (% de la hora + 1) | Horas con ahorro posible | Ahorro posible | Reactivo | IA `base` | IA `aviso_perfecto` | Cambios de ruta reactivo / IA `base` |
|---|---|---|---|---|---|---|---|---|
| Falla mecánica | 55 | 89,838 (51.8%) | 4 | 23 | 11 | 2 | 17 | 743 / 57 |
| Incidente de plataforma | 29 | 57,628 (33.2%) | 10 | **22,499 (99.9%)** | 19,875 | 7,262 | 16,375 | 9,687 / 2,201 |
| Lluvia | 2 | 25,868 (14.9%) | 0 | 0 | 0 | −1 | −1 | 12 / 26 |

El mismo patrón se ve en la hora 0 y la hora + 2. En la hora 0, la lluvia son 3 horas y
37,290 casos (13%) con ahorro posible de 1 min, y la falla son 157,216 casos (56%) con 49
min. En la hora + 2 no hay lluvia; la falla suma 21 min de ahorro posible y el incidente
91.

### 11.4 Lectura

1. **Requisito de calidad del aviso: al menos ~96% de acierto en "termina / sigue".** La
   IA supera al reactivo con P ≥ 95% solo si el aviso se equivoca en ±1 hora en 10% de los
   casos o menos (95.9% de acierto en "termina / sigue"). Con 10% de error rinde igual que
   el aviso perfecto (Δ −45 min, P = 48%). Con 20% de error (92.4% de acierto) pierde
   −5,566 min frente al perfecto (IC excluye 0) y P baja a 84%. Con 33% o más de error
   queda igual que sin aviso (48–50% del ahorro posible). La caída es abrupta entre 10% y
   20%, no gradual. Es consistente con la Sección 9.2: el aviso solo sirve si mueve la
   probabilidad de continuación lejos del ~39% que el modelo ya estima. Unos puntos de
   error bastan para que el modelo lo descuente, y el RMSE con evento sube de forma
   monótona (1.027 → 1.038 → 1.072 → 1.102). **Requisito para un despliegue real:** el
   aviso del STC debe acertar si el servicio se restablece en la hora en curso en al menos
   ~96% de los casos. En pasos de 1 hora, eso equivale a errar la hora de restablecimiento
   en no más de 1 de cada 10 avisos. No hay datos para saber si el STC cumple ese nivel.
2. **La ventaja de la IA en la hora + 2 no sobrevive a un reactivo menos ingenuo.** Con la
   regla de duración, el reactivo pasa de 20,154 a 39,069 min (79.4% del ahorro posible) y
   supera a la IA sin aviso en −15,575 min (IC [−35,180; −261], P = 2.3% de que la IA
   ahorre más). La regla elimina las pérdidas del reactivo en la hora + 2 (−18,852 → +9
   min) sin modelo alguno. Allí también le gana a la IA, que pierde −863 min. Es decir, la
   ventaja de la IA que la Sección 10.2 atribuyó al final del evento se explica por
   completo por el error del reactivo ingenuo. Una regla fija de una línea la captura
   mejor.
3. **Ni con aviso perfecto la IA supera al reactivo con regla.** La brecha se reduce a
   −2,608 min (P = 28%) y no es significativa. La IA gana en la hora 0 (+1,568, P = 68%) y
   pierde en la hora + 1 (−3,496, P = 12%) y en la hora + 2 (−680). Con aviso de 20% de
   error o peor, la regla vuelve a ganar con claridad.
4. **Lo que sí sobrevive: la IA pierde menos.** Contra la regla, la IA `base` pierde −5,483
   min frente a −10,031 (P = 97.7% de que pierda menos). Con un aviso preciso, la ventaja
   es mayor (+6,100 a +6,700 min, P ≈ 99%, IC que excluye 0). No es robusta en todas las
   variantes: con `hgb` (P = 81%), `aviso_error_33` y `aviso_aditivo` el IC toca el 0. Las pérdidas de la regla están todas en las horas 0 y + 1 (−7,402 y −2,629),
   donde la regla no se activa: se desvía por eventos jóvenes que terminan antes de la
   hora siguiente. Esta ventaja es más chica que la reportada contra el reactivo ingenuo
   (+4,548 min en vez de +23,542) y su IC queda cerca de 0 ([75; 9,959]).
5. **La hora + 1 es un problema de incidentes de plataforma, no de lluvia.** La lluvia es
   el 15% de los casos de la hora + 1 y no tiene ahorro posible: afecta líneas completas,
   así que no hay ruta alternativa que la evite. Las fallas mecánicas son el 52% de los
   casos, con 23 min de ahorro posible. Los incidentes de plataforma, con 33% de los
   casos, concentran el 99.9% del ahorro posible (22,499 de 22,521 min). Toda la brecha de
   la IA en la hora + 1 (7,262 vs. 19,875 min del reactivo) está en ese tipo. Ahí la IA sin
   aviso casi no se desvía (2,201 cambios de ruta vs. 9,687), y con aviso perfecto llega a
   16,375 min, todavía por debajo del reactivo. La hipótesis de la Sección 10.3 (la
   lluvia domina el número de casos) no se confirma. La lluvia pesa en casos, pero en
   minutos es irrelevante, y solo aparece en 2–3 horas del set de prueba.
6. **Para la tesis.** La formulación de las Secciones 9.2 y 10.2 se acota otra vez:
   - Contra un reactivo ingenuo, anticipar reduce el riesgo (P = 100%) y, con un aviso que
     acierte ≥ 96% en "termina / sigue", además ahorra más (P ≈ 96%).
   - Contra un reactivo con una regla simple de duración, la IA **no ahorra más** en
     ninguna variante (P ≤ 30%). Sin aviso ahorra significativamente menos. Su única
     ventaja es perder menos en las primeras dos horas del evento (P ≈ 98–99% con RF, con o
     sin aviso preciso; no significativa con `hgb` ni con avisos de 33% de error o más).
   - El baseline de comparación debe ser el reactivo con regla de duración, no el reactivo
     ingenuo. Reportar solo el segundo sobrestima el valor de la IA.
   - **Salvedad:** el umbral de la regla (edad ≥ 2) se eligió después de ver la Sección 10
     y coincide con la duración media simulada de los incidentes (2 h). En la red real
     habría que fijarlo con la distribución histórica de duraciones. Aun así, un operador
     conoce esa distribución, así que la regla no usa información que un sistema real no
     tendría.
     **Actualización (Sección 12):** la ventaja de la regla solo existe con edad ≥ 2; con
     ≥ 1 o ≥ 3 rinde como el reactivo ingenuo (39–41% del ahorro posible).

### 11.5 Siguiente paso (puntos 1 y 2 ejecutados — ver [Sección 12](#12-sistema-híbrido-y-sensibilidad-del-umbral-de-la-regla-2026-09-27))

1. **Sistema híbrido:** la IA en los tramos con evento joven (edad 0–1) y la regla de
   duración en edad ≥ 2. Por construcción combinaría las menores pérdidas de la IA en las
   primeras horas con las nulas pérdidas de la regla al final. Es el candidato natural a
   superar a ambos, y se puede evaluar con `--solo-ruteo` sin reentrenar.
2. **Sensibilidad del umbral de la regla** (edad ≥ 1, ≥ 2, ≥ 3), para ver si el resultado
   depende de haber elegido el umbral que coincide con la duración media simulada.
3. Sin cambios respecto a la Sección 9.3: limitación #4 (horizonte sub-horario, donde el
   requisito de ~96% en "termina / sigue" se traduce a minutos) y #5 (perfiles O-D reales),
   y documentar las fuentes del aviso en el capítulo de metodología.

## 12. Sistema híbrido y sensibilidad del umbral de la regla (2026-09-27)

Ejecución de los puntos 1 y 2 de la Sección 11.5. No se resimuló ni se reentrenó: todo sale
de reevaluar el ruteo (`--solo-ruteo`) con los datasets y modelos ya guardados de las 8
variantes × 30 semillas.

Cambios de código:

- `ruteo_anticipatorio.py`:
  - Nuevo sistema `hibrido`. Usa la congestión que proyecta la IA, salvo en los tramos
    cuyo evento tiene edad ≥ umbral; ahí aplica la regla de duración de la Sección 11
    (perfil histórico sin evento del tramo).
  - La regla y el híbrido se evalúan con tres umbrales: edad ≥ 1, ≥ 2 y ≥ 3. Los sistemas
    con umbral 1 y 3 llevan sufijo (`reactivo_duracion_1`, `hibrido_3`, ...). Sin sufijo
    son el umbral por defecto (≥ 2), así que `reactivo_duracion` conserva su significado.
  - `congestion_con_regla_duracion` recibe la columna base (congestión observada o
    proyección de la IA) y el umbral.
- `experimento_semillas.py`:
  - El bootstrap agrega, para cada umbral, las diferencias pareadas híbrido − regla,
    híbrido − IA, híbrido − reactivo y regla − IA (ahorro y pérdidas) y sus probabilidades.
  - Nuevo `--hibrido-umbral`, que arma la tabla resumen y la guarda en
    `modelos/resultados_hibrido_umbral.csv`.

Ejecución:

```
python experimento_semillas.py --solo-ruteo --variante base hgb aviso_perfecto aviso_ruidoso aviso_aditivo aviso_error_10 aviso_error_20 aviso_error_33
python experimento_semillas.py --hibrido-umbral --variante <las mismas 8>
```

Pruebas:

- Semilla 42 (`base`): todas las columnas de la evaluación de la Sección 11 son idénticas.
- Tras reevaluar las 8 variantes, los totales de reactivo, regla ≥ 2, IA y oráculo
  reproducen exactamente la Sección 11, y la tabla de sensibilidad de la Sección 11.1 sale
  idéntica.
- Con umbral 3, el híbrido coincide con la IA casi al minuto (diferencia ≤ 4 min): solo 5
  conglomerados tienen eventos de edad ≥ 3.

### 12.1 Sensibilidad del umbral de la regla

La regla sola, sin modelo, es la misma en todas las variantes:

| Umbral | Ahorro total (min) [IC 95%] | Pérdidas (min) | % del ahorro posible [IC 95%] | Hora 0 / + 1 / + 2 (min) |
|---|---|---|---|---|
| Sin regla (reactivo) | 20,154 [−16,622; 57,996] | −29,024 | 40.9 [−66.0; 78.8] | 19,174 / 19,887 / −18,852 |
| Edad ≥ 1 | 19,309 | −7,400 | 39.2 [−1.8; 72.2] | 19,160 / **141** / 9 |
| **Edad ≥ 2** | **39,069 [9,850; 72,775]** | −10,031 | **79.4 [44.2; 92.5]** | 19,174 / 19,887 / **9** |
| Edad ≥ 3 | 20,209 | −28,969 | 41.1 [−65.9; 78.9] | 19,174 / 19,887 / −18,852 |

Ahorro posible (oráculo) por edad: 26,591 / 22,521 / 112 min.

P de que la regla ahorre más que la IA, según el umbral:

| Umbral | IA `base` | IA `aviso_perfecto` | IA `aviso_error_20` |
|---|---|---|---|
| Edad ≥ 1 | 24.9% | 1.5% | 5.6% |
| Edad ≥ 2 | **97.7%** | 71.7% | 91.3% |
| Edad ≥ 3 | 39.7% | 3.8% | 16.4% |

Probabilidad de que un incidente o falla siga activo en `t + 1` según su edad. Se midió en
los datasets de `base` (30 semillas). La duración simulada es `1 + Poisson(1)`, recortada
a 4 h.

| Edad | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| P(sigue en t + 1) | 56.9% | 38.0% | 24.1% | 0% |
| Filas | 830 | 502 | 191 | 46 |

### 12.2 Sistema híbrido

**Umbral por defecto (edad ≥ 2)**. IC 95% por conglomerados; en las diferencias, positivo
significa que el híbrido ahorra más o pierde menos:

| Variante | Ahorro híbrido (min) | % del ahorro posible [IC 95%] | Pérdidas híbrido | Δ ahorro vs. regla [IC 95%] | P(híbrido > regla) | Δ pérdidas vs. regla [IC 95%] | Δ ahorro vs. IA [IC 95%] | P(híbrido > IA) |
|---|---|---|---|---|---|---|---|---|
| `base` | 24,349 | 49.5 [26.1; 69.7] | −4,616 | −14,721 [−34,300; 685] | 3.2% | **+5,415 [1,111; 10,790]** | +854 [71; 1,904] | 99.8% |
| `hgb` | 22,271 | 45.2 [14.3; 68.8] | −7,119 | **−16,798 [−38,362; −1,019]** | 1.5% | +2,913 [−707; 6,979] | +1,022 [30; 2,689] | 99.7% |
| `aviso_perfecto` | 37,124 | 75.4 [60.8; 82.9] | −3,219 | −1,945 [−10,942; 6,705] | 33.5% | **+6,812 [1,650; 12,952]** | +663 [−9; 1,704] | 96.9% |
| `aviso_ruidoso` | 37,172 | 75.5 [61.9; 83.0] | −2,672 | −1,897 [−11,367; 7,464] | 35.0% | **+7,359 [1,803; 14,068]** | +624 [−9; 1,618] | 96.9% |
| `aviso_error_10` | 37,076 | 75.3 [60.7; 83.1] | −3,064 | −1,994 [−11,114; 6,854] | 33.5% | **+6,967 [1,746; 13,176]** | +660 [−2; 1,683] | 97.3% |
| `aviso_error_20` | 31,564 | 64.1 [37.8; 79.3] | −4,268 | −7,506 [−21,631; 3,787] | 10.8% | **+5,764 [1,202; 11,355]** | +668 [−1; 1,706] | 97.5% |
| `aviso_error_33` | 25,612 | 52.0 [26.6; 72.9] | −4,040 | −13,457 [−33,778; 1,858] | 5.3% | **+5,991 [1,471; 11,532]** | +1,809 [40; 4,170] | 98.7% |
| `aviso_aditivo` | 25,435 | 51.7 [26.8; 70.8] | −5,016 | −13,635 [−32,133; 806] | 3.5% | **+5,015 [1,039; 9,964]** | +828 [17; 1,915] | 98.5% |

Por edad (`base` / `aviso_perfecto`, min):

| Edad | Ahorro posible | Regla ≥ 2 | IA | Híbrido ≥ 2 |
|---|---|---|---|---|
| Hora 0 | 26,591 | 19,174 | 17,095 / 20,742 | 17,095 / 20,742 |
| Hora + 1 | 22,521 | 19,887 | 7,262 / 16,391 | 7,262 / 16,391 |
| Hora + 2 | 112 | 9 | −863 / −672 | −9 / −9 |

**Otros umbrales del híbrido** (P de que el híbrido ahorre más que la regla con el mismo
umbral / que la IA):

| Variante | Umbral 1: % capturado, P(> regla), P(> IA) | Umbral 3: % capturado, P(> regla), P(> IA) |
|---|---|---|
| `base` | 35.0%, 34.7%, 5.2% | 47.7%, 60.3%, 86.4% |
| `aviso_perfecto` | 42.4%, 68.0%, 1.3% | 74.1%, **96.2%**, 86.4% |
| `aviso_error_20` | 39.8%, 52.9%, 3.1% | 62.8%, 83.6%, 86.4% |

### 12.3 Lectura

1. **El resultado de la Sección 11 depende del umbral.** Solo edad ≥ 2 funciona: la regla
   captura 79.4% del ahorro posible. Con edad ≥ 1 cae a 39.2%, y con edad ≥ 3 a 41.1%,
   igual que el reactivo ingenuo.
   - Con **edad ≥ 1** la regla deja de desviar en la hora + 1. Ahí el 38% de los
     incidentes sigue, y concentran 22,521 min de ahorro posible; la regla captura 141.
     Pierde casi todo lo que el reactivo ganaba en esa hora.
   - Con **edad ≥ 3** la regla no toca la hora + 2 y conserva las −18,852 min de
     pérdidas del reactivo. Además, a los 3 h de edad ningún evento sigue (duración máxima
     4 h), así que la regla solo actúa en 5 conglomerados.
   - Edad ≥ 2 acierta porque en la hora + 2 el costo de seguir desviando (−18,852 min del
     reactivo) es mucho mayor que el ahorro posible (112 min), aunque el 24% de los
     incidentes siga activo. En la hora + 1 pasa lo contrario.
   - **Consecuencia:** la ventaja de la regla es de filo de navaja. Solo existe si el
     umbral cae justo en la edad donde el balance se invierte, y esa edad es una propiedad
     de la distribución de duraciones simulada. La salvedad de la Sección 11.4 (punto 6) se
     confirma: con otra distribución de duraciones, el umbral correcto sería otro. Para que
     la regla sea un baseline legítimo en la red real, el umbral debe fijarse con
     duraciones históricas del STC, no con este experimento.
2. **El híbrido mejora a la IA, pero poco.** Con umbral 2, el híbrido ahorra más que la IA
   en todas las variantes (P entre 96.9% y 99.8%), pero solo por +600 a +1,800 min. Toda la
   mejora está en la hora + 2, donde las pérdidas de la IA (−863 min sin aviso) bajan a −9.
   En las horas 0 y + 1 el híbrido es idéntico a la IA.
3. **El híbrido no supera a la regla sola en ahorro.** Con umbral 2, la regla gana en todas
   las variantes:
   - Sin aviso, la regla ahorra 14,721 min más (P = 3.2% de que el híbrido ahorre más; el
     IC de dos colas, [−34,300; 685], apenas toca el 0).
   - Con un aviso preciso (perfecto, ruidoso o 10% de error), la brecha baja a ~−1,950 min
     y deja de ser significativa (P ≈ 34%).
   - La causa es la hora + 1: el híbrido usa ahí la IA, y la IA se desvía menos que el
     reactivo cuando el incidente sigue (7,262 vs. 19,887 min sin aviso; 16,391 con aviso
     perfecto). La regla, que en la hora + 1 es el reactivo, supone continuidad y acierta
     más.
   - El punto 1 de la Sección 11.5 suponía que el híbrido combinaría lo mejor de los dos.
     No es así: combina las menores pérdidas de la IA con su menor ahorro en la hora + 1.
4. **Lo que el híbrido sí gana: menos pérdidas que la regla, con IC que excluye 0.** Con
   umbral 2, el híbrido pierde 5,000 a 7,400 min menos que la regla en todas las variantes
   con RF (P ≥ 99.6%, IC que excluye 0 en las siete), incluidas las de aviso impreciso.
   Contra la regla, la IA sola ganaba en pérdidas con IC que tocaba el 0 en `aviso_error_33`
   y `aviso_aditivo` (Sección 11.4, punto 4). El híbrido corrige eso porque ya no pierde en
   la hora + 2. Con `hgb` la ventaja no es significativa (IC [−707; 6,979]).
5. **El híbrido y la regla ofrecen un trade-off, no un ganador.** Con un aviso preciso,
   el híbrido captura 75% del ahorro posible (la regla, 79%; diferencia no significativa)
   con 3,200 min de pérdidas en lugar de 10,000. Es decir, ahorra lo mismo y pierde la
   tercera parte. Sin aviso, la regla ahorra mucho más (79% vs. 50%) a cambio del doble de
   pérdidas.
6. **Umbral 3 con aviso: el único caso en que la IA supera a una regla con P ≥ 95%**
   (96.2%). No vale como argumento, porque la regla ≥ 3 es casi el reactivo ingenuo, y eso
   ya se sabía (Sección 8.2, punto 4).
7. **Para la tesis.** La formulación final queda:
   - Contra un reactivo ingenuo, anticipar reduce el riesgo, y con un aviso que acierte
     ≥ 96% en "termina / sigue", además ahorra más (Secciones 8–11).
   - Contra un reactivo con la regla de duración bien calibrada, ni la IA ni el híbrido
     ahorran más. El híbrido con aviso preciso ahorra lo mismo (diferencia no
     significativa) y pierde tres veces menos (P = 99.8%).
   - La ventaja de la regla depende de fijar el umbral justo (edad ≥ 2 en esta simulación);
     con un umbral una hora antes o después rinde como el reactivo ingenuo o peor. La IA
     no necesita ese ajuste: aprende la probabilidad de continuación de los datos.
   - Por eso el argumento defendible a favor de la IA no es que ahorre más, sino que
     **pierde menos y no depende de calibrar a mano una regla** cuya forma correcta cambia
     con la distribución de duraciones de cada red.

### 12.4 Siguiente paso (punto 3, limitación #4, ejecutado — ver [Sección 13](#13-horizonte-sub-horario-bloques-de-15-minutos-2026-09-29); puntos 1 y 2 descartados)

1. **Hora + 1 de incidentes de plataforma**: es donde está toda la brecha entre la IA (o el
   híbrido) y la regla (Secciones 11.3 y 12.3). Hay dos opciones:
   - Un umbral de decisión asimétrico para la IA en tramos con incidente: desviar si la
     probabilidad de que siga supera cierto valor, en lugar de usar el valor esperado.
   - Un modelo de dos partes: probabilidad de continuación × retraso si sigue.
2. **Robustez del umbral de la regla ante otra distribución de duraciones.** Resimular con
   duraciones más largas (p. ej. `1 + Poisson(2)`) y comprobar si el umbral óptimo de la
   regla se mueve mientras la IA se adapta sola. Eso requiere resimular y reentrenar, no
   `--solo-ruteo`.
3. Sin cambios: limitación #4 (horizonte sub-horario) y #5 (perfiles O-D reales), y
   documentar las fuentes del aviso en el capítulo de metodología.


## 13. Horizonte sub-horario: bloques de 15 minutos (2026-09-29)

Esta sección resuelve la limitación #4 (punto 3 de la Sección 12.4). Los puntos 1 y 2 de la
Sección 12.4 (umbral asimétrico o modelo de dos partes para la hora + 1, y robustez del
umbral de la regla) se descartan: la fase de optimización con el horizonte de 1 hora se
dio por cerrada y no se harán más ajustes de umbrales ni de modelos.

Todo el pipeline pasa de pasos de 1 hora a bloques de 15 minutos. El modelo predice la
congestión a 15, 30, 45 y 60 minutos, que cubre el rango de 10–60 min del objetivo de la
tesis.

### 13.1 Cambios de código

- `simulador_congestion.py`:
  - 76 bloques de 15 min por día (05:00 a 23:45). Cada fila trae `hora`, `minuto` y
    `bloque`.
  - `capacidad_tramo()` devuelve la capacidad del bloque: la horaria entre 4 (los trenes
    de un bloque son la cuarta parte de los de la hora). Aplica igual a los transbordos.
  - Nueva `interpolar_demanda_bloques()`: reparte la carga horaria por sentido de tramo
    en 4 bloques (Sección 13.2).
  - `tasa_evento()` divide la tasa horaria entre 4. `sortear_severidad_duracion()` sortea
    la duración en bloques. `retraso_por_suspension()` depende de los bloques restantes.
  - `horas_restantes_anunciadas()` → `bloques_restantes_anunciados()`: el aviso y sus
    errores se miden en bloques (±1 = ±15 min). La columna del dataset se renombra igual.
  - `edad_evento` se mide en bloques.
  - Snapshots con 4 rezagos (`congestibilidad_t_minus_1..4`) y 4 objetivos
    (`target_congestibilidad_t_plus_1..4` = 15, 30, 45 y 60 min). Se descartan los
    primeros y los últimos 4 bloques de cada día.
- `entrenador_anticipatorio.py`:
  - `HORIZONTES = {15: t+1, 30: t+2, 45: t+3, 60: t+4}`; un modelo RF + línea/tramo por
    horizonte, con su propio target encoding del tramo. `TARGET` es el de 15 min.
  - Features nuevas: `minuto` y `congestibilidad_t_minus_2..4`.
  - La comparación de candidatos y la gráfica de importancia usan el horizonte de 15 min.
  - Nuevo `--sin-comparacion`. El `.pkl` guarda `modelos_por_horizonte` e
    `inicio_test = (fecha, bloque)`.
- `ruteo_anticipatorio.py`:
  - Evalúa cada bloque con evento en los 4 horizontes. El reactivo usa la congestión de
    ahora (su ruta no depende del horizonte); la IA y el híbrido usan la predicción a
    t + k; el oráculo y la verdad de referencia, la congestión real de t + k.
  - Umbrales de la regla de duración convertidos a bloques sin reajustarlos: 1, 2 y 3 h =
    4, 8 y 12 bloques. Los sufijos `_1` y `_3` siguen significando horas.
  - El perfil sin evento de la regla se calcula por (tramo, tipo de día, bloque).
  - La evaluación guarda `edad_evento` (bloques) y `hora_evento` (0, + 1, + 2, ...).
  - Arma un DataFrame por bloque: acumular ~500 mil diccionarios por semilla llegaba a
    2.1 GB de memoria; ahora el pico es 0.8 GB, con la misma salida byte a byte.
- `experimento_semillas.py`:
  - Todos los agregados se reportan por horizonte. El conglomerado del bootstrap es un
    bloque con evento (semilla, fecha, bloque).
  - Los `evaluacion_ruteo.csv` pesan ~100 MB por semilla (antes 4 MB): se agregan semilla
    por semilla en vez de concatenarlos.
  - Las corridas multi-semilla entrenan con `--sin-comparacion`.
  - Nueva tabla `modelos/resultados_horizontes.csv` (`--tabla-horizontes`), con las
    métricas del modelo por horizonte leídas de cada `.pkl`.

### 13.2 Supuestos nuevos

| Elemento | Antes (horario) | Ahora (bloques de 15 min) |
|---|---|---|
| Demanda | Matriz O-D por hora | La misma matriz; la carga horaria de cada sentido de tramo se interpola linealmente entre los centros de horas vecinas y se reescala para conservar el total de la hora |
| Capacidad | Pasajeros/hora por línea | La horaria / 4 |
| Tasa de falla e incidente | λ por línea-hora | λ / 4 por línea-bloque (mismo número esperado por hora) |
| Duración | `1 + Poisson(1)` h, tope 4 h | `1 + Poisson(7)` bloques, tope 16 (misma media, ~2 h) |
| Retraso por suspensión | `30·s²` min por hora | `s²·R/2`, con R = minutos de cierre que faltan contando el bloque actual (`30·s²` al inicio de un evento de 1 h; decae a 0) |
| Error del aviso | ±1 h | ±1 bloque (15 min) |
| Regla de duración | Edad ≥ 1, 2, 3 h | Edad ≥ 4, 8, 12 bloques |

Sin la interpolación, dividir la capacidad entre 4 no cambia nada: si la demanda horaria
también se reparte en 4 partes iguales, V/C queda igual y los 4 bloques de una hora salen
idénticos. Con la interpolación el rango medio de la congestibilidad dentro de una hora es
de 0.018 min (p99 0.32 min).

La secuencia aleatoria del simulador cambió (se sortea por bloque), así que los datasets no
son comparables semilla por semilla con los de las Secciones 7–12. Los resultados horarios
se archivaron en `datos_procesados/horario/` y `modelos/horario/`.

### 13.3 Ejecución

```
python simulador_congestion.py && python entrenador_anticipatorio.py && python ruteo_anticipatorio.py
python experimento_semillas.py --variante base aviso_perfecto aviso_ruidoso aviso_aditivo aviso_error_10 aviso_error_20 aviso_error_33 --paralelo 3
python experimento_semillas.py --sensibilidad-aviso
python experimento_semillas.py --hibrido-umbral --variante <las mismas 7>
python experimento_semillas.py --comparar aviso_perfecto aviso_error_10
```

- 7 variantes × 30 semillas = 210 corridas, unas 7.5 h. `hgb` se omitió porque es un
  experimento de modelo.
- La comparación de candidatos se corrió solo en el pipeline principal (semilla 42), a
  15 min. RF + línea/tramo vuelve a tener el menor RMSE con evento en la validación por
  días: 0.125, contra 0.133 de GradientBoosting + línea/tramo y 0.363 de HistGB +
  línea/tramo. Con bloques, la comparación tarda ~10 min por semilla.

Pruebas:

- La interpolación conserva exactamente el total de cada hora.
- El retraso de un incidente decae linealmente hasta 0 y el aviso cuenta hacia atrás.
- Con bloques, las magnitudes de la congestión son las del dataset horario: media 0.022
  contra 0.021 min y máximo 12.15 en ambos.
- La tabla por horizonte de `experimento_semillas.py` reproduce el resumen del ruteo de la
  semilla 42.

### 13.4 Resultados por horizonte

> **Reemplazado por la [Sección 14](#14-retraso-por-suspensión-sin-filtración-de-la-duración-2026-09-30).** Estos resultados tienen la filtración de
> la limitación #8; se conservan como registro (`datos_procesados/bloques_v1/`,
> `modelos/bloques_v1/`).

Variante `base` (sin aviso), 30 semillas: 1,164 bloques con evento y 3,071,024 casos O-D en
cada horizonte. % del ahorro posible que captura cada sistema, con IC 95% bootstrap por
conglomerados; la regla usa el umbral por defecto (edad ≥ 2 h).

| Horizonte | Ahorro posible (min) | Reactivo | Regla ≥ 2 h | IA | Híbrido ≥ 2 h | IA − reactivo (min) [IC 95%] | P(IA > reactivo) |
|---|---|---|---|---|---|---|---|
| 15 min | 430,306 | **94.8** [92.9; 96.2] | 94.5 [92.4; 96.0] | 93.8 [92.0; 95.3] | 92.9 [90.9; 94.6] | −4,618 [−12,986; 2,963] | 12.5% |
| 30 min | 298,927 | 77.0 [69.2; 82.5] | 78.0 [70.5; 83.2] | **88.2** [84.6; 91.0] | 88.0 [84.3; 90.8] | **+33,477** [20,734; 45,875] | 100% |
| 45 min | 206,134 | 32.9 [9.4; 49.1] | 35.4 [13.1; 51.2] | **75.5** [65.7; 83.0] | 75.6 [65.8; 83.1] | **+87,980** [67,774; 109,542] | 100% |
| 60 min | 139,428 | −53.9 [−111.8; −16.2] | −49.0 [−105.6; −12.3] | **60.0** [44.8; 72.7] | 60.2 [44.9; 72.8] | **+158,785** [123,841; 195,900] | 100% |

Pérdidas (casos en que el sistema queda peor que la ruta estática, min):

| Horizonte | Reactivo | IA | Híbrido |
|---|---|---|---|
| 15 min | −19,648 | −15,749 | −15,388 |
| 30 min | −64,745 | −21,508 | −20,732 |
| 45 min | −125,886 | −22,161 | −21,438 |
| 60 min | −195,825 | −20,763 | −20,460 |

Error del modelo en el 20% final (media de 30 semillas; RMSE con evento ponderado por las
4,012 filas con evento):

| Horizonte | RMSE (min) | RMSE con evento (min) |
|---|---|---|
| 15 min | 0.057 | 0.759 |
| 30 min | 0.072 | 0.764 |
| 45 min | 0.082 | 0.725 |
| 60 min | 0.090 | 0.685 |

### 13.5 Por hora del evento y por tipo

Ahorro total (min) en `base`: posible / reactivo / IA.

| Hora del evento | Bloques | 15 min | 30 min | 45 min | 60 min |
|---|---|---|---|---|---|
| Hora 0 | 633 | 361,852 / **350,771** / 338,208 | 261,188 / 225,898 / **231,735** | 178,836 / 104,579 / **142,214** | 115,335 / −9,524 / **81,985** |
| Hora + 1 | 489 | 64,366 / 55,711 / **61,606** | 35,736 / 7,199 / **31,145** | 26,467 / −31,711 / **13,626** | 23,849 / −58,886 / **1,984** |
| Hora + 2 | 133 | 4,079 / 1,591 / **3,613** | 2,001 / −2,725 / **765** | 830 / −4,895 / −108 | 244 / −6,462 / −287 |

La única celda donde el reactivo gana es la hora 0 a 15 min (P(IA > reactivo) = 0%). En
todas las demás, P(IA > reactivo) ≥ 88%.

Por tipo, el incidente de plataforma concentra casi todo el ahorro posible: 425,393 de
430,306 min a 15 min y 120,612 de 139,428 a 60 min. La lluvia aporta 769,752 casos (una
cuarta parte) en solo 81 bloques, pero casi no hay nada que ahorrar (≤ 1,583 min).

### 13.6 Aviso de restablecimiento: ya no aporta

| Variante | P(error) | Aviso exacto | Termina/sigue correcto | % capturado IA a 15 / 30 / 45 / 60 min |
|---|---|---|---|---|
| `base` (sin aviso) | — | — | — | 93.8 / 88.2 / 75.5 / 60.0 |
| `aviso_perfecto` | 0 | 100% | 100% | 93.6 / 88.1 / 76.0 / 62.2 |
| `aviso_error_10` | 0.10 | 90.6% | 99.0% | 93.7 / 88.0 / 75.7 / 62.0 |
| `aviso_error_20` | 0.20 | 80.8% | 97.2% | 93.5 / 88.1 / 75.4 / 61.3 |
| `aviso_error_33` | 0.33 | 68.9% | 95.8% | 93.7 / 88.0 / 75.8 / 61.2 |
| `aviso_ruidoso` | σ = 0.5 | — | — | 94.0 / 88.5 / 75.5 / 60.3 |
| `aviso_aditivo` | 2/3 | 38.0% | 92.3% | 93.6 / 87.8 / 75.5 / 62.7 |

Todas las variantes quedan a menos de 3 puntos de `base`, dentro de sus IC. La comparación
pareada `aviso_error_10` − `aviso_perfecto` tiene IC que incluye 0 en los 4 horizontes. En
la versión horaria el aviso era lo que más movía el resultado (47.7% → 74.2%, Sección 8).

**Causa: el simulador filtra la duración restante (limitación #8).** El retraso de un tramo
suspendido es `s²·R/2`. La severidad `s` es una feature y el retraso observado del bloque
actual es parte de `congestibilidad_t`, así que el modelo puede despejar R, los minutos de
cierre que faltan, sin necesidad del aviso. En la versión horaria el retraso era `30·s²`,
no dependía de R, y la única pista de la duración era el aviso.

Consecuencias:

- El aviso sale redundante porque la misma información ya está en el estado actual.
- Parte de la ventaja de la IA a 30–60 min, que se concentra en incidentes de plataforma,
  se debe a esta filtración y no a que el modelo anticipe mejor. En un sistema real, la
  congestión observada no revela cuánto va a durar el cierre.
- Los porcentajes de la Sección 13.4 **no deben citarse todavía** como resultado de la
  tesis. La degradación del reactivo (pierde valor rápido con el horizonte y a 60 min
  queda peor que la ruta estática) no depende de la filtración, porque el reactivo no usa
  la severidad; sí depende de que el retraso decaiga durante el evento, que es un supuesto
  del simulador.

### 13.7 Lectura

1. **La limitación #4 está resuelta en lo técnico.** El pipeline completo (simulación,
   entrenamiento, ruteo y experimento multi-semilla) trabaja en bloques de 15 min y evalúa
   los 4 horizontes del objetivo de la tesis.
2. **El reactivo se degrada con el horizonte.** Captura 94.8% a 15 min, 77.0% a 30, 32.9% a
   45 y −53.9% a 60 min. A una hora de distancia seguir la congestión de ahora es peor que
   no hacer nada: en ese tiempo los incidentes terminan o su retraso ya bajó, y el reactivo
   desvía por un retraso que ya no existe (−195,825 min de pérdidas).
3. **A 15 min anticipar no sirve.** La congestión de ahora es casi la de dentro de 15 min;
   el reactivo gana en la hora 0 del evento y la diferencia total no es significativa
   (P = 12.5%).
4. **A 30–60 min la IA supera al reactivo con P = 100%**, y sus pérdidas se mantienen
   estables (~−21,000 min) en vez de crecer con el horizonte. La magnitud está inflada por
   la limitación #8 (Sección 13.6).
5. **La regla de duración y el híbrido casi no cambian nada.** La regla ≥ 2 h queda a ≤ 5
   puntos del reactivo (mejora significativa solo a 45 y 60 min: +5,161 y +6,729 min). El
   híbrido queda a ≤ 1 punto de la IA y a 15 min pierde 3,556 min [1,132; 6,628] contra
   ella.
6. **Los IC pueden ser algo optimistas.** El conglomerado es el bloque, y los bloques
   consecutivos de un mismo evento están correlacionados (lo mismo pasaba con las horas
   consecutivas en la versión horaria).

### 13.8 Siguiente paso (punto 1 ejecutado — ver [Sección 14](#14-retraso-por-suspensión-sin-filtración-de-la-duración-2026-09-30))

1. **Corregir la limitación #8.** El retraso por suspensión no debe depender de la duración
   real que falta. Hay que definirlo con información observable (por ejemplo, la duración
   esperada según la edad del evento) o hacer que la severidad observada sea ruidosa.
   Luego se resimula, se reentrena y se reevalúa para obtener los números citables.
2. Conglomerar el bootstrap por evento en vez de por bloque.
3. Sin cambios: limitación #5 (perfiles O-D reales) y documentar las fuentes del aviso en
   el capítulo de metodología.


## 14. Retraso por suspensión sin filtración de la duración (2026-09-30)

Esta sección resuelve la limitación #8 (punto 1 de la Sección 13.8) y reemplaza los
resultados de las Secciones 13.4–13.6. El pipeline de bloques de 15 minutos de la Sección 13
no cambia; solo cambia cómo se calcula el retraso de un tramo suspendido.

### 14.1 El problema

En la Sección 13 el retraso de un tramo suspendido era `s²·R/2`, con R los minutos de cierre
que faltan. La severidad `s` es una feature del modelo, así que de la congestión observada se
despeja R: el modelo sabía cuánto duraría el incidente sin necesidad del aviso. Por eso:

- El aviso de restablecimiento no aportaba nada: todas sus variantes quedaban a menos de 3
  puntos de `base`.
- La ventaja de la IA a 30–60 min estaba inflada.

No basta con cambiar la forma de la función. Cualquier retraso que dependa de R (lineal o
no) se puede invertir de la misma manera si es monótono en R.

### 14.2 Cambio en el simulador

`retraso_por_suspension(severidad)` ya no recibe los bloques restantes:

```
retraso = s × (MINUTOS_CICLO_CIERRE · s) / 2 = 30·s²   (MINUTOS_CICLO_CIERRE = 60)
```

Mientras el incidente está activo, el servicio del tramo se interrumpe en cierres de 60·s
minutos por cada ciclo de 60. Una fracción s de los pasajeros llega durante un cierre y
espera en promedio la mitad. El retraso es constante mientras dura el evento y vuelve a 0
al terminar.

- Es la misma magnitud que el modelo horario de las Secciones 7–12 y que la fórmula de la
  Sección 13 al inicio de un evento de 1 h.
- El retraso depende solo de s. Lo único que dice si el incidente sigue en t + k es su edad
  (la probabilidad de que continúe) o el aviso.
- `bloques_restantes_de()` sigue existiendo, pero solo para calcular el aviso.

Nada más cambió: demanda, capacidad, tasas, duraciones, aviso, modelo, ruteo y experimento
son los de la Sección 13. Con la misma semilla los eventos sorteados son idénticos; solo
cambia el retraso de los incidentes de plataforma.

Pruebas (semilla 42):

- En las 126 filas con incidente de plataforma, `congestibilidad_t / s²` vale entre 29.9 y
  34.9 (mediana 30.0; lo que varía es la parte BPR). Su correlación con la duración real
  restante es 0.115; antes el retraso la determinaba exactamente.
- La IA a 60 min captura 79.0% del ahorro posible sin aviso y 98.4% con aviso perfecto;
  antes del cambio daban lo mismo. Con esta prueba se decidió lanzar el experimento
  completo.

### 14.3 Ejecución

Los mismos comandos de la Sección 13.3: pipeline principal, 7 variantes × 30 semillas
(210 corridas, 7.5 h) y las tablas de sensibilidad, más
`python experimento_semillas.py --comparar base aviso_perfecto`. Los resultados de la
Sección 13 se archivaron en `datos_procesados/bloques_v1/` y `modelos/bloques_v1/`.

### 14.4 Resultados por horizonte

> **Actualizado en la [Sección 15.7](#15-perfiles-de-estación-derivados-de-datos-2026-09-30).** Estas tablas usan la demanda con perfiles
> fijos y sin los viajes de las 11 estaciones sin nodo (Sección 15.3); se conservan como
> registro (`datos_procesados/bloques_v2/`, `modelos/bloques_v2/`).

Variante `base` (sin aviso), 30 semillas: 1,164 bloques con evento y 3,071,024 casos O-D por
horizonte. % del ahorro posible que captura cada sistema, con IC 95% bootstrap por
conglomerados; la regla y el híbrido usan el umbral por defecto (edad ≥ 2 h).

| Horizonte | Ahorro posible (min) | Reactivo | Regla ≥ 2 h | IA | Híbrido ≥ 2 h | IA − reactivo (min) [IC 95%] | P(IA > reactivo) |
|---|---|---|---|---|---|---|---|
| 15 min | 343,352 | 86.5 [78.1; 92.8] | 87.3 [79.3; 93.2] | 87.3 [81.3; 91.5] | 85.7 [79.4; 90.4] | +3,005 [−9,495; 17,691] | 66.1% |
| 30 min | 287,708 | 68.4 [52.8; 80.0] | 74.3 [60.5; 84.5] | 75.5 [64.6; 83.3] | 76.0 [65.0; 83.7] | +20,440 [−2,350; 45,238] | 95.6% |
| 45 min | 234,292 | 41.8 [14.0; 60.6] | 50.2 [25.3; 67.3] | **63.4** [51.7; 72.0] | 64.2 [52.6; 72.6] | **+50,526** [15,681; 86,811] | 99.9% |
| 60 min | 180,804 | −0.4 [−46.2; 29.5] | 11.8 [−29.8; 39.2] | **40.7** [19.3; 55.7] | 41.4 [20.3; 56.3] | **+74,296** [36,543; 114,557] | 100% |

Misma tabla con **aviso perfecto** (`aviso_perfecto`; reactivo y regla no usan el aviso y
no cambian):

| Horizonte | IA | Híbrido ≥ 2 h | P(IA > reactivo) | P(IA > regla) |
|---|---|---|---|---|
| 15 min | 89.5 [83.6; 93.3] | 87.7 [81.3; 92.2] | 93.8% | 82.8% |
| 30 min | **82.1** [71.8; 89.2] | 81.5 [71.1; 88.7] | 100% | 98.7% |
| 45 min | **73.9** [63.8; 81.7] | 73.2 [62.8; 81.1] | 100% | 100% |
| 60 min | **58.8** [40.1; 71.9] | 58.5 [39.6; 71.5] | 100% | 100% |

Pérdidas en `base` (casos en que el sistema queda peor que la ruta estática, min):

| Horizonte | Reactivo | IA | Híbrido |
|---|---|---|---|
| 15 min | −45,700 | −28,518 | −22,978 |
| 30 min | −89,488 | −42,168 | −38,036 |
| 45 min | −132,805 | −43,496 | −41,231 |
| 60 min | −176,104 | −56,639 | −55,182 |

Error del modelo en el 20% final (media de 30 semillas; RMSE con evento ponderado por las
4,012 filas con evento), `base` / `aviso_perfecto`:

| Horizonte | RMSE (min) | RMSE con evento (min) |
|---|---|---|
| 15 min | 0.038 / 0.036 | 0.587 / 0.540 |
| 30 min | 0.050 / 0.047 | 0.702 / 0.603 |
| 45 min | 0.058 / 0.055 | 0.764 / 0.638 |
| 60 min | 0.066 / 0.063 | 0.807 / 0.641 |

Frente a la Sección 13 (con filtración), la IA sin aviso baja de 93.8 / 88.2 / 75.5 / 60.0%
a 87.3 / 75.5 / 63.4 / 40.7%. El reactivo también cambia (de 94.8 / 77.0 / 32.9 / −53.9% a
86.5 / 68.4 / 41.8 / −0.4%), porque ahora el retraso de un incidente no decae: si el
incidente sigue en t + k, su retraso es el mismo que el reactivo observa.

### 14.5 El aviso recupera su valor

| Variante | Aviso exacto | Termina/sigue correcto | % capturado IA a 15 / 30 / 45 / 60 min |
|---|---|---|---|
| `base` (sin aviso) | — | — | 87.3 / 75.5 / 63.4 / 40.7 |
| `aviso_perfecto` | 100% | 100% | 89.5 / 82.1 / 73.9 / 58.8 |
| `aviso_error_10` | 90.6% | 99.0% | 88.8 / 81.2 / 72.9 / 56.6 |
| `aviso_error_20` | 80.8% | 97.2% | 88.6 / 81.5 / 72.1 / 57.5 |
| `aviso_error_33` | 68.9% | 95.8% | 88.6 / 78.6 / 71.1 / 56.3 |
| `aviso_ruidoso` (σ = 0.5) | — | — | 88.9 / 80.3 / 67.0 / 46.4 |
| `aviso_aditivo` | 38.0% | 92.3% | 88.3 / 78.7 / 69.8 / 54.4 |

Comparación pareada `aviso_perfecto` − `base` (mismos bloques; positivo = el aviso ayuda):

| Horizonte | Δ ahorro (min) [IC 95%] | Δ pérdidas (min) [IC 95%] | P(aviso ahorra más) |
|---|---|---|---|
| 15 min | +7,431 [3,987; 11,458] | +2,101 [122; 4,988] | 100% |
| 30 min | +18,814 [12,206; 26,186] | +4,714 [1,285; 8,607] | 100% |
| 45 min | +24,561 [16,226; 34,234] | +4,500 [1,181; 8,405] | 100% |
| 60 min | +32,806 [21,720; 46,350] | +6,573 [3,260; 10,233] | 100% |

Con 10% de avisos errados (±15 min), la pérdida frente al aviso perfecto es pequeña: el IC
de `aviso_error_10` − `aviso_perfecto` incluye 0 a 30, 45 y 60 min; solo a 15 min es
significativa (−2,337 min [−4,368; −699]).

### 14.6 Por hora del evento y por tipo

Ahorro total (min) en `base`: posible / reactivo / IA; entre paréntesis, P(IA > reactivo).

| Hora del evento | Bloques | 15 min | 30 min | 45 min | 60 min |
|---|---|---|---|---|---|
| Hora 0 | 633 | 211,422 / **201,036** / 195,738 (0%) | 198,231 / **177,470** / 168,083 (0%) | 176,585 / **142,542** / 132,010 (11%) | 145,850 / **90,131** / 72,766 (3%) |
| Hora + 1 | 489 | 119,228 / 98,547 / 98,467 (46%) | 84,255 / 36,221 / **50,313** (91%) | 54,026 / −25,139 / **18,236** (100%) | 32,771 / −68,960 / **2,109** (100%) |
| Hora + 2 | 133 | 12,120 / −2,300 / **6,274** (100%) | 5,196 / −15,592 / **−753** (100%) | 3,669 / −18,122 / **−1,735** (100%) | 2,184 / −20,580 / **−1,336** (100%) |

Por tipo, el incidente de plataforma concentra casi todo el ahorro posible: 339,467 de
343,352 min a 15 min y 175,114 de 180,804 a 60 min. En incidentes, a 60 min el reactivo
ahorra 2,004 min y la IA 75,305.

### 14.7 Lectura

1. **La limitación #8 está resuelta.** El aviso vuelve a ser la variable que más mueve el
   resultado: con aviso perfecto la IA gana 7,431 min a 15 min y 32,806 a 60 min (P = 100%
   en los 4 horizontes). Un 10% de avisos errados cuesta poco.
2. **A 15 min anticipar no sirve**, con o sin aviso: la diferencia con el reactivo no es
   significativa (P = 66% sin aviso, 94% con aviso perfecto).
3. **A 30 min la ventaja depende del aviso.** Sin aviso, P = 95.6% y el IC incluye 0; con
   aviso perfecto, la IA captura 82.1% contra 68.4% del reactivo (P = 100%).
4. **A 45–60 min la IA supera al reactivo con o sin aviso.** A 60 min el reactivo no ahorra
   nada (−0.4%) y la IA captura 40.7% sin aviso y 58.8% con aviso perfecto.
5. **En la hora 0 del evento el reactivo gana en todos los horizontes.** Un incidente que
   acaba de empezar casi siempre sigue (la duración media es de 2 h): el reactivo ve el
   retraso completo y desvía, mientras que la IA sin aviso proyecta el valor esperado, que
   es menor. La ventaja de la IA está en la hora + 2 y, desde 30 min, en la hora + 1
   (a 15 min empatan, P = 46%), donde el reactivo sigue desviando por incidentes que ya
   terminaron. Es el mismo patrón que en la versión horaria
   (Sección 10).
6. **La regla de duración ahora sí ayuda al reactivo a 30–60 min** (+17,021 a +22,065 min,
   IC excluye 0), pero sin aviso la IA solo la supera con claridad a 45 y 60 min
   (P = 97.2% y 99.9%). El híbrido queda a ≤ 2 puntos de la IA.
7. Los IC siguen conglomerados por bloque y pueden ser algo optimistas (Sección 13.7,
   punto 6).

### 14.8 Siguiente paso (punto 2, limitación #5, ejecutado — ver [Sección 15](#15-perfiles-de-estación-derivados-de-datos-2026-09-30); punto 1 descartado)

1. Conglomerar el bootstrap por evento en vez de por bloque.
2. Sin cambios: limitación #5 (perfiles O-D reales) y documentar las fuentes del aviso en
   el capítulo de metodología.


## 15. Perfiles de estación derivados de datos (2026-09-30)

Esta sección resuelve la limitación #5. Antes, 10 estaciones eran "origen" y 10 "destino"
por listas escritas a mano, y las otras 143 eran "mixto". Ahora el perfil de cada estación
sale de la afluencia real y de la topología de la red. El punto 1 de la Sección 14.8
(conglomerar el bootstrap por evento) se descarta: la evaluación estadística se dio por
cerrada.

### 15.1 Qué permiten los datos

`afluenciastc_desglosado_01_2026.csv` trae **entradas diarias** por estación y tipo de pago
(2021-01-01 a 2026-01-31, 163 estaciones), **sin hora ni salidas**. Eso tiene dos
consecuencias:

- No se puede comparar el pico matutino contra el vespertino ni medir la curva horaria de
  cada estación.
- El total diario no distingue origen de destino. Una estación residencial registra sus
  entradas en la mañana y una laboral en la tarde, pero los totales del día se parecen,
  porque la gente regresa.

Sí se puede medir qué tan "de traslado al trabajo" es una estación. En domingo caen las
entradas en los dos extremos del viaje al trabajo, mientras que las estaciones de ocio o
turismo se mantienen o suben. En 2025, la mediana de domingo / día laboral va de 0.18
(Norte 45, zona industrial) a 1.21 (La Villa y Basílica). La mezcla de tipos de pago casi
no varía entre estaciones (desviación estándar 0.03) y no se usa.

### 15.2 Método

Para cada estación se calculan dos índices en [0, 1]:

- **Traslado `c`** (CSV, año 2025):
  - `r_dom = mediana de entradas en domingo / mediana en día laboral`.
  - La mediana laboral excluye las vacaciones escolares (1–6 ene, Semana Santa y 22–31 dic).
  - Se descartan los días con menos del 20% de la mediana de la estación (cierres u obras).
  - `c = 1 − rango percentil de r_dom`. Alto = traslado al trabajo; bajo = ocio o turismo.
- **Origen `a`** (grafo GTFS): `a = (rango percentil del tiempo medio de viaje a las demás
  estaciones + es_terminal) / 2`. La periferia sola no basta: por tiempo de viaje Pantitlán
  sale "central" (percentil 0.32) por ser un nodo de 4 líneas. Las terminales de línea
  (19 en el grafo) concentran el transporte alimentador (CETRAM), que es lo que las hace
  origen de los viajes de la mañana.

Pesos de cada estación (suman 1): `w_origen = c·a`, `w_destino = c·(1 − a)` y
`w_mixto = 1 − c`. Una estación de traslado se reparte entre origen y destino según su
posición en la red; una de ocio queda como mixta.

- **Curva horaria de entradas** = `w_origen·curva_origen + w_destino·curva_destino +
  w_mixto·curva_mixta`. Cada estación tiene su propia curva.
- **Atractividad como destino** en el modelo gravitacional = la misma mezcla aplicada a la
  tabla de atractividad por periodo (mañana, tarde, valle).
- **Perfil discreto** = el peso mayor. Solo se usa para reportar y para las columnas
  `perfil_*` de la matriz O-D.

Siguen siendo supuestos documentados: la forma de las tres curvas horarias, la curva de fin
de semana derivada de la laboral y la tabla de atractividad. Observatorio y Juanacatlán no
tienen estadística en 2025 (cerradas casi todo el año) y reciben `c = 0.5`.

Los resultados por estación se guardan en `datos_procesados/perfiles_estaciones.csv`.

### 15.3 Bug corregido: estaciones sin nodo en el grafo

12 nombres del CSV no coincidían con los del grafo (11 estaciones; Peñón Viejo aparece con
dos grafías). Por ejemplo: `Zócalo/Tenochtitlan` contra `Zócalo`, `Chapultepec` contra
`Chapultepec ` (con espacio), `Garibaldi/Lagunilla` contra `Garibaldi y Lagunilla` y
`Peñón Viejo` contra `Penón Viejo`. El simulador no encontraba su nodo y **descartaba en
silencio todos los viajes que entraban o salían de ellas**: 398,096 de 3,058,747 viajes el
2026-01-13 (13%). Se corrigió con `ALIAS_ESTACIONES` en el generador. Ahora las 163
estaciones coinciden.

**Todos los resultados anteriores a esta sección se calcularon sin esos viajes.**

### 15.4 Resultado de la clasificación

98 estaciones quedan como mixto, 56 como destino y 9 como origen (antes 143 / 10 / 10).

| Estación | r_dom | c | Periferia | Terminal | a | w_origen | w_destino | w_mixto | Perfil |
|---|---|---|---|---|---|---|---|---|---|
| Tláhuac | 0.41 | 0.73 | 1.00 | sí | 1.00 | 0.73 | 0.00 | 0.27 | origen |
| Universidad | 0.34 | 0.88 | 0.85 | sí | 0.93 | 0.81 | 0.06 | 0.12 | origen |
| El Rosario | 0.41 | 0.72 | 0.79 | sí | 0.89 | 0.64 | 0.08 | 0.28 | origen |
| Pantitlán | 0.47 | 0.56 | 0.32 | sí | 0.66 | 0.37 | 0.19 | 0.44 | mixto |
| Indios Verdes | 0.57 | 0.27 | 0.67 | sí | 0.83 | 0.22 | 0.04 | 0.73 | mixto |
| Polanco | 0.29 | 0.94 | 0.67 | no | 0.34 | 0.32 | 0.62 | 0.06 | destino |
| Insurgentes | 0.42 | 0.70 | 0.23 | no | 0.11 | 0.08 | 0.62 | 0.30 | destino |
| Juárez | 0.40 | 0.78 | 0.12 | no | 0.06 | 0.05 | 0.74 | 0.22 | destino |
| Zócalo | 0.77 | 0.06 | 0.11 | no | 0.06 | 0.00 | 0.06 | 0.94 | mixto |
| Bellas Artes | 0.96 | 0.01 | 0.05 | no | 0.02 | 0.00 | 0.01 | 0.99 | mixto |
| La Villa y Basílica | 1.21 | 0.00 | 0.65 | no | 0.33 | 0.00 | 0.00 | 1.00 | mixto |

- Mayor peso de destino: Niños Héroes, Hospital General, Doctores, Balderas, Colegio
  Militar, Eugenia, Patriotismo y Juárez (oficinas, hospitales y juzgados).
- Mayor peso de mixto: La Villa y Basílica, Bosque de Aragón, Bellas Artes, Merced,
  Garibaldi, Autobuses del Norte, Lagunilla y Pino Suárez (turismo, comercio y terminales
  foráneas).
- Mayor peso de origen: Barranca del Muerto, Universidad, Tláhuac, Politécnico, El Rosario,
  La Paz, Martín Carrera y Mixcoac.

**Limitación:** los grandes CETRAM (Pantitlán, Indios Verdes, Ciudad Azteca, Tasqueña y
Cuatro Caminos) quedan como mixto. Mantienen mucha afluencia en domingo (r_dom 0.47–0.57)
por viajes regionales y de comercio, así que su `c` es bajo. Conservan un peso de origen de
0.22–0.37 y su curva sigue teniendo pico matutino, pero menor que con la lista anterior. Se
dejó así en vez de forzarlo con un ajuste a mano; una fuente horaria (por ejemplo, la
Encuesta Origen-Destino 2017 del INEGI) lo resolvería.

### 15.5 Integración con los bloques de 15 minutos

No hubo que cambiar el simulador. La interpolación de la Sección 13 reparte la carga horaria
ya enrutada, así que funciona igual con cualquier curva horaria por estación. Prueba con la
semilla 42 (simulador → entrenador → ruteo):

| Métrica | Perfiles fijos (Sec. 14) | Perfiles de datos |
|---|---|---|
| Entradas totales en los 14 días | 44,864,272 | 45,015,150 |
| Carga media por tramo y bloque | 2,206 | 2,535 (+15%, viajes recuperados) |
| Congestibilidad media (min) | 0.0221 | 0.0258 |
| Rango medio dentro de una hora (min) | 0.0168 | 0.0191 |

El pipeline completo corre sin cambios. Los resultados multi-semilla de la Sección 14 aún
usan las matrices O-D anteriores (respaldadas en `datos_procesados/od_perfiles_fijos/`).

### 15.6 Siguiente paso (ejecutado — ver Sección 15.7)

1. Reejecutar el experimento completo (7 variantes × 30 semillas) con las matrices O-D
   nuevas, para actualizar la tabla de la Sección 14.4 con la demanda corregida.

### 15.7 Resultados con la demanda corregida (2026-10-01)

Mismos comandos de la Sección 13.3 más `--comparar base aviso_perfecto`: 210 corridas,
7.7 h. Los bloques con evento y los casos O-D evaluados son los mismos que en la Sección 14
(1,164 bloques y 3,071,024 casos por horizonte), porque dependen de los eventos y de las
rutas estáticas, no de la demanda.

**Variante `base` (sin aviso)**, % del ahorro posible con IC 95% bootstrap por
conglomerados; la regla y el híbrido usan el umbral por defecto (edad ≥ 2 h):

| Horizonte | Ahorro posible (min) | Reactivo | Regla ≥ 2 h | IA | Híbrido ≥ 2 h | IA − reactivo (min) [IC 95%] | P(IA > reactivo) |
|---|---|---|---|---|---|---|---|
| 15 min | 349,701 | 86.6 [78.4; 92.9] | 87.4 [79.6; 93.2] | 86.4 [79.6; 91.1] | 84.7 [77.6; 89.9] | −680 [−14,013; 14,293] | 46.0% |
| 30 min | 291,937 | 68.8 [53.6; 80.1] | 74.6 [61.3; 84.4] | 75.0 [63.3; 83.2] | 76.0 [64.4; 83.9] | +18,192 [−3,066; 42,266] | 94.8% |
| 45 min | 236,729 | 42.2 [15.6; 60.8] | 50.5 [26.3; 67.0] | **62.5** [51.9; 70.5] | 63.0 [52.4; 71.0] | **+48,121** [14,125; 84,665] | 99.8% |
| 60 min | 182,190 | 0.0 [−44.9; 29.5] | 12.2 [−28.1; 38.7] | **44.2** [26.0; 57.3] | 44.6 [26.4; 57.6] | **+80,459** [41,510; 121,539] | 100% |

**Variante `aviso_perfecto`** (reactivo y regla no cambian):

| Horizonte | IA | Híbrido ≥ 2 h | P(IA > reactivo) | P(IA > regla) |
|---|---|---|---|---|
| 15 min | 88.2 [81.5; 92.6] | 86.4 [79.4; 91.4] | 76.7% | 61.0% |
| 30 min | **79.9** [68.6; 87.6] | 80.1 [69.0; 87.6] | 100% | 92.2% |
| 45 min | **71.3** [62.0; 78.7] | 71.1 [61.5; 78.6] | 100% | 100% |
| 60 min | **62.7** [48.5; 73.0] | 62.0 [47.3; 72.4] | 100% | 100% |

Pérdidas en `base` (min) y error del modelo (media de 30 semillas, `base` / `aviso_perfecto`):

| Horizonte | Pérdidas reactivo | Pérdidas IA | Pérdidas híbrido | RMSE (min) | RMSE con evento (min) |
|---|---|---|---|---|---|
| 15 min | −45,935 | −34,150 | −28,157 | 0.040 / 0.039 | 0.571 / 0.532 |
| 30 min | −89,582 | −51,412 | −45,256 | 0.054 / 0.052 | 0.692 / 0.614 |
| 45 min | −133,082 | −44,011 | −42,192 | 0.064 / 0.062 | 0.759 / 0.657 |
| 60 min | −176,519 | −50,092 | −49,327 | 0.072 / 0.069 | 0.773 / 0.656 |

**Aviso** (% capturado por la IA a 15 / 30 / 45 / 60 min):

| Variante | % capturado IA |
|---|---|
| `base` (sin aviso) | 86.4 / 75.0 / 62.5 / 44.2 |
| `aviso_perfecto` | 88.2 / 79.9 / 71.3 / 62.7 |
| `aviso_error_10` | 87.7 / 78.5 / 70.1 / 59.4 |
| `aviso_error_20` | 87.4 / 79.0 / 70.4 / 62.0 |
| `aviso_error_33` | 87.5 / 77.2 / 68.9 / 58.5 |
| `aviso_aditivo` | 87.1 / 77.5 / 66.8 / 57.5 |

`aviso_perfecto` − `base`: +6,238 [2,342; 10,390], +14,238 [7,687; 22,435], +20,792
[12,260; 30,969] y +33,696 [23,061; 45,966] min a 15 / 30 / 45 / 60 min (P ≥ 99.8%).
`aviso_error_10` − `aviso_perfecto`: −1,675, −4,072, −2,772 y −6,060 min; el IC excluye 0
a 15, 30 y 60 min.

Lectura:

1. **Las conclusiones de la Sección 14.7 se mantienen con la demanda corregida.** Los
   porcentajes cambian menos de 4 puntos:
   - A 15 min anticipar no sirve (P = 46% sin aviso).
   - A 30 min la ventaja depende del aviso.
   - A 45–60 min la IA supera al reactivo con o sin aviso; a 60 min el reactivo no ahorra
     nada (0.0%) y la IA captura 44.2% sin aviso y 62.7% con aviso perfecto.
2. **El aviso sigue siendo la variable que más mueve el resultado** (+6,238 a +33,696 min).
   Con la demanda corregida, un 10% de avisos errados ya tiene un costo pequeño pero
   significativo en 3 de los 4 horizontes.

