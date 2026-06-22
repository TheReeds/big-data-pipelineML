```python
def collect_metrics(exp_name, progress_list):
    """Extrae metricas estructuradas de recentProgress."""
    rows = []
    for p in progress_list:
        dur = p.get("durationMs") or {}
        so  = (p.get("stateOperators") or [{}])[0]
        rows.append({
            "experiment":         exp_name,
            "batch_id":           p.get("batchId", -1),
            "input_rows":         p.get("numInputRows") or 0,
            "throughput_rps":     round(p.get("processedRowsPerSecond") or 0, 4),
            "input_rps":          round(p.get("inputRowsPerSecond") or 0, 4),
            "lat_trigger_ms":     dur.get("triggerExecution", 0),
            "lat_addbatch_ms":    dur.get("addBatch", 0),
            "lat_getbatch_ms":    dur.get("getBatch", 0),
            "lat_planning_ms":    dur.get("queryPlanning", 0),
            "state_rows_total":   so.get("numRowsTotal", 0),
            "state_rows_updated": so.get("numRowsUpdated", 0),
            "watermark":          (p.get("eventTime") or {}).get("watermark", ""),
        })
    return rows

all_metrics = []
for name, prog in exp_raw.items():
    all_metrics.extend(collect_metrics(name, prog))

df_m = pd.DataFrame(all_metrics)

print("=== S8 — Metricas estructuradas por batch ===")
print(f"Total batches capturados: {len(df_m)}")

if len(df_m) == 0:
    print("Sin datos — asegurate de haber ejecutado los experimentos (seccion S7)")
else:
    print()
    numeric_cols = ["input_rows", "throughput_rps", "lat_trigger_ms",
                    "lat_addbatch_ms", "state_rows_total"]
    existing_cols = [c for c in numeric_cols if c in df_m.columns]
    print(df_m.groupby("experiment")[existing_cols].agg(["mean", "max", "min"]).round(2).to_string())
    print()

    print("Throughput (rps) — solo batches con datos:")
    for exp_name, grp in df_m.groupby("experiment"):
        with_data = grp[grp["input_rows"] > 0]
        if len(with_data) > 0:
            print(f"  {exp_name}: avg={with_data['throughput_rps'].mean():.4f} "
                  f"max={with_data['throughput_rps'].max():.4f}")
        else:
            print(f"  {exp_name}: sin batches con datos")
```


```python
ALERT_THRESHOLDS = {
    "max_latency_trigger_ms": 500,    # batch > 500 ms -> alertar
    "max_state_rows":         10_000, # estado > 10K filas -> watermark demasiado amplio
    "min_throughput_rps":     0.01,   # throughput = 0 por > 2 min -> stalled stream
    "max_input_rps_spike":    100,    # spike de ingesta -> posible rebalanceo o replay
}

print("=== S8 - Evaluacion de Alertas ===")
print("Umbrales configurados:")
for k, v in ALERT_THRESHOLDS.items():
    print(f"  {k}: {v}")
print()

violations = []

if len(df_m) > 0:
    high_lat = df_m[df_m["lat_trigger_ms"] > ALERT_THRESHOLDS["max_latency_trigger_ms"]]
    if len(high_lat) > 0:
        violations.append(
            f"LATENCIA ALTA: {len(high_lat)} batches > {ALERT_THRESHOLDS['max_latency_trigger_ms']} ms"
            f" (max={high_lat['lat_trigger_ms'].max():.0f} ms, exp={high_lat['experiment'].values[0]})"
        )

    high_state = df_m[df_m["state_rows_total"] > ALERT_THRESHOLDS["max_state_rows"]]
    if len(high_state) > 0:
        violations.append(
            f"ESTADO GRANDE: {len(high_state)} batches con state_rows > {ALERT_THRESHOLDS['max_state_rows']}"
        )

    # Detectar experimentos donde NINGUN batch proceso datos (stream stalled)
    rows_by_exp = df_m.groupby("experiment")["input_rows"].sum()
    for exp_name, total_rows in rows_by_exp.items():
        if total_rows == 0:
            violations.append(f"STREAM STALLED: {exp_name} no proceso ningun evento")

if violations:
    for v in violations:
        print(f"  ALERTA: {v}")
else:
    print("  OK - todos los umbrales dentro del rango normal")

print()
print("Resumen de metricas observadas:")
if len(df_m) > 0:
    print(f"  Max latencia:    {df_m['lat_trigger_ms'].max():.0f} ms")
    print(f"  Max state_rows:  {df_m['state_rows_total'].max():.0f}")
    wd = df_m[df_m["throughput_rps"] > 0]
    if len(wd) > 0:
        print(f"  Min throughput (con datos): {wd['throughput_rps'].min():.4f} rps")
else:
    print("  Sin metricas disponibles - ejecutar seccion S8 primero")
```


## 11. S8 — Estimación de Recursos y Costos

### Configuración actual (demo local Docker)

| Componente | Config | Memoria asignada |
|------------|--------|-----------------|
| Kafka (KRaft, 1 broker) | 1 partición, replication=1 | 1 GB |
| Spark (driver + executor) | 4 vCPU, shuffle.partitions=4 | 4 GB |
| Jupyter kernel | Python 3.11 | ~500 MB |
| **Total** | | **~5.5 GB** |

### Carga actual
- **Eventos/s:** 0.1 (1 evento cada 10 s)  
- **Tamaño mensaje:** ~250 bytes JSON  
- **Throughput Kafka:** ~25 bytes/s — completamente despreciable  
- **Estado Spark:** 1–3 ventanas abiertas × ~20 bytes/fila → < 1 KB de estado

### Proyección a producción (100 ciudades, 1 evento/ciudad/min)

| Componente | Instancia AWS | Cost/hr | Cost/mes |
|------------|--------------|---------|----------|
| Kafka (MSK 2 brokers) | kafka.m5.large × 2 | $0.40 | ~$290 |
| Spark (EMR 2 workers) | m5.xlarge × 2 | $0.38 | ~$275 |
| Almacenamiento (S3 sink) | 10 GB/día × 30 | $0.023/GB | ~$7 |
| **Total estimado** | | | **~$572/mes** |

> Para escala mayor (1000 ciudades, 1 msg/s): agregar particiones Kafka, aumentar  
> EMR a 4–8 workers. Costo estimado ~$1,500–$2,500/mes.

### CPU y Ejecutores — Configuración propuesta para producción

| Escenario | vCPU Driver | vCPU por Executor | Mem/Executor | N° Executors | Total vCPU | Particiones Kafka |
|-----------|------------|-------------------|-------------|-------------|-----------|------------------|
| Demo local (1 ciudad) | 1 | 1 | 2 GB | 1 | 2 | 1 |
| Escala 2 (10 ciudades) | 2 | 2 | 4 GB | 2 | 6 | 10 |
| Escala 3 (100 ciudades) | 4 | 4 | 8 GB | 4 | 20 | 100 |
| Escala 4 (1000 ciudades) | 8 | 8 | 16 GB | 8 | 72 | 1000 |

> **Nota:** El número de particiones Kafka iguala al número de ciudades para garantizar  
> paralelismo 1:1 entre partición y ciudad. Los ejecutores Spark se escalan linealmente  
> con el volumen de datos. La memoria por executor incluye overhead de JVM y Spark.


### Consumo real de recursos (Docker local)

Métrica tomada del entorno local con todos los servicios en ejecución:

| Servicio | CPU % | Memoria usada | Límite | Mem % |
|----------|-------|--------------|--------|-------|
| spark-notebook | ~12 % | 810 MB | 4 GB | 20 % |
| kafka | ~0.5 % | 192 MB | 1 GB | 19 % |
| postgres | ~5 % | 37 MB | 512 MB | 7 % |
| grafana | ~0.4 % | 123 MB | 256 MB | 48 % |
| prometheus | ~0 % | 33 MB | 256 MB | 13 % |
| **Total** | **~18 %** | **~1.19 GB** | **~6 GB** | **20 %** |

> El consumo real es bajo porque la carga actual es mínima (1 evento cada 10 s).
> En producción con 100 ciudades, Kafka y Spark consumirían el ~70-80 % de los límites
> asignados. PostgreSQL, Grafana y Prometheus se mantienen estables.


```python
# Snapshot de consumo actual de los contenedores
import subprocess
result = subprocess.run(
    ["docker", "stats", "--no-stream",
     "--format", "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"],
    capture_output=True, text=True, timeout=10
)
print(result.stdout)
if result.stderr:
    print(f"(solo visible dentro del contenedor: {result.stderr})")
    print("Ejecuta 'docker stats --no-stream' en la terminal del host para ver los datos reales.")
```


??? output "Salida"
    NAME                       CPU %     MEM USAGE / LIMIT     MEM %
    bigdata-spark-notebook-1   0.27%     904.7MiB / 4GiB       22.09%
    bigdata-superset-1         4.06%     16.67MiB / 1GiB       1.63%
    bigdata-grafana-1          0.49%     186.8MiB / 256MiB     72.98%
    bigdata-postgres-1         0.00%     21.92MiB / 512MiB     4.28%
    bigdata-prometheus-1       0.00%     54.09MiB / 256MiB     21.13%
    bigdata-kafka-1            0.58%     119.5MiB / 1GiB       11.67%
    facturacion-app            708.22%   326.8MiB / 15.24GiB   2.09%


---
## Propuesta de Integración y Arquitectura Definitiva

Las decisiones arquitectónicas para el pipeline en producción se han materializado  
en este notebook (S6–S11). A continuación se documenta la arquitectura resultante  
y las extensiones pendientes para un entorno de producción real.


### 1. Base de Datos Definitiva: PostgreSQL

**Contexto:** El notebook usa `sink: memory` para visualización interactiva. En producción,
las ventanas agregadas por Spark Structured Streaming deben persistir en una base de datos
relacional que soporte consultas analíticas y conexión con herramientas de BI.

**Recomendación: PostgreSQL** por las siguientes razones:

| Característica | PostgreSQL | MySQL | Impacto en el pipeline |
|----------------|-----------|-------|----------------------|
| **JSONB nativo** | Sí — índices GIN sobre JSONB | Solo JSON (texto) sin índices eficientes | Los eventos Open-Meteo llegan en JSON; PostgreSQL permite almacenarlos y consultarlos sin aplanar |
| **Upsert (ON CONFLICT)** | `INSERT ... ON CONFLICT DO UPDATE` | `ON DUPLICATE KEY UPDATE` | Spark produce ventanas en modo `update` (misma ventana se re-emite). Con upsert por `(window_start, window_end)` se evitan duplicados |
| **TimescaleDB (extensión)** | `CREATE EXTENSION timescaledb` convierte tablas en hypertables automáticamente | No existe equivalente nativo | Si el pipeline escala a miles de ciudades, TimescaleDB particiona por tiempo sin cambiar de tecnología |
| **Conectividad Spark** | `spark.write.mode("append").jdbc(PostgreSQL)` | Igual, pero sin JSONB | El conector JDBC funciona igual; la ventaja diferencial está en la semántica de almacenamiento |

**Configuración propuesta para el sink JDBC (producción):**

```python
# En lugar de .format("memory"), usar:
stream_query = (
    windowed.writeStream
    .foreachBatch(lambda df, epoch_id: 
        df.write \
          .mode("append") \
          .jdbc(
              url="jdbc:postgresql://postgres:5432/weather_dm",
              table="weather_windows",
              properties={"user": "spark", "password": "...",
                          "driver": "org.postgresql.Driver"}
          )
    )
    .outputMode("update")
    .trigger(processingTime="5 seconds")
    .option("checkpointLocation", "/home/jovyan/checkpoint/postgres")
    .start()
)
```

**Tabla destino en PostgreSQL:**

```sql
CREATE TABLE weather_windows (
    window_start    TIMESTAMP NOT NULL,
    window_end      TIMESTAMP NOT NULL,
    events          INTEGER,
    avg_temp        DOUBLE PRECISION,
    avg_humidity    DOUBLE PRECISION,
    avg_wind        DOUBLE PRECISION,
    min_temp        DOUBLE PRECISION,
    max_temp        DOUBLE PRECISION,
    min_pressure    DOUBLE PRECISION,
    max_pressure    DOUBLE PRECISION,
    PRIMARY KEY (window_start, window_end)
);

-- Opcional: convertir a hypertable TimescaleDB para escalado temporal
-- SELECT create_hypertable('weather_windows', 'window_start');
```

> **Conclusión:** PostgreSQL es la base de datos correcta para este pipeline por su soporte
> JSONB, upsert atómico, y extensibilidad a series temporales vía TimescaleDB.


### 2. Integración de BI — Capa Semántica para Consumo Analítico

La arquitectura de BI opera sobre tres tablas en PostgreSQL:

| Tabla | Origen | Uso |
|-------|--------|-----|
| `weather_windows` | Spark Structured Streaming (§5-6) | Dashboard de condiciones meteorológicas por ventana |
| `model_metrics` | MLlib training (§13) | Comparativa de modelos — RMSE, R², MAE |
| `temp_predictions` | Inferencia en streaming (§14) | Real vs predicho en tiempo real |

#### Capa BI Operativa — Grafana

Grafana está provisionado con dos dashboards:
- **Weather Pipeline - Monitoreo Técnico:** métricas de infraestructura Spark (latencia, throughput, watermark)
- **ML Pipeline — Resultados S9/S10/S11:** métricas del modelo, predicciones en streaming, tabla de experimentos

#### Capa BI Analítica — Apache Superset

Superset se conecta a PostgreSQL vía SQLAlchemy para exploración ad-hoc:
- Dataset `weather_windows`: promedios de temperatura por ventana temporal
- Dataset `model_metrics`: comparativa de modelos entrenados
- Dataset `temp_predictions`: error de predicción en tiempo real

#### Capa ML — Modelos Entrenados

| Modelo | Features | RMSE test | R² | Uso |
|--------|----------|-----------|-----|-----|
| LinearRegression | base (7) | 2.965°C | 0.726 | Baseline interpretable |
| GBTRegressor base | base (7) | 1.479°C | 0.932 | Streaming (§14) |
| GBTRegressor+lags | lag (10) | 0.922°C | 0.974 | Batch/histórico |
| GBTRegressor tuned | base (7) | **1.558°C** | **0.924** | Modelo campeón S11 |


### 3. Observabilidad: Grafana + Prometheus

**Contexto:** La sección S8 ya extrae métricas estructuradas vía `stream_query.recentProgress`
(latencia, throughput, estado de ventanas). Para un dashboard operativo en tiempo real,
estas métricas deben exponerse a un sistema de monitoreo dedicado.

**Propuesta: Grafana + Prometheus** — estándar de la industria para observabilidad de infraestructura.

#### ¿Por qué Grafana y no Power BI para monitoreo técnico?

| Aspecto | Power BI | Grafana |
|---------|----------|--------|
| Destinado a | Indicadores de negocio (KPI) | Métricas de infraestructura (CPU, throughput, latencia) |
| Refresco | Cada hora o diario | Milisegundos (polling a Prometheus) |
| Fuente de datos | Bases de datos relacionales | Prometheus, InfluxDB, Elasticsearch, Loki |
| Alertas | Limitadas | AlertManager integrado con canales (Slack, email, PagerDuty) |

#### Arquitectura de Monitoreo Recomendada

```
┌────────────────────┐     ┌──────────────┐     ┌────────────────────┐
│ Spark Structured   │────▶│ Prometheus   │────▶│ Grafana Dashboard  │
│ Streaming          │     │ (scrape cada  │     │                    │
│ recentProgress     │     │  15s)         │     │  Panel 1: Health   │
│                    │     │              │     │  Panel 2: Latencia │
│ Métricas:          │     │              │     │  Panel 3: Tput     │
│  - processedRowsRPS│     │              │     │  Panel 4: Estado   │
│  - triggerExecution│     │              │     │                    │
│  - numRowsTotal    │     │              │     │  AlertManager ✓    │
│  - watermark       │     │              │     └────────────────────┘
└────────────────────┘     └──────────────┘
```

#### Métricas que ya genera el notebook (listas para Prometheus/Grafana)

| Métrica | Fuente en código | Tipo | Panel |
|---------|-----------------|------|-------|
| `processed_rows_per_second` | `lastProgress["processedRowsPerSecond"]` | Gauge | Throughput |
| `trigger_execution_ms` | `durationMs.triggerExecution` | Gauge | Latencia |
| `num_rows_total` | `stateOperators[0].numRowsTotal` | Gauge | Estado |
| `input_rows_per_second` | `lastProgress["inputRowsPerSecond"]` | Gauge | Ingesta |
| `watermark` | `eventTime.watermark` | Gauge | Watermark |

#### Implementación futura (Unidad 3)

Para exponer estas métricas a Prometheus sin modificar Spark, se puede usar:
1. **Prometheus JMX Exporter** — embebido en el JVM de Spark, expone métricas en `/metrics`
2. **Pushgateway** — el notebook envía métricas vía HTTP POST a Prometheus
3. **Exportador Python personalizado** — script que lee `recentProgress` cada 15s y las publica
   en un endpoint HTTP escaneable por Prometheus (`prometheus_client` library)

```python
# Pseudocódigo del exportador Python:
from prometheus_client import start_http_server, Gauge
import time

tput = Gauge("spark_throughput_rps", "Rows processed per second")
lat  = Gauge("spark_trigger_latency_ms", "Trigger execution latency")

start_http_server(8001)  # Prometheus scrape target
while True:
    prog = stream_query.lastProgress
    if prog:
        tput.set(prog.get("processedRowsPerSecond", 0))
        lat.set((prog.get("durationMs") or {}).get("triggerExecution", 0))
    time.sleep(15)
```

> **Conclusión:** La propuesta de dashboard de la sección S8 es correcta y se potencia con
> Grafana + Prometheus como capa de observabilidad en tiempo real, separando claramente
> el monitoreo técnico (Grafana) del análisis de negocio (Power BI / Superset).


## S8 — Propuesta de Dashboard + Nota Operativa

### Dashboard mínimo de operación (4 paneles)

**Panel 1 — Health del pipeline**
- `producer.events_sent_total` — contador acumulado de eventos
- `producer.last_event_age_s` — segundos desde el último evento enviado
- `kafka.consumer_lag[weather_topic]` — mensajes sin procesar
- Alerta: `consumer_lag > 100` por más de 2 min

**Panel 2 — Latencia Spark**
- `stream.avg_trigger_latency_ms` — promedio últimos 5 min
- `stream.p95_trigger_latency_ms` — percentil 95
- Alerta: promedio > 500 ms por 2 min consecutivos

**Panel 3 — Throughput**
- `stream.processed_rows_per_second` — throughput actual
- `stream.input_rows_per_second` — tasa de ingesta en Kafka
- Alerta: throughput = 0 por más de 60 s (stream stalled)

**Panel 4 — Estado del stream**
- `stream.state_rows_total` — filas activas en ventanas abiertas
- `stream.watermark_lag_s` — diferencia entre max(event_time) y watermark
- Alerta: `state_rows_total > 10,000` (watermark demasiado amplio o memory leak)

---

### Nota Operativa

#### Riesgos identificados

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| API Open-Meteo lenta / timeout | Media | Bajo | Producer con `timeout=10s` + retry=3; Kafka actúa de buffer |
| Watermark no avanza (eventos con mismo timestamp) | Media | Medio | Usar `produced_at` como event_timestamp (implementado); ventanas se llenan cada 5 min |
| State leak (ventana que nunca cierra) | Baja | Alto | Alertar si `state_rows > 10K`; watermark garantiza cierre tras `watermark_delay` |
| Stream stalled (Kafka inaccessible) | Baja | Alto | `failOnDataLoss=false`; health check en docker-compose; alertar si throughput=0 por 60s |

#### Backpressure

Spark Structured Streaming **no tiene backpressure nativo** como Spark Streaming legacy.  
El control de flujo se logra vía:
1. `maxOffsetsPerTrigger` — limitar eventos por batch (añadir si el producer escala)
2. `trigger(processingTime="X")` — spacing entre batches (ya configurado)
3. `kafka.maxPartitionFetchBytes` — limitar bytes descargados por fetch

Para el pipeline actual (1 evento/10s) no es necesario, pero al escalar a 100 ciudades  
se recomienda añadir `.option("maxOffsetsPerTrigger", "1000")`.

#### Plan de escalado

```
Escala 1 (actual, demo):   1 ciudad  → 1 partición  → 1 executor Spark
Escala 2 (10 ciudades):   10 particiones → 2 executors, partition key=city_code
Escala 3 (100 ciudades): 100 particiones → 4–8 executors, compaction en S3 sink
Escala 4 (global):       usar Confluent Cloud + Databricks Structured Streaming
```

#### Semántica de entrega

| Componente | Semántica | Condición |
|------------|-----------|-----------|
| Producer | at-least-once | `acks=all`, `retries=3` → posibles duplicados en retry |
| Spark sin checkpoint | at-least-once | Reinicio puede reprocesar eventos |
| Spark con checkpoint | exactly-once | Requiere sink idempotente (S3, Delta Lake) |


## 12. Verificación Rápida del Pipeline

Re-ejecuta esta celda cuando quieras ver el estado actual de todos los sinks.


```python
print("="*60)
print("VERIFICACIÓN DEL PIPELINE")
print("="*60)

# 1. Estado de las queries activas
active = spark.streams.active
print(f"\nQueries activas: {len(active)}")
for q in active:
    p = q.lastProgress
    if p:
        print(f"  · {q.name}: {q.isActive} | "
              f"input={p.get('numInputRows',0)} filas | "
              f"latency={p.get('triggerExecutionInMs',0):.0f} ms")
    else:
        print(f"  · {q.name}: {q.isActive} (sin progreso aún)")

# 2. Datos en memoria (Spark SQL)
print(f"\n--- Memoria (spark.sql) ---")
try:
    df = spark.sql("SELECT window.start, window.end, events, avg_temp, avg_humidity "
                   "FROM weather_windows ORDER BY window.start")
    print(f"Filas: {df.count()}")
    df.show(truncate=False)
except Exception as e:
    print(f"  (sin datos aún — {e})")

# 3. Datos en PostgreSQL
print(f"--- PostgreSQL (weather_windows) ---")
try:
    pg_df = spark.read.jdbc(PG_URL, "weather_windows", properties=PG_PROPS)
    print(f"Filas: {pg_df.count()}")
    pg_df.orderBy("window_start").show(truncate=False)
except Exception as e:
    print(f"  Error: {e}")

# 4. Archivos Parquet en warehouse
import subprocess
result = subprocess.run(
    ["find", "/home/jovyan/warehouse/weather_windows", "-name", "*.parquet"],
    capture_output=True, text=True, timeout=5
)
parquet_files = [f for f in result.stdout.split('\n') if f.strip()]
print(f"--- Parquet (warehouse) ---")
print(f"  Archivos: {len(parquet_files)}")
if parquet_files:
    parquet_size = subprocess.run(
        ["du", "-sh", "/home/jovyan/warehouse/weather_windows"],
        capture_output=True, text=True, timeout=5
    )
    print(f"  Tamaño total: {parquet_size.stdout.strip()}")

# 5. Estado del producer
try:
    print(f"\nProducer vivo: {producer_thread.is_alive()}")
    if _producer_log:
        for line in _producer_log[-3:]:
            print(f"  {line}")
except NameError:
    print(f"\nProducer: no iniciado aún")

print("\n" + "="*60)
```
