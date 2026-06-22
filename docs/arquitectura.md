## Arquitectura Completa del Pipeline (S6–S11)

```
┌─────────────┐    ┌────────────────┐    ┌──────────────────────────────────┐
│ Open-Meteo  │    │ Kafka          │    │ Spark Structured Streaming        │
│ API (c/10s) │───▶│ weather_topic  │───▶│ watermark 10min + ventana 5min   │
└─────────────┘    │ 1 partición    │    │                                  │
                   │ KRaft (S6)     │    │  Sink 1: memory → spark.sql()    │
                   └────────────────┘    │  Sink 2: foreachBatch → PG       │──▶ weather_windows
                                         │  Sink 3: Parquet warehouse        │
                                         └──────────────┬───────────────────┘
                                                        │
                                         ┌──────────────▼───────────────────┐
                                         │ MLlib (S9–S11)                   │
                                         │ Dataset: 741 registros históricos │
                                         │ LinearRegression   RMSE=2.97°C   │
                                         │ GBTRegressor base  RMSE=1.48°C   │──▶ model_metrics (PG)
                                         │ GBTRegressor+lags  RMSE=0.92°C   │
                                         │ GBT tuned (S11)    RMSE=1.56°C   │
                                         └──────────────┬───────────────────┘
                                                        │ model.transform(stream)
                                         ┌──────────────▼───────────────────┐
                                         │ Inferencia Streaming (S10)        │
                                         │ GBT base → pred_temp en tiempo   │──▶ temp_predictions (PG)
                                         │ MAE stream ≈ 0.33°C              │
                                         └──────────────────────────────────┘

PostgreSQL (weather_dm)
  ├── weather_windows     ◀── Spark streaming aggregations
  ├── model_metrics       ◀── MLlib training results
  └── temp_predictions    ◀── Streaming inference output
         │
         ├──▶ Grafana (localhost:3000)
         │      Dashboard 1: Weather Pipeline — Monitoreo Técnico (Prometheus + PG)
         │      Dashboard 2: ML Pipeline — Resultados S9/S10/S11 (PG)
         │
         └──▶ Apache Superset (localhost:8088)
                SQL Lab + Dashboards analíticos

Prometheus (localhost:9090) ◀── spark-notebook:8001/metrics (cada 15s)
  └──▶ Grafana — paneles de latencia, throughput, watermark, state_rows
```

**Puertos de acceso:**

| Servicio | URL | Credenciales |
|---|---|---|
| Jupyter (Spark) | http://localhost:8888 | token en logs |
| Grafana | http://localhost:3000 | admin / admin |
| Superset | http://localhost:8088 | admin / admin |
| Spark UI | http://localhost:4040 | — |
| Prometheus | http://localhost:9090 | — |


## Apache Superset — BI Analítico

### Conexión
- **URL:** http://localhost:8088
- **Usuario:** `admin` / **Contraseña:** `admin`

### Guía rápida para crear tu primer dashboard

1. **Ir a http://localhost:8088** y loguearse con `admin`/`admin`
2. **Conectar dataset:**
   - Settings → Database Connections → ya existe **"Weather DM (PostgreSQL)"**
   - Ir a **Data → Datasets** → click **"+ Dataset"**
   - Database: `Weather DM (PostgreSQL)`, Schema: `public`, Table: `weather_windows`
3. **Crear un chart:**
   - **Temperatura promedio por ventana:**
     - Chart type: **Time Series Line Chart**
     - X: `window_start` (time)
     - Metrics: `AVG(avg_temp)`
   - **Eventos por ventana:**
     - Chart type: **Bar Chart**
     - X: `window_start` (time)
     - Metrics: `SUM(events)`
4. **Dashboard:**
   - Crear dashboard → Add charts → organizar paneles
   - Filtro por rango de tiempo (días u horas)

### Arquitectura actualizada

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────┐    ┌───────────────────┐
│ Open-Meteo    │    │  Kafka            │    │  Spark Structured       │    │  PostgreSQL        │
│ API           │───▶│  weather_topic    │───▶│  Streaming              │───▶│  (Data Mart)       │
│ (c/10s)       │    │  (1 partición)    │    │  watermark + ventana    │    │  weather_windows   │
└──────────────┘    └──────────────────┘    │  sink1: memory (notebook) │    └────────┬──────────┘
                                              │  sink2: foreachBatch(JDBC)│             │
                                              │  sink3: Parquet warehouse │             │
                                              │  checkpointLocation       │             │
                                              └─────────────┬───────────────┘             │
                                                            │                             │
                     ┌───────────────────────────────────────┼─────────────────────────────┤
                     │                         │             │                           │
                     ▼                         ▼             ▼                           ▼
         ┌──────────────────────┐   ┌──────────────┐   ┌──────────────┐        ┌──────────────────────┐
         │  Prometheus           │   │  Grafana      │   │  Superset    │        │  Power BI /          │
         │  scrape :8001/metrics │   │  dashboards   │   │  BI analítico│        │  Tableau (futuro)    │
         │  cada 15s             │   │  ops +        │   │  SQL Lab     │        │  Conexión JDBC       │
         └──────────┬───────────┘   │  Observabilidad│   │  Dashboards  │        └──────────────────────┘
                    ▼               └────────────────┘   └──────────────┘
         ┌──────────────────────┐
         │  Grafana             │
         │  (datasource:        │
         │   Prometheus + PG)   │
         └──────────────────────┘
```

**Nota:** Superset consume directamente de PostgreSQL, no de la memoria de Spark. Los datos persisten aunque el notebook reinicie.

### Conexión técnica
| Parámetro | Valor |
|-----------|-------|
| Host | `postgres` (docker network) |
| Port | `5432` |
| Database | `weather_dm` |
| Usuario | `spark` |
| Password | `spark123` |
| JDBC URI | `postgresql://spark:spark123@postgres:5432/weather_dm` |
| Superset URI | `http://localhost:8088` |
