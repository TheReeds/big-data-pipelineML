# Arquitectura del Pipeline

Visión completa de los componentes, flujos de datos y decisiones de diseño del pipeline S6–S11.

---

## Diagrama de componentes

```mermaid
flowchart TB
    subgraph ingesta["📥 Ingesta (S6)"]
        API["🌤️ Open-Meteo API\nforecast + archive"]
        PROD["kafka-python\nProducer Thread"]
        KAFKA["Apache Kafka\nweather_topic\n1 partición · KRaft"]
        API -->|"GET /v1/forecast\ncada 10 s"| PROD
        PROD -->|"JSON event"| KAFKA
    end

    subgraph streaming["⚡ Streaming (S7)"]
        SPARK["Spark Structured Streaming\nwatermark 10 min\nventana 5 min"]
        KAFKA -->|"ReadStream\nkafka format"| SPARK
    end

    subgraph sinks["💾 Persistencia"]
        PG[("PostgreSQL\nweather_dm")]
        PARQUET["Parquet\nwarehouse/"]
        MEM["Memory Sink\nspark.sql()"]
        SPARK -->|"foreachBatch\nJDBC upsert"| PG
        SPARK -->|"writeStream\nmode=append"| PARQUET
        SPARK -->|"format=memory"| MEM
    end

    subgraph ml["🤖 ML (S9-S11)"]
        HIST["Open-Meteo Archive\n30 días · 741 registros"]
        LR["LinearRegression\nRMSE=2.97°C · R²=0.73"]
        GBT["GBTRegressor base\nRMSE=1.48°C · R²=0.93"]
        LAG["GBT + lag features\nRMSE=0.92°C · R²=0.97"]
        TUNE["TrainValidationSplit\n6 configs LR · 6 configs GBT"]
        MODEL["Modelo Guardado\nGBT base · 7 features"]
        HIST --> LR & GBT & LAG
        GBT --> TUNE
        TUNE --> MODEL
    end

    subgraph inference["🔮 Inferencia (S10)"]
        INF["PipelineModel.load\nStreaming Transform"]
        PREDS[("temp_predictions\nPostgreSQL")]
        MODEL -->|"load"| INF
        KAFKA -->|"ReadStream"| INF
        INF -->|"foreachBatch JDBC"| PREDS
    end

    subgraph observabilidad["📊 Observabilidad (S8)"]
        PROM["Prometheus\nscrape :8001"]
        GRAF["Grafana\n2 dashboards"]
        SUPER["Apache Superset\nBI analítico"]
        PG --> GRAF & SUPER
        PREDS --> GRAF
        PROM --> GRAF
    end

    style ingesta fill:#fef3c7,stroke:#f59e0b
    style streaming fill:#ede9fe,stroke:#8b5cf6
    style sinks fill:#d1fae5,stroke:#10b981
    style ml fill:#fce7f3,stroke:#ec4899
    style inference fill:#e0e7ff,stroke:#6366f1
    style observabilidad fill:#e0f2fe,stroke:#0ea5e9
```

---

## Flujo de datos detallado

```mermaid
sequenceDiagram
    participant API as Open-Meteo API
    participant PROD as Producer Thread
    participant KAFKA as Kafka<br/>weather_topic
    participant SPARK as Spark Streaming
    participant PG as PostgreSQL
    participant MODEL as GBT Model
    participant GRAF as Grafana

    loop cada 10 segundos
        PROD->>API: GET /v1/forecast
        API-->>PROD: JSON {temp, humidity, wind...}
        PROD->>KAFKA: produce(event_id, timestamp, weather_data)
    end

    SPARK->>KAFKA: ReadStream (poll continuo)
    KAFKA-->>SPARK: micro-batch de eventos

    SPARK->>SPARK: watermark(10min) + window(5min)
    SPARK->>PG: foreachBatch → UPSERT weather_windows
    SPARK->>MODEL: transform(stream_features)
    MODEL-->>SPARK: prediction (pred_temp)
    SPARK->>PG: INSERT temp_predictions

    PG-->>GRAF: SQL queries (cada 30s)
    GRAF-->>GRAF: render dashboards
```

---

## Decisiones de diseño

### Kafka — KRaft sin ZooKeeper

```mermaid
flowchart LR
    OLD["❌ Arquitectura antigua\nKafka + ZooKeeper\n2 procesos · complejo"] 
    NEW["✅ KRaft mode\nKafka standalone\n1 proceso · simple"]
    OLD -.->|"Kafka 3.3+"| NEW
```

!!! info "Por qué KRaft"
    Para un pipeline de demo/educativo, ZooKeeper añade complejidad innecesaria.
    KRaft (Kafka Raft) integra el coordinador en el mismo broker desde Kafka 3.3.

---

### Spark — Watermark + Ventana tumbling

!!! note "Parámetros de ventana"
    | Parámetro | Valor | Razonamiento |
    |-----------|-------|--------------|
    | Watermark | 10 min | Tolerancia a eventos tardíos de la API |
    | Ventana | 5 min | Granularidad suficiente para temperatura |
    | Trigger | processingTime=10s | Balance latencia/throughput |
    | Output mode | update | Solo emite ventanas modificadas |

---

### ML — Por qué GBT sobre Linear Regression

```mermaid
flowchart LR
    A["Temperatura\n≈ f(hora, humedad,\nviento, presión)"]
    B["Relación\nlineal?"]
    C["LinearRegression\nR²=0.73 ❌"]
    D["Relación\nno-lineal"]
    E["GBTRegressor\nR²=0.93 ✅"]
    A --> B
    B -->|"Asumimos sí"| C
    B -->|"Validamos no"| D
    D --> E
```

!!! tip "Lag features"
    La temperatura actual depende fuertemente de la temperatura de las últimas horas.
    Añadir `temp_lag1/2/3` (lags de 1-3 horas) mejora R² de 0.93 → 0.97, pero requiere
    estado temporal — incompatible con streaming stateless. Por eso el modelo de producción
    usa solo las 7 features sin lags.

---

## Puertos de acceso

| Servicio | URL | Credenciales |
|----------|-----|-------------|
| Jupyter (Spark) | `http://localhost:8888` | token en logs |
| Grafana | `http://localhost:3000` | admin / admin |
| Apache Superset | `http://localhost:8088` | admin / admin |
| Spark UI | `http://localhost:4040` | — |
| Prometheus | `http://localhost:9090` | — |

---

## Tablas PostgreSQL

```mermaid
erDiagram
    weather_windows {
        timestamp window_start PK
        timestamp window_end
        double avg_temp
        double min_temp
        double max_temp
        double avg_humidity
        double avg_wind
        bigint event_count
    }
    model_metrics {
        serial id PK
        varchar model_name
        varchar features
        double rmse
        double mae
        double r2
        double rmse_sigma
        timestamp trained_at
    }
    temp_predictions {
        int event_id
        double real_temp
        double pred_temp
        double error_abs
        int day_of_year
        timestamp produced_at
        timestamp inserted_at
    }
```
