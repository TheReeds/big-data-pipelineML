# Arquitectura del Pipeline

Visión completa de los componentes, flujos de datos y decisiones de diseño del pipeline S6–S11,
incluyendo CRISP-DM, forecasting y observabilidad con 3 dashboards.

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

    subgraph sinks["💾 Persistencia (5 tablas)"]
        PG_WW[("weather_windows\nventanas 5 min")]
        PG_TP[("temp_predictions\ninferencia stream")]
        PG_MM[("model_metrics\n4 modelos")]
        PG_S11[("s11_experiments\n12 experimentos")]
        PG_WF[("weather_forecast\n+24h Open-Meteo")]
        SPARK -->|"foreachBatch JDBC"| PG_WW
        SPARK -->|"foreachBatch JDBC"| PG_TP
    end

    subgraph ml["🤖 ML CRISP-DM (S9-S11)"]
        HIST["Open-Meteo Archive\n30 días · 741 registros"]
        LR["LinearRegression\nRMSE=2.97°C · R²=0.73"]
        GBT["GBTRegressor base\nRMSE=1.48°C · R²=0.93"]
        LAG["GBT + lag features\nRMSE=0.92°C · R²=0.97"]
        TUNE["TrainValidationSplit\n6 LR · 6 GBT · S11"]
        CHAMP["GBT Tuned S11\nRMSE=1.56°C · R²=0.924"]
        HIST --> LR & GBT & LAG
        GBT --> TUNE --> CHAMP
        CHAMP -->|"INSERT"| PG_MM & PG_S11
    end

    subgraph forecast["🔮 Forecasting (S10)"]
        F1H["+1h: GBT+lags\nlag-shift actual→lag1\n±RMSE=0.92°C"]
        F24H["+24h: GBT base\nOpen-Meteo hourly API"]
        F24H -->|"JDBC"| PG_WF
    end

    subgraph inference["🔮 Inferencia Streaming (S10)"]
        INF["PipelineModel.load\nGBT base · 7 features"]
        GBT -->|"save/load"| INF
        KAFKA -->|"ReadStream"| INF
        INF -->|"foreachBatch"| PG_TP
    end

    subgraph observabilidad["📊 Observabilidad (S8) — 3 dashboards"]
        PROM["Prometheus\nscrape :8001"]
        GRAF1["Grafana\nWeather Pipeline\n18 paneles"]
        GRAF2["Grafana\nMétricas ML\n16 paneles"]
        GRAF3["Grafana\nForecasting\n18 paneles"]
        PG_WW --> GRAF1 & GRAF2
        PG_MM & PG_S11 --> GRAF2
        PG_WF & PG_TP --> GRAF3
        PROM --> GRAF1
    end

    style ingesta fill:#fef3c7,stroke:#f59e0b
    style streaming fill:#ede9fe,stroke:#8b5cf6
    style sinks fill:#d1fae5,stroke:#10b981
    style ml fill:#fce7f3,stroke:#ec4899
    style forecast fill:#e0e7ff,stroke:#6366f1
    style inference fill:#dbeafe,stroke:#3b82f6
    style observabilidad fill:#e0f2fe,stroke:#0ea5e9
```

---

## Flujo CRISP-DM

```mermaid
flowchart LR
    subgraph CRISP["Metodología CRISP-DM"]
        BU["① Business\nUnderstanding\nS9 crisp_bu"]
        DU["② Data\nUnderstanding\nEDA 5 gráficas"]
        DP["③ Data\nPreparation\n5 transformaciones"]
        MOD["④ Modeling\nLR · GBT · GBT+lags\n+ Tuning S11"]
        EVAL["⑤ Evaluation\nRMSE/σ = 0.28 ✅\n< 0.4 objetivo"]
        DEP["⑥ Deployment\nStreaming + Forecast\n+ Grafana"]
        BU --> DU --> DP --> MOD --> EVAL --> DEP
    end

    DEP -->|"Inferencia"| KAFKA2["Kafka Stream\n→ temp_predictions"]
    DEP -->|"Forecasting"| FC2["+1h y +24h\n→ weather_forecast"]
    DEP -->|"Dashboards"| GF["3 Dashboards\nGrafana"]

    style BU fill:#fef3c7,stroke:#f59e0b
    style DU fill:#ede9fe,stroke:#8b5cf6
    style DP fill:#d1fae5,stroke:#10b981
    style MOD fill:#fce7f3,stroke:#ec4899
    style EVAL fill:#dbeafe,stroke:#3b82f6
    style DEP fill:#e0f2fe,stroke:#0ea5e9
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
    participant FC as Forecast (+1h/+24h)
    participant GRAF as Grafana (3 dash)

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

    Note over FC: Batch (S10)
    FC->>API: GET /v1/forecast?hourly (24h)
    API-->>FC: 24 registros horarios
    FC->>MODEL: transform(hourly_features)
    MODEL-->>FC: 24 predicciones
    FC->>PG: INSERT weather_forecast (24 rows)

    PG-->>GRAF: SQL queries cada 30s
    GRAF-->>GRAF: render 3 dashboards
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
    KRaft integra el coordinador en el mismo broker desde Kafka 3.3.

---

### ML — Por qué GBT sobre Linear Regression

```mermaid
flowchart LR
    A["Temperatura\n≈ f(hora, humedad,\nviento, presión)"]
    B{"Relación\nlineal?"}
    C["LinearRegression\nR²=0.73 ❌"]
    D["Relación\nno-lineal"]
    E["GBTRegressor\nR²=0.93 ✅"]
    F["GBT + lags\nR²=0.97 ⭐"]
    A --> B
    B -->|"Asumimos sí"| C
    B -->|"Validamos no"| D
    D --> E --> F
```

!!! tip "Lag features vs. Streaming"
    Los lags (temp última hora) mejoran R² de 0.93 → 0.97, pero requieren estado temporal —
    incompatible con streaming stateless. El modelo de producción usa 7 features base.
    El modelo con lags se usa solo en batch (forecasting +1h).

---

### Forecasting — Dos horizontes

!!! note "Estrategias de forecasting"
    | Horizonte | Estrategia | Precisión |
    |-----------|-----------|-----------|
    | **+1 hora** | GBT+lags · lag-shift (actual→lag1) | ±0.92°C (RMSE del modelo) |
    | **+24 horas** | GBT base · Open-Meteo hourly API como features | MAE vs. referencia API |

    El forecast +24h no predice desde cero — usa la predicción de la API meteorológica como
    *feature de entrada* al modelo GBT, añadiendo la corrección aprendida del histórico.

---

## Puertos de acceso

| Servicio | URL | Credenciales |
|----------|-----|-------------|
| Jupyter (Spark) | `http://localhost:8888` | token en logs |
| Grafana | `http://localhost:3000` | admin / admin |
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
    s11_experiments {
        serial id PK
        varchar model_type
        double max_depth
        double max_iter
        double reg_param
        double elastic_net
        double val_rmse
        double test_rmse
        timestamp created_at
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
    weather_forecast {
        timestamp forecast_time PK
        double temperature_2m
        double relative_humidity_2m
        double wind_speed_10m
        double pressure_msl
        double pred_temp
        timestamp created_at
    }

    weather_windows ||--o{ temp_predictions : "ventana genera predicciones"
    model_metrics ||--o{ s11_experiments : "experimentos del tuning"
    weather_forecast }o--|| model_metrics : "usa GBT base"
```
