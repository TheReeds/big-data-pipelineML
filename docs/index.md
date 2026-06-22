# Big Data Pipeline — Unidad 2

Pipeline de ingesta, procesamiento en streaming y ML distribuido sobre datos meteorológicos reales.

---

## Flujo general

```mermaid
flowchart LR
    A["🌤️ Open-Meteo API\ncada 10 s"] -->|JSON| B["📨 Kafka\nweather_topic\nKRaft mode"]
    B -->|ReadStream| C["⚡ Spark\nStructured Streaming\nwatermark 10 min"]
    C -->|foreachBatch| D[("🐘 PostgreSQL\nweather_windows")]
    C -->|writeStream| E["📦 Parquet\nwarehouse"]
    D --> F["📊 Grafana\n+ Prometheus"]
    D -->|datos históricos| G["🤖 MLlib\nS9 · S10 · S11"]
    G -->|GBT model| H["🔮 Inferencia\nStreaming"]
    H --> D

    style A fill:#0ea5e9,color:#fff
    style B fill:#f59e0b,color:#fff
    style C fill:#8b5cf6,color:#fff
    style D fill:#10b981,color:#fff
    style G fill:#ec4899,color:#fff
    style H fill:#6366f1,color:#fff
```

---

## Sesiones cubiertas

| # | Sesión | Tecnología | Resultado |
|---|--------|-----------|-----------|
| S6 | Kafka — tópico, productor, consumidor | KRaft · kafka-python | Contrato de evento JSON validado |
| S7 | Structured Streaming — ventanas, watermark | Spark 3.5 | Latencia p95 < 200 ms |
| S8 | Observabilidad — métricas, alertas, costos | Prometheus · Grafana | 2 dashboards operativos |
| S9 | ML distribuido — regresión con MLlib | VectorAssembler · GBT | R² = 0.974 con lag features |
| S10 | Series de tiempo + inferencia streaming | PipelineModel.load | MAE stream ≈ 0.33°C |
| S11 | Tuning distribuido — TrainValidationSplit | ParamGridBuilder | 12 experimentos, GBT campeón |

---

## Resultados de modelos

```mermaid
xychart-beta
    title "RMSE por modelo (°C) — menor es mejor"
    x-axis ["LinearRegression", "GBT Tuned S11", "GBT base", "GBT+lags"]
    y-axis "RMSE (°C)" 0 --> 3.5
    bar [2.965, 1.558, 1.479, 0.922]
```

| Modelo | Features | RMSE | MAE | R² | RMSE/σ |
|--------|----------|-----:|----:|---:|-------:|
| LinearRegression | base (7) | 2.965°C | 2.435°C | 0.726 | 0.538 |
| GBT Tuned (S11) | base (7) | 1.558°C | — | 0.924 | 0.283 |
| GBTRegressor base | base (7) | 1.479°C | 1.029°C | 0.932 | 0.269 |
| **GBT + lag features** | lag (10) | **0.922°C** | **0.587°C** | **0.974** | **0.167** |

!!! success "Modelo campeón de producción"
    **GBT base** (7 features, sin lags) se usa en **streaming** — compatible con datos en tiempo real sin estado.
    **GBT + lags** es el mejor modelo batch con RMSE/σ = 0.17 (37.7% mejor que GBT base).

---

## Stack tecnológico

=== "Ingesta"
    - **Apache Kafka 7.5** — KRaft mode, sin ZooKeeper
    - **Open-Meteo API** — datos meteorológicos gratuitos cada 10 s
    - **kafka-python** — producer daemon thread

=== "Procesamiento"
    - **Apache Spark 3.5** Structured Streaming
    - Watermark 10 min + ventanas tumbling 5 min
    - Sinks: PostgreSQL · Parquet · Memory

=== "Machine Learning"
    - **MLlib** — LinearRegression, GBTRegressor, Pipeline
    - **TrainValidationSplit** — grid search distribuido
    - Features: cyclic hour encoding, day_of_year, lag features

=== "Observabilidad"
    - **Prometheus** — scrape métricas Spark cada 15 s
    - **Grafana** — dashboard infra + dashboard ML results
    - **Apache Superset** — BI analítico sobre PostgreSQL

=== "Infraestructura"
    - Docker Compose — 5 servicios (Kafka, Spark, PG, Grafana, Superset)
    - Jupyter PySpark notebook integrado
    - GitHub Actions → MkDocs → GitHub Pages
