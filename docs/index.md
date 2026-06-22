# Pipeline: Open-Meteo → Kafka → Spark Streaming → PostgreSQL → ML

Pipeline integrado con PostgreSQL sink, Prometheus metrics, Grafana y MLlib.  
Cubre los criterios **S6** a **S11**:

| Sesión | Objetivo | Evidencia en este notebook |
|--------|----------|---------------------------|
| S6 | Kafka: tópicos, productor, consumidor, contrato de evento | §2 contrato, §4 producer, §8 consumer verify |
| S7 | Structured Streaming: ventanas, watermark, latencia/throughput | §5 stream, §6 watermark, §9 benchmarking |
| S8 | Observabilidad (Grafana+Prometheus), costos, escalado | §10 métricas, §11 alertas, §12 costos |
| S9 | ML distribuido: regresión con MLlib (LR + GBT + lags) | §13 dataset, modelos, tabla comparativa |
| S10 | Series de tiempo e inferencia en streaming | §14 patrones horarios, inferencia Kafka→modelo |
| S11 | Tuning y experimentación distribuida (TrainValidationSplit) | §15 grid search, tabla de 12 experimentos |


## 1. Imports y configuración


```python
import requests, json, time, threading, subprocess
from datetime import datetime
from kafka import KafkaProducer, KafkaAdminClient, KafkaConsumer, TopicPartition
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window, avg,
    min as spark_min, max as spark_max, count,
    to_timestamp, round as spark_round
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)
from prometheus_client import start_http_server, Gauge

BOOTSTRAP_SERVERS = "kafka:9092"
TOPIC_NAME        = "weather_topic"
PG_URL            = "jdbc:postgresql://postgres:5432/weather_dm"
PG_PROPS          = {"user": "spark", "password": "spark123", "driver": "org.postgresql.Driver"}
NYC_LAT, NYC_LON  = 40.7128, -74.0060
API_URL           = "https://api.open-meteo.com/v1/forecast"
API_PARAMS = {
    "latitude": NYC_LAT, "longitude": NYC_LON,
    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,weather_code",
    "timezone": "America/New_York"
}
print("Imports OK")
```


??? output "Salida"
    Imports OK


## Resultados Clave

| Modelo | Features | RMSE | MAE | R² | RMSE/σ |
|--------|----------|------|-----|-----|--------|
| LinearRegression | base (7) | 2.965°C | 2.435°C | 0.726 | 0.538 |
| GBTRegressor base | base (7) | 1.479°C | 1.029°C | 0.932 | 0.269 |
| GBTRegressor+lags | lag (10) | 0.922°C | 0.587°C | 0.974 | 0.167 |
| **GBT Tuned (S11)** | base (7) | **1.558°C** | — | **0.924** | 0.283 |

> **GBT+lags** es el mejor modelo batch (RMSE/σ = 0.17, mejora 37.7% vs GBT base).
> **GBT Tuned** es el modelo de producción seleccionado en S11 (streaming-compatible).
