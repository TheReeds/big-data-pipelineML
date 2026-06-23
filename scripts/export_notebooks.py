"""
Exporta 03_04_kafka_pipeline_integrado.ipynb → páginas Markdown para MkDocs.
"""
import json
import textwrap
from pathlib import Path

NOTEBOOK = Path("notebooks/03_04_kafka_pipeline_integrado.ipynb")
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

SECTIONS = {
    "s6": {
        "file": "s6_kafka.md",
        "title": "S6 — Apache Kafka",
        "intro": """\
!!! abstract "Objetivo S6"
    Implementar la capa de ingesta con Apache Kafka en modo KRaft (sin ZooKeeper):
    creación del tópico, contrato de evento JSON, producer en background y consumer de verificación.

```mermaid
flowchart LR
    A["🌤️ Open-Meteo API"] -->|"GET cada 10 s"| B["kafka-python\nProducer Thread"]
    B -->|"JSON event"| C["Kafka\nweather_topic\n1 partición · KRaft"]
    C -->|"KafkaConsumer\nverificación S6"| D["Consumer\n(últimos 5 msg)"]
    C -->|"ReadStream"| E["Spark Streaming\n(S7 →)"]
    style C fill:#f59e0b,color:#fff
```
""",
        "cells": ["c03", "c04", "c05", "c10", "c11", "c12", "c13", "c18", "c19"],
    },
    "s7": {
        "file": "s7_streaming.md",
        "title": "S7 — Structured Streaming",
        "intro": """\
!!! abstract "Objetivo S7"
    Procesar el stream de Kafka con watermark y ventanas tumbling.
    Comparar triggers y medir latencia/throughput en 3 experimentos.

```mermaid
flowchart LR
    K["Kafka\nweather_topic"] -->|"ReadStream"| P["parse JSON\nSchema tipado"]
    P -->|"withWatermark\n10 min"| W["window\n5 min tumbling"]
    W --> AGG["avg/min/max\nhumedad · viento"]
    AGG -->|"foreachBatch JDBC"| PG[("PostgreSQL\nweather_windows")]
    AGG -->|"memory sink"| MEM["spark.sql()"]
    style W fill:#8b5cf6,color:#fff
    style PG fill:#10b981,color:#fff
```

!!! info "Parámetros clave"
    | Parámetro | Valor | Motivo |
    |-----------|-------|--------|
    | Watermark | 10 min | Tolera retrasos de la API externa |
    | Ventana | 5 min | Granularidad adecuada para temperatura |
    | Trigger | 10 s | Latencia baja sin overhead |
    | Output mode | `update` | Solo emite ventanas modificadas |
""",
        "cells": ["c15", "c16", "c14", "c14b", "c14c", "c17", "c20", "c21", "c22", "c23", "c24"],
    },
    "s8": {
        "file": "s8_observabilidad.md",
        "title": "S8 — Observabilidad",
        "intro": """\
!!! abstract "Objetivo S8"
    Exportar métricas del pipeline a Prometheus, visualizarlas en Grafana
    y estimar costos de operación en producción.

```mermaid
flowchart LR
    SPARK["Spark\nStreaming Query"] -->|"recentProgress"| EXP["Prometheus\nExporter :8001"]
    EXP -->|"scrape 15 s"| PROM["Prometheus\n:9090"]
    PROM -->|"datasource"| GRAF["Grafana\n:3000"]
    PG[("PostgreSQL")] -->|"SQL datasource"| GRAF

    subgraph metricas["5 métricas expuestas"]
        M1["throughput_rows_per_sec"]
        M2["latency_trigger_ms"]
        M3["state_rows"]
        M4["watermark_lag_s"]
        M5["input_rows_total"]
    end
    EXP --> metricas
    style PROM fill:#e65100,color:#fff
    style GRAF fill:#f57c00,color:#fff
```
""",
        "cells": [
            "c25", "c26", "c27", "c27a", "c27a_code",
            "c27b", "c27c", "c27d", "c27e", "c28",
        ],
    },
    "s9": {
        "file": "s9_mllib.md",
        "title": "S9 — ML Distribuido con MLlib (CRISP-DM)",
        "intro": """\
!!! abstract "Objetivo S9"
    Entrenar modelos de regresión con MLlib siguiendo la metodología **CRISP-DM**.
    Dataset: 741 registros históricos de Open-Meteo Archive (30 días).

```mermaid
flowchart TB
    subgraph CRISP["CRISP-DM"]
        BU["① Business\nUnderstanding"]
        DU["② Data\nUnderstanding\nEDA"]
        DP["③ Data\nPreparation"]
        MOD["④ Modeling"]
        EVAL["⑤ Evaluation"]
        DEP["⑥ Deployment"]
        BU --> DU --> DP --> MOD --> EVAL --> DEP
    end

    DP -->|"7 features base\n+ lags batch"| MOD
    MOD --> LR["LinearRegression\nRMSE=2.97 · R²=0.73"]
    MOD --> GBT["GBTRegressor base\nRMSE=1.48 · R²=0.93"]
    MOD --> LAG["GBT + lags\nRMSE=0.92 · R²=0.97"]
    GBT -->|"save"| SAVED["MODEL_PATH\nweather_temp_model"]
    EVAL -->|"RMSE/σ=0.27 ✅"| DEP
    SAVED -->|"S10 Inferencia"| INF["Streaming\nInference"]

    style GBT fill:#ec4899,color:#fff
    style LAG fill:#6366f1,color:#fff
```
""",
        "cells": [
            "s9_md", "crisp_bu",
            "s9_data", "crisp_du_md", "crisp_eda",
            "crisp_dp_md", "s9_spark_df",
            "s9_lr", "s9_gbt",
            "s9_lag_md", "s9_lag",
            "s9_pg_save", "s9_preds",
        ],
    },
    "s10": {
        "file": "s10_series_tiempo.md",
        "title": "S10 — Series de Tiempo, Inferencia y Forecasting",
        "intro": """\
!!! abstract "Objetivo S10"
    Tres partes: (1) análisis de patrones temporales, (2) forecast +1h y +24h,
    (3) inferencia en streaming sobre Kafka.

```mermaid
flowchart TB
    subgraph analisis["① Análisis Temporal"]
        HIST["Serie 30 días\n741 puntos"] --> HOUR["Ciclo diario\npeak 16:00=26.4°C\ncold 06:00=17.4°C"]
        HIST --> AUTO["Autocorr lag-24h\n= 0.739 ≥ 0.7 ✅"]
    end

    subgraph forecast["② Forecasting"]
        F1H["+1h → GBT+lags\nlag-shift actual→lag1\n±RMSE=0.92°C"]
        F24H["+24h → GBT base\nOpen-Meteo hourly API\nMAE vs API reference"]
        F24H -->|"JDBC"| WF[("weather_forecast\nPostgreSQL")]
    end

    subgraph streaming["③ Inferencia Streaming"]
        KAFKA["Kafka\nweather_topic"] --> MODEL["PipelineModel\nGBT base"]
        MODEL -->|"foreachBatch"| TP[("temp_predictions\nPostgreSQL")]
    end

    style WF fill:#10b981,color:#fff
    style TP fill:#10b981,color:#fff
```
""",
        "cells": [
            "s10_md", "s10_ts",
            "s10_forecast_md", "s10_forecast_1h", "s10_forecast_24h",
            "s10_stream",
        ],
    },
    "s11": {
        "file": "s11_tuning.md",
        "title": "S11 — Tuning y Experimentación Distribuida",
        "intro": """\
!!! abstract "Objetivo S11"
    Optimizar hiperparámetros con `TrainValidationSplit`.
    12 experimentos (6 LR + 6 GBT). Campeón persiste en PostgreSQL.

```mermaid
flowchart TB
    TRAIN["Training Set\n597 registros"] --> TVS["TrainValidationSplit\ntrainRatio=0.8"]

    subgraph LR_GRID["LR — 6 configs\nregParam × elasticNet"]
        L1["0.01/0.0"] & L2["0.01/0.5"] & L3["0.1/0.0"] & L4["0.1/0.5"] & L5["1.0/0.0"] & L6["1.0/0.5"]
    end
    subgraph GBT_GRID["GBT — 6 configs\nmaxDepth × maxIter"]
        G1["3×50 ⭐"] & G2["5×50"] & G3["5×30"] & G4["3×30"] & G5["7×50"] & G6["7×30"]
    end

    TVS --> LR_GRID & GBT_GRID
    G1 -->|"test RMSE=1.558°C"| FINAL["GBT Tuned S11\nR²=0.924 · RMSE/σ=0.28 ✅"]
    FINAL -->|"save"| FPATH["weather_temp_model_final"]
    FINAL -->|"INSERT"| PG[("model_metrics\ns11_experiments\nPostgreSQL")]

    style G1 fill:#10b981,color:#fff
    style FINAL fill:#ec4899,color:#fff
```

!!! success "Hallazgo S11"
    `maxDepth=7` es el **peor** resultado — con 597 muestras los árboles profundos sobreajustan.
    `maxDepth=3` generaliza mejor: **test RMSE=1.558°C, 47.4% mejor que LR campeón**.
""",
        "cells": [
            "s11_md", "s11_lr_tune", "s11_gbt_tune",
            "s11_final", "s11_pg_save", "crisp_deploy",
        ],
    },
    "arquitectura": {
        "file": "arquitectura.md",
        "title": "Arquitectura del Pipeline",
        "intro": "",   # esta página se escribe manualmente abajo
        "cells": [],
    },
}

SKIP_CELLS = {"c30"}

nb = json.loads(NOTEBOOK.read_text())
cell_map = {c.get("id", ""): c for c in nb["cells"]}


def cell_to_md(cell):
    src = "".join(cell["source"]).rstrip()
    if not src:
        return ""
    if cell["cell_type"] == "markdown":
        return src + "\n"
    if cell["cell_type"] == "code":
        return "```python\n" + src + "\n```\n"
    return ""


def outputs_to_md(cell):
    parts = []
    for out in cell.get("outputs", []):
        text = out.get("text", []) or out.get("data", {}).get("text/plain", [])
        if text:
            parts.append("".join(text).rstrip())
    if not parts:
        return ""
    combined = "\n".join(parts)
    lines = combined.split("\n")
    if len(lines) > 35:
        lines = lines[:35] + [f"... ({len(lines)-35} líneas omitidas)"]
        combined = "\n".join(lines)
    return f'\n??? output "Salida"\n{textwrap.indent(combined, "    ")}\n'


for section_key, cfg in SECTIONS.items():
    if section_key == "arquitectura":
        continue  # se genera manualmente más abajo

    chunks = [f"# {cfg['title']}\n", cfg.get("intro", ""), "---\n"]
    for cid in cfg["cells"]:
        if cid in SKIP_CELLS:
            continue
        cell = cell_map.get(cid)
        if cell is None:
            continue
        md = cell_to_md(cell)
        if md:
            chunks.append(md)
            if cell["cell_type"] == "code" and cell.get("outputs"):
                out = outputs_to_md(cell)
                if out:
                    chunks.append(out)
        chunks.append("")

    out_path = DOCS_DIR / cfg["file"]
    out_path.write_text("\n".join(chunks).strip() + "\n", encoding="utf-8")
    print(f"  ✓  {out_path}  ({out_path.stat().st_size:,} bytes)")

print("\nExportación completada.")
