"""
Exporta 03_04_kafka_pipeline_integrado.ipynb a páginas Markdown enriquecidas para MkDocs.
Genera admonitions, bloques de código con copy, y secciones con contexto.
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
    Implementar un pipeline Kafka completo: tópico, contrato de evento, producer en background
    y consumer de verificación. Se usa **KRaft mode** (sin ZooKeeper) con un solo broker.

```mermaid
flowchart LR
    A["🌤️ Open-Meteo API"] -->|"GET cada 10 s"| B["kafka-python\nProducer Thread"]
    B -->|"JSON event"| C["Kafka\nweather_topic\n1 partición"]
    C -->|"KafkaConsumer\nverificación"| D["Consumer\n(S6 verify)"]
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
    Procesar el stream de Kafka con Spark Structured Streaming aplicando watermark
    y ventanas de tiempo. Comparar triggers y medir latencia/throughput.

```mermaid
flowchart LR
    K["Kafka\nweather_topic"] -->|"ReadStream"| P["parse JSON\nSchema tipado"]
    P -->|"withWatermark\n10 min"| W["window\n5 min tumbling"]
    W -->|"avg/min/max"| AGG["Agregaciones"]
    AGG -->|"foreachBatch"| PG[("PostgreSQL\nweather_windows")]
    AGG -->|"memory sink"| MEM["spark.sql()"]

    style W fill:#8b5cf6,color:#fff
    style PG fill:#10b981,color:#fff
```

!!! info "Parámetros clave"
    | Parámetro | Valor | Motivo |
    |-----------|-------|--------|
    | Watermark | 10 min | Tolera retrasos de la API externa |
    | Ventana | 5 min | Granularidad adecuada para temperatura |
    | Trigger | 10 s | Latencia baja sin overhead excesivo |
    | Output mode | `update` | Solo emite ventanas modificadas |
""",
        "cells": ["c15", "c16", "c14", "c14b", "c14c", "c17", "c20", "c21", "c22", "c23", "c24"],
    },
    "s8": {
        "file": "s8_observabilidad.md",
        "title": "S8 — Observabilidad",
        "intro": """\
!!! abstract "Objetivo S8"
    Exponer métricas del pipeline vía Prometheus, visualizarlas en Grafana
    y estimar costos de operación en producción.

```mermaid
flowchart LR
    SPARK["Spark\nStreaming Query"] -->|"recentProgress\ncada batch"| EXP["Prometheus\nExporter :8001"]
    EXP -->|"scrape 15 s"| PROM["Prometheus\n:9090"]
    PROM -->|"datasource"| GRAF["Grafana\n:3000"]
    PG[("PostgreSQL")] -->|"SQL datasource"| GRAF

    subgraph metricas["Métricas expuestas"]
        M1["spark_throughput_rows_per_sec"]
        M2["spark_latency_trigger_ms"]
        M3["spark_state_rows"]
        M4["spark_watermark_lag_s"]
    end

    EXP --> metricas

    style PROM fill:#e65100,color:#fff
    style GRAF fill:#f57c00,color:#fff
```

!!! warning "Nota sobre tmpfs"
    El docker-compose monta `/tmp` como `tmpfs` sin permisos de ejecución.
    Esto impide que Snappy cargue su librería nativa (`.so`).
    **Solución:** usar `spark.sql.parquet.compression.codec = uncompressed` al guardar modelos.
""",
        "cells": [
            "c25", "c26", "c27", "c27a", "c27a_code",
            "c27b", "c27c", "c27d", "c27e", "c28",
        ],
    },
    "s9": {
        "file": "s9_mllib.md",
        "title": "S9 — ML Distribuido con MLlib",
        "intro": """\
!!! abstract "Objetivo S9"
    Entrenar modelos de regresión con MLlib para predecir temperatura.
    Dataset: 741 registros históricos de Open-Meteo Archive (30 días).
    Comparar LinearRegression vs GBTRegressor con y sin lag features.

```mermaid
flowchart TB
    HIST["Open-Meteo Archive\n30 días · 741 registros"] -->|"feature engineering"| FE

    subgraph FE["Feature Engineering"]
        F1["hour_sin / hour_cos\ncyclic encoding"]
        F2["day_of_year\nestacionalidad"]
        F3["temp_lag1/2/3\nlag features (batch only)"]
    end

    FE --> SPLIT["Train/Test split\nrandomSplit 80/20\n597 / 144"]

    SPLIT --> LR["LinearRegression\nStandardScaler\nRMSE=2.97 · R²=0.73"]
    SPLIT --> GBT["GBTRegressor\nbase 7 features\nRMSE=1.48 · R²=0.93"]
    SPLIT --> LAG["GBTRegressor\n+ lags 10 features\nRMSE=0.92 · R²=0.97"]

    GBT -->|"save model"| SAVED["MODEL_PATH\n/work/models/weather_temp_model"]
    SAVED -->|"S10 →"| INF["Streaming\nInference"]

    style GBT fill:#ec4899,color:#fff
    style LAG fill:#6366f1,color:#fff
    style SAVED fill:#10b981,color:#fff
```

!!! success "Resultados"
    | Modelo | RMSE | R² | RMSE/σ | Uso |
    |--------|------|-----|--------|-----|
    | LinearRegression | 2.965°C | 0.726 | 0.54 | Baseline |
    | GBTRegressor base | 1.479°C | 0.932 | 0.27 | **Streaming** |
    | GBT + lag features | **0.922°C** | **0.974** | **0.17** | Batch/histórico |

    RMSE/σ < 0.6 → modelo aceptable para producción.
""",
        "cells": [
            "s9_md", "s9_data", "s9_spark_df",
            "s9_lr", "s9_gbt",
            "s9_lag_md", "s9_lag",
            "s9_pg_save", "s9_preds",
        ],
    },
    "s10": {
        "file": "s10_series_tiempo.md",
        "title": "S10 — Series de Tiempo e Inferencia Streaming",
        "intro": """\
!!! abstract "Objetivo S10"
    Analizar patrones temporales en la serie de temperatura (ciclo diario, autocorrelación)
    y aplicar inferencia en tiempo real: el stream de Kafka pasa por el modelo GBT guardado.

```mermaid
flowchart LR
    subgraph ts["Análisis Temporal"]
        HIST["Serie histórica\n741 puntos"] --> HOUR["Promedio por hora\npeak 16:00 = 26.4°C\ncold 06:00 = 17.4°C"]
        HIST --> AUTO["Autocorrelación\nlag-24h = 0.739 ≥ 0.7\n✅ ciclo diario confirmado"]
    end

    subgraph inf["Inferencia Streaming"]
        KAFKA["Kafka\nweather_topic"] -->|"ReadStream"| FEAT["Feature\nEngineering\nhour_sin/cos · day_of_year"]
        FEAT -->|"transform"| MODEL["GBT Model\n(cargado)"]
        MODEL -->|"foreachBatch"| PG[("temp_predictions\nPostgreSQL")]
        PG --> GRAF["Grafana\nReal vs Predicho"]
    end

    style MODEL fill:#ec4899,color:#fff
    style PG fill:#10b981,color:#fff
```

!!! info "Autocorrelación lag-24h"
    Un valor ≥ 0.7 confirma que la temperatura de hoy a las 14:00 es un buen predictor
    de la temperatura de mañana a las 14:00 — ciclo circadiano estadísticamente significativo.
""",
        "cells": ["s10_md", "s10_ts", "s10_stream"],
    },
    "s11": {
        "file": "s11_tuning.md",
        "title": "S11 — Tuning y Experimentación Distribuida",
        "intro": """\
!!! abstract "Objetivo S11"
    Optimizar hiperparámetros con `TrainValidationSplit` (equivalente distribuido de GridSearchCV).
    Se evalúan 6 configuraciones de LR y 6 de GBT → tabla de 12 experimentos.

```mermaid
flowchart TB
    TRAIN["Training Set\n597 registros"] --> TVS

    subgraph TVS["TrainValidationSplit\ntrainRatio=0.8"]
        subgraph LR_GRID["LR Grid — 6 configs"]
            LR1["regParam=0.01\nelasticNet=0.0"]
            LR2["regParam=0.01\nelasticNet=0.5"]
            LR3["regParam=0.1 ..."]
        end
        subgraph GBT_GRID["GBT Grid — 6 configs"]
            G1["maxDepth=3\nmaxIter=50"]
            G2["maxDepth=5\nmaxIter=50"]
            G3["maxDepth=7 ..."]
        end
    end

    TVS -->|"best LR"| BEST_LR["LR campeón\nregParam=0.01 · elasticNet=0.0\nval RMSE=2.648°C"]
    TVS -->|"best GBT"| BEST_GBT["GBT campeón\nmaxDepth=3 · maxIter=50\nval RMSE=1.738°C"]

    BEST_GBT -->|"test set"| FINAL["Test RMSE=1.558°C\nR²=0.924\n47.4% mejor que LR"]
    FINAL -->|"save"| MODEL["FINAL_MODEL_PATH\nweather_temp_model_final"]

    style BEST_GBT fill:#10b981,color:#fff
    style FINAL fill:#ec4899,color:#fff
```

!!! success "Hallazgo S11"
    Con `day_of_year` como feature, `maxDepth=7` ya no sobreajusta (antes era el peor).
    Features más ricas permiten árboles más profundos sin overfitting.
    **Champion: GBT maxDepth=3, test RMSE=1.558°C, R²=0.924.**
""",
        "cells": ["s11_md", "s11_lr_tune", "s11_gbt_tune", "s11_final"],
    },
}

SKIP_CELLS = {"c30"}

nb = json.loads(NOTEBOOK.read_text())
cell_map = {c.get("id", ""): c for c in nb["cells"]}


def cell_to_md(cell: dict) -> str:
    src = "".join(cell["source"]).rstrip()
    if not src:
        return ""
    if cell["cell_type"] == "markdown":
        return src + "\n"
    if cell["cell_type"] == "code":
        return "```python\n" + src + "\n```\n"
    return ""


def outputs_to_md(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs", []):
        text = (
            out.get("text", [])
            or out.get("data", {}).get("text/plain", [])
        )
        if text:
            parts.append("".join(text).rstrip())
    if not parts:
        return ""
    combined = "\n".join(parts)
    lines = combined.split("\n")
    if len(lines) > 35:
        lines = lines[:35] + [f"... ({len(lines)-35} líneas omitidas)"]
        combined = "\n".join(lines)
    indented = textwrap.indent(combined, "    ")
    return f'\n??? output "Salida"\n{indented}\n'


for section_key, cfg in SECTIONS.items():
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
                out_md = outputs_to_md(cell)
                if out_md:
                    chunks.append(out_md)
        chunks.append("")

    content = "\n".join(chunks).strip() + "\n"
    out_path = DOCS_DIR / cfg["file"]
    out_path.write_text(content, encoding="utf-8")
    print(f"  ✓  {out_path}  ({len(content):,} bytes)")

print("\nExportación completada.")
