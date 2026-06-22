"""
Convierte 03_04_kafka_pipeline_integrado.ipynb en páginas Markdown para MkDocs.
Cada sección del notebook (S6-S11 + intro + arquitectura) genera su propio .md.
"""
import json
import re
import textwrap
from pathlib import Path

NOTEBOOK = Path("notebooks/03_04_kafka_pipeline_integrado.ipynb")
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

# ── Mapeo celda → sección ─────────────────────────────────────────────────
SECTIONS = {
    "index": {
        "file": "index.md",
        "cells": ["c00", "c01", "c02"],
    },
    "s6": {
        "file": "s6_kafka.md",
        "cells": ["c03", "c04", "c05", "c10", "c11", "c12", "c13", "c18", "c19"],
    },
    "s7": {
        "file": "s7_streaming.md",
        "cells": ["c15", "c16", "c14", "c14b", "c14c", "c17", "c20", "c21", "c22", "c23", "c24"],
    },
    "s8": {
        "file": "s8_observabilidad.md",
        "cells": [
            "c25", "c26", "c27", "c27a", "c27a_code",
            "c27b", "c27c", "c27d", "c27e", "c28",
            "verify_md", "verify_code",
        ],
    },
    "s9": {
        "file": "s9_mllib.md",
        "cells": [
            "s9_md", "s9_data", "s9_spark_df",
            "s9_lr", "s9_gbt",
            "s9_lag_md", "s9_lag",
            "s9_pg_save", "s9_preds",
        ],
    },
    "s10": {
        "file": "s10_series_tiempo.md",
        "cells": ["s10_md", "s10_ts", "s10_stream"],
    },
    "s11": {
        "file": "s11_tuning.md",
        "cells": ["s11_md", "s11_lr_tune", "s11_gbt_tune", "s11_final"],
    },
    "arquitectura": {
        "file": "arquitectura.md",
        "cells": ["c31", "c31_superset"],
    },
}

# Celdas a omitir (cleanup, internals sin valor documental)
SKIP_CELLS = {"c30"}

# ── Cargar notebook ───────────────────────────────────────────────────────
nb = json.loads(NOTEBOOK.read_text())
cell_map = {c.get("id", ""): c for c in nb["cells"]}


def cell_to_md(cell: dict) -> str:
    src = "".join(cell["source"]).rstrip()
    if not src:
        return ""

    if cell["cell_type"] == "markdown":
        return src + "\n"

    if cell["cell_type"] == "code":
        # Strip leading shebang-style comments that are only noise
        lines = src.split("\n")
        return "```python\n" + "\n".join(lines) + "\n```\n"

    return ""


def outputs_to_md(cell: dict) -> str:
    """Render cell outputs (print / text) as a collapsed admonition."""
    parts = []
    for out in cell.get("outputs", []):
        text = []
        if out.get("output_type") in ("stream", "display_data", "execute_result"):
            text = out.get("text", []) or out.get("data", {}).get("text/plain", [])
        if text:
            parts.append("".join(text).rstrip())
    if not parts:
        return ""
    combined = "\n".join(parts)
    # Trim very long outputs
    lines = combined.split("\n")
    if len(lines) > 40:
        lines = lines[:40] + [f"... ({len(lines)-40} líneas omitidas)"]
        combined = "\n".join(lines)
    indented = textwrap.indent(combined, "    ")
    return f'\n??? output "Salida"\n{indented}\n'


# ── Generar páginas ────────────────────────────────────────────────────────
for section_key, cfg in SECTIONS.items():
    chunks = []
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
        chunks.append("")  # blank line between cells

    content = "\n".join(chunks).strip() + "\n"
    out_path = DOCS_DIR / cfg["file"]
    out_path.write_text(content, encoding="utf-8")
    print(f"  ✓  {out_path}  ({len(content):,} bytes)")

# ── Página de inicio extra: tabla de resultados fija ─────────────────────
results_extra = """
## Resultados Clave

| Modelo | Features | RMSE | MAE | R² | RMSE/σ |
|--------|----------|------|-----|-----|--------|
| LinearRegression | base (7) | 2.965°C | 2.435°C | 0.726 | 0.538 |
| GBTRegressor base | base (7) | 1.479°C | 1.029°C | 0.932 | 0.269 |
| GBTRegressor+lags | lag (10) | 0.922°C | 0.587°C | 0.974 | 0.167 |
| **GBT Tuned (S11)** | base (7) | **1.558°C** | — | **0.924** | 0.283 |

> **GBT+lags** es el mejor modelo batch (RMSE/σ = 0.17, mejora 37.7% vs GBT base).
> **GBT Tuned** es el modelo de producción seleccionado en S11 (streaming-compatible).
"""

idx_path = DOCS_DIR / "index.md"
idx_path.write_text(idx_path.read_text() + "\n" + results_extra, encoding="utf-8")
print(f"  ✓  Tabla de resultados añadida a index.md")
print("\nExportación completada.")
