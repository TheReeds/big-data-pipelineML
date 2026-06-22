---
## 15. S11 — Tuning y Experimentación Distribuida

**Objetivo:** optimizar el modelo vía búsqueda de hiperparámetros con  
`TrainValidationSplit` (alternativa más rápida a `CrossValidator` para demostraciones).

| Hiperparámetro | Valores a probar |
|----------------|-----------------|
| `regParam` (LR) | 0.01, 0.1, 1.0 |
| `elasticNetParam` (LR) | 0.0, 0.5 |
| `maxDepth` (GBT) | 3, 5, 7 |
| `maxIter` (GBT) | 30, 50 |


```python
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

print("=== S11 — Tuning LinearRegression ===")

assembler_t = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
scaler_t    = StandardScaler(inputCol="features_raw", outputCol="features",
                             withMean=True, withStd=True)
lr_t        = LinearRegression(featuresCol="features", labelCol=LABEL_COL)
pipe_lr_t   = Pipeline(stages=[assembler_t, scaler_t, lr_t])

param_grid_lr = (
    ParamGridBuilder()
    .addGrid(lr_t.regParam,       [0.01, 0.1, 1.0])
    .addGrid(lr_t.elasticNetParam, [0.0, 0.5])
    .build()
)

tvs_lr = TrainValidationSplit(
    estimator=pipe_lr_t,
    estimatorParamMaps=param_grid_lr,
    evaluator=evaluator_rmse,
    trainRatio=0.8,
    parallelism=2,
)

print(f"Evaluando {len(param_grid_lr)} configuraciones LR (train/val split)...")
tvs_model_lr = tvs_lr.fit(train_df)

# Tabla de resultados
lr_rows = []
for params, metric in zip(param_grid_lr, tvs_model_lr.validationMetrics):
    lr_rows.append({
        "modelo":          "LinearRegression",
        "regParam":        params[lr_t.regParam],
        "elasticNetParam": params[lr_t.elasticNetParam],
        "maxDepth":        "-",
        "maxIter":         "-",
        "val_RMSE":        round(metric, 4),
    })

df_tune_lr = pd.DataFrame(lr_rows).sort_values("val_RMSE")
print(df_tune_lr.to_string(index=False))
best_lr_row = df_tune_lr.iloc[0]
print(f"\nMejor LR — regParam={best_lr_row['regParam']}, "
      f"elasticNetParam={best_lr_row['elasticNetParam']}, "
      f"val_RMSE={best_lr_row['val_RMSE']}")
```


??? output "Salida"
    === S11 — Tuning LinearRegression ===
    Evaluando 6 configuraciones LR (train/val split)...
              modelo  regParam  elasticNetParam maxDepth maxIter  val_RMSE
    LinearRegression      1.00              0.0        -       -    2.7747
    LinearRegression      0.10              0.5        -       -    2.7767
    LinearRegression      0.10              0.0        -       -    2.7867
    LinearRegression      0.01              0.5        -       -    2.7973
    LinearRegression      0.01              0.0        -       -    2.7987
    LinearRegression      1.00              0.5        -       -    2.9351

    Mejor LR — regParam=1.0, elasticNetParam=0.0, val_RMSE=2.7747


```python
print("=== S11 — Tuning GBTRegressor ===")

assembler_g = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
gbt_t       = GBTRegressor(featuresCol="features", labelCol=LABEL_COL, stepSize=0.1)
pipe_gbt_t  = Pipeline(stages=[assembler_g, gbt_t])

param_grid_gbt = (
    ParamGridBuilder()
    .addGrid(gbt_t.maxDepth, [3, 5, 7])
    .addGrid(gbt_t.maxIter,  [30, 50])
    .build()
)

tvs_gbt = TrainValidationSplit(
    estimator=pipe_gbt_t,
    estimatorParamMaps=param_grid_gbt,
    evaluator=evaluator_rmse,
    trainRatio=0.8,
    parallelism=2,
)

print(f"Evaluando {len(param_grid_gbt)} configuraciones GBT...")
tvs_model_gbt = tvs_gbt.fit(train_df)

gbt_rows = []
for params, metric in zip(param_grid_gbt, tvs_model_gbt.validationMetrics):
    gbt_rows.append({
        "modelo":          "GBTRegressor",
        "regParam":        "-",
        "elasticNetParam": "-",
        "maxDepth":        params[gbt_t.maxDepth],
        "maxIter":         params[gbt_t.maxIter],
        "val_RMSE":        round(metric, 4),
    })

df_tune_gbt = pd.DataFrame(gbt_rows).sort_values("val_RMSE")
print(df_tune_gbt.to_string(index=False))
best_gbt_row = df_tune_gbt.iloc[0]
print(f"\nMejor GBT — maxDepth={best_gbt_row['maxDepth']}, "
      f"maxIter={best_gbt_row['maxIter']}, "
      f"val_RMSE={best_gbt_row['val_RMSE']}")
```


??? output "Salida"
    === S11 — Tuning GBTRegressor ===
    Evaluando 6 configuraciones GBT...
          modelo regParam elasticNetParam  maxDepth  maxIter  val_RMSE
    GBTRegressor        -               -         3       50    1.5774
    GBTRegressor        -               -         5       50    1.6797
    GBTRegressor        -               -         5       30    1.7383
    GBTRegressor        -               -         3       30    1.7857
    GBTRegressor        -               -         7       50    2.3047
    GBTRegressor        -               -         7       30    2.3238

    Mejor GBT — maxDepth=3, maxIter=50, val_RMSE=1.5774


```python
# Tabla unificada de todos los experimentos (S9 baseline + S11 tuning)
print("=== S11 — Tabla Completa de Experimentos ===")

df_all_experiments = pd.concat([df_tune_lr, df_tune_gbt], ignore_index=True)
df_all_experiments = df_all_experiments.sort_values("val_RMSE").reset_index(drop=True)
df_all_experiments.index += 1
print(df_all_experiments.to_string())

# Evaluar el mejor modelo de cada familia en test set (holdout real)
best_lr_final  = tvs_model_lr.bestModel
best_gbt_final = tvs_model_gbt.bestModel

preds_lr_final  = best_lr_final.transform(test_df)
preds_gbt_final = best_gbt_final.transform(test_df)

rmse_lr_f  = evaluator_rmse.evaluate(preds_lr_final)
rmse_gbt_f = evaluator_rmse.evaluate(preds_gbt_final)
r2_lr_f    = evaluator_r2.evaluate(preds_lr_final)
r2_gbt_f   = evaluator_r2.evaluate(preds_gbt_final)

print()
print("=== S11 — Evaluación en TEST SET (modelo óptimo de cada familia) ===")
print(f"  LinearRegression (best): RMSE={rmse_lr_f:.3f} °C | R²={r2_lr_f:.4f}")
print(f"  GBTRegressor     (best): RMSE={rmse_gbt_f:.3f} °C | R²={r2_gbt_f:.4f}")
print()

# Observación sobre profundidad de árboles
gbt_depth_summary = (
    df_tune_gbt.groupby("maxDepth")["val_RMSE"]
    .mean().round(4).sort_index()
)
print("Efecto de maxDepth (val_RMSE promedio):")
for d, rmse in gbt_depth_summary.items():
    note = " ← mejor generalización" if d == gbt_depth_summary.idxmin() else ""
    print(f"  maxDepth={d}: {rmse}{note}")
print("  (más profundidad → más overfitting con 720 muestras)")

# Siempre guardar el mejor GBT (pipeline sin StandardScaler — serialización limpia)
FINAL_MODEL_PATH = "/home/jovyan/work/models/weather_temp_model_final"
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
best_gbt_final.write().overwrite().save(FINAL_MODEL_PATH)
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")

champ_rmse = rmse_gbt_f
print(f"\nModelo campeón (GBTRegressor, maxDepth=3) guardado en {FINAL_MODEL_PATH}")
print(f"RMSE test: {champ_rmse:.3f} °C | R²: {r2_gbt_f:.4f}")
print(f"Mejora sobre LR baseline: {((rmse_lr_f - rmse_gbt_f) / rmse_lr_f * 100):.1f}% menos RMSE")
```


??? output "Salida"
    === S11 — Tabla Completa de Experimentos ===
                  modelo regParam elasticNetParam maxDepth maxIter  val_RMSE
    1       GBTRegressor        -               -        3      50    1.5774
    2       GBTRegressor        -               -        5      50    1.6797
    3       GBTRegressor        -               -        5      30    1.7383
    4       GBTRegressor        -               -        3      30    1.7857
    5       GBTRegressor        -               -        7      50    2.3047
    6       GBTRegressor        -               -        7      30    2.3238
    7   LinearRegression      1.0             0.0        -       -    2.7747
    8   LinearRegression      0.1             0.5        -       -    2.7767
    9   LinearRegression      0.1             0.0        -       -    2.7867
    10  LinearRegression     0.01             0.5        -       -    2.7973
    11  LinearRegression     0.01             0.0        -       -    2.7987
    12  LinearRegression      1.0             0.5        -       -    2.9351

    === S11 — Evaluación en TEST SET (modelo óptimo de cada familia) ===
      LinearRegression (best): RMSE=3.093 °C | R²=0.7021
      GBTRegressor     (best): RMSE=1.558 °C | R²=0.9244

    Efecto de maxDepth (val_RMSE promedio):
      maxDepth=3: 1.6816 ← mejor generalización
      maxDepth=5: 1.709
      maxDepth=7: 2.3142
      (más profundidad → más overfitting con 720 muestras)

    Modelo campeón (GBTRegressor, maxDepth=3) guardado en /home/jovyan/work/models/weather_temp_model_final
    RMSE test: 1.558 °C | R²: 0.9244
    Mejora sobre LR baseline: 49.6% menos RMSE
