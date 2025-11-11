# EdgeGCN para red vial de Sogamoso

Esta guía explica cómo funciona el modelo EdgeGCN que entrena y predice variables por arista (ej. velocidad, congestión) sobre el grafo vial original. Cubre entradas/salidas, arquitectura, librerías, entrenamiento, inferencia, visualización y resolución de problemas.

## Qué problema resuelve

Dado un grafo vial G=(V,E) y un snapshot con el “estado de las vías” (features por arista), el modelo predice una o varias variables objetivo por arista (p. ej. velocidad, congestión) considerando:
- Embeddings de nodos (intersecciones) aprendidos con GCN sobre la topología original
- Atributos de cada arista (longitud, carriles, señales dinámicas, etc.)

## Estructura del dataset base (`gnn_dataset/`)

Ficheros clave:
- `edge_index.npy`: matriz (2, E) con pares (u, v) de cada arista usando índices de nodo 0..N-1
- `node_features.npy`: matriz (N, F_n) con features por nodo (si F_n=2 y parecen lon/lat, se usan como posiciones)
- `train_snapshots.pkl` y `val_snapshots.pkl` (o `temporal_snapshots.pkl` + `train_index.csv` / `val_index.csv`): lista de snapshots; cada snapshot contiene una matriz 2D de features de arista con forma (E, F_e_total)
- `graph_structure.json`: metadatos incluyendo `node_ids` (orden canónico de nodos)
- `positions.npy` (opcional; se genera si falta): (N, 2) lon/lat por nodo

Notas:
- Si hay `temporal_snapshots.pkl`, usa los índices CSV para reconstruir train/val.
- La clave dentro de cada snapshot que contiene la matriz de aristas puede variar; el script detecta automáticamente la matriz 2D con E filas (o E columnas, y entonces la transpone). También puedes forzar una clave con `--edge-attr-key`.

## Entradas y salidas del modelo

Entradas (por snapshot):
- Grafo estático: `edge_index` (2, E), `node_features` (N, F_n)
- Features de arista del snapshot: `edge_attr` (E, F_e_total)
- Selección de objetivos: índices `target_cols` dentro de `edge_attr` (p. ej. `[speed, congestion]`)
- Entradas del modelo = todas las columnas de `edge_attr` excepto `target_cols`

Salidas:
- Predicción por arista `y_hat`: (E, T) con T = número de objetivos (ej. 1 o 2), desnormalizada a unidades originales
- Artefactos del entrenamiento: `edge_gcn_best_model.pt`, `edge_gcn_model_ts.pt` (TorchScript opcional), `edge_gcn_scalers.json` (medias/std), `positions.npy`

## Normalización y fuga de datos

- Se calculan medias y desviaciones estándar solo con snapshots de entrenamiento (train) y se guardan en `edge_gcn_scalers.json`.
- Se aplican esas estadísticas a validación/inferencia. Esto evita fuga de datos (data leakage).

## Arquitectura

- Bloque GCN sobre nodos: 2 capas `GCNConv` (configurable) para generar embeddings de nodos h ∈ R^{N×H}
- Bloque por arista: para cada arista (u→v), se concatena `[h_u, h_v, edge_inputs]` y se pasa por una MLP para producir la(s) salida(s) por arista

Diagrama simplificado:
```
node_features (N,F_n) --GCNConv--> (N,H) --GCNConv--> (N,H) = h
for each edge (u,v): z = concat(h[u], h[v], edge_feat[u,v]) --MLP--> y_hat[u,v]
```

Hiperparámetros principales:
- `hidden` (dimensión H), `gcn_layers` (capas GCN), `dropout` en la MLP
- Pérdida: MSE sobre objetivos normalizados; métricas en escala original (MAE/RMSE)
- Early stopping por `val_loss`

## Librerías

- PyTorch: entrenamiento (`torch`, `torch.nn`, `torch.optim`)
- PyTorch Geometric: capas GCN (`torch_geometric.nn.GCNConv`)
- Numpy, Matplotlib

## Entrenamiento (script `train_edge_gcn.py`)

El script se auto-configura para encontrar `gnn_dataset` esté:
- junto al script
- o si ejecutas desde dentro de `gnn_dataset`

Opciones útiles:
- `--epochs 60` épocas de entrenamiento
- `--target-cols 0 1` selecciona índices de columnas objetivo en `edge_attr`
- `--target-name-substrings speed congestion` si tienes un JSON con nombres de columnas (`--feature-names-json`)
- `--edge-attr-key features` fuerza la clave del snapshot que contiene la matriz (E, F_e_total)

Artefactos generados en `gnn_dataset/`:
- `edge_gcn_best_model.pt`
- `edge_gcn_model_ts.pt` (si el tracing tiene éxito)
- `edge_gcn_scalers.json`
- `positions.npy` (si no existía)

## Inferencia

Pasos:
1) Carga artefactos (`edge_index.npy`, `node_features.npy`, `edge_gcn_best_model.pt`, `edge_gcn_scalers.json`)
2) Prepara el `edge_attr` del snapshot a evaluar (misma estructura y orden de columnas que en train)
3) Separa entradas (`edge_inputs = edge_attr[:, ~mask_targets]`), normaliza con `in_mean/in_std`
4) Pasa por el modelo y desnormaliza con `t_mean/t_std`
5) (Opcional) Guarda CSV por arista y visualiza

Un ejemplo completo de inferencia está incluido en los mensajes anteriores; puedes convertirlo en `infer_edge_gcn.py` si lo deseas.

## Visualización

- El script de entrenamiento genera `positions.npy` automáticamente si:
  - `node_features` parece lon/lat (rango plausible), o
  - encuentra `nodes.geojson` y `graph_structure.json` para mapear `node_ids` a lon/lat
- La visualización colorea aristas por la predicción del primer objetivo usando `matplotlib` (LineCollection) con escala por percentiles [2,98].

## Edición manual de estado vial (idea de UI)

Se puede crear una pequeña interfaz (ej. Streamlit) para:
- Editar columnas específicas de `edge_attr` (density, speed_limit, incidentes) por arista o en bloque
- Ejecutar inferencia con el EdgeGCN y visualizar
- Exportar el `edge_attr` modificado y las predicciones para comparar con una corrida SUMO real

Un prototipo de Streamlit fue incluido en la conversación; puede añadirse como `apps/edge_editor_app.py`.

## Resolución de problemas

- "No encuentra torch_geometric": probablemente está instalado en otro intérprete. Asegúrate de activar el venv y usa `python -m pip install ...` con ese intérprete.
- "KeyError: 'edge_attr'": la clave del snapshot no es `edge_attr`. El script ya detecta la matriz de aristas automáticamente o usa `--edge-attr-key`.
- "index out of bounds en TorchScript": ya se corrigió el tracing para usar todos los nodos y un subconjunto de aristas; si aún falla, el script lo omite con un warning.
- "positions.npy faltante": se genera desde `node_features` si son lon/lat o desde `data/nodes.geojson` + `graph_structure.json`.
- Doble `graph_structure.json`: usa el de `gnn_dataset` como fuente canónica (IDs OSM). 
- `processed_gnn_dataset` describe un line graph (nodos=aristas); para EdgeGCN usa SIEMPRE `gnn_dataset`.

## Buenas prácticas

- Mantén documentados `target_cols` y el orden de columnas de `edge_attr` para reproducibilidad
- No incluyas las columnas objetivo como entradas del modelo (evita fuga)
- Calcula estadísticas de normalización solo con train
- Guarda las versiones de librerías usadas (torch, torch_geometric)

## Glosario rápido

- N: número de nodos (intersecciones)
- E: número de aristas (segmentos viales)
- F_n: dimensión de features de nodo
- F_e_total: dimensión total de features de arista
- T: número de objetivos (targets) por arista

## Archivos relacionados

- `train_edge_gcn.py`: entrenamiento + exportación + visualización
- `gnn_dataset/`: carpeta con datos y artefactos
- `README_EDGE_GCN.md`: este documento

---
Si quieres, puedo agregar un script `infer_edge_gcn.py` y una app Streamlit `apps/edge_editor_app.py` para edición e inferencia interactiva. Pídelo y lo integro al repo.
