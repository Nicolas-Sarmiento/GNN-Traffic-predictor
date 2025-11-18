# GNN-Traffic-predictor
This is a GNN for traffic metrics prediction such as congestion and road wear level.

## Authors
- Diego Andrés Combariza Puerto
- Nicolas Sarmiento Vargas

## Requisites

- Python 3.9
- install the requiriments.txt first :D

## Functionality

For training run the file `train.py`; for more details you can check `config.py` to see the available hyper-parameters (including `--dataset-dir` to point at a dataset and `--show-plots` to pop up the generated graphs). Each run writes the model/scalers plus per-epoch loss and accuracy (R²) curves into `gnn_dataset/reports`.
After training, run `evaluate.py` to load the saved checkpoint, compute MAE/MSE/RMSE/R² metrics, and generate the accuracy-style plots stored under `gnn_dataset/reports` by default.
An example:
```
 python train.py --epochs 1000 --patience 1000 --gcn-layer 8 --dropout 0.4 --weight-decay 1e-3
```

Then you can run the model interface which is `path_finder.py` with this interface you can visualize an specific snapshot and find a path 
between two nodes ( OSMD ID ) where the used algorithm is A* with the weights as the predictions and the heuristic the distance. To run this 
make sure you already have trained the model first and there is a `.pt` file in the `gnn_dataset` folder. Run this command:

``` 
python path_finder.py --start "<osmid start>" --end "<osmid end>" --snapshot-index 1
 ```

--start flag is for start node osmid, --end stands for end node osmid and --snapshot-index is the argument of the used input snapshot. In this repository contains a snapshot with 20 indexes from 0 to 19.

