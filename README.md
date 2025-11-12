# GNN-Traffic-predictor
This is a GNN for traffic metrics prediction such as congestion and road wear level.

## Authors
- Diego Andrés Combariza Puerto
- Nicolas Sarmiento Vargas

## Requisites

- Python 3.9
- install the requiriments.txt first :D

## Functionality

For training run the file `train.py` for more details you can check the file `config.py` to see the multiple options for training.
Then you can run the model interface which is `path_finder.py` with this interface you can visualize an specific snapshot and find a path 
between two nodes ( OSMD ID ) where the used algorithm is A* with the weights as the predictions and the heuristic the distance. To run this 
make sure you already have trained the model first and there is a `.pt` file in the `gnn_dataset` folder. Run this command:

``` 
python path_finder.py --start "9293223530" --end "4038648450" --snapshot-index 20
 ```

