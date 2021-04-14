#!/bin/bash

# setcover
python branch2learn/02_train.py -p setcover -m mlp1 -g 0
python branch2learn/02_train.py -p setcover -m mlp2 -g 0

python branch2learn/04_evaluate.py -p setcover -m mlp1 -g 0
python branch2learn/04_evaluate.py -p setcover -m mlp2 -g 0

python branch2learn/04_evaluate.py -p setcover -m mlp1 -g -1
python branch2learn/04_evaluate.py -p setcover -m mlp2 -g -1