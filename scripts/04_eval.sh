#!/bin/bash

# setcover
python branch2learn/04_evaluate.py -p setcover -m mlp1 -g 0
python branch2learn/04_evaluate.py -p setcover -m mlp2 -g 0
python branch2learn/04_evaluate.py -p setcover -m mlp3 -g 0
python branch2learn/04_evaluate.py -p setcover -m gnn1 -g 0
python branch2learn/04_evaluate.py -p setcover -m gnn2 -g 0

python branch2learn/04_evaluate.py -p setcover -m mlp1 -g -1
python branch2learn/04_evaluate.py -p setcover -m mlp2 -g -1
python branch2learn/04_evaluate.py -p setcover -m mlp3 -g -1
python branch2learn/04_evaluate.py -p setcover -m gnn1 -g -1
python branch2learn/04_evaluate.py -p setcover -m gnn2 -g -1

# cauctions
python branch2learn/04_evaluate.py -p cauctions -m mlp1 -g 0
python branch2learn/04_evaluate.py -p cauctions -m mlp2 -g 0
python branch2learn/04_evaluate.py -p cauctions -m mlp3 -g 0
python branch2learn/04_evaluate.py -p cauctions -m gnn1 -g 0
python branch2learn/04_evaluate.py -p cauctions -m gnn2 -g 0

python branch2learn/04_evaluate.py -p setcover -m mlp1 -g -1
python branch2learn/04_evaluate.py -p setcover -m mlp2 -g -1
python branch2learn/04_evaluate.py -p setcover -m mlp3 -g -1
python branch2learn/04_evaluate.py -p setcover -m gnn1 -g -1
python branch2learn/04_evaluate.py -p setcover -m gnn2 -g -1

# indset
python branch2learn/01_generate_data.py -p indset

python branch2learn/04_evaluate.py -p indset -m mlp1 -g 0
python branch2learn/04_evaluate.py -p indset -m mlp2 -g 0
python branch2learn/04_evaluate.py -p indset -m mlp3 -g 0
python branch2learn/04_evaluate.py -p indset -m gnn1 -g 0
python branch2learn/04_evaluate.py -p indset -m gnn2 -g 0

python branch2learn/04_evaluate.py -p setcover -m mlp1 -g -1
python branch2learn/04_evaluate.py -p setcover -m mlp2 -g -1
python branch2learn/04_evaluate.py -p setcover -m mlp3 -g -1
python branch2learn/04_evaluate.py -p setcover -m gnn1 -g -1
python branch2learn/04_evaluate.py -p setcover -m gnn2 -g -1

# facilities
#python branch2learn/04_evaluate.py -p facilities -m mlp1 -g 0
#python branch2learn/04_evaluate.py -p facilities -m mlp2 -g 0
#python branch2learn/04_evaluate.py -p facilities -m gnn1 -g 0
#python branch2learn/04_evaluate.py -p facilities -m gnn2 -g 0