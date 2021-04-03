#!/bin/bash

# setcover
python branch2learn/03_test.py -p setcover -m mlp1 -g 0
python branch2learn/03_test.py -p setcover -m mlp2 -g 0
python branch2learn/03_test.py -p setcover -m mlp3 -g 0
python branch2learn/03_test.py -p setcover -m gnn1 -g 0
python branch2learn/03_test.py -p setcover -m gnn2 -g 0

# cauctions
python branch2learn/03_test.py -p cauctions -m mlp1 -g 0
python branch2learn/03_test.py -p cauctions -m mlp2 -g 0
python branch2learn/03_test.py -p cauctions -m mlp3 -g 0
python branch2learn/03_test.py -p cauctions -m gnn1 -g 0
python branch2learn/03_test.py -p cauctions -m gnn2 -g 0

# indset
python branch2learn/03_test.py -p indset -m mlp1 -g 0
python branch2learn/03_test.py -p indset -m mlp2 -g 0
python branch2learn/03_test.py -p indset -m mlp3 -g 0
python branch2learn/03_test.py -p indset -m gnn1 -g 0
python branch2learn/03_test.py -p indset -m gnn2 -g 0

# facilities
#python branch2learn/03_test.py -p facilities -m mlp1 -g 0
#python branch2learn/03_test.py -p facilities -m mlp2 -g 0
#python branch2learn/03_test.py -p facilities -m gnn1 -g 0
#python branch2learn/03_test.py -p facilities -m gnn2 -g 0