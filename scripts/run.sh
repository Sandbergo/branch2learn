#!/bin/bash

problems=( setcover cauctions facilities indset )
models=( mlp1 mlp2 mlp3 gnn1 gnn2)
policies=( fsb pc rpc)


for PROBLEM in "${problems[@]}"
do
    python branch2learn/01_generate_data.py -p $PROBLEM

	for MODEL in "${models[@]}"
    do
        python branch2learn/02_train.py    -p $PROBLEM -m $MODEL -g 0
        python branch2learn/03_test.py     -p $PROBLEM -m $MODEL -g 0
        python branch2learn/04_evaluate.py -p $PROBLEM -m $MODEL -g 0
        python branch2learn/04_evaluate.py -p $PROBLEM -m $MODEL -g -1
        
    done

    for POLICY in "${policies[@]}"
    do
        python branch2learn/05_evaluate_standard.py -p $PROBLEM -b $POLICY
    done
done
