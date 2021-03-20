<h1 align="center">Branch to Learn</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
Learning the optimal branching variable in Mixed Integer Linear Programming Brach & Bound algorithms with graph convolutional neural networks and multi-layer perceptrons. The project uses _Ecole_ by DS4DM. 
</p>
<br> 


## About <a name = "about"></a>

This project is a continued exploration and development of ideas from two articles:

Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, Andrea Lodi [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://github.com/ds4dm/learn2branch) (2019).

Prateek Gupta, Maxime Gasse, Elias B. Khalil, M. Pawan Kumar, Andrea Lodi, Yoshua Bengio: [Hybrid Models for Learning to Branch](https://arxiv.org/abs/2006.15212) (2020)

And implements variations of these ideas in the freamework _Ecole_, which is found in the article:

Antoine Prouvost, Justin Dumouchelle, L. Scavuzzo, Maxime Gasse, D. Chételat, A. Lodi: [Ecole: A Gym-like Library for Machine Learning in Combinatorial Optimization Solvers](https://www.semanticscholar.org/paper/Ecole%3A-A-Gym-like-Library-for-Machine-Learning-in-Prouvost-Dumouchelle/e5f3f6d89be2f29eda70133fd83913229650d008)

This is the code for my project thesis for a Master of Science in Engineering Cybernetics at the Norwegian University of Science and Technology. The title of the project is "Multi-Layer Perceptrons for Branching in Mixed-Integer Linear Programming". Feel free to contact me about any and all details of this project. 

## Installation


Follow installation instructions of [learn2branch](https://github.com/Sandbergo/learn2branch/blob/master/INSTALL.md) to install [SCIP](https://www.scipopt.org/) and PySCIPOpt.

Following python dependencies were used to run the code in this repository
```
torch==1.4.0.dev20191031
scipy==1.5.2
numpy==1.18.1
networkx==2.4
Cython==0.29.13
PySCIPOpt==2.1.5
scikit-learn==0.20.2
```

To setup this repo, follow
```bash
git clone https://github.com/Sandbergo/learn2branch.git
cd branch2learn
```

## How to run it?
In the instructions below we assumed that a bash variable `PROBLEM` exists. For example,
```bash
PROBLEM=setcover
```
Below instructions assume access to `data/` folder in the repo. Please look at the argument flags in each of the script to use another folder.

### Generate Instances
```bash
# generate instances
python learn2branch/01_generate_instances.py $PROBLEM
```

### Generate dataset
```bash
# generate dataset
python 02_generate_dataset.py $PROBLEM
```
### Train models

```bash
# GNN
python 03_train_gcnn_torch.py $PROBLEM # PyTorch version of learn2branch GNN

# COMP
python learn2branch/03_train_competitor.py $PROBLEM -m extratrees --hybrid_data_structure
python learn2branch/03_train_competitor.py $PROBLEM -m svmrank --hybrid_data_structure
python learn2branch/03_train_competitor.py $PROBLEM -m lambdamart --hybrid_data_structure

# MLP
python 03_train_mlp.py $PROBLEM

# Hybrid models
python 03_train_hybrid.py $PROBLEM -m concat --no_e2e # (pre)
python 03_train_hybrid.py $PROBLEM -m concat --no_e2e --distilled # (pre + KD)

python 03_train_hybrid.py $PROBLEM -m film --no_e2e # (pre)
python 03_train_hybrid.py $PROBLEM -m film --no_e2e --distilled # (pre + KD)

## CONCAT
python 03_train_hybrid.py $PROBLEM -m concat # (e2e)
python 03_train_hybrid.py $PROBLEM -m concat --distilled # (e2e + KD)

## FILM
python 03_train_hybrid.py $PROBLEM -m film # (e2e)
python 03_train_hybrid.py $PROBLEM -m film --distilled # (e2e + KD)

## HybridSVM
python 03_train_hybrid.py $PROBLEM -m hybridsvm # (e2e)
python 03_train_hybrid.py $PROBLEM -m hybridsvm --distilled  # (e2e + KD)

## HybridSVM-FiLM
python 03_train_hybrid.py $PROBLEM -m hybridsvm-film # (e2e)
python 03_train_hybrid.py $PROBLEM -m hybridsvm-film --distilled  # (e2e + KD)

# Auxiliary task (AT)
python 03_train_hybrid.py $PROBLEM -m film --at ED --beta_at 0.001 # (e2e + AT)
python 03_train_hybrid.py $PROBLEM -m film --distilled --at ED --beta_at 0.001 # (e2e + KD + AT)

# l2 regularization
python 03_train_hybrid.py $PROBLEM -m film --at ED --beta_at 0.001 --l2 0.001
```

### Test model performance
```bash
# test models

python 04_test_gcnn_torch.py $PROBLEM # GNN
python 04_test_mlp.py $PROBLEM # MLP

# ml-comp (COMP is the one with best accuracy)
python learn2branch/04_test.py $PROBLEM --no_gnn --ml_comp_brancher svmrank_khalil --hybrid_data_structure
python learn2branch/04_test.py $PROBLEM --no_gnn --ml_comp_brancher lambdamark_khalil --hybrid_data_structure
python learn2branch/04_test.py $PROBLEM --no_gnn --ml_comp_brancher extratrees_gcnn_agg --hybrid_data_structure

# Hybrid models
python 04_test_hybrid.py $PROBLEM # tests all available hybrid models in trained_models/$PROBLEM
```

### Evaluate models
```bash
# evaluate models

python 05_evaluate_gcnn_torch.py $PROBLEM -g -1 # GNN-CPU
python 05_evaluate_gcnn_torch.py $PROBLEM -g 0 # GNN-GPU
python 05_evaluate_mlp.py $PROBLEM -g -1

# COMP
python learn2branch/05_evaluate.py $PROBLEM --ml_comp_brancher use_best_performing_ml_competitor_folder_name --time_limit 2700 --no_gnn --hybrid_data_structure -g -1


# FiLM
python 05_evaluate_hybrid.py $PROBLEM -g -1 --model_string use_best_performing_model_folder_name


# internal branchers
python learn2branch/05_evaluate.py $PROBLEM --internal_brancher pscost --time_limit 2700 --no_gnn -g -1 --hybrid_data_structure # PB
python learn2branch/05_evaluate.py $PROBLEM --internal_brancher relpscost --time_limit 2700 --no_gnn  -g -1 --hybrid_data_structure # RPB
python learn2branch/05_evaluate.py $PROBLEM --internal_brancher fullstrong --time_limit 2700 --no_gnn  -g -1 --hybrid_data_structure # FSB
```

Follow instructions [here](https://github.com/Sandbergo/learn2branch/blob/master/RESULTS.md) to reproduce the evaluation results (Table 4).

## Citation
Please cite the original paper if you use this code in your work, and feel free to contact @Sandbergo for the specifics of my addition.


## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs.
