<h1 align="center">Branch to Learn</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
Learning the optimal branching variable in Mixed Integer Linear Programming branch-and-bound algorithms with graph convolutional neural networks and multi-layer perceptrons. This repository contains the source code for my master's thesis <a href="https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2781067">Ablating a Graph convolutional Network for Branching in Mixed-Integer Linear Programming</a>. The project uses <a href="https://github.com/ds4dm/ecole">Ecole by DS4DM</a>. 
</p>
<br> 


## About <a name = "about"></a>

This thesis is a continued exploration and development of ideas from two articles:

Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, Andrea Lodi [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://github.com/ds4dm/learn2branch) (2019).

Prateek Gupta, Maxime Gasse, Elias B. Khalil, M. Pawan Kumar, Andrea Lodi, Yoshua Bengio: [Hybrid Models for Learning to Branch](https://arxiv.org/abs/2006.15212) (2020)

And implements variations of these ideas in the freamework _Ecole_, which is found in the article:

Antoine Prouvost, Justin Dumouchelle, L. Scavuzzo, Maxime Gasse, D. Chételat, A. Lodi: [Ecole: A Gym-like Library for Machine Learning in Combinatorial Optimization Solvers](https://www.semanticscholar.org/paper/Ecole%3A-A-Gym-like-Library-for-Machine-Learning-in-Prouvost-Dumouchelle/e5f3f6d89be2f29eda70133fd83913229650d008)

This is the code for my master's thesis for a Master of Science in Engineering Cybernetics at the Norwegian University of Science and Technology. The title of the project is "Ablating a Graph Neural Network for Branching in Mixed-Integer Linear Programming". Feel free to contact me about any and all details of this project. 

## Installation


Installation instructions are given in [INSTALL.md](INSTALL.md)

## How to run it?
To recreate results, choose a problem, model and device. Following is an example of how to do this.

First, set the variables, for example:
```bash
PROBLEM=setcover
MODEL=mlp2
DEVICE=0
```

### Generate Dataset
```bash
python branch2learn/01_generate_data.py -p $PROBLEM
```

### Train model
```bash
python branch2learn/02_train.py -p $PROBLEM -m $MODEL -g $DEVICE
```

### Test model performance
```bash
python branch2learn/03_test.py -p $PROBLEM -m $MODEL -g $DEVICE
```

### Evaluate models
```bash
python branch2learn/04_evaluate.py -p $PROBLEM -m $MODEL -g $DEVICE
```

### Evaluate default policies
```bash
python branch2learn/05_evaluate_standard.py -p $PROBLEM -m fsb
```

Or, you can run all experiments with one bash script:
### Run all experiments
```bash
./scripts/run.sh
```

## Citation
Please cite the three original papers as well as my thesis if you use this code in your work, 
and feel free to contact @Sandbergo about anything related to the project.


## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs.
