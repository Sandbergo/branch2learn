
# SCIP solver

SCIP Optimization suite 7.0.2 (free for academic uses)

```
https://www.scipopt.org/download.php?fname=SCIPOptSuite-7.0.2-Linux-ubuntu.deb
```


# Python dependencies

## Conda
```
conda env create -f dev/conda.yaml
```


## PySCIPOpt

SCIP's python interface (modified version)

```
conda install -c conda-forge ecole pyscipopt
```

## Ecole
```
git clone https://github.com/ds4dm/ecole
cd ecole
cmake -B build/
cmake --build build/ --parallel
python -m pip install build/python
cd ..
```