
# SCIP solver

SCIP Optimization suite 7.0.2 (free for academic uses)

```
sudo apt-get install gfortran liblapack3 libtbb2 libcliquer1 libopenblas-dev libgsl23 -y
https://www.scipopt.org/download.php?fname=SCIPOptSuite-7.0.2-Linux-ubuntu.deb
sudo apt install ./SCIPOptSuite-7.0.2-Linux-ubuntu.deb
```


# Python dependencies

## Conda
```
https://docs.conda.io/en/latest/miniconda.html
git clone ...
cd branch2learn
conda env create -n ecole -f dev/conda.yaml
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

## Cuda

```
conda install -c anaconda cudatoolkit
```

## Extra Dependencies

```
conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
```