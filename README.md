# BitBooster
This is the repository for the KDD 2021 paper

*BitBooster: Effective Approximation of the Euclidean Distance Metric via Binary Operations* 

## Introduction
The idea of this work is that we lower the precision of the data (to $n$ bits per value) and as a result lower the computational complexity of the Euclidean distance metric. The actual implementation of BitBooster (and its competitors) are part of the submodule BitBooster, this repo contains the experimental evaluation implementation.

## Reproducibility
This page is dedicated to the reproducibility of the paper. We discuss how to obtain the data, how to redo the experiments, and how to compute the tables and figures from the paper.

### Dependencies
The easiest way to run the code is to create a virtual Python 3.8.6 environment. All commands are below are executed from the directory that contains this git (bitbooster folder). Running the following in a Python environment installs all dependencies:

```
pip install -r bitbooster/requirements.txt
```

### Data
The Real experiments use data from the UCI repository https://archive.ics.uci.edu/ml/datasets. Full details on all datasets can be found in the file bitbooster/data/preprocessing, which also specifies which files should be downloaded for each dataset. After downloading, the datasets can be preprocessed

```
python bitbooster/data/preprocessing.py all
```

Any files that are not downloaded will be skipped

### Experiments
Each experiment can be executed separately:

```
python bitbooster/experiment_execution_script distance_experiment
```
```
python bitbooster/experiment_execution_script synthetic_experiment
```
```
python bitbooster/experiment_execution_script real_experiment
```

The real experiment requires user input to determine the eps values. Alternatively, you can call
```
python bitbooster/experiment_execution_script real_experiment skip_eps
```
which uses the same eps values for dbscan used by the paper.

### Results
After execution, the results can be extracted:
```
python bitbooster/paper.py
```
which creates several images and tex files that are used in the paper. These files are saved in ./paper_results