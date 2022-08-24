# Coefficient Mutation in GP-GOMEA

This repo contains the code for the paper:
```
@inproceedings{virgolin2022coefficient,
author = {Virgolin, Marco and Bosman, Peter A. N.},
title = {Coefficient Mutation in the Gene-Pool Optimal Mixing Evolutionary Algorithm for Symbolic Regression},
year = {2022},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3520304.3534036},
doi = {10.1145/3520304.3534036},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
pages = {2289–2297},
numpages = {9},
}
```

## How to reproduce the experiments
You must install [GP-GOMEA](https://github.com/marcovirgolin/gp-gomea) and [SRBench](https://github.com/cavalab/srbench) to reproduce the experiments. 
See the respective READMEs.
A copy of [the GP benchmark collection by Luiz Otavio Vilas Boas Oliveira et al.](https://github.com/laic-ufmg/gp-benchmarks) is duplicated in this repo, no need to install this separately.

To reproduce the first part of the experiments, use `run.py` (`runbatch.sh` calls `run.py`).
To reproduce the second part, you must use `SRBench`, see the README of `SRBench`'s repo.
To obtain plots and info, use `analysis.ipynb`.
