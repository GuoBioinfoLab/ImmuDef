# ImmuDef

![Workflow](./Fig_1.tif)

## Package: `ImmuDef: a Novel Method for Quantitative Evaluation of Anti-infection Immune Defense Function and its Application`

We created a python package called "ImmuDef" that uses RNA-seq data to compute quantitative assessments of an individual's immune defense function.

# Requirements
## Python Package Requirements
- Python 3.10
- scikit-learn
- numpy
- pandas
## R Package Requirements
- R 4.3.3
- getopt
- tidyverse
- GSVA
- clusterProfiler
- msigdbr
  
## Test

Run test.py by `python ./main.py `

## Transfer Learn to Your Own Dataset

- Prepare your dataset as a csv file which is ssGSEA data or RNA-seq data.

## Start Compution
- Import this package.
  `from immune_score.score_caculator import Immune_Score_Caculator`
- Read your `csv` file as a `pandas.DataFrame`.
  `data = pd.read_csv('Data/data/ssgsea_TB_c7.csv.bz2', index_col=0)`.
- Compute immune scores.
  `isc = Immune_Score_Caculator()`
  `immune_scores = isc.score_compute(data=data, use_all_to_nor=True)`

# Cite

If you make use of the code/experiment in your work, please cite our paper (Bibtex below).

@article{
title={''},
author={''},
year={2024}
}
