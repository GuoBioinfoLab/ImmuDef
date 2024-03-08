# Immune_Score_Caculator

![Workflow](./process.png)

## Package: `TOSICA`

We created the python package called `TOSICA` that uses `scanpy` ans `torch` to explainablely annotate cell type on single-cell RNA-seq data.

# Requirements
- Python 3.9
- scikit-learn
- numpy
- jupyter
- notebook
- pandas

# Datasets
The SMRT dataset is collect from [this paper](https://doi.org/10.1038/s41467-019-13680-7)
Datasets for transfer learning is download from [PredRet](http://predret.org/)

# Usage

## Validation and Test

Run test.py by `python ./test.py `

## Transfer Learn to Your Own Dataset

- Prepare your dataset as a csv file which has "InChI" and "RT" columns.
- Rename it as "data.csv" at the root directory.
- download the pre-trained model from [huggingface](https://huggingface.co/spaces/Xue-Jun/RT-Transformer/tree/main).
- Run transfer.py

You can also follow this [jupyter notebook](./) to fine-tuning the model.

## Retrain the Model
- Prepare your dataset as a csv file which has "InChI" and "RT" columns.
- Rename it as "data.csv" at the root directory.
- Run train.py

## Pretrained Model Files

best_state_download_dict.pth The Best model of RT-Transformer train from retained data.
best_state_dict.pth The Best model of RT-Transformer train from full data.

# Cite

If you make use of the code/experiment in your work, please cite our paper (Bibtex below).

@article{xue2023rt,
title={RT-Tranformer: Retention Time Prediction for Metabolite Annotation to Assist in Metabolite Identification},
author={Jun Xue and Bingyi Wang and Hongchao Ji and Weihua Li },
year={2023}
}
