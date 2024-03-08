# Immune_Score_Caculator

![Workflow](./process.png)

## Package: `TOSICA`

We created the python package called `TOSICA` that uses `scanpy` ans `torch` to explainablely annotate cell type on single-cell RNA-seq data.

### Requirements

+ Linux/UNIX/Windows system
+ Python >= 3.8
+ torch == 1.7.1

### Create environment

```
conda create -n TOSICA python=3.8 scanpy
conda activate TOSICA
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2 cudatoolkit=10.1 -c pytorch
```

### Installation
