# ImmuDef

![Workflow](./process.svg)

## Project: `A deep learning method for quantitative immune defense function evaluation and its clinical prognostication`

The immune defense function protecting body from invasive patho-gens is a key indicator of health. However, it lacks of methods for quantita-tive and precise evaluation. This study introduces ImmuDef, a novel algo-rithm that provides a precise and quantitative assessment of anti-infection immune defense function based on RNA-seq data. ImmuDef employs the following pipeline: (1) transforming gene expression into robust immune signatures of MSigDB C7 immunologic gene sets; (2) selecting immune re-lated features through comparisons of AIDS vs. healthy controls (HC) and dead-survived sepsis patients; (3) dimensionality reduction and latent space construction via a variational autoencoder model, QImmuDef-VAE. Build-ing on QImmuDef-VAE’s latent representation, a defense immune score (DImmuScore) was calculated by measuring the distance between a patient and HC within the latent space, using healthy individuals as reference. We validated ImmuDef on 3,202 samples across HC and different infectious diseases, which were stratified into four immune states: immunodeficiency, immunocompromised, immunocompetent, and immunoactive. As a result, DImmuScore calculated by ImmuDef achieves high classification accuracy (AUC: 0.79-1.00) among samples with various immune states and infection. And in AIDS patients, lower DImmuScore predicted mortality, aligning with immune reconstitution dynamics. Furthermore, DImmuScore can serve as a metric for disease severity across hepatitis B, tuberculosis, and COVID-19 cohorts, where its gradient directly quantifies disease progression. As an ap-plication, the DImmuScore can be a strong prognostic indicator, effectively stratifying mortality/survival in both sepsis and COVID-19 patients with same symptom. These results indicate that ImmuDef can accurately reflect immune function and disease states of infectious diseases. By applying deep learning on immune pathway features, we designed the first algorithm for quantitatively calculating immunity of anti-infection. It enables accurate evaluation of the immune defense function, offering clinical potential for disease surveillance and prognosis prediction.

# Requirements
## Python Package Requirements
- Python 3.10
- scikit-learn 1.2.2
- numpy 1.23.5
- pandas 1.5.3
- pytorch 2.0.1
## R Package Requirements
- R 4.3.2
- getopt 1.20.4 
- tidyverse 2.0.0
- GSVA 1.50.1
- clusterProfiler 4.10.1
- msigdbr 7.5.1
# 🧬 ssGSEA Immune Pathway Enrichment Analysis (C7 Gene Sets)

This R script performs **single-sample Gene Set Enrichment Analysis (ssGSEA)** using the **MSigDB C7 immunologic signature gene sets**. It takes a gene expression matrix as input and outputs enrichment scores for each sample across all C7 pathways.

---

## 📦 Required R Packages

Make sure the following packages are installed:


> ⚠️ **Note**: This script only supports **human (Homo sapiens)** data due to the use of `org.Hs.eg.db`.

---

## 📁 Input Format

### ✅ Requirements
- **File type**: CSV (comma-separated)
- **Rows**: Genes (using official **HGNC gene symbols**, e.g., `TP53`, `CD3D`)
- **Columns**: Samples
- **Values**: Expression values  
  - Use **raw counts** (e.g., from RNA-seq) → script uses `kcdf="Poisson"`  
  - For **log-transformed data** (e.g., log2(TPM+1)), you must modify the script to use `kcdf="Gaussian"`



## Usage
Example (`input.csv`):
```bash
Rscript Rscript_ssGSEA.R -i input_file.csv -o output_file.csv
```
