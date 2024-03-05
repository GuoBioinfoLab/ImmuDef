#!/path/to/Rscript
library('getopt')
library(tidyverse)
library(GSVA)
library(clusterProfiler)
library(msigdbr)

# 定义参数规格
spec = matrix(c(
  'input', 'i', 1, "character",
  'output', 'o', 1, "character"
), byrow=TRUE, ncol=4)

# 解析命令行参数
opt = getopt(spec)

# 在这里添加你的代码
tempfile <- opt$input
cov_group <- read.csv(tempfile, row.names = 1)

gene <- rownames(cov_group)
#开始ID转换，会有丢失
gene = bitr(gene,fromType="SYMBOL",toType="ENTREZID",OrgDb="org.Hs.eg.db") 

#去重
gene <- dplyr::distinct(gene,SYMBOL,.keep_all=TRUE)
cov_group <- cov_group[gene$SYMBOL,]
rownames(cov_group) <- gene$ENTREZID

##C7 Geneset
C7 <-  msigdbr(species = "Homo sapiens", # Homo sapiens or Mus musculus
               category = "C7") 
c7_df <- dplyr::select(C7,
                       gs_name,
                       gs_exact_source,
                       human_entrez_gene)

c7_list <- split(c7_df$human_entrez_gene, c7_df$gs_name) ##按照gs_name给gene_symbol

ssgsea_mat <- gsva(expr=as.matrix(cov_group),
                   method = "ssgsea",
                   gset.idx.list=c7_list, 
                   kcdf="Poisson" ,#"Gaussian" for logCPM,logRPKM,logTPM, "Poisson" for counts
                   verbose=T, 
                   parallel.sz = parallel::detectCores())#调用所有核

# 将DataFrame写入CSV文件
write.csv(t(as.data.frame(ssgsea_mat)), opt$output)