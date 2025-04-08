# 导入所需的库
import sklearn
import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.svm import SVC  # 支持向量机
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report  # 用于评估准确率和分类报告
import xgboost as xgb
import joblib  # 用于保存模型

# 加载数据
with open("/new_immudef/dis_data_df.pkl", "rb") as f:
    data_dict = pickle.load(f)

data_dict = {k:v[np.load('D:/new_immudef/infection_fea_619.npy')] for k,v in data_dict.items()}
for k,v in data_dict.items():
    v['label'] = k
dataset = pd.concat(data_dict.values())

dataset['label'] = dataset['label'].replace({
    'COVID': 'COVID-19',
    'Control': 'Healthy Control',
    'TB': 'Tuberculosis',
    'Candida': 'Fungus',
    'HIV': 'AIDS'
})

# 步骤 1: 标签转换 (如果标签是字符串，则需要转换为数值)
label_encoder = LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['label'])

# 保存LabelEncoder
label_encoder_path = "label_encoder.pkl"
joblib.dump(label_encoder, label_encoder_path)
print(f"LabelEncoder已保存到 {label_encoder_path}")

# 步骤 2: 切分特征和标签
X = dataset.drop('label', axis=1)  # 特征部分
y = dataset['label']  # 标签部分

# 使用 sklearn 工具将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# SVM模型
svm_params = {
    'C': 0.5,
    'cache_size': 2048,
    'class_weight': 'balanced',
    'gamma': 'scale',
    'kernel': 'poly',
    'max_iter': 1000,
    'probability': True,
    'shrinking': True,
    'tol': 0.0001
}

svm_model = SVC(**svm_params)
svm_model.fit(X_train, y_train)  # 训练SVM模型
svm_pred = svm_model.predict(X_test)  # 在测试集上预测

# 输出SVM模型的准确率
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM模型在测试集上的准确率: {svm_accuracy * 100:.2f}%")

# 输出SVM模型每个类别的分类报告
print("SVM模型每个类别的分类报告:")
print(classification_report(y_test, svm_pred, target_names=label_encoder.classes_))

# 保存SVM模型
svm_model_path = "svm_model.pkl"
joblib.dump(svm_model, svm_model_path)
print(f"SVM模型已保存到 {svm_model_path}")

# Logistic回归模型
log_reg_params = {
    'C': 1.0,
    'class_weight': 'balanced',
    'max_iter': 1000,
    'multi_class': 'ovr',
    'n_jobs': -1,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'tol': 0.0001,
    'warm_start': True
}

log_reg_model = LogisticRegression(**log_reg_params)
log_reg_model.fit(X_train, y_train)  # 训练Logistic回归模型
log_reg_pred = log_reg_model.predict(X_test)  # 在测试集上预测

# 输出Logistic回归模型的准确率
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
print(f"Logistic回归模型在测试集上的准确率: {log_reg_accuracy * 100:.2f}%")

# 输出Logistic回归模型每个类别的分类报告
print("Logistic回归模型每个类别的分类报告:")
print(classification_report(y_test, log_reg_pred, target_names=label_encoder.classes_))

# 保存Logistic回归模型
log_reg_model_path = "log_reg_model.pkl"
joblib.dump(log_reg_model, log_reg_model_path)
print(f"Logistic回归模型已保存到 {log_reg_model_path}")

# XGBoost模型
xgb_params = {
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss',
    'gamma': 0,
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 200,
    'num_class': 9,
    'objective': 'multi:softmax',
    'predictor': 'gpu_predictor',
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'subsample': 0.8,
    'tree_method': 'gpu_hist'
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train)  # 训练XGBoost模型
xgb_pred = xgb_model.predict(X_test)  # 在测试集上预测

# 输出XGBoost模型的准确率
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost模型在测试集上的准确率: {xgb_accuracy * 100:.2f}%")

# 输出XGBoost模型每个类别的分类报告
print("XGBoost模型每个类别的分类报告:")
print(classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))

# 保存XGBoost模型
xgb_model_path = "xgb_model.pkl"
joblib.dump(xgb_model, xgb_model_path)
print(f"XGBoost模型已保存到 {xgb_model_path}")
