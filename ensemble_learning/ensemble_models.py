import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import copy
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report  # 用于评估准确率和分类报告
import joblib

def ensemable_classcification(X, y = None, rt_type='Prob'):
    # 加载 LabelEncoder
    label_encoder_path = "label_encoder.pkl"
    label_encoder = joblib.load(label_encoder_path)
    print(f"LabelEncoder已加载，类别标签为: {label_encoder.classes_}")

    # 加载模型
    svm_model_path = "svm_model.pkl"
    svm_model = joblib.load(svm_model_path)
    print(f"SVM模型已加载")

    log_reg_model_path = "log_reg_model.pkl"
    log_reg_model = joblib.load(log_reg_model_path)
    print(f"Logistic回归模型已加载")

    xgb_model_path = "xgb_model.pkl"
    xgb_model = joblib.load(xgb_model_path)
    print(f"XGBoost模型已加载")

    if y:
        print(classification_report(y, svm_model.predict(X), target_names=label_encoder.classes_))
        print(classification_report(y, log_reg_model.predict(X), target_names=label_encoder.classes_))
        print(classification_report(y, xgb_model.predict(X), target_names=label_encoder.classes_))

    if rt_type == 'Class':
        svm_result = svm_model.predict(X)
        log_reg_result = log_reg_model.predict(X)
        xgb_result = xgb_model.predict(X)
        return svm_result, log_reg_result, xgb_result
    elif rt_type == 'Prob':
        svm_result = svm_model.predict_proba(X)
        log_reg_result = log_reg_model.predict_proba(X)
        xgb_result = xgb_model.predict_proba(X)
        return svm_result, log_reg_result, xgb_result
    elif rt_type == 'mean_Prob':
        svm_result = svm_model.predict_proba(X)
        log_reg_result = log_reg_model.predict_proba(X)
        xgb_result = xgb_model.predict_proba(X)
        mean_prob = {label_encoder.classes_[i]: np.mean([svm_result[:, i], log_reg_result[:, i], xgb_result[:, i]], axis=0)
                     for i in range(len(label_encoder.classes_))}
        return mean_prob
    else:
        raise TypeError('Please provide a legal return param.')
