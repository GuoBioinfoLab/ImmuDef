import copy
import functools
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc

__all__ = ['Feature_Selction']

class Feature_Selction(object):
    def __init__(self):
        """
        """
        self.fitted = False
        self.case_df = None
        self.control_df = None
        self.label_all = None

    def __call__(self, *args, **kwargs):
        pass

    def fit(self, case_df, control_df):
        """

        :param case_df:
        :param control_df:
        :return:
        """
        self.case_df = copy.deepcopy(case_df)
        self.control_df = copy.deepcopy(control_df)
        self.case_control_df, self.label_all = self.get_data_label(case_df=self.case_df, control_df=self.control_df)
        self.metric_df = self.get_feature_metric(case_df=self.case_control_df,
                                                 labels=self.label_all)
        # Reverse
        self.t_case_control_df, self.t_label_all = self.get_data_label(case_df=self.control_df, control_df=self.case_df)
        self.t_metric_df = self.get_feature_metric(case_df=self.t_case_control_df,
                                                   labels=self.t_label_all)
        self.fitted = True
        self.scaler = joblib.load('../Data/scaler.pkl')

    @property
    def __str__(self):
        pass

    @property
    def _get_control_data(self):
        return self.control_df

    @property
    def _get_case_data(self):
        return self.case_df

    @property
    def _get_feture_metric(self):
        return self.metric_df

    @property
    def _get_feture_metric(self):
        return self.metric_df

    def single_feature_select(self, data, label):
        """
        该函数的目的是以Logistic回归检验单个特征的实际分类效果
        :param data:
        :param label:
        :return:
        """
        # 单特征下的Logistic回归不存在过拟合现象
        clf = LogisticRegressionCV(cv=3, random_state=0, class_weight='balanced').fit(X=data, y=label)
        y_score = clf.decision_function(X=data)
        fpr, tpr, _ = roc_curve(label, y_score)
        del _

        auc_score = auc(fpr, tpr)
        acc = accuracy_score(label, clf.predict(data))

        try:
            f1_score = precision_recall_fscore_support(y_true=label, y_pred=clf.predict(data))[:-1]
            acc1, acc2, recall1, recall2 = f1_score[0][1], f1_score[0][1], f1_score[1][0], f1_score[1][1]
        except Warning as e:
            pass

        return {"clf": clf, "acc": acc,
                "auc": auc_score,
                "acc1": acc1,
                "acc2": acc2,
                "recall1": recall1,
                "recall2": recall2}

    def get_feature_metric(self, case_control_df, labels):
        """
        :param case_crol_df:
        :param labels:
        :return:
        """
        # 根据Logistic回归的结果计算出特征的各类指标
        res = [self.single_feature_select(case_control_df[[index]], label=labels) for index in case_control_df.columns.to_numpy()]
        dic = {"AUC": [i["auc"] for i in res],
             "acc":  [i["acc"] for i in res],
             "acc1": [i["acc1"] for i in res],
             "acc2": [i["acc2"] for i in res],
             "recall1": [i["recall1"] for i in res],
             "recall2": [i["recall2"] for i in res]}

        metric_df = pd.DataFrame(dic)
        metric_df.index = case_control_df.columns.to_numpy()
        metric_df.sort_values(by="AUC", ascending=False)
        return metric_df

    def get_data_label(self, case_df, control_df):
        """

        :param case_df:
        :param crol_df:
        :return:
        """
        # 根据输入的数据自动生成 0-1 label
        case_crol = pd.concat([case_df, control_df])
        label_all = np.zeros((case_df.shape[0]+control_df.shape[0]))
        label_all[case_df.shape[0]:] = 1
        return case_crol, label_all

    def select_mertics_cutoff(self, metric_df, lower=False, cut_off=0.90):
        """
        # 接收一个有四项特征指标构成的 DataFrame
        # 按照默认参数 cut_off的阈值对所有的筛选结果取交集
        # 默认的参数筛选阈值为0.9
        # 如果要筛选阈值低于某一值的参数，则需要调整lower参数
        :param metric_df:
        :param lower:
        :param cut_off:
        :return:
        """

        if lower:
            # 要求筛选的所有指标都必须低于阈值，所以对每个字典取交集
            metric_dict = {metric: metric_df[metric_df[metric] < cut_off].index.to_numpy() for metric in
                           metric_df.columns.to_numpy()}
            metric_arr = functools.reduce(np.intersect1d, [i for i in metric_dict.values()])
        else:
            # 要求筛选的所有指标都必须高于阈值，所以对每个字典取交集
            metric_dict = {metric: metric_df[metric_df[metric] > cut_off].index.to_numpy() for metric in
                           metric_df.columns.to_numpy()}
            metric_arr = functools.reduce(np.intersect1d, [i for i in metric_dict.values()])

        metrics_df = metric_df.loc[metric_arr]
        metrics_df = metrics_df.sort_values(by=metrics_df.columns.to_list(), ascending=False)

        return metrics_df

    def Get_feature_with_metric(self, df_1, df_2, lower_state=False, cut_off_value=0.90):
        """

        :param df_1:
        :param df_2:
        :param lower_state:
        :param cut_off_value:
        :return:
        """
        # 给定两个疾病的分类特征，对任意两类数据之间取特征的交集和并集
        metric_df1 = self.select_mertics_cutoff(df_1, lower=lower_state, cut_off=cut_off_value)
        metric_df2 = self.select_mertics_cutoff(df_2, lower=lower_state, cut_off=cut_off_value)
        union_feature_names = np.union1d(metric_df1.index.to_numpy(), metric_df1.index.to_numpy())
        # 并集
        intersect_feature_names = np.union1d(metric_df2.index.to_numpy(), metric_df1.index.to_numpy())
        # 交集
        return metric_df1, metric_df2, union_feature_names, intersect_feature_names
