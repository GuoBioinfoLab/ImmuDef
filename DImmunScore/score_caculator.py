import pandas as pd
import numpy as np
import sklearn
import os
import copy
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report  



def manhattan_distance2(array, dataframe):
    # 检查参数是否合法
    if not isinstance(array, np.ndarray) or not isinstance(dataframe, pd.DataFrame):
        print("Invalid arguments. Please provide a numpy array and a pandas dataframe.")
        return None
    # 获取数组和Dataframe的维度
    array_shape = array.shape
    dataframe_shape = dataframe.shape
    # 检查数组和Dataframe的维度是否匹配
    if array_shape[0] != dataframe_shape[1]:
        print("Dimension mismatch. The array and the dataframe must have the same number of rows.")
        return None
    # 创建一个空的Dataframe，用于存储结果
    distance_arr = np.array([])

    # 遍历Dataframe的每一列
    for ind in dataframe.index:
        # 计算数组和Dataframe的该列的曼哈顿距离
        if dataframe.loc[ind]['y'] >= array[1]:
            distance = 1 +  np.linalg.norm(array - dataframe.loc[ind].values, axis=0, ord=1) / np.sum(np.abs(array))
        # elif  np.sum(np.abs(dataframe.loc[ind].values)) < np.sum(np.abs(array)):
        else:
            distance = 1 - np.linalg.norm(array - dataframe.loc[ind].values, axis=0, ord=1) / np.sum(np.abs(array))
        # 将结果添加到结果Dataframe中
        distance_arr = np.append(distance_arr, distance)
    # 返回结果Dataframe
    return distance_arr


def immune_scror_calculate2(input_data_dict, features, refer, return_t='dict'):
    if isinstance(input_data_dict, pd.DataFrame):
        immune_score_df = copy.deepcopy(input_data_dict)
        immune_score_df['Defense Immune Score'] = manhattan_distance2(array=refer[features].to_numpy(),
                                                                   dataframe=immune_score_df[features])
        return immune_score_df

    elif isinstance(input_data_dict, dict):
        input_data_df = copy.deepcopy(input_data_dict)

        for k, v in input_data_df.items():
            v["Disease Group"] = k
            v['Defense Immune Score'] = manhattan_distance2(array=refer[features].to_numpy(),
                                                           dataframe=v[features])
        if return_t == 'dict':
            immune_score_df = {k: v[['Disease Group', 'Defense Immune Score']] for k, v in input_data_df.items()}
        else:
            # 检查是否存在列 'Disease Group'
            if 'Disease Group' in input_data_df.columns:
                immune_score_df = input_data_df[['Disease Group', 'Defense Immune Score']]
            else:
                immune_score_df = input_data_df[['Defense Immune Score']]

        return immune_score_df
    else:
        raise ValueError('to this error ')
