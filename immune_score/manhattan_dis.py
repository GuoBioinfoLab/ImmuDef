import numpy as np
import pandas as pd

__all__ = ['Motify_Manhattan_Distance']


# 定义一个函数，接受一个numpy数组和一个pandas的Dataframe作为参数
def Motify_Manhattan_Distance(array, dataframe):
    """

    :param array:
    :param dataframe:
    :return:
    """


    # 检查参数是否合法
    if not isinstance(array, np.ndarray) or not isinstance(dataframe, pd.DataFrame):
        print("Invalid arguments. Please provide a numpy array or a pandas dataframe.")
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
        if np.sum(np.abs(dataframe.loc[ind].values)) >= np.sum(np.abs(array)):
            distance = 1 + np.sum(np.abs(array - dataframe.loc[ind].values), axis=0) / np.sum(np.abs(array))
        # elif  np.sum(np.abs(dataframe.loc[ind].values)) < np.sum(np.abs(array)):
        else:
            distance = 1 - np.sum(np.abs(array - dataframe.loc[ind].values), axis=0) / np.sum(np.abs(array))
        # 将结果添加到结果Dataframe中
        distance_arr = np.append(distance_arr, distance)
    # 返回结果Dataframe
    return distance_arr
