import joblib
import numpy as np
import pandas
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
import os

__all__ = ['Normalizer']


def Normalizer(data,
               data_for_nor=None,
               sigma=4,
               methods='Standar',
               ues_fitted=True,
               **kwargs) -> dict or pandas.DataFrame:
    """
    :param data:
    :param sigma:
    :param methods:
    :param kwargs:
    :return: a normalize dataframe and a fitted scaler
    """

    # Check if data is either a Pandas DataFrame or a dictionary
    if not(isinstance(data, pd.DataFrame)) and not(isinstance(data, dict)):
        raise TypeError('Invalid arguments. Please provide a pandas dataframe.')
    elif isinstance(data, dict):
        # raise Warning('Please make sure that all')
        for k, v in data.items():
            if not(isinstance(v, pd.DataFrame)):

                raise TypeError("dict: has a invalid arguments. Please provide a pandas dataframe.")

    if isinstance(data, dict):
        for key, v in data.items():
            # Calculate the mean and standard deviation of each column
            mean = v.mean()
            std = v.std()
            # Use a boolean mask to find values that are more than 4 standard deviations from the mean
            mask = (v - mean).abs() > sigma * std
            # Replace these outliers with np.nan
            v = v.where(~mask, np.nan)
            # Update the dictionary with the normalized data
            data[key] = v


    elif isinstance(data, pd.DataFrame):
        # Calculate the mean and standard deviation of each column
        mean = data.mean()
        std = data.std()
        # Use a boolean mask to find values that are more than 4 standard deviations from the mean
        mask = (data - mean).abs() > sigma * std
        # Replace these outliers with np.nan
        data = data.where(~mask, np.nan)
        # Update the dictionary with the normalized data
        data = data

    # Impute NA values using KNN imputation
    if isinstance(data, dict):
        for k, v in data.items():
            imp_knn = KNNImputer()
            fill_data = imp_knn.fit_transform(v)
            data[k] = pd.DataFrame(fill_data, index=v.index, columns=v.columns)

    elif isinstance(data, pd.DataFrame):
        imp_knn = KNNImputer()
        fill_data = imp_knn.fit_transform(data)
        data = pd.DataFrame(fill_data, index=data.index, columns=data.columns)

    if ues_fitted:
        print("Using fitted data scaler...")
        scaler = joblib.load('Data/params/scaler.pkl')
    elif methods == 'Standar':
        print('Using StandardScaler...')
        scaler = StandardScaler()
    elif methods == 'QuantileTransformer':
        print('Using QuantileTransformer...')
        scaler = QuantileTransformer(output_distribution='normal')
    elif methods == 'MinMaxScaler':
        print('Using MinMaxScaler...')
        scaler = MinMaxScaler()
    else:
        raise ValueError('Please use correct normalize params.')

    if isinstance(data, dict):
        # Apply the scaler to the data
        # Data with different key will be seemed as different group data!

        dis_df = pd.concat(list(data.values()))
        if ues_fitted:
            values = scaler.transform(X=dis_df)
        else:
            values = scaler.fit_transform(X=dis_df)

        # dis_df = {k: (v - data_for_nor.min()) / (data_for_nor.max() - data_for_nor.min()) for k, v in dis_df.items()}
        dis_df = pd.DataFrame(values, index=dis_df.index, columns=dis_df.columns)
        dis_data_df = {k: dis_df.loc[v.index.to_numpy(), :] for k, v in data.items()}

    elif isinstance(data, pd.DataFrame):
        # Apply the scaler to the data,如果只使用Dataframe则会将数值视为同一类型
        if ues_fitted:
            values = scaler.transform(X=data)
        else:
            values = scaler.fit_transform(X=data)
        dis_data_df = pd.DataFrame(values, index=data.index, columns=data.columns)

    del values
    return dis_data_df
