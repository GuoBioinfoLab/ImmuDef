import json
import numpy as np
import pandas as pd
import sys
import os
sys.path.append("..")

from .calculate import immune_score_calculate
from feature_selection_methods.normalizmethods import Normalizer
from .gsea_func import R_ssGSEA_func


class Immune_Score_Caculator:
    def __init__(self,
                 reference_data=None,
                 immune_fea_dict=None,
                 use_fitted_normalizer=True,
                 normalizer=Normalizer,
                 immune_features=None
                 ) -> None:
        """

        :param reference_data:
        :param immune_fea_dict:
        :param normalizer:
        :param immune_features:
        """
        if immune_fea_dict:
            self.immune_fea_dict = immune_fea_dict
        else:
            'Use default immune score features...'
            # 打开JSON文件
            with open("Data/params/Infection_Sepcific_Features_Sep.json", "r") as f:
                # 读取文件内容
                fea_dict = json.load(f)
            self.immune_fea_dict = fea_dict

        if immune_features:
            pass
        else:
            self.immune_features = np.load("Data/params/union_dis_fea_dict_0.9.npy",
                                           allow_pickle=True).item()['Control']


        if reference_data:
            self.reference_data = reference_data
        else:
            'Use default healthy reference samples...'
            self.reference_data = pd.read_csv('Data/data/ssgsea_Control_C7.csv.bz2',
                                              index_col=0).transpose()

        #
        self.normalizer = normalizer
        #
        self.use_fitted_normalizer = use_fitted_normalizer

        #
        self.data_for_nor = {files.split('.')[0].split('_')[-2]: pd.read_csv(files, index_col=0).transpose() for files
                             in
                             [os.path.join("Data/data/", files) for files in os.listdir("Data/data/")]}


        self.data_for_nor = self.normalizer(self.data_for_nor)
        self.data_for_nor = pd.concat(list(self.data_for_nor.values()), axis=0)
        self.data_for_nor = self.data_for_nor[self.immune_features]

        self.R_ssGSEA_func = R_ssGSEA_func


        #
        if self.use_fitted_normalizer:
            self.reference_data = self.normalizer(self.reference_data)[self.immune_features]
            self.reference_data = (self.reference_data - self.data_for_nor.min()) / (
                        self.data_for_nor.max() - self.data_for_nor.min())
        else:
            self.reference_data = self.normalizer(self.reference_data)[self.immune_features]

        self.reference_sample = self.reference_data.median()

        #
        self.immune_score_calculate = immune_score_calculate


    def __call__(self,
                 data,
                 normalize=True) -> dict or pd.DataFrame:
        """

        :param data:
        :param normalize:
        :return:
        """
        if normalize:
            data = self.normalizer(data,
                                   ues_fitted=self.use_fitted_normalizer,
                                   data_for_nor=self.data_for_nor)
            data = data[self.immune_features]
            data = (data - self.data_for_nor.min()) / (
                    self.data_for_nor.max() - self.data_for_nor.min())
        else:
            raise Warning()

        return self.immune_score_calculate(data,
                                           control_median_data=self.reference_sample,
                                           fea_dict=self.immune_fea_dict)

    def score_compute(self,
                 data,
                 data_type='ssGSEA',
                 use_all_to_nor=False,
                 normalize=True) -> dict or pd.DataFrame:
        """

        :param data:
        :param data_type:
        :param use_all_to_nor:
        :param normalize:
        :return:
        """
        if data_type == 'ssGSEA':
            pass
        elif data_type == 'Matrix':
            print("Data Transforming...")
            data = self.R_ssGSEA_func(data)
        #
        if use_all_to_nor == False:
            if normalize:
                data = self.normalizer(data,
                                       ues_fitted=self.use_fitted_normalizer,
                                       data_for_nor=self.data_for_nor)
                data = data[self.immune_features]
                data = (data - self.data_for_nor.min()) / (
                        self.data_for_nor.max() - self.data_for_nor.min())
            else:
                raise Warning("'Data hasn't been normalized! It may lead to a wrong result!")

        else:
            #
            self.reference_data = pd.read_csv('Data/data/ssgsea_Control_C7.csv.bz2',
                                              index_col=0).transpose()
            print(self.reference_data.shape)

            self.data_for_nor = {files.split('.')[0].split('_')[-2]: pd.read_csv(files, index_col=0).transpose() for
                                 files in
                                 [os.path.join("Data/data/", files) for files in os.listdir("Data/data/")]}

            self.data_for_nor['new'] = data

            self.data_for_nor = self.normalizer(self.data_for_nor, ues_fitted=False,)
            self.data_for_nor = pd.concat(list(self.data_for_nor.values()), axis=0)
            self.data_for_nor = self.data_for_nor[self.immune_features]

            self.data_for_nor = (self.data_for_nor - self.data_for_nor.min()) / (self.data_for_nor.max() - self.data_for_nor.min())

            self.reference_data = self.data_for_nor.loc[self.reference_data.index]
            self.reference_sample = self.reference_data.median()

            data = self.data_for_nor.loc[data.index]
            data = data.drop_duplicates()

        return self.immune_score_calculate(data,
                                           control_median_data=self.reference_sample,
                                           fea_dict=self.immune_fea_dict
                                           )
