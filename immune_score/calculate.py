import copy
import pandas as pd
from .manhattan_dis import Motify_Manhattan_Distance

__all__ = ['immune_score_calculate']


def immune_score_calculate(input_data_dict,
                           control_median_data,
                           fea_dict=None) -> pd.DataFrame:
    """

    :param input_data_dict:
    :param fea_dict:
    :return:
    """



    if isinstance(input_data_dict, pd.DataFrame):
        for k_, v_ in fea_dict.items():

            if len(control_median_data[v_].to_numpy()) == len(input_data_dict[v_]):
                input_data_dict[k_] = Motify_Manhattan_Distance(array=control_median_data[v_].to_numpy(),
                                                                dataframe=input_data_dict[v_])
            else:
                print(f"Length mismatch: {len(control_median_data[v_].to_numpy())} != {len(input_data_dict[v_])}")

            print(control_median_data[v_].to_numpy().shape)
            print(input_data_dict[v_].to_numpy().shape)
            input_data_dict[k_] = Motify_Manhattan_Distance(array=control_median_data[v_].to_numpy(),
                                                            dataframe=input_data_dict[v_])

        input_data_df = input_data_dict
        input_data_df = input_data_df[['Anti-Bacterium Immune Score',
                                       'Anti-Virus Immune Score',
                                       'Anti-Fungus Immune Score']]
    #
    elif isinstance(input_data_dict, dict):
        input_data_df = copy.deepcopy(input_data_dict)
        for k, v in input_data_df.items():
            v["Disease Group"] = k
            for k_, v_ in fea_dict.items():
                v[k_] = Motify_Manhattan_Distance(array=control_median_data[v_].to_numpy(),
                                                  dataframe=v[v_])

        input_data_df = pd.concat(list(input_data_df.values()), axis=0)
        input_data_df = input_data_df[['Anti-Bacterium Immune Score',
                                       'Anti-Virus Immune Score',
                                       'Anti-Fungus Immune Score',
                                       "Disease Group"]]
    return input_data_df
