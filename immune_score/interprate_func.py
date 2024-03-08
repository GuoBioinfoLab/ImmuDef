import pandas as pd
import numpy as np

def interprete_func(array,
                    dataframe,
                    key,
                    refer_sample) -> dict:
    """

    :param array:
    :param dataframe:
    :param key:
    :param refer_sample:
    :return:
    """
    # 检查参数是否合法
    if not isinstance(array, np.ndarray) or not isinstance(dataframe, pd.DataFrame):
        print("Invalid arguments. Please provide a numpy array and a pandas dataframe.")
        return None

    interp_df = pd.DataFrame(pd.DataFrame(dataframe[array] - refer_sample[array].to_numpy()).median(),columns=[key]).sort_index(axis=0)
    sum_score = np.sum(np.abs(pd.DataFrame(dataframe[array] - refer_sample[array].to_numpy()).median()))
    abs_df = interp_df.abs()
    abs_df = abs_df.sort_values(by=key, ascending=False)
    abs_df = abs_df/sum_score
    mean_tribute = np.mean(abs_df)
    return abs_df, mean_tribute


def score_contribute_vis_func(data_df,
                              fea_dict,
                              refer_sample=None,
                              if_use_ref=False,
                              ):

    if not if_use_ref:
        print('Use default reference...')
        interp_df_dict = {k: interprete_func(dataframe=data_df,
                                            key=k,
                                             refer_sample=refer_sample,
                                            array=np.array(fea_dict[k]))[0] for k, v in fea_dict.items()}
        mean_dict = {k: interprete_func(dataframe=data_df,
                                       key=k,
                                        refer_sample=refer_sample,
                                       array=np.array(fea_dict[k]))[1] for k, v in fea_dict.items()}
    else:
        print('Use self-build reference...')
        interp_df_dict = {k: interprete_func(dataframe=data_df,
                                             key=k,
                                             array=np.array(fea_dict[k]),
                                             refer_sample=refer_sample)[0] for k,v in fea_dict.items()}

        mean_dict = {k: interprete_func(dataframe=data_df,
                                        key=k,
                                        array=np.array(fea_dict[k]),
                                        refer_sample=refer_sample)[1] for k, v in fea_dict.items()}

    return interp_df_dict
