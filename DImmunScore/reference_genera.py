import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
from vae_model import BetaVAE
import pickle
import joblib
import matplotlib
from sklearn.preprocessing import MinMaxScaler

def reference_sample_genarate(file="dis_data_df.pkl",
                              model_file='anneal_betaFalse_anneal_steps20000enc512_128_32_lr0.0001_bs16_beta0.6.pth',
                              featurefile='infection_fea_619.npy',
                              dataaugmentation_file = '1e6_sampling_data_is_axis.h5',
                              labelencoder_file='label_encoder.pkl',
                              if_calculate=True,
                              scaler=True):
    # 检查文件是否存在
    if os.path.exists(file):
        # 读取 pkl 文件为字典
        with open(file, 'rb') as f:
            data_dict = pickle.load(f)
        print("Dictionary successfully loaded")
        dis_data_df = copy.deepcopy(data_dict)        
    else:
        print(f"File '{file_name}' does not exist in the current directory.")
        
    immune_features = np.load(featurefile)
    # 将数据转换为 PyTorch 张量并移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict = {k: torch.tensor(v[immune_features].values, dtype=torch.float32).to(device) for k, v in data_dict.items()}
    print("Data successfully moved to:", device)

    model = torch.load(model_file,
                   map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      weights_only=False)
    model.eval()
    vae_results = {k:model(v) for k,v in data_dict.items()}
    latent_var = {k:model.reparameterize(v[1],v[2]) for k,v in vae_results.items()}
    
    # 将每个键的值从 GPU 移到 CPU，并转为 NumPy 数组
    latent_var = {k: v.detach().cpu().numpy() for k, v in latent_var.items()}
    print("All tensors have been moved to CPU and converted to NumPy arrays.")
    output_is = pd.read_hdf(dataaugmentation_file, key='df')

    vis_df = pd.concat(
    [
        pd.DataFrame(v, columns=['x', 'y'], index=dis_data_df[k].index)
        .join(dis_data_df[k])
        .assign(Disease_Group=k)
        for k, v in latent_var.items()
    ],
    axis=0)

    vis_df.rename(columns={'Disease_Group': 'Disease Group'}, inplace=True)

    vis_df['Disease Group'] = vis_df['Disease Group'].replace({
        'COVID': 'COVID-19',
        'Control': 'Healthy Control',
        'TB': 'Tuberculosis',
        'Candida': 'Fungus',
        'HIV': 'AIDS'
    })
    
    output_is_for_cls = output_is[output_is.columns[:-3]]
    svm_result, log_reg_result, xgb_result = ensemable_classcification(output_is_for_cls,rt_type='Class')

    models = ['SVM', 'Logistic', 'XGBoost']
    results = [svm_result, log_reg_result, xgb_result]
    label_encoder = joblib.load(labelencoder_file)
    
    position_labeled_df = output_is[output_is.columns[-3:]].assign(
        **{f"{m} Result": label_encoder.inverse_transform(r) for m, r in zip(models, results)}
    )

    cols = ['SVM Result', 'Logistic Result', 'XGBoost Result']
    selected_label_df = position_labeled_df[position_labeled_df[cols].nunique(axis=1).eq(1)]

    # 计算 SVM Result, Logistic Result, XGBoost Result 的均值
    position_labeled_df['mean_result'] = ensemable_classcification(output_is_for_cls,rt_type='mean_Prob')['Healthy Control']

    mn_scaler = MinMaxScaler()
    vis_df_2  = vis_df[['x','y','Disease Group']]
    vis_df_2[['x', 'y']] = mn_scaler.fit_transform(vis_df_2[['x', 'y']])
    
    if if_calculate:
        return vis_df_2,mn_scaler
    else:
        return vis_df

def generate_control_samples(
    latentspace_df,
    trainingdata_df,
    nm_scaler,
    control_threshold=0.95,
    disease_group_col="Disease Group",
    control_group="Healthy Control",
    coord_cols=["x", "y"]):
    """
    生成标准化控制组样本
    
    参数：
    latentspace_df (DataFrame): 潜在空间数据框
    trainingdata_df (DataFrame): 可视化坐标数据框
    control_threshold (float): 控制组筛选阈值
    disease_group_col (str): 疾病分组列名
    control_group (str): 控制组名称
    coord_cols (list): 坐标列名称列表
    
    返回：
    DataFrame: 标准化后的控制组样本
    """
    # 初始化缩放器
    scaler = nm_scaler
    
    try:
        # 初步筛选控制组样本
        control_mask = latentspace_df[control_group].gt(control_threshold).any(axis=1)
        control_df = latentspace_df.loc[control_mask].copy()
        
        # 获取坐标范围
        control_vis = trainingdata_df[trainingdata_df[disease_group_col] == control_group]
        ranges = {
            'x': (control_vis['x'].min(), control_vis['x'].max()),
            'y': (control_vis['y'].min(), control_vis['y'].max())
        }
        
        # 空间范围筛选
        spatial_filter = (
            control_df['x'].between(*ranges['x']) &
            control_df['y'].between(*ranges['y'])
        )
        filtered_df = control_df.loc[spatial_filter]
        
        # 数据标准化
        if not filtered_df.empty:
            filtered_df.loc[:, coord_cols] = scaler.fit_transform(filtered_df[coord_cols])
        
        return filtered_df
    
    except KeyError as e:
        print(f"列不存在错误: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return pd.DataFrame()
