# compute_immune_score.py
import torch
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from QImmuDef_VAE import BetaVAE
from ensemble_learning import ensemable_classcification, immune_scror_calculate
from typing import Dict, Any, Optional, Union
from feature_selection_methods.normalizmethods import Normalizer
import argparse


def _load_external_df(maybe_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load an external DataFrame. If input is already a DataFrame, return it.
    If it's a string, try common loaders (pickle, parquet, csv).
    Raise ValueError on failure.
    """
    if maybe_path is None:
        return None
    if isinstance(maybe_path, pd.DataFrame):
        return maybe_path.copy()
    if not isinstance(maybe_path, str):
        raise ValueError("extra input must be a pandas DataFrame or a path string.")
    # try common formats in order
    if not os.path.exists(maybe_path):
        raise FileNotFoundError(f"Extra input path not found: {maybe_path}")
    # try pd.read_pickle first (for .pkl)
    try:
        return pd.read_pickle(maybe_path)
    except Exception:
        pass
    # parquet
    try:
        return pd.read_parquet(maybe_path)
    except Exception:
        pass
    # csv
    try:
        return pd.read_csv(maybe_path, index_col=0)
    except Exception:
        pass

    # last resort: try generic pickle load
    try:
        with open(maybe_path, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, pd.DataFrame):
                return obj
    except Exception:
        pass

    raise ValueError(f"Unable to load extra DataFrame from path: {maybe_path}")


def compute_latent_immune_score(
    data_pkl: str = "dis_data_df.pkl",
    model_path: str = "anneal_betaFalse_anneal_steps20000enc512_128_32_lr0.0001_bs16_beta0.1.pth",
    features_npy: str = "infection_fea_619.npy",
    generated_h5: str = "1e6_sampling_data_is_axis.h5",
    h5_key: str = "df",
    prob_threshold: float = 0.90,
    latent_dim: int = 2,
    control_group_name: str = "Control",
    healthy_class_name: str = "Healthy Control",
    device: str = None,
    extra_input: Optional[Union[str, pd.DataFrame]] = None,
    extra_key: str = "Extra"
) -> pd.DataFrame:
    """
    一键计算基于生成高置信Control的潜空间免疫评分
    支持传入额外一个 DataFrame（或其文件路径），该 DataFrame 会先通过 Normalizer 做
    标准化（在该额外数据上 fit 标准化器），并被加入到 data_dict[extra_key] 中，参与后续流程。
    返回: vis_df_final (含 x, y, Disease Group, immune_score)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # 1. 加载特征 & 模型
    features = np.load(features_npy)
    model = torch.load(model_path, map_location=device, weights_only=False).eval()

    # 2. 加载真实数据
    with open(data_pkl, 'rb') as f:
        data_dict = pickle.load(f)  # expected dict[str, pd.DataFrame]

    # 2.1 如果提供了额外数据，则读取并标准化后加入 data_dict
    if extra_input is not None:
        extra_df = _load_external_df(extra_input)
        if extra_df is None:
            raise ValueError("Failed to load extra input DataFrame.")
        # 确保 extra_df 至少包含 features 中的一部分列
        missing_features = [f for f in features if f not in extra_df.columns]
        if missing_features:
            # 如果缺少部分特征，尝试只保留相交的特征；如果没有交集，则报错
            inter = [f for f in features if f in extra_df.columns]
            if len(inter) == 0:
                raise ValueError(f"Extra DataFrame does not contain any of required features. Required features length: {len(features)}")
            # 提示：只使用交集特征
            extra_df = extra_df[inter].copy()
        else:
            # 只保留并按照 features 顺序排列（以便后续一致）
            extra_df = extra_df[list(features)].copy()

        # 使用 Normalizer 对额外数据进行默认的 Standard 标准化（fit on extra_df）
        normalized_extra = Normalizer(extra_df, ues_fitted=False, methods='Standar')
        # Normalizer 可能返回 DataFrame 或 dict，这里确保拿到 DataFrame
        if isinstance(normalized_extra, dict):
            # 如果返回 dict，则取第一个元素
            normalized_extra = list(normalized_extra.values())[0]

        # 将标准化后的 DataFrame 加入 data_dict
        data_dict[extra_key] = normalized_extra

    # 3. 将每个 group 的特征转为 tensor 并送入 VAE 编码
    tensor_data = {}
    for k, v in data_dict.items():
        # v 可能不是 DataFrame（谨慎检查）
        if not isinstance(v, pd.DataFrame):
            raise TypeError(f"data_dict[{k}] is not a pandas DataFrame.")
        # 确保 v 含有 features（或其子集），并按 features 顺序取列
        inter_feats = [f for f in features if f in v.columns]
        if len(inter_feats) == 0:
            raise ValueError(f"data_dict[{k}] does not contain any of the required features.")
        # If some features are missing, we will only feed available columns.
        # The VAE model was trained on full feature set; if shape mismatch happens,
        # the model.encode will raise an error — user should ensure feature alignment.
        arr = v[inter_feats].values
        tensor_data[k] = torch.tensor(arr, dtype=torch.float32).to(device)

    # 4. VAE 潜空间投影
    latent_z = {}
    for k, x in tensor_data.items():
        mu, logvar = model.encode(x)
        latent_z[k] = model.reparameterize(mu, logvar)

    # 5. 合并生成 vis_df（x,y）
    dfs = []
    for k, z in latent_z.items():
        df = pd.DataFrame(z.detach().cpu().numpy(),
                          columns=[f'z{i}' for i in range(latent_dim)],
                          index=data_dict[k].index)
        df['Disease Group'] = k
        dfs.append(df)
    vis_df = pd.concat(dfs).rename(columns={'z0': 'x', 'z1': 'y'} if latent_dim >= 2 else {'z0': 'x'})

    # 6. 生成高置信Control参考
    gen_df = pd.read_hdf(generated_h5, key=h5_key)
    probs = ensemable_classcification(gen_df[features], rt_type='mean_Prob')
    gen_df['Healthy Control Prob'] = probs[healthy_class_name]

    high_conf_ctrl = gen_df[gen_df['Healthy Control Prob'] > prob_threshold].copy()
    real_ctrl = vis_df[vis_df['Disease Group'] == control_group_name]

    # 动态列名兼容（支持更高维）
    coord_cols = ['x', 'y'][-latent_dim:]
    for col in coord_cols:
        if col not in high_conf_ctrl.columns:
            high_conf_ctrl[col] = high_conf_ctrl[f'z{coord_cols.index(col)}']

    # 限制在真实Control包围盒内（若 real_ctrl 为空会出错）
    if real_ctrl.empty:
        raise ValueError(f"No real control samples found for control_group_name='{control_group_name}' in vis_df.")

    mask = True
    for col in coord_cols:
        mn, mx = real_ctrl[col].min(), real_ctrl[col].max()
        mask &= high_conf_ctrl[col].between(mn, mx)
    gen_ctrl_filtered = high_conf_ctrl[mask]

    # 7. 统一归一化（基于真实数据范围）
    scaler = MinMaxScaler().fit(vis_df[coord_cols])
    vis_df[coord_cols] = scaler.transform(vis_df[coord_cols])
    gen_ctrl_filtered[coord_cols] = scaler.transform(gen_ctrl_filtered[coord_cols])

    # 8. 计算免疫评分（以生成Control中位数为参考点）
    refer_point = gen_ctrl_filtered[coord_cols].median()
    vis_df_final = immune_scror_calculate(
        input_data_dict=vis_df,
        features=coord_cols,
        refer=refer_point
    )
    return vis_df_final


# --------------------- 命令行入口 ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute latent-space immune score using high-confidence generated controls")
    parser.add_argument("--data", default="dis_data_df.pkl", help="Path to real disease data pickle")
    parser.add_argument("--model", default="anneal_betaFalse_anneal_steps20000enc512_128_32_lr0.0001_bs16_beta0.1.pth")
    parser.add_argument("--features", default="infection_fea_619.npy", help="Feature list numpy file")
    parser.add_argument("--generated", default="1e6_sampling_data_is_axis.h5", help="Generated samples HDF5")
    parser.add_argument("--threshold", type=float, default=0.90, help="Healthy control probability threshold")
    parser.add_argument("--output", default="vis_df_with_immune_score.parquet", help="Output file path")
    parser.add_argument("--extra", default=None, help="Optional extra DataFrame path (pkl/parquet/csv) or omit to skip")
    parser.add_argument("--extra-key", default="Extra", help="Key name to use for the extra DataFrame inside data_dict")

    args = parser.parse_args()

    result_df = compute_latent_immune_score(
        data_pkl=args.data,
        model_path=args.model,
        features_npy=args.features,
        generated_h5=args.generated,
        prob_threshold=args.threshold,
        extra_input=args.extra,
        extra_key=args.extra_key
    )

    result_df.to_parquet(args.output)
    print(f"Success: Immune score computed! Saved to {args.output}")
    print(result_df['immune_score'].describe())
