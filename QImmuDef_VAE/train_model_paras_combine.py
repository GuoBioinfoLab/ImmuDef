import subprocess
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

def generate_configs():
    """
    生成不同的参数组合
    """
    configs = {
        'batch_size': [16],
        'lr': [1e-4],
        'epochs': [20000],
        'beta': [0.1, 0.3, 0.5, 0.6],
        'anneal_steps': [20000],
        'encoder_layers': [
            [512, 128, 32],  # Example of one combination
        ],
        'loss_fun': ['mse'],
        'save_model': [True]
    }

    # 生成所有可能的参数组合
    all_configs = product(*configs.values())
    config_names = list(configs.keys())
    return all_configs, config_names

def run_model_on_gpu(config_dict, gpu_id):
    """
    在指定的 GPU 上运行模型
    """
    # 将 config_dict 转换为命令行参数
    cmd = ['python', './vae_model.py']
    for key, value in config_dict.items():
        if isinstance(value, list):  # If it's a list (e.g., encoder_layers)
            value = " ".join(map(str, value))  # Join list into a string
        cmd.append(f'--{key}')
        cmd.append(str(value))

    # 执行命令
    subprocess.run(cmd)

def run_experiment():
    # 生成所有参数组合
    all_configs, config_names = generate_configs()

    # 假设只有一个 GPU（GPU 0）
    gpu_id = 0

    # 设置最大并行任务数（比如 4 个任务并行）
    max_parallel = 4

    # 使用 ProcessPoolExecutor 并行执行模型训练
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = []

        # 遍历所有参数组合并提交任务
        for i, config in enumerate(all_configs):
            config_dict = dict(zip(config_names, config))
            print(f"\nStarting experiment with configuration: {config_dict}")

            # 提交任务到 executor，分配 GPU
            futures.append(executor.submit(run_model_on_gpu, config_dict, gpu_id))

        # 等待所有任务完成
        for future in as_completed(futures):
            pass  # 可以在这里处理每个任务的返回结果

if __name__ == '__main__':
    run_experiment()

