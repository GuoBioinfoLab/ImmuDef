import pandas as pd
import tempfile
import subprocess


def R_ssGSEA_func(expr_matrix,
                  save=False,
                  save_dir=None
                  ):
    """

    :param expr_matrix:
    :param save:
    :param save_dir:
    :return:
    """
    # 创建一个pandas DataFrame
    expr_matrix = expr_matrix

    # 使用tempfile库创建一个临时文件
    with tempfile.NamedTemporaryFile(delete=False) as in_temp, tempfile.NamedTemporaryFile(delete=False) as out_temp:
        # 将DataFrame写入临时文件
        expr_matrix.to_csv(in_temp.name)

        # 使用subprocess库运行R脚本，将临时文件的路径作为参数传递给R脚本
        command = 'Rscript'
        path2script = './Data/Rscript.R'

        # 注意：你需要在你的R脚本中添加代码来读取这个临时文件
        args = [command, path2script,
                '--input', in_temp.name,
                '--output', out_temp.name]

        subprocess.run(args, check=True)
        ssgsea_df = pd.read_csv(out_temp.name, index_col=0)

    if save:
        # 读取R脚本写入的CSV文件
        if dir:
            ssgsea_df.to_csv(save_dir)
        else:
            raise ValueError("Save dictionary param 'save_dir' can't be 'None'!")

    return ssgsea_df
