# 这是一个示例 Python 脚本。
import pandas as pd
import subprocess
import tempfile

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

    import pandas as pd
    from immune_score.score_caculator import Immune_Score_Caculator

    isc = Immune_Score_Caculator()
    # 创建一个pandas DataFrame
    df = pd.read_csv('D:/GSEA/SRP318559.csv', index_col=0)
    data = isc.score_compute(data=df,
                             data_type='Matrix',
                             use_all_to_nor=True)
    print(data)

    data = pd.read_csv('Data/data/ssgsea_TB_c7.csv.bz2', index_col=0).transpose()
    isc = Immune_Score_Caculator()
    isc(data=data)
    print(isc.score_compute(data=data, use_all_to_nor=True))
