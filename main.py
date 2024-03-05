# This is a test Python Script.
import pandas as pd
import subprocess
import tempfile

def print_hi(name):

    print(f'Hi, {name}')  

# Test
if __name__ == '__main__':
    print_hi('PyCharm')

    import pandas as pd
    from immune_score.score_caculator import Immune_Score_Caculator

    isc = Immune_Score_Caculator()
    # Create a pandas DataFrame
    df = pd.read_csv('D:/GSEA/SRP318559.csv', index_col=0)
    data = isc.score_compute(data=df,
                             data_type='Matrix',
                             use_all_to_nor=True)
    print(data)

    data = pd.read_csv('Data/data/ssgsea_TB_c7.csv.bz2', index_col=0).transpose()
    isc = Immune_Score_Caculator()
    isc(data=data)
    print(isc.score_compute(data=data, use_all_to_nor=True))
