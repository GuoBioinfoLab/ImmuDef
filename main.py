# This is a test Python Script.
import pandas as pd
import subprocess
import tempfile

"""This is a script test if ImmuDef is working correctlt."""
# Test

if __name__ == '__main__':
    print('Test start.')

    import pandas as pd
    from immune_score.score_caculator import Immune_Score_Caculator

    isc = Immune_Score_Caculator()
    # Create a pandas DataFrame
    data = pd.read_csv('./Data/data/SRP318559.csv', index_col=0)
    data = isc.score_compute(data=df,data_type='Matrix',use_all_to_nor=True)
    print(data)
