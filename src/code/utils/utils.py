import pandas as pd




def rebuild_table(table):
    return pd.DataFrame(
        data=[
            [entry_data['text'] 
             for entry_data in entry]
            for entry in table['tableData']
        ],
        columns=table['tableHeaders'][0]
        )