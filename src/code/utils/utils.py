import pandas as pd


def round_to(n, precision):
    if n >= 0 or n < 0:
        correction = 0.5 if n >= 0 else -0.5
        return int(n / precision + correction) * precision
    else:
        return n


def round_to_05(n):
    return float(format(round_to(n, 0.05), ".2f"))


def rebuild_table(table):
    return pd.DataFrame(
        data=[
            [entry_data['text'] 
             for entry_data in entry]
            for entry in table['tableData']
        ],
        columns=table['tableHeaders'][0]
        )