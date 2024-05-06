import pandas as pd
from copy import deepcopy

def stable_marriage(match_table: pd.DataFrame, columns=['A', 'B', 'sim']):
    matches = pd.DataFrame(columns=columns)
    c1, c2, c3 = columns
    # match_table = deepcopy(match_table)
    match_table = match_table.sort_values([c3], ascending=[False])
    
    while True:
        R = match_table.loc[(~match_table[c1].isin(matches[c1])) & (~match_table[c2].isin(matches[c2]))]
        if len(R) == 0:
            break
        x = R.iloc[0, :]
        matches.loc[len(matches), matches.columns] = x
    
    return matches


def simmetric_best_match(match_table: pd.DataFrame, columns=['A', 'B', 'sim']):
    CMT = deepcopy(match_table)
    # CMT = match_table
    c1, c2, c3 = columns
    CMT['A_RowNo'] = CMT \
        .sort_values([c3], ascending=[False]) \
        .groupby([c1]) \
        .cumcount() + 1

    CMT['B_RowNo'] = CMT \
        .sort_values([c3], ascending=[False]) \
        .groupby([c2]) \
        .cumcount() + 1

    return CMT[(CMT.A_RowNo==1) & (CMT.B_RowNo==1)] \
        .drop(columns=['A_RowNo', 'B_RowNo']) \
        .sort_values([c3], ascending=[False], ignore_index=True)