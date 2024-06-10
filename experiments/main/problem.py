import pprint
import pandas as pd

df = pd.read_csv('/data4/nanni/tesi-magistrale/data/josie-tests/full/results/alshforest_mbag_k10_extracted.csv')
print(df.head())

rv = df.groupby(by='query_id')['result_id']

for id, v in rv:
    print(id, v.values.tolist())
    print()