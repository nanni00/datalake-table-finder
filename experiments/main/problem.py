import pprint
import pandas as pd

df = pd.read_csv('/data4/nanni/tesi-magistrale/data/josie-tests/full/results/alshforest_mbag_k10_extracted.csv')
print(df.head())

rv = df.groupby(by='query_id')[['result_id', 'sloth_overlap']]

s = [(6266, 2), (8312, 2), (8728, 2), (5474, 2), (8890, 2), (3317, 2), (2516, 2), (516, 1), (1647, 1), (4341, 1), (4933, 1), (3232, 1), (9357, 1), (7217, 1), (612, 1), (4926, 1), (2441, 1), (2881, 1), (12114, 1), (2753, 1), (4853, 1), (4188, 1), (1727, 1), (9571, 1), (8827, 1), (8404, 1)]

x = [x[0] for x in s[:5]]

print(x)