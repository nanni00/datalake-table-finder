import os
import pandas as pd


josie = (
    pd.read_csv(f"{os.environ['DTFPATH']}/data/examples/wikitables/josie/results_josie.csv.raw")[['query_id', 'results']]
    .fillna('')
    .set_index('query_id')
    .to_dict()['results']
)

with open(f"{os.environ['DTFPATH']}/data/examples/wikitables/bf/results_bf.csv") as fr:
    bf = {int(k): v for k, v in [line.strip().split(',') for line in fr.readlines()]}

print(len(bf), len(josie), bf.keys() == josie.keys())

for bfq, bfqv in bf.items():
    jv = josie[bfq]
    assert hash(''.join(sorted(bfqv))) == hash(''.join(sorted(jv)))

print("JOSIE and Brute-force results are equals")
