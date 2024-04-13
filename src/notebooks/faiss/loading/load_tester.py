import os
from itertools import product


# n_tables = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
# batch_sizes = [1000, 10000, 100000]

n_tables = [10]
batch_sizes = [100]
to_remove = False

for id_test, (n, add_label, batch_size) in enumerate(product(n_tables, [True], batch_sizes)):
    # to_remove = n < 25000
    if n < 10000 and batch_size > 10000: continue

    os.system(f"python analysis_loading_v1.py {id_test} {n} {add_label} {batch_size} {to_remove}")

