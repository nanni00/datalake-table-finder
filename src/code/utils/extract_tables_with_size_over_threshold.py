import os
import sys

import jsonlines
import polars as pl
from tqdm import tqdm

from code.utils.settings import DefaultPath



ROW_THRESHOLD = 5
COLUMN_THRESHOLD = 2
AREA_THRESHOLD = 50

IN_SLOTH_RESULTS = True

JSONL_FILE = DefaultPath.data_path.wikitables + 'train_tables.jsonl'
NTOTAL_TABLES = 570171

directory = f'threshold_r{ROW_THRESHOLD}-c{COLUMN_THRESHOLD}-a{AREA_THRESHOLD}/'
jsonl_file = f'sloth-tables-r{ROW_THRESHOLD}-c{COLUMN_THRESHOLD}-a{AREA_THRESHOLD}.jsonl'

if os.path.exists(DefaultPath.data_path.wikitables + directory):
    if input(f'Directory "{DefaultPath.data_path.wikitables + directory}" already exists, drop it? (yes/no)\t') == 'yes':
        os.system(f'rm -rf {DefaultPath.data_path.wikitables + directory}')
    else:
        sys.exit()
    os.mkdir(DefaultPath.data_path.wikitables + directory)

all_sloth_results = pl.scan_csv(DefaultPath.data_path.wikitables + 'train_set_turl_malaguti.csv')

# I load all the table IDs of train_tables.jsonl used in the sloth tests, and read just them

print(f'Reading table IDs from train_set_turl_malaguti.csv...')
sloth_tables_ids = set(pl.concat([all_sloth_results.select('r_id').collect().to_series(), 
                              all_sloth_results.select('s_id').collect().to_series()]) \
                              .to_list())
print('IDs read.')

final_sample_ids = set()

print(f'Reading jsonl file with wikitables from {JSONL_FILE}...')
with jsonlines.open(JSONL_FILE) as reader:
    to_write = []
    with jsonlines.open(DefaultPath.data_path.wikitables + directory + jsonl_file, mode='w') as writer:
        for i, read_json_table in tqdm(enumerate(reader), total=NTOTAL_TABLES):
            if read_json_table['_id'] not in sloth_tables_ids:
                continue

            if len(read_json_table['tableData']) >= ROW_THRESHOLD \
                and len(read_json_table['tableHeaders'][0]) >= COLUMN_THRESHOLD \
                and len(read_json_table['tableData']) * len(read_json_table['tableHeaders'][0]) >= AREA_THRESHOLD:
                to_write.append(read_json_table)
                final_sample_ids.add(read_json_table['_id'])
            
            # write by step of 5%
            if len(to_write) >= len(sloth_tables_ids) * 0.05:
                writer.write_all(to_write)
                to_write = []
        if to_write:
            writer.write_all(to_write)
        
print(f'Completed. Wrote {len(to_write)} tables.')



sub_res = all_sloth_results.filter((pl.col('r_id').is_in(final_sample_ids)) & (pl.col('s_id').is_in(final_sample_ids))).collect()
sub_res.write_csv(DefaultPath.data_path.wikitables + directory + f'sloth-results-r{ROW_THRESHOLD}-c{COLUMN_THRESHOLD}-a{AREA_THRESHOLD}.csv')

