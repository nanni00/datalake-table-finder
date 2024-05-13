import os
import sys

import jsonlines
import polars as pl
from tqdm import tqdm

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import rebuild_table

ROW_THRESHOLD = 5
COLUMN_THRESHOLD = 2
AREA_THRESHOLD = 50

IN_SLOTH_RESULTS = True

input_csv_file = defpath.data_path.wikitables + '/train_set_turl_malaguti.csv'
FULL_JSON_FILE = defpath.data_path.wikitables + '/train_tables.jsonl'
NTOTAL_TABLES = 570171

directory = defpath.data_path.wikitables + '/sloth-subset'

output_sloth_csv_file = directory + f'/sloth-results-r{ROW_THRESHOLD}-c{COLUMN_THRESHOLD}-a{AREA_THRESHOLD}.csv'
output_csv_tables_dir = directory + '/csv'

if os.path.exists(directory):
    if input(f'Directory "{directory}" already exists, drop it? (yes/no)\t') == 'yes':
        os.system(f'rm -rf {directory}')
    else:
        sys.exit()
    os.mkdir(directory)
else:
    os.system(f'mkdir -p {directory}')
    os.system(f'mkdir -p {output_csv_tables_dir}')


all_sloth_results = pl.scan_csv(input_csv_file)

# I load all the table IDs of train_tables.jsonl used in the sloth tests, and read just them
print('Reading table IDs from train_set_turl_malaguti.csv...')
sloth_tables_ids = set(pl.concat([all_sloth_results.select('r_id').collect().to_series(), 
                              all_sloth_results.select('s_id').collect().to_series()]) \
                              .to_list())
print('IDs read.')

final_sample_ids = set()

print(f'Reading jsonl file with wikitables from {FULL_JSON_FILE}...')
ntabread = 0
with jsonlines.open(FULL_JSON_FILE) as reader:
    for i, jtable in tqdm(enumerate(reader), total=NTOTAL_TABLES):
        if jtable['_id'] not in sloth_tables_ids:
            continue

        if len(jtable['tableData']) >= ROW_THRESHOLD \
            and len(jtable['tableHeaders'][0]) >= COLUMN_THRESHOLD \
            and len(jtable['tableData']) * len(jtable['tableHeaders'][0]) >= AREA_THRESHOLD:
                
            rebuild_table(jtable).to_csv(output_csv_tables_dir + f'/{jtable["_id"]}', index=False)
            final_sample_ids.add(jtable['_id'])
            ntabread += 1

print(f'Completed. Wrote {ntabread} tables.')

sub_res = all_sloth_results.filter((pl.col('r_id').is_in(final_sample_ids)) & (pl.col('s_id').is_in(final_sample_ids))).collect()
sub_res.write_csv(output_sloth_csv_file)

