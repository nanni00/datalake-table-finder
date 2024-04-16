import os
from sys import getsizeof
from code.utils.utils import rebuild_table
from code.utils.settings import DefaultPath
import polars as pl

import jsonlines


ROW_THRESHOLD = 5
COLUMN_THRESHOLD = 2
AREA_THRESHOLD = 50

directory = f'threshold_r{ROW_THRESHOLD}-c{COLUMN_THRESHOLD}-a{AREA_THRESHOLD}/'
jsonl_file = f'sloth-tables-r{ROW_THRESHOLD}-c{COLUMN_THRESHOLD}-a{AREA_THRESHOLD}.jsonl'

if not os.path.exists(DefaultPath.data_path.wikitables + directory):
    os.mkdir(DefaultPath.data_path.wikitables + directory)


ids = set()

with jsonlines.open(DefaultPath.data_path.wikitables + 'sloth_tables.jsonl') as reader:
    to_write = []
    with jsonlines.open(DefaultPath.data_path.wikitables + directory + jsonl_file, mode='w') as writer:
        for i, read_json_table in enumerate(reader):
            if len(read_json_table['tableData']) >= ROW_THRESHOLD \
                and len(read_json_table['tableHeaders'][0]) >= COLUMN_THRESHOLD \
                and len(read_json_table['tableData']) * len(read_json_table['tableHeaders'][0]) >= AREA_THRESHOLD:
                to_write.append(read_json_table)
                ids.add(read_json_table['_id'])               
        writer.write_all(to_write)
        print(f'Completed. Wrote {len(to_write)} tables ({len(to_write)*100//i}%).')


all_sloth_results = pl.scan_csv(DefaultPath.data_path.wikitables + 'train_set_turl_malaguti.csv')

sub_res = all_sloth_results.filter((pl.col('r_id').is_in(ids)) & (pl.col('s_id').is_in(ids))).collect()
sub_res.write_csv(DefaultPath.data_path.wikitables + directory + f'sloth-results-r{ROW_THRESHOLD}-c{COLUMN_THRESHOLD}-a{AREA_THRESHOLD}.csv')

