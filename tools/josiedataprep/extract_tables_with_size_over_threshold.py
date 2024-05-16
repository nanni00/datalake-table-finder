import os
import sys

import jsonlines
import numpy as np
import polars as pl
from tqdm import tqdm

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import print_info, rebuild_table

@print_info(msg_before="Extracting tables from JSONL file...")
def extract_tables_from_jsonl_as_csv_folder(
        extraction_root_directory,
        input_tables_jsonl_file,
        input_sloth_results_csv_file,
        output_csv_tables_directory,
        output_sloth_results_csv_file,
        thresholds:dict[str:int],
        ntotal_tables=570171
        ):
    if os.path.exists(extraction_root_directory):
        if input(f'Directory "{extraction_root_directory}" already exists, drop it? (yes/no) ') == 'yes':
            os.system(f'rm -rf {extraction_root_directory}')
        else:
            sys.exit()

    os.system(f'mkdir -p {extraction_root_directory}')
    os.system(f'mkdir -p {output_csv_tables_directory}')

    MIN_ROW_THRESHOLD = 0       if 'min_rows' not in thresholds else thresholds['min_rows']
    MAX_ROW_THRESHOLD = np.inf  if 'max_rows' not in thresholds else thresholds['max_rows']

    MIN_COLUMN_THRESHOLD = 0       if 'min_columns' not in thresholds else thresholds['min_columns']
    MAX_COLUMN_THRESHOLD = np.inf  if 'max_columns' not in thresholds else thresholds['max_columns']

    MIN_AREA_THRESHOLD = 0       if 'min_area' not in thresholds else thresholds['min_area']
    MAX_AREA_THRESHOLD = np.inf  if 'max_area' not in thresholds else thresholds['max_area']

    all_sloth_results = pl.scan_csv(input_sloth_results_csv_file)

    # I load all the table IDs of train_tables.jsonl used in the sloth tests, and read just them
    print(f'Reading table IDs from {input_sloth_results_csv_file}...')
    sloth_tables_ids = set( \
        pl.concat( \
            [all_sloth_results.select('r_id').collect().to_series(), 
            all_sloth_results.select('s_id').collect().to_series()]
            ) \
                .to_list()
        )

    final_sample_ids = set()

    print(f'Reading jsonl file with wikitables from {input_tables_jsonl_file}...')
    with jsonlines.open(input_tables_jsonl_file) as reader:
        for i, jtable in tqdm(enumerate(reader), total=ntotal_tables):
            if jtable['_id'] not in sloth_tables_ids:
                continue

            nrows, ncols = len(jtable['tableData']), len(jtable['tableHeaders'][0]) 
            area = len(jtable['tableData']) * len(jtable['tableHeaders'][0])

            if (MIN_ROW_THRESHOLD <= nrows <= MAX_ROW_THRESHOLD) \
                and (MIN_COLUMN_THRESHOLD <= ncols <= MAX_COLUMN_THRESHOLD) \
                and (MIN_AREA_THRESHOLD <= area <= MAX_AREA_THRESHOLD):

                rebuild_table(jtable, mode='pandas').to_csv(output_csv_tables_directory + f'/{jtable["_id"]}')
                # with open(output_csv_tables_dir + f'/{jtable["_id"]}', 'w') as f:
                #     f.write(rebuild_table(jtable, mode='text')) # forse non funziona proprio benissimissimo
                final_sample_ids.add(jtable['_id'])
                
    print(f'Wrote {i + 1} tables.')

    # Selecting only the records whose tables appear in the selected ids
    sub_res = all_sloth_results \
        .filter( \
            (pl.col('r_id').is_in(final_sample_ids)) & \
                (pl.col('s_id').is_in(final_sample_ids)) \
            ) \
                .collect() \
                    .sort(by='overlap_area')

    sub_res.write_csv(output_sloth_results_csv_file)



if __name__ == '__main__':
    ROW_THRESHOLD = 5
    COLUMN_THRESHOLD = 2
    AREA_THRESHOLD = 50

    IN_SLOTH_RESULTS = True

    input_csv_file = defpath.data_path.wikitables + '/original_sloth_results.csv'
    FULL_JSON_FILE = defpath.data_path.wikitables + '/original_turl_train_tables.jsonl'
    NTOTAL_TABLES = 570171

    extraction_root_directory = defpath.data_path.wikitables + '/sloth-subset'

    output_sloth_csv_file = extraction_root_directory + f'/sloth-results-r{ROW_THRESHOLD}-c{COLUMN_THRESHOLD}-a{AREA_THRESHOLD}.csv'
    output_csv_tables_directory = extraction_root_directory + '/csv'
