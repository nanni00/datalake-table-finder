import re
import polars as pl
from code.utils.settings import DefaultPath
from code.utils.utils import get_int_from_


def extract_from_josie_results_pairs_and_overlaps(josie_results_file, 
                                                  pairs_with_overlap_file):
    josie_results = pl.scan_csv(josie_results_file).select(['query_id', 'results'])

    josie_results = josie_results \
        .with_columns(pl.col('results') \
                      .map_elements(get_int_from_, pl.List(pl.Int64))                      
                      ).with_columns(pl.col('results').list.to_struct()).unnest('results')                                                
    
    mapping = {
        'query_id': 'set_id',
        'field_0': 's1', 'field_1': 'o1', 
        'field_2': 's2', 'field_3': 'o2', 
        'field_4': 's3', 'field_5': 'o3'}
    
    josie_results = josie_results.collect().rename(mapping).sort(by=['o1', 'o2', 'o3'], descending=True).drop_nulls()

    josie_results.write_csv(pairs_with_overlap_file)


if __name__ == '__main__':
    n_tables = 500
    ROOT_TEST_DIR = DefaultPath.josie_stuff_path.base + f'josie-test-{n_tables}/'

    josie_to_sloth_id_file =    ROOT_TEST_DIR + 'id_table.csv'
    josie_results_file =        ROOT_TEST_DIR + f'result_k_3.csv'
    pairs_with_overlap_file =   ROOT_TEST_DIR + f'pairs_overlap.csv'

    extract_from_josie_results_pairs_and_overlaps(josie_results_file, pairs_with_overlap_file)
