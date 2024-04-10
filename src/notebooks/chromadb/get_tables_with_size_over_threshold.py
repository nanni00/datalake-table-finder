from code.utils.utils import rebuild_table
from code.utils.settings import DefaultPath

import jsonlines


ROW_THRESHOLD = 5
COLUMN_THRESHOLD = 2


filename = 'sloth_tables_over_threshold.jsonl'


with jsonlines.open(DefaultPath.data_path.wikitables + 'sloth_tables.jsonl') as reader:
    with jsonlines.open(DefaultPath.data_path.wikitables + filename, mode='w') as writer:
        for table in reader:
            



