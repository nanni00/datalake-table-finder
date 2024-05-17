import pandas as pd

from tools.josiedataprep.preparation_functions import _create_set_with_set_semantic
root_dir = '/home/nanni/unimore/tesi-magistrale/data/josie-tests/n45673-mbag'

tables_csv_dir =        '/home/nanni/unimore/tesi-magistrale/data/turl_sloth_wikitables/tables-subset/csv'
josie_sloth_ids_file =  root_dir + '/josie_sloth_ids.csv'
tables_file =           root_dir + '/tables.set'


josie_sloth_ids = pd.read_csv(josie_sloth_ids_file)


s_id1 = josie_sloth_ids[josie_sloth_ids['josie_id'] == 1]['sloth_id'].values[0]

print(pd.read_csv(tables_csv_dir + '/' + s_id1))
