import argparse
from pprint import pprint


from thesistools.myutils.basicconfig import DATALAKES
from thesistools.myutils.datalake import SimpleDataLakeHelper
from thesistools.myutils.misc import apply_sloth




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('r_id', type=int)
    parser.add_argument('s_id', type=int)
    parser.add_argument('--datalake', required=False, default='wikiturlsnap', choices=DATALAKES)
    args = parser.parse_args()
    
    r_id, s_id, datalake = args.r_id, args.s_id, args.datalake

    match datalake:
        case "wikiturlsnap":
            datalake_location = 'mongodb'
            mapping_id_file = numeric_columns_file = None
            blacklist = []
        case "gittables":
            datalake_location = 'mongodb'
            mapping_id_file = numeric_columns_file = None
            blacklist = {"comment", "story", "{\"$numberDouble\": \"NaN\"}"}
        case "santoslarge":
            datalake_location = "/data4/nanni/data/santos_large/datalake"
            mapping_id_file = "/data4/nanni/data/santos_large/mapping_id.pickle"
            numeric_columns_file = "/data4/nanni/data/santos_large/numeric_columns.pickle"
            blacklist = []
        case _:
            raise ValueError(f"Unknown dataset {datalake}")

    dlh = SimpleDataLakeHelper(datalake_location, datalake, 'standard', mapping_id_file, numeric_columns_file)

    r_table_obj = dlh.get_table_by_numeric_id(r_id)
    s_table_obj = dlh.get_table_by_numeric_id(s_id)

    r, m = apply_sloth(r_table_obj['content'], s_table_obj['content'], r_table_obj['numeric_columns'], s_table_obj['numeric_columns'], blacklist=blacklist)
    #  1print('results:')
    #  1pprint(r)
    #  1print()
    print('metrics:')
    print(m)
    print()

