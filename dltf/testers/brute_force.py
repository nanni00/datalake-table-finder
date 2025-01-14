from collections import defaultdict

from tqdm import tqdm

from dltf.utils.loghandler import info
from dltf.testers.base_tester import AlgorithmTester
from dltf.utils.tables import table_to_tokens, is_valid_table



class BruteForceTester(AlgorithmTester):
    def __init__(self, mode, blacklist, dlh, token_translators):
        super().__init__(mode, blacklist, dlh, token_translators)

    def data_preparation(self):
        pass

    def query(self, results_file, k, query_ids, *args):
        info("Creating tokenized tables...")
        tokenized = {}
        for table in tqdm(self.dlh.scan_tables(), total=self.dlh.count_tables()):
            if is_valid_table(table['content'], table['valid_columns']):
                tokenized[table['_id_numeric']] = set(table_to_tokens(
                                                    table['content'],
                                                    table['valid_columns'],
                                                    self.mode,
                                                    encode=None,
                                                    blacklist=[],
                                                    string_translators=self.string_translators))
        info("Tokenized tables created.")

        info("All-to-all comparisons...")
        results = defaultdict(list)
        for qid in tqdm(query_ids):
            tq = tokenized[qid]
            for i in range(self.dlh.count_tables()):
                if i == qid or i not in tokenized:
                    continue
                ti = tokenized[i]
                overlap = [x for x in tq if x in ti]
                if len(overlap) == 0: continue
                results[qid].append((i, len(overlap)))
            results[qid] = sorted(results[qid], key=lambda x: x[1], reverse=True)[:k]
        print("Queries completed.")

        with open(results_file, 'w') as fw:
            for qid, res in results.items():
                fw.write(f"{qid},{''.join(f's{i}o{o}' for i, o in res)}\n")

    
    def clean(self):
        pass


        
if __name__ == '__main__':
    import os

    from dltf.utils.datalake import DataLakeHandlerFactory
    from dltf.utils.loghandler import logging_setup
    from dltf.utils.settings import DefaultPath as dp
    from dltf.utils.query import sample_queries, read_query_ids
    from dltf.utils.misc import whitespace_translator, punctuation_translator, lowercase_translator

    mode = 'set'
    datalake = 'gittables'
    blacklist = []
    mongo_datasets = ['datasets.gittables']

    num_cpu = 8
    dlh = DataLakeHandlerFactory.create_handler('mongodb', 'gittables', mongo_datasets)
    token_translators = [whitespace_translator, punctuation_translator, lowercase_translator]
        
    test_dir = f"{dp.data_path.base}/examples/{datalake}/bf"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    logfile = f"{test_dir}/.logfile"
    db_stat_file = f"{test_dir}/.dbstat"
    query_file = f"{test_dir}/../queries.json"
    results_file = f"{test_dir}/results_bf.csv"

    logging_setup(logfile)
    tester = BruteForceTester(mode, blacklist, dlh, token_translators)

    # print(tester.data_preparation())
    # sample_queries(query_file, 10, 8, 'mongodb', 'wikitables', mongo_datasets)
    
    token_table_on_memory = True
    tester.query(results_file, 10, read_query_ids(query_file))






