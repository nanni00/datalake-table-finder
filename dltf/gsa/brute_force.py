from collections import defaultdict

from tqdm import tqdm

from dltf.utils.loghandler import info
from dltf.gsa.base_tester import AbstractGlobalSearchAlgorithm
from dltf.utils.tables import table_to_tokens, is_valid_table



class BruteForceGS(AbstractGlobalSearchAlgorithm):
    def __init__(self, mode, blacklist, dlh, string_translators, string_patterns):
        super().__init__(mode, blacklist, dlh, string_translators, string_patterns)

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
                                                    string_blacklist=[],
                                                    string_translators=self.string_translators,
                                                    string_patterns=self.string_patterns))
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


