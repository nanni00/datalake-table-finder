import bisect
import json


class LUT:
    def __init__(self, idxs=None, table_ids=None) -> None:
        self.idxs = idxs
        self.table_ids = table_ids

    def load(self, json_lut_filepath:str):
        with open(json_lut_filepath, 'r') as reader:
            json_lut = json.load(reader)
            try:
                assert len(json_lut['idxs']) == len(json_lut['table_ids'])
                self.idxs = json_lut['idxs']
                self.table_ids = json_lut['table_ids']
            except AssertionError:
                print(f"idxs and table ids have different lengths: {len(json_lut['idxs'])}, {len(json_lut['table_ids'])}")

    def lookup(self, vector_id):
        return self.table_ids[bisect.bisect_left(self.idxs, vector_id)]

    @property
    def ntotal(self):
        return len(self.idxs)