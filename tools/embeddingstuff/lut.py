import bisect
import json


class LUT:
    """
    LookUp Table used for keep the ids of indexed vectors on FAISS index
    """
    def __init__(self, idxs=None, ids=None) -> None:
        self.idxs = idxs if idxs else []
        self.ids = ids if ids else []

    def insert_index(self, numitems, id):
        " Insert a new record in the LUT, assuming ordered insert. "
        self.idxs.append(numitems - 1 if len(self.idxs) == 0 else self.idxs[-1] + numitems)
        self.ids.append(id)

    def _load(self, json_lut_filepath:str):
        with open(json_lut_filepath, 'r') as reader:
            json_lut = json.load(reader)
            try:
                assert len(json_lut['idxs']) == len(json_lut['ids'])
                self.idxs = json_lut['idxs']
                self.ids = json_lut['ids']
            except AssertionError:
                print(f"idxs and table ids have different lengths: {len(json_lut['idxs'])}, {len(json_lut['table_ids'])}")

    def save(self, json_lut_filepath:str):
        with open(json_lut_filepath, 'w') as writer:
            json.dump(
                {
                    'idxs': self.idxs,
                    'ids': self.ids
                },
                writer
            )

    def lookup(self, vector_id):
        return self.ids[bisect.bisect_left(self.idxs, vector_id)]

    @property
    def ntotal(self):
        return len(self.idxs)
    

def load_json(path_to_json_lut):
    _lut = LUT()
    _lut._load(path_to_json_lut)
    return _lut