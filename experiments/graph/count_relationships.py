from tqdm import tqdm

from tools.utils.utils import _create_token_set
from tools.utils.utils import get_mongodb_collections


mongodb, collections = get_mongodb_collections(small=False)


collection = collections[0]

tot_rel = 0
ntables = sum(collection.count_documents({}) for collection in collections)
for collection in collections:
    print(collection)
    for doc in tqdm(collection.find({}, projection={'content': 1, 'numeric_columns': 1})):
        table = doc['content']
        nc = doc['numeric_columns']
        
        tot_rel += len(_create_token_set(table, 'set', nc))


# anche considerando tutte le tabelle e le possibili relazioni, si verrebbe ad ottenere un 
# grafo di circa ~6GB
# è comunque meno di JOSIE-Bag, e se il tempo di query fosse accettabile sarebbe top
# risulterebbe un grafo bipartito, 
# 2'697'723 75'859'073

print(ntables, tot_rel)

mongodb.close()
