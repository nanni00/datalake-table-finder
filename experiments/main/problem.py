import re
import pprint
import pandas as pd

from tools.josiestuff.db import JosieDB


josiedb = JosieDB('nanni', 'alltests_mset')
josiedb.open()
stat = pd.DataFrame(josiedb.get_statistics())

def convert_to_giga(x):
    if x.upper().endswith('MB'):
        return int(re.match(r'\d+', x).group()) / 1024
    elif x.upper().endswith('KB'):
        return int(re.match(r'\d+', x).group()) / (1024 ** 2)

dbsize = pd.DataFrame([['josie', 'set', stat['total_size'].apply(convert_to_giga).sum()]], columns=['algorithm', 'mode', 'size(GB)'])

print(dbsize)

josiedb.close()