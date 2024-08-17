import logging.config
import os
import psycopg
import pymongo
import statistics
import pandas as pd
from tqdm import tqdm
from collections import defaultdict 
import binascii



# mc, coll = get_mongodb_collections('gittables', False)
# doc = get_one_document_from_mongodb_by_key('_id_numeric', 0, *coll)
# print(pd.DataFrame(doc['content']))
# mc.close()
"""

_dbconn = psycopg.connect(f"port=5442 host=/tmp dbname=nanni")


show_stat = 0
show_token = not show_stat

if show_stat == 1:
    res = _dbconn.execute("SELECT frequency FROM test1_dgittables_mset_inverted_lists;")
    res = [r[0] for r in res.fetchall()]
    print(statistics.mean(res))
    print(statistics.stdev(res))
    print(statistics.median(res))
    print(min(res))
    print(max(res))

if show_token == 1:
    res = _dbconn.execute("SELECT raw_token, frequency FROM a_test_dwikipedia_mbag_inverted_lists ORDER BY frequency DESC LIMIT 20;")
    for r in res:
        rt, f = r
        rt = binascii.unhexlify(rt).decode('utf-8')
        print(rt, f)

_dbconn.close()


import multiprocessing as mp

def work(chunk):
    return [i for i in chunk]

sequence = range(100)
chunk_size = 10

def chunks(sequence):
    # Chunks of 1000 documents at a time.
    for j in range(0, len(sequence), chunk_size):
        yield sequence[j:j + chunk_size]

with mp.Pool(5) as pool:
    for j in pool.imap(work, chunks(sequence)):
        print(j)
"""
