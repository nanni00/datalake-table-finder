import faiss
import numpy as np

import pandas as pd

from tools.embedding import FastTextTableEmbedder

xb = np.zeros((10, 5), dtype='float32')
xb[:, 0] = np.arange(10) + 1000

index = faiss.index_factory(5, "Flat,IDMap2")
index.train(xb)

# ids = np.arange(10, dtype='int64')
ids = np.array([0] * 10, dtype='int64')
index.add_with_ids(xb, ids)



print(index.reconstruct_batch(xb))
