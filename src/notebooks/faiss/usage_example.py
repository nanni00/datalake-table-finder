import time
import faiss
import numpy as np

d = 64
nb = 10 ** 5
nq = 10 ** 4

np.random.seed(1234)


xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


index = faiss.IndexFlatL2(d)
print(index.is_trained)
start_add = time.time()
index.add(xb)
print(f"add time: {round(time.time() - start_add, 3)}s")
print(index.ntotal)

start_sanity_check = time.time()
k = 4
D, I = index.search(xb[:5], k)
print(f"sanity check time: {round(time.time() - start_sanity_check, 3)}s")
print(I)
print(D)

start_search = time.time()
D, I = index.search(xq, k)
print(f"search time:  {round(time.time() - start_search, 3)}s")
print(I[:5])
print(I[-5:])