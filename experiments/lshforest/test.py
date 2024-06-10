from datasketch import MinHashLSHForest, MinHash


with open('experiments/lshforest/mining_massive_datasets_rows.txt') as freader:
    data = [line.split() for line in freader.readlines()]


num_perm = 256
def _create_minhash(d):
    m = MinHash(num_perm)
    m.update_batch([w.encode('utf-8') for w in d])
    return m

minhashes = [(i, _create_minhash(d)) for i, d in enumerate(data, start=1)]

# Create a MinHash LSH Forest with the same num_perm parameter
forest = MinHashLSHForest(num_perm=num_perm, l=16)

for i, m in minhashes:
    forest.add(i, m)

# IMPORTANT: must call index() otherwise the keys won't be searchable
forest.index()

# Check for membership using the key

qid = 1
qminhash = minhashes[qid - 1][1]

result = forest.query(qminhash, 10)
print(qid, ' '.join(data[qid - 1]))
print()

qset = set(data[qid - 1])
for rid in  result:
    rset = set(data[rid - 1])

    print(rid, '\t', round(len(qset.intersection(rset)) / len(qset.union(rset)), 3), '\t', ' '.join(data[rid - 1]))
