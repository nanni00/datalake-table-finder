from datasketch import MinHash
from itertools import combinations
from pymongo import MongoClient

import bz2
import json
import pickle


def parse_tables(path_tables, table_collection, milestone, n):
    """
    :param path_tables: path of the JSONL file containing the tables
    :param table_collection: MongoDB collection to store the tables
    :param milestone: table interval to use for tracking the progress
    :param n: maximum number of tables to load into the MongoDB collection
    """
    tables = list()  # empty list to store the parsed tables
    counter = 0  # counter of the parsed tables
    with open(path_tables, "r") as input_file:  # open the JSONL file containing the tables
        for i, line in enumerate(input_file):  # for every table contained in the JSONL file
            if counter % milestone == 0 and len(tables) > 0:  # track the progress
                print(counter, end='\r')
                table_collection.insert_many(tables)
                tables = []
            raw_table = json.loads(line)  # load the table in its raw form

            if i >= n:
                break
            
            raw_table_content = raw_table["tableData"]  # load the table content in its raw form
            table = dict()  # empty dictionary to store the parsed table

            table["_id"] = raw_table["_id"]  # unique identifier of the table
            table["_id_numeric"] = counter
            
            table["content"] = [[cell["text"] for cell in row] for row in raw_table_content]  # table content (only cell text)
            
            caption = raw_table["tableCaption"] if "tableCaption" in raw_table else ""
            table["context"] = raw_table["pgTitle"] + " | " + raw_table["sectionTitle"] + " | " +  caption  # table context
            
            # basic format
            table["headers"] = raw_table['tableHeaders']

            # for WikiTables format
            # table["headers"] = [o['text'] for o in raw_table["tableHeaders"][0]]
            table['rows'] = raw_table['numDataRows']
            table['columns'] = raw_table['numCols']
            tables.append(table)  # append the parsed table to the list
            counter += 1  # increment the counter
    table_collection.insert_many(tables)
    print("Table parsing completed: " + str(counter) + " tables read and stored in the database.")


def compute_minhash(table_collection, minhash_size, milestone):
    """
    :param table_collection: MongoDB collection to store the tables
    :param minhash_size: number of bits to use for the minhash
    :param milestone: table interval to use for tracking the progress
    """
    table_ids = [table["_id"] for table in table_collection.find({}, {"_id": 1})]  # retrieve all table identifiers
    counter = 0  # table counter
    for id in table_ids:  # for every table
        if counter % milestone == 0:  # track the progress
                print(counter)
        minhash = MinHash(num_perm=minhash_size)  # initialize the MinHash object
        table_content = list(table_collection.find({"_id": id}, {"content": 1}))[0]["content"]  # retrieve the table content
        cell_values = {cell for row in table_content for cell in row}  # set of all (distinct) cell values in the table
        for cell_value in cell_values:
            minhash.update(cell_value.encode("utf8"))  # update the minhash for every cell value
        table_collection.update_one({"_id": id}, {"$set": {"minhash": minhash.hashvalues.tolist()}})  # update the table document to store its minhash
        counter += 1  # increment the counter
    print("Minhash computation completed for " + str(counter) + " tables.")


def lsh_banding(table_collection, path_clusters, minhash_size, num_bands, milestone):
    """
    :param table_collection: MongoDB collection to store the tables
    :param path_clusters: path of the pickle file to store the obtained clusters of tables
    :param minhash_size: number of bits to use for the minhash
    :param num_bands: number of bands to use for the LSH banding
    :param milestone: table interval to use for tracking the progress
    """
    num_rows = int(minhash_size / num_bands)  # number of rows in each band
    pins = [i for i in range(0, minhash_size, num_rows)]  # bit intervals to decompose the minhash into bands
    pins.append(minhash_size)  # include the last bit to denote the last band
    buckets = [dict() for _ in range(0, num_bands)]  # list of dictionaries to store the buckets for each band
    tables = [(table["_id"], table["minhash"]) for table in table_collection.find({}, {"_id": 1, "minhash": 1})]  # retrieve for each table its identifier and its minhash
    counter = 0  # table counter
    for table in tables:  # for every table
        if counter % milestone == 0:  # track the progress
                print(counter)
        for i in range(0, num_bands):  # decompose the minhash into bands
            hash_value = hash("".join([str(h) for h in table[1][pins[i]:pins[i + 1]]]))  # value for the current band
            if hash_value not in buckets[i].keys():  # if the value is not present in buckets, create a new bucket for that band
                buckets[i][hash_value] = [table[0]]
            else:  # otherwise, insert the table into the existing bucket
                buckets[i][hash_value].append(table[0])
        counter += 1  # increment the counter
    print("LSH banding completed for " + str(counter) + " tables.")
    clusters = list()  # list of all clusters determined by the generated buckets
    for i in range(0, num_bands):  # for each band, read its buckets
        for bucket in buckets[i].items():
            if len(bucket[1]) > 1:  # if the bucket contains more than one element (i.e., generates candidate table pairs)
                cluster = dict()  # initialize the corresponding cluster as a dictionary
                cluster["_id"] = str(bucket[0]) + "-" + str(i)  # discriminate among same bucket identifiers in different bands
                cluster["tables"] = bucket[1]  # list of tables
                clusters.append(cluster)
    with bz2.BZ2File(path_clusters, "wb") as output_file:  # store the clusters into the dedicated file
        pickle.dump(clusters, output_file)
        output_file.close()


def generate_candidate_set(path_clusters, path_candidates, milestone):
    """
    :param path_clusters: path of the pickle file that stores the clusters of tables
    :param path_candidates: path of the pickle file to store the candidate table pairs (i.e., candidates)
    :param milestone: cluster interval to use for tracking the progress
    """
    with bz2.BZ2File(path_clusters, "rb") as input_file:  # read the clusters from their file
        clusters = pickle.load(input_file)
        input_file.close()
    candidates = set()  # set to store the distinct candidate table pairs
    counter = 0  # cluster counter
    for cluster in clusters:  # for each cluster
        if counter % milestone == 0:  # track the progress
                print(counter)
        cluster_candidates = list(combinations(sorted(cluster["tables"]), 2))  # compute the candidate table pairs
        for candidate in cluster_candidates:  # add each table pair to the candidate set
            candidates.add(candidate)
        counter += 1  # increment the counter
    print("Candidate generation completed for " + str(counter) + " clusters: " + str(len(candidates)) + " generated candidates.")
    with bz2.BZ2File(path_candidates, "wb") as output_file:  # store the candidate set into the dedicated file
        pickle.dump(list(candidates), output_file)
        output_file.close()


def main():
    # PARSE THE TABLES FROM THE JSONL FILE AND STORE THEM IN A DEDICATED MONGODB COLLECTION
    # path_tables = "datasets/train_tables.jsonl"  # path of the JSONL file containing the tables
    path_tables = '/home/nanni/datasets_datalakes/WikiTables/tables.json'
    
    client = MongoClient()  # connect to MongoDB
    db = client.datasets  # define the database to use
    table_collection = db.wikitables  # define the collection in the database to store the tables
    milestone = 10000  # table interval to use for tracking the progress
    n = 100
    parse_tables(path_tables, table_collection, milestone, n)
    

    # COMPUTE THE MINHASH FOR ALL TABLES IN THE COLLECTION
    """
    client = MongoClient()  # connect to MongoDB
    db = client.optitab  # define the database to use
    table_collection = db.turl_training_set  # define the collection in the database containing the tables
    minhash_size = 128  # number of bits to use for the minhash
    milestone = 1000  # table interval to use for tracking the progress
    compute_minhash(table_collection, minhash_size, milestone)
    """

    # GENERATE TABLE CLUSTERS THROUGH LSH BANDING
    """
    client = MongoClient()  # connect to MongoDB
    db = client.optitab  # define the database to use
    table_collection = db.turl_training_set  # define the collection in the database containing the tables
    minhash_size = 128  # number of bits to use for the minhash
    num_bands = 16  # number of bands to use for the LSH banding (16 bands ~ 0.7 Jaccard similarity)
    path_clusters = "candidates/clusters_128_16.pkl"  # path of the pickle file to store the obtained clusters of tables
    milestone = 1000  # table interval to use for tracking the progress
    lsh_banding(table_collection, path_clusters, minhash_size, num_bands, milestone)
    """

    # GENERATE CANDIDATE TABLE PAIRS FROM THE TABLE CLUSTERS
    """
    path_clusters = "candidates/clusters_128_16.pkl"  # path of the pickle file that stores the clusters of tables
    path_candidates = "candidates/candidates_128_16.pkl"  # path of the pickle file to store the candidate table pairs
    milestone = 1000  # cluster interval to use for tracking the progress
    generate_candidate_set(path_clusters, path_candidates, milestone)
    """


if __name__ == "__main__":
    main()
