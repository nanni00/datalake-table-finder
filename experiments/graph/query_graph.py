import random
from time import time
from neo4j import GraphDatabase



small = True

USER = "neo4j"
PASSWD = "12345678"

AUTH = (USER, PASSWD)
DATABASE = "neo4j"
URI = f"bolt://localhost:7687"
       
query_ids = random.choices(range(1, 20000), k=100)
query = """
    UNWIND $query_ids as query_id
    WITH query_id
    MATCH (n:Table WHERE n.table_id = query_id)-[r:HAS]->(t:Token)<-[p:HAS]-(m:Table)
    RETURN 
        query_id, 
        CASE WHEN r.token_count <= p.token_count THEN r.token_count ELSE p.token_count END as token_count,
        // apoc.coll.min([r.token_count, p.token_count]) AS token_count,
        m.table_id as result_id
"""

start = time()
with GraphDatabase.driver(uri=URI, auth=AUTH) as driver:
    with driver.session(database=DATABASE) as session:
        results = session.run(query=query, parameters={"query_ids": query_ids}).values()
        # results.pivot_table(values=['token_count'], index=['query_id', 'result_id'], aggfunc='sum')

print(time() - start)
