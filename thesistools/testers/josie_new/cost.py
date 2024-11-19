import time

from db import JOSIEDBHandler


# Constants
min_read_cost = 1000000.0
read_set_cost_slope = 1253.19054300781
read_set_cost_intercept = -9423326.99507381
read_list_cost_slope = 1661.93366983753
read_list_cost_intercept = 1007857.48225696


def read_list_cost(length: int) -> float:
    cost = read_list_cost_slope * float(length) + read_list_cost_intercept
    if cost < min_read_cost:
        cost = min_read_cost
    return cost / 1000000.0


def read_set_cost(size: int) -> float:
    cost = read_set_cost_slope * float(size) + read_set_cost_intercept
    if cost < min_read_cost:
        cost = min_read_cost
    return cost / 1000000.0


def read_set_cost_reduction(size: int, truncation: int) -> float:
    return read_set_cost(size) - read_set_cost(size - truncation)



def sample_read_set_cost(db:JOSIEDBHandler):
    # the table should have been already created when the JOSIE has built the main indexes
    sample_set_ids = db.get_queries_agg_id()

    for set_id in sample_set_ids:
        start = time.time()
        s = db.get_set_tokens(set_id)
        duration = time.time() - start  # Duration in seconds
        db.insert_read_set_cost(set_id, len(s), int(duration * 1e9))
 

def sample_read_list_cost(db:JOSIEDBHandler, 
                          min_length:int = 0, 
                          max_length:int = 20_000, 
                          step:int = 500, 
                          sample_size_per_step:int = 10):
    for l in range(min_length, max_length, step):
        db.insert_read_list_cost(l, l+step, sample_size_per_step)
        # count = db.count_token_from_read_list_cost(l, l+step)

    sample_list_tokens = db.get_array_agg_token_read_list_cost()
    for token in sample_list_tokens:
        start = time.time()
        db.get_inverted_list(token)
        duration = time.time() - start

        db.update_read_list_cost(token, int(duration * 1e9))


def sample_cost(db:JOSIEDBHandler, 
                min_length:int = 0, 
                max_length:int = 20_000, 
                step:int = 500, 
                sample_size_per_step:int = 10):
    sample_read_set_cost(db)
    sample_read_list_cost(db, min_length, max_length, step, sample_size_per_step)
