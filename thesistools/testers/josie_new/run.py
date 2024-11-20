import os
import time

from thesistools.testers.josie_new.db import JOSIEDBHandler
from thesistools.testers.josie_new.tokentable import TokenTableMem, TokenTableDisk
from thesistools.testers.josie_new.josie_alg import JOSIE
from thesistools.testers.josie_new.exp import write_all_results

from thesistools.utils.logging_handler import info


def run_experiments(db:JOSIEDBHandler, k: int, output_dir: str, results_file: str, use_mem_token_table: bool, query_ignore_self: bool, verbose: bool):
    db.reset_cost_function_parameters(verbose)

    if verbose:
        info("Counting total number of sets")
    total_number_of_sets = float(db.count_number_of_sets())
    if verbose:
        info(f"Number of sets: {total_number_of_sets}")

    if use_mem_token_table:
        if verbose: info("Creating token table on memory...")
        tb = TokenTableMem(db, query_ignore_self)
    else:
        if verbose: info("Creating token table on disk...")
        tb = TokenTableDisk(db, query_ignore_self)
    
    queries = db.get_query_sets()

    if verbose: info(f"Begin experiment for {k=}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    perfs = []
    start = time.time()

    for q in queries:
        perfs.append(JOSIE(db, tb, q, k, query_ignore_self)[1])

    if verbose: info(f"Finished experiment for {k=} in {round((time.time() - start) / 60, 3)} minutes")
    
    write_all_results(perfs, results_file)
