from sloth import sloth
from utils import parse_arguments, parse_table

import bz2
import csv
import math
import os
import pandas as pd
import pickle as pkl
import sys
import time


path_candidates = "dataset/table_querying_candidates.pkl"
path_tables = "dataset/tables_csv/"
path_dir_results = "results/table_querying/"
path_res_cand_base = "res_cand.csv"
path_res_run_base = "res_run.csv"

res_cand_headers = ["id", "r_id", "r_w", "r_h", "r_a", "r_tokens", "s_id", "s_w", "s_h", "s_a", "s_tokens", "int_size",
                    "jsim", "jsim_time", "josie", "josie_time"]
res_run_headers = ["cand_id", "seeds", "seed_init_time", "algo", "bw", "setup_time", "gen_cands", "gen_time", "ver_cands", "ver_time",
                   "o_num", "o_w", "o_h", "o_a", "total_time"]


def main(mode, r_id=-1, s_id=-1, first_id=-1, num_cand=-1, num_res=1, min_w=0, max_w=math.inf, min_h=0, max_h=math.inf, min_a=0):
    """
    Perform the task defined by the user
    :param mode: the type of task to be performed, i.e., "s" (single candidate) or "m" (batch of candidates)
    :param r_id: the id of the table R(X) (for single mode)
    :param s_id: the id of the table S(Y) (for single mode)
    :param first_id: the index of the first candidate to evaluate (for batch mode)
    :param num_cand: the number of candidates to evaluate (for batch mode)
    :param num_res: the cardinality of the result, i.e., "o" (only the first largest overlap) or "a" (all)
    :param min_w: the minimum overlap width (default 0)
    :param max_w: the maximum overlap width (default infinite)
    :param min_h: the minimum overlap height (default 0)
    :param max_h: the maximum overlap height (default infinite)
    :param min_a: the minimum overlap area (default 0)
    """
    start_time = time.time()

    # Load the candidates
    if mode == "s":
        candidates = [(r_id, s_id)]
    else:
        with open(path_candidates, "rb") as input_file:
            candidates = pkl.load(input_file)
            input_file.close()
        print("Number of candidates: " + str(len(candidates)))

    # Write the headers in the result file describing the candidates
    if mode == "m" and not os.path.exists(path_dir_results):
        os.makedirs(path_dir_results)
    path_res_cand = path_res_cand_base.split("_")
    path_res_cand = path_res_cand[0] + "_" + str(first_id) + "_" + path_res_cand[1]
    path_res_cand = path_dir_results + path_res_cand
    if mode == "m":
        with open(path_res_cand, "w", newline="") as output_file:
            csv.writer(output_file).writerow(res_cand_headers)
            output_file.close()

    # Evaluate the candidates
    results = list()  # list of the detected largest overlaps
    tab_cache = dict()  # used to store the tables whose information have already been computed
    tot_cand = len(candidates)
    iter_id = 0

    while iter_id < num_cand and first_id + iter_id < tot_cand:
        cand_id = first_id + iter_id
        cand = candidates[cand_id] if mode == "m" else candidates[0]

        r_id = cand[0]  # left table identifier
        s_id = cand[1]  # right table identifier

        if mode == "m":
            if iter_id % 50 == 0:
                print(iter_id)
    
        # Get the information about the tables
        if r_id in tab_cache.keys():
            r_obj = tab_cache[r_id]
        else:
            r_df = pd.read_csv(path_tables + r_id, lineterminator="\n")
            r_obj = dict()
            r_obj["content"] = r_df.values.tolist()
            r_obj["headers"] = list(r_df.columns)
            r_obj["num_header_rows"] = 0
            r_obj["num_columns"] = len(r_obj["content"][0])
            tab_cache[r_id] = r_obj

        if s_id in tab_cache.keys():
            s_obj = tab_cache[s_id]
        else:
            s_df = pd.read_csv(path_tables + s_id, lineterminator="\n")
            s_obj = dict()
            s_obj["content"] = s_df.values.tolist()
            s_obj["headers"] = list(s_df.columns)
            s_obj["num_header_rows"] = 0
            s_obj["num_columns"] = len(s_obj["content"][0])
            tab_cache[s_id] = s_obj

        # Put the tables into a list of lists (columns) format
        r_tab = parse_table(r_obj["content"], r_obj["num_columns"], r_obj["num_header_rows"])
        s_tab = parse_table(s_obj["content"], s_obj["num_columns"], s_obj["num_header_rows"])

        # Compute the Jaccard similarity and the overlap set similarity (Josie) between the sets of cells of the two tables
        r_tokens = {cell for col in r_tab for cell in col}
        s_tokens = {cell for col in s_tab for cell in col}

        int_size = len(r_tokens.intersection(s_tokens))

        jsim_start_time = time.time()
        try:
            jsim = len(r_tokens.intersection(s_tokens)) / len(r_tokens.union(s_tokens))
        except ZeroDivisionError:
            jsim = None
        jsim_time = time.time() - jsim_start_time

        josie_start_time = time.time()
        try:
            josie = len(r_tokens.intersection(s_tokens)) / min(len(r_tokens), len(s_tokens))
        except ZeroDivisionError:
            josie = None
        josie_time = time.time() - josie_start_time

        # Compute the size of the two tables
        r_w = len(r_tab)  # width (i.e., number of columns) of the table R(X)
        try:
            r_h = len(r_tab[0])  # height (i.e., number of rows) of the table R(X)
        except IndexError:
            r_h = 0
        r_a = r_w * r_h  # area of the table R(X)

        s_w = len(s_tab)  # width (i.e., number of columns) of the table S(Y)
        try:
            s_h = len(s_tab[0])  # height (i.e., number of rows) of the table S(Y)
        except IndexError:
            s_h = 0
        s_a = s_w * s_h  # area of the table S(Y)

        # Save the information about the current pair in the result file describing the candidates
        if mode == "m":
            metrics = [cand_id, r_id, r_w, r_h, r_a, len(r_tokens), s_id, s_w, s_h, s_a, len(s_tokens), int_size,
                       jsim, jsim_time, josie, josie_time]
            with open(path_res_cand, "a", newline="") as input_file:
                csv_writer = csv.writer(input_file)
                csv_writer.writerow(metrics)
                input_file.close()

        # Write the headers in the result file describing the metrics of the performed run
        path_res_run = path_res_run_base.split("_")
        path_res_run = path_res_run[0] + "_" + str(first_id) + "_" + path_res_run[1]
        path_res_run = path_dir_results + path_res_run
        if mode == "m":
            if iter_id == 0:
                with open(path_res_run, "w", newline="") as out_file:
                    csv.writer(out_file).writerow(res_run_headers)
                    out_file.close()
        metrics = [cand_id] if mode == "m" else None

        # Detect the largest overlaps
        verbose = True if mode == "s" else False
        results, metrics = sloth(r_tab, s_tab, min_a=min_a, metrics=metrics, verbose=verbose,
                                 min_w=min_w, max_w=max_w, min_h=min_h, max_h=max_h)

        if mode == "m":
            while len(metrics) < len(res_run_headers):
                metrics.append(None)
            with open(path_res_run, "a", newline="") as output_file:
                csv.writer(output_file).writerow(metrics)
                output_file.close()

        iter_id += 1

    if mode == "m":
        print("Number of performed comparisons: " + str(iter_id))
    print("Total elapsed time: " + str(time.time() - start_time) + " seconds")

    return results


if __name__ == "__main__":
    parsed_args = parse_arguments(sys.argv)
    mode = parsed_args["mode"]
    if mode == "s":
        results = main(mode, r_id=parsed_args["r_id"], s_id=parsed_args["s_id"], num_cand=1,
                       min_a=parsed_args["delta"] if "delta" in parsed_args.keys() else 0,
                       min_w=parsed_args["min_w"] if "min_w" in parsed_args.keys() else 0,
                       max_w=parsed_args["max_w"] if "max_w" in parsed_args.keys() else math.inf,
                       min_h=parsed_args["min_h"] if "min_h" in parsed_args.keys() else 0,
                       max_h=parsed_args["max_h"] if "max_h" in parsed_args.keys() else math.inf)
    else:
        results = main(mode, first_id=parsed_args["first_id"], num_cand=parsed_args["num_cand"],
                       min_a=parsed_args["delta"] if "delta" in parsed_args.keys() else 0,
                       min_w=parsed_args["min_w"] if "min_w" in parsed_args.keys() else 0,
                       max_w=parsed_args["max_w"] if "max_w" in parsed_args.keys() else math.inf,
                       min_h=parsed_args["min_h"] if "min_h" in parsed_args.keys() else 0,
                       max_h=parsed_args["max_h"] if "max_h" in parsed_args.keys() else math.inf)
