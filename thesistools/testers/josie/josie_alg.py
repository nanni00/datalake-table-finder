import time

from db import JOSIEDBHandler
from josie_io import RawTokenSet
from tokentable import TokenTable
from common import overlap
from exp import ExperimentResult, write_result_string

from heap import (
    SearchResultHeap, 
    push_candidate, 
    kth_overlap, 
    kth_overlap_after_push, 
    ordered_results
)

from josie_util import (
    CandidateEntry, 
    process_candidates_init,
    process_candidates_update,
    upperbound_overlap_unknown_candidate, 
    next_batch_distinct_lists, 
    read_set_benefit,
    read_lists_benefit_for_candidate
)


def JOSIE(
    db: JOSIEDBHandler,
    tb: TokenTable,
    query: RawTokenSet,
    k: int,
    batch_size: int = 20,
    expensive_estimation_budget: int = 5000,
    ignore_self: bool = True
):
    exp_result = ExperimentResult()

    start = time.time()
    tokens, freqs, gids = tb.process(query)
    read_list_costs = [0.0] * len(freqs)
    for i in range(len(freqs)):
        if i == 0:
            read_list_costs[i] = db.read_list_cost(freqs[i] + 1)
        else:
            read_list_costs[i] = read_list_costs[i - 1] + db.read_list_cost(freqs[i] + 1)

    exp_result.preproc_duration = int((time.time() - start) * 1000)  # in milliseconds
    start = time.time()

    query_size = len(tokens)
    counter = {}
    ignores = {query.set_id: True} if ignore_self else {}
    h = SearchResultHeap()
    num_skipped = 0

    curr_batch_lists = batch_size

    for i in range(query_size):
        token = tokens[i]
        skipped_overlap = num_skipped
        max_overlap_unseen_candidate = upperbound_overlap_unknown_candidate(query_size, i, skipped_overlap)

        if kth_overlap(h, k) >= max_overlap_unseen_candidate and len(counter) == 0:
            break
        
        entries = db.get_inverted_list(token)
        exp_result.num_list_read += 1
        exp_result.max_list_size_read = max(exp_result.max_list_size_read, len(entries))

        for entry in entries:
            if entry.set_id in ignores:
                continue
            if entry.set_id in counter:
                ce = counter[entry.set_id]
                ce.update(entry.match_position, skipped_overlap)
                continue
            if kth_overlap(h, k) >= max_overlap_unseen_candidate:
                continue
            counter[entry.set_id] = CandidateEntry(entry.set_id, entry.size, entry.match_position, i, skipped_overlap)

        if i == query_size - 1:
            break

        if len(counter) == 0 or (len(counter) < k and len(h) < k) or curr_batch_lists > 0:
            curr_batch_lists -= 1
            continue

        curr_batch_lists = batch_size
        next_batch_end_index = next_batch_distinct_lists(tokens, gids, i, batch_size)
        merge_lists_cost = read_list_costs[next_batch_end_index] - read_list_costs[i]

        merge_lists_benefit, num_with_benefit, candidates = process_candidates_init(
            db, query_size, i, next_batch_end_index, kth_overlap(h, k), batch_size, counter, ignores
        )

        exp_result.max_counter_size = max(exp_result.max_counter_size, len(counter))

        if num_with_benefit == 0 or len(candidates) == 0:
            continue

        candidates.sort(key=lambda c: c.estimated_overlap, reverse=True)
        prev_kth_overlap = kth_overlap(h, k)
        num_candidate_expensive = 0
        fast_estimate = False
        fast_estimate_kth_overlap = None

        for candidate in candidates:
            if candidate is None:
                continue
            kth = kth_overlap(h, k)
            if candidate.estimated_overlap <= kth:
                break
            if len(h) >= k:
                num_candidate_expensive += 1
                if not fast_estimate and num_candidate_expensive * len(candidates) > expensive_estimation_budget:
                    fast_estimate = True
                    fast_estimate_kth_overlap = prev_kth_overlap
                if not fast_estimate:
                    merge_lists_benefit = process_candidates_update(db, kth, candidates, counter, ignores)

                probe_set_benefit = read_set_benefit(
                    query_size, kth, kth_overlap_after_push(h, k, candidate.estimated_overlap),
                    candidates, read_list_costs, fast_estimate
                )
                probe_set_cost = candidate.estimated_cost

                if probe_set_benefit - probe_set_cost < merge_lists_benefit - merge_lists_cost:
                    break

            if fast_estimate or (num_candidate_expensive + 1) * len(candidates) > expensive_estimation_budget:
                merge_lists_benefit -= read_lists_benefit_for_candidate(db, candidate, fast_estimate_kth_overlap)

            candidate.read = True
            ignores[candidate.id] = True
            del counter[candidate.id]

            if candidate.maximum_overlap <= kth:
                continue

            total_overlap = 0
            if candidate.suffix_length() > 0:
                s = db.get_set_tokens_by_suffix(candidate.id, candidate.latest_match_position + 1)
                if s == None:
                    print('>>>>>  ', candidate.id, s)
                exp_result.num_set_read += 1
                exp_result.max_set_size_read = max(exp_result.max_set_size_read, len(s))
                suffix_overlap = overlap(s, tokens[i + 1:])
                total_overlap = suffix_overlap + candidate.partial_overlap
            else:
                total_overlap = candidate.partial_overlap

            prev_kth_overlap = kth
            push_candidate(h, k, candidate.id, total_overlap)

    for ce in counter.values():
        push_candidate(h, k, ce.id, ce.partial_overlap)

    results = ordered_results(h)

    exp_result.duration = int((time.time() - start) * 1000)  # in milliseconds
    exp_result.results = write_result_string(results)
    exp_result.query_id = query.set_id
    exp_result.query_size = len(query.raw_tokens)
    exp_result.num_result = len(results)
    exp_result.ignore_size = len(ignores)
    exp_result.query_num_token = len(tokens)
    
    return results, exp_result
