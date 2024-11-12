"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""
import numpy as np
from typing import List, Dict
import csv
import pandas as pd


def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    precision = 0
    num_rel = 0
    if search_result_relevances:
        for i in range(0, cut_off):
            if search_result_relevances[i] == 1: # or search_result_relevances[i] == 2:
                num_rel += 1
                precision += num_rel / (i + 1)

    if num_rel != 0:
        map_score = precision / cut_off
        return map_score
    else:
        return 0


def ndcg_score(search_result_relevances: list[float],
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    dcg = 0
    idcg = 0

    if search_result_relevances and ideal_relevance_score_ordering:
        for i in range(0, cut_off):
            # if search_result_relevances[i] == 1:
            #     relevant = 1
            # elif search_result_relevances[i] == 2:
            #     relevant = 2
            # else:
            #     continue

            # if ideal_relevance_score_ordering[i] == 1:
            #     i_relevant = 1
            # elif ideal_relevance_score_ordering[i] == 2:
            #     i_relevant = 2
            # else:
            #     i_relevant = 0
            relevant = search_result_relevances[i]
            i_relevant = ideal_relevance_score_ordering[i]

            if i == 0:
                dcg += relevant
                idcg += i_relevant
            else:
                log = np.log2(i+1)
                dcg += (relevant / log)
                idcg += (i_relevant / log)

    if idcg != 0:
        ndcg = dcg / idcg
        return ndcg
    else:
        return 0


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.

    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset

    # TODO: Run each of the dataset's queries through your ranking function

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.

    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    print("New Test")
    relevance_data = {}
    with open(relevance_data_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)
        for row in csv_reader:
            query = row[0]
            doc_id = int(row[2])
            score = int(row[4])
            if query not in relevance_data:
                relevance_data[query] = []
            relevance_data[query].append((doc_id, score))

    map_scores = []
    ndcg_scores = []

    # Run each query through the ranking function
    for query, relevance_judgments in relevance_data.items():
        ranked_docs = ranker.query(query)

        # Extract the doc_ids and relevance scores from the ranked results
        ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]

        actual_relevances = [1 if score >= 4 else 0 for doc_id, score in relevance_judgments]
        actual_relevances_full = [score for doc_id, score in relevance_judgments]

        # Create the search result relevances list
        search_result_relevances = [
            relevance_judgment
            for doc_id in ranked_doc_ids
            for doc_id_judgment, relevance_judgment in relevance_judgments
            if doc_id == doc_id_judgment
        ]

        # Ensure the search_result_relevances length matches the cut-off
        while len(search_result_relevances) < 10:
            search_result_relevances.append(0)  # Pad with zeros if less than cut-off


        # Calculate the ideal relevance scores for NDCG
        ideal_relevance_score_ordering = sorted(actual_relevances_full, reverse=True)

        # Calculate MAP and NDCG for the current query
        try:
            map_score_query = map_score(search_result_relevances, cut_off=10)
            ndcg_score_query = ndcg_score(search_result_relevances, ideal_relevance_score_ordering, cut_off=10)
        except IndexError as e:
            print(f"Error calculating scores for query '{query}': {e}")
            continue

        map_scores.append(map_score_query)
        ndcg_scores.append(ndcg_score_query)


    avg_map = np.mean(map_scores)
    avg_ndcg = np.mean(ndcg_scores)

    return {
        'map': avg_map,
        'ndcg': avg_ndcg,
        'map_list': map_scores,
        'ndcg_list': ndcg_scores
    }


if __name__ == '__main__':
    pass
