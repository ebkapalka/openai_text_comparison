from fuzzywuzzy import fuzz
from typing import Callable

from utils import export_combined_results


def find_best_matches(descriptions: list[str], targets: list[str], method: Callable) -> list[tuple[str, float]]:
    """
    Find the best matches for a list of descriptions from a list of targets using the specified model.
    :param descriptions: descriptions (incoming data) to match
    :param targets: targets (existing data) to match against
    :param method: fuzzywuzzy method to use for matching
    :return: list of best matches
    """
    best_matches = []
    for description in descriptions:
        closest_match = ({}, 0.0)
        for target in targets:
            ratio = method(description, target)
            if ratio > closest_match[1]:
                closest_match = (target, float(ratio))
        best_matches.append(closest_match)
    return best_matches


def fuzzy_match(descriptions: list[str], targets: list[str]):
    """
    Perform fuzzy matching on the descriptions and targets.
    :param descriptions: descriptions (incoming data) to match
    :param targets: targets (existing data) to match against
    :return: None
    """
    methods = {
        "Ratio": fuzz.ratio,
        "Partial Ratio": fuzz.partial_ratio,
        "Token Sort Ratio": fuzz.token_sort_ratio,
        "Token Set Ratio": fuzz.token_set_ratio
    }

    all_results = {}
    for method in methods:
        func = methods[method]
        print(f"Matching using {method}...")
        matches = find_best_matches(descriptions, targets, func)
        # export_to_csv(method, descriptions, matches)
        all_results[method] = matches
    export_combined_results(list(methods.keys()),
                            descriptions,
                            all_results)
    return all_results
