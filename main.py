from utils import csv_to_list, export_to_csv
from gpt_match import find_best_matches


if __name__ == '__main__':
    models = [
        "text-embedding-3-small",  # 62_500 pages per 1USD, 62.3% performance
        "text-embedding-ada-002",  # 12,500 pages per 1USD, 61.0%% performance
        "text-embedding-3-large",  # 9,615 pages per 1USD, 64.6% performance
    ]
    descriptions_path = "input/cappex_majors.csv"
    targets_path = "input/banner_majors.csv"
    print_results = False

    descriptions = sorted(csv_to_list(descriptions_path)[:20])
    targets = csv_to_list(targets_path)
    for model in models:
        matches = find_best_matches(descriptions, targets, model=model)
        export_to_csv(model, descriptions, matches)

        if print_results:
            print(f"Using model {model}")
            for d, (match, sim) in zip(descriptions, matches):
                print(f"'{d}' ?= '{match}': similarity {sim:.2f}")
            print('\n')

"""
Models come from 
"""
