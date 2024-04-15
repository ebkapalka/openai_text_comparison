from utils import (csv_to_list, export_to_csv,
                   export_combined_results)
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

    descriptions = sorted(csv_to_list(descriptions_path))
    targets = csv_to_list(targets_path)
    all_results = {}
    for model in models:
        matches = find_best_matches(descriptions, targets, model=model)
        export_to_csv(model, descriptions, matches)
        all_results[model] = matches
        if print_results:
            print(f"\nUsing model {model}")
            for d, (match, sim) in zip(descriptions, matches):
                print(f"'{d}' ?= '{match}': {sim:.2f}")
            print()
    export_combined_results(models, descriptions, all_results)
