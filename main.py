from utils import csv_to_list, export_to_csv
from gpt_match import find_best_matches


if __name__ == '__main__':
    models = [
        "text-embedding-3-small",  # 62_500 pages per 1USD, 62.3% performance
        "text-embedding-3-large",  # 9,615 pages per 1USD, 64.6% performance
        "text-embedding-ada-002"   # 12,500 pages per 1USD, 61.0%% performance
    ]
    descriptions = csv_to_list("input/banner_majors.csv")
    targets = csv_to_list("input/cappex_majors.csv")

    for model in models:
        print(f"Using model {model}")
        matches = find_best_matches(descriptions, targets, model=model)
        for d, (match, sim) in zip(descriptions, matches):
            print(f"Best match for '{d}' is '{match}' with similarity {sim:.2f}")
        export_to_csv(model, descriptions, matches)
        print('\n')

"""
Models come from https://platform.openai.com/docs/guides/embeddings/embedding-models
"""
