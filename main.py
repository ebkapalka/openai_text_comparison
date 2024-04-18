from matching_tools.fuzzy_match import fuzzy_match
from matching_tools.gpt_match import gpt_match
from utils import csv_to_list

from pprint import pprint


if __name__ == '__main__':
    descriptions_path = "input/cappex_majors.csv"
    targets_path = "input/banner_majors_2.csv"
    descriptions = sorted(csv_to_list(descriptions_path))
    targets = sorted(csv_to_list(targets_path))

    # Fuzzy matching
    fuzzy_matches = fuzzy_match(descriptions, targets)
    gpt_matches = gpt_match(descriptions, targets)
