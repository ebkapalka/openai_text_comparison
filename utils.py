from datetime import datetime
import csv


def csv_to_list(path: str, encoding='utf-8') -> list:
    """
    Load a CSV file into a list.
    Assumes each row contains only one item of interest.
    :param path: path to the CSV file
    :param encoding: encoding of the CSV file
    :return: list containing the first item from each row
    """
    with open(path, mode='r', newline='', encoding=encoding) as csvfile:
        reader = csv.reader(csvfile)
        return [row[0] for row in reader if row]


def export_to_csv(model, descriptions, matches, folder='output'):
    """
    Export the results to a CSV file
    :param model: model used in matching
    :param descriptions: descriptions used in matching
    :param matches: matches found
    :param folder: folder to save the file
    :return: None
    """
    folder = folder.rstrip('/')
    folder = folder.rstrip('\\')
    filename = f"{folder}/{model}_{timestamp()}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Major', 'Closest Match', 'Similarity'])  # Header
        for desc, (match, sim) in zip(descriptions, matches):
            writer.writerow([desc, match, f"{sim:.2f}"])  # Write each row
    print(f"Results exported to {filename}")


def export_combined_results(filenames: list[str], descriptions, all_results, folder='output'):
    """
    Export combined results to a CSV file with each model's results in separate columns.
    :param filenames: the method or model names used for matching
    :param descriptions: descriptions (incoming data) used in matching
    :param all_results: dictionary of matches from each model
    :param folder: output directory
    """
    folder = folder.rstrip('/').rstrip('\\')
    filename = f"{folder}/combined_results_{timestamp()}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Prepare header with accurate names
        headers = ['Major'] + [f'{name} Match' for name in filenames]
        writer.writerow(headers)

        # Iterate over each description and write rows for each
        for i, desc in enumerate(descriptions):
            row = [desc]
            for name in filenames:
                match, sim = all_results[name][i]  # Retrieve match and similarity for this column
                row.append(f"{match} ({sim:.2f})")  # Append result in format "Match (Similarity)"
            writer.writerow(row)

    print(f"Combined results exported to {filename}")


def timestamp(dt_obj: datetime = None, fmt="%Y%m%d_%H%M%S") -> str:
    """
    Get the current timestamp in the desired format
    :param dt_obj: datetime object to use
    :param fmt: format of the timestamp
    :return: timestamp string
    """
    if dt_obj is None:
        dt_obj = datetime.now()
    return dt_obj.strftime(fmt)
