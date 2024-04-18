from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np
import os

from utils import export_combined_results

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)


def get_embeddings(text_list, model: str, batch_size: int) -> list[list[float]]:
    """
    Get embeddings for a list of texts using the specified model in batches.
    :param text_list: list of texts to get embeddings for
    :param model: model to use for embeddings
    :param batch_size: number of texts to process at once
    :return: list of embeddings
    """
    embeddings = []
    total = len(text_list)
    for i in range(0, total, batch_size):
        batch = text_list[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings


def find_best_matches(descriptions: list[str], targets: list[str],
                      model: str, batch_size: int) -> list[tuple[str, float]]:
    """
    Find the best matches for a list of descriptions from a list of targets using the specified model.
    :param descriptions: descriptions (incoming data) to match
    :param targets: targets (existing data) to match against
    :param model: model to use for embeddings
    :param batch_size: size of the batches to process
    :return: list of best matches
    """
    # Get embeddings for descriptions and targets
    desc_embeddings = get_embeddings(descriptions, model=model, batch_size=batch_size)
    target_embeddings = get_embeddings(targets, model=model, batch_size=batch_size)
    desc_embeddings = np.array(desc_embeddings)
    target_embeddings = np.array(target_embeddings)

    # Calculate cosine similarities between descriptions and targets
    similarities = cosine_similarity(desc_embeddings, target_embeddings)

    # Find the best match for each description
    best_matches = []
    for index, similarity_vector in enumerate(similarities):
        best_match_idx = np.argmax(similarity_vector)
        best_matches.append((targets[best_match_idx], similarity_vector[best_match_idx]))

    return best_matches


def gpt_match(descriptions: list[str], targets: list[str], batch_size=50):
    """
    Match descriptions to targets using OpenAI embedding filenames
    :param descriptions: descriptions (incoming data) to match
    :param targets: targets (existing data) to match against
    :param batch_size: batch size for processing
    :return: None
    """
    models = [
        "text-embedding-3-small",  # 62_500 pages per 1USD, 62.3% MTEB benchmark
        "text-embedding-ada-002",  # 12_500 pages per 1USD, 61.0%% MTEB benchmark
        "text-embedding-3-large",  # 9_615 pages per 1USD, 64.6% MTEB benchmark
    ]

    all_results = {}
    for model in models:
        print(f"Matching using {model}...")
        matches = find_best_matches(descriptions,
                                    targets,
                                    model=model,
                                    batch_size=batch_size)
        # export_to_csv(model, descriptions, matches)
        all_results[model] = matches
    export_combined_results(models,
                            descriptions,
                            all_results)
    return all_results
