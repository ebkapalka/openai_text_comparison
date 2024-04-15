from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np
import os

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)


def get_embeddings(text_list, model: str, batch_size: int, print_every: int) -> list:
    """
    Get embeddings for a list of texts using the specified model in batches.
    :param text_list: list of texts to get embeddings for
    :param model: model to use for embeddings
    :param batch_size: number of texts to process at once
    :param print_every: print progress every n texts
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

        # Define more clear variables for progress tracking
        if print_every > 1:
            current_count = i + len(batch)  # Current position in the text list
            is_progress_step = (current_count % print_every == 0)  # Check if it's time to print progress
            is_final_batch = current_count >= total  # Check if this is the final batch

            if is_progress_step or is_final_batch:
                print(f"   Processed {current_count} of {total} texts")

    return embeddings


def find_best_matches(descriptions: list[str], targets: list[str], model="text-embedding-3-large",
                      batch_size=50, print_every=100) -> list:
    """
    Find the best matches for a list of descriptions from a list of targets using the specified model.
    :param descriptions: descriptions to match
    :param targets: targets to match against
    :param model: model to use for embeddings
    :param batch_size: size of the batches to process
    :param print_every: print progress every n texts
    :return: list of best matches
    """
    print("Getting embeddings for descriptions...")
    desc_embeddings = get_embeddings(descriptions,
                                     model=model,
                                     batch_size=batch_size,
                                     print_every=print_every)
    print("Getting embeddings for targets...")
    target_embeddings = get_embeddings(targets,
                                       model=model,
                                       batch_size=batch_size,
                                       print_every=print_every)
    desc_embeddings = np.array(desc_embeddings)
    target_embeddings = np.array(target_embeddings)

    print("Calculating similarities...")
    similarities = cosine_similarity(desc_embeddings, target_embeddings)
    best_matches = []
    for index, similarity_vector in enumerate(similarities):
        best_match_idx = np.argmax(similarity_vector)
        best_matches.append((targets[best_match_idx], similarity_vector[best_match_idx]))
        if print_every > 1 and index % print_every == 0:
            print(f"   Processed {index} similarities")

    if print_every > 1 and len(descriptions) % print_every != 0:
        print(f"   Processed {len(descriptions)} similarities")
    return best_matches
