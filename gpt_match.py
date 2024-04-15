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
        current_count = i + len(batch)  # Current position in the text list
        is_progress_step = (current_count % print_every == 0)  # Check if it's time to print progress
        is_final_batch = current_count >= total  # Check if this is the final batch

        if is_progress_step or is_final_batch:
            print(f"   Processed {current_count} of {total} texts")

    return embeddings



def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    :param vec1: vector 1
    :param vec2: vector 2
    :return: dot product of the two vectors divided by the product of their magnitudes
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_best_matches(descriptions: list[str], targets: list[str], model="text-embedding-3-large",
                      batch_size=50, print_every=100) -> list:
    """
    Find the best matches for a list of descriptions and targets.
    :param descriptions: list of descriptions
    :param targets: list of targets
    :param model: model to use for embeddings
    :param batch_size: number of descriptions to process at once
    :param print_every: print progress every n descriptions
    :return: best matches for each description
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

    best_matches = []
    print("Calculating similarities...")
    for index, desc_emb in enumerate(desc_embeddings):
        similarities = [cosine_similarity(desc_emb, targ_emb) for targ_emb in target_embeddings]
        best_match_idx = np.argmax(similarities)
        best_matches.append((targets[best_match_idx], max(similarities)))
        if index % print_every == 0:
            print(f"   Processed {index} similarities")
    if len(descriptions) % print_every != 0:
        print(f"   Processed {len(descriptions)} similarities")
    return best_matches
