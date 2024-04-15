from openai import OpenAI
import numpy as np
import os

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)


def get_embeddings(text_list, model: str, print_every) -> list:
    """
    Get embeddings for a list of texts using the specified model.
    :param text_list: list of texts to get embeddings for
    :param model: model to use for embeddings
    :param print_every: print progress every n texts
    :return: list of embeddings
    """
    embeddings = []
    for index, text in enumerate(text_list):
        response = client.embeddings.create(
            model=model,
            input=text
        )
        embeddings.append(response.data[0].embedding)
        if index % print_every == 0:
            print(f"   Processed {index} texts")
    if len(text_list) % print_every != 0:
        print(f"   Processed {len(text_list)} texts")
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


def find_best_matches(descriptions: list[str], targets: list[str],
                      model="text-embedding-3-large", print_every=100) -> list:
    """
    Find the best matches for a list of descriptions and targets.
    :param descriptions: list of descriptions
    :param targets: list of targets
    :param model: model to use for embeddings
    :param print_every: print progress every n descriptions
    :return: best matches for each description
    """
    print("Getting embeddings for descriptions...")
    desc_embeddings = get_embeddings(descriptions,
                                     model=model,
                                     print_every=print_every)
    print("Getting embeddings for targets...")
    target_embeddings = get_embeddings(targets,
                                       model=model,
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
