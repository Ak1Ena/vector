import random
import json
import os

# Load learned vectors if they exist
LEARNED_VECTORS_PATH = "learned_vectors.json"

def load_learned_vectors():
    if os.path.exists(LEARNED_VECTORS_PATH):
        with open(LEARNED_VECTORS_PATH, "r") as f:
            return json.load(f)
    return {}

LEARNED_VECTORS = load_learned_vectors()

def tokenizer(vocab):
    embeddings = {}
    for word in vocab:
        # Use learned vector if available, otherwise random
        if word in LEARNED_VECTORS:
            embeddings[word] = LEARNED_VECTORS[word]
        else:
            embeddings[word] = [random.uniform(-1, 1) for _ in range(8)]
    return embeddings
