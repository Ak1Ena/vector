import numpy as np
import json

def train():
    # 1. Load the text
    with open("corpus.txt", "r") as f:
        lines = f.readlines()

    # 2. Build vocabulary
    words = []
    for line in lines:
        words.extend(line.strip().split())
    vocab = sorted(list(set(words)))
    word_to_id = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    # 3. Create Co-occurrence Matrix
    co_matrix = np.zeros((vocab_size, vocab_size))
    
    for line in lines:
        sentence_words = line.strip().split()
        for i, word1 in enumerate(sentence_words):
            for j, word2 in enumerate(sentence_words):
                if i != j:
                    id1 = word_to_id[word1]
                    id2 = word_to_id[word2]
                    co_matrix[id1, id2] += 1

    # 4. Dimensionality Reduction (Simplified SVD)
    U, S, Vh = np.linalg.svd(co_matrix)
    
    # We take the first 8 columns of U as our embeddings
    embeddings_raw = U[:, :8]
    
    # Normalize
    embeddings_raw = embeddings_raw / np.abs(embeddings_raw).max()

    # 5. Save results
    learned_vectors = {}
    for i, word in enumerate(vocab):
        learned_vectors[word] = embeddings_raw[i].tolist()

    with open("learned_vectors.json", "w") as f:
        json.dump(learned_vectors, f)

    print(f"Training complete! Learned {vocab_size} words.")
    print("Vectors saved to learned_vectors.json")

if __name__ == "__main__":
    train()
