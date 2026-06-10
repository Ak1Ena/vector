import tokenizer
from plot import plot_embeddings

# Let's use words from our corpus to see the learned relations
vocab = ["cat", "dog", "rat", "car", "truck", "road", "mat", "sits", "drives"]

plot_embeddings(tokenizer.tokenizer(vocab))