import matplotlib.pyplot as plt

def plot_embeddings(embeddings):
    # Plot using first 2 dimensions
    for word, vector in embeddings.items():
        x = vector[0]
        y = vector[1]

        plt.scatter(x, y)
        plt.annotate(word, (x, y))

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Word Embeddings")
    plt.grid(True)

    plt.show()
