import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class TinyTransformer:
    def __init__(self, d_model=8):
        self.d_model = d_model
        # 1. Attention Weights (Query, Key, Value)
        # In a real transformer, these are learned. Here we initialize them.
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

    def self_attention(self, x):
        # x shape: (seq_len, d_model)
        
        # Linear projections
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # 2. Scaled Dot-Product Attention
        # Score = (Q * K^T) / sqrt(d_k)
        d_k = Q.shape[-1]
        scores = (Q @ K.T) / np.sqrt(d_k)
        
        # 3. Attention Weights (Softmax)
        weights = softmax(scores)
        
        # 4. Final Output
        output = weights @ V
        return output, weights

def demonstrate_transformer():
    import json
    # Load our SVD embeddings to use as input to the transformer
    with open("learned_vectors.json", "r") as f:
        vectors = json.load(f)
    
    # Let's take a sample sentence: "the cat sits"
    sentence = ["the", "cat", "sits"]
    x = np.array([vectors[w] for w in sentence])
    
    # Initialize our Tiny Transformer
    model = TinyTransformer(d_model=8)
    
    # Run the attention mechanism
    context_vectors, attn_weights = model.self_attention(x)
    
    print("--- Tiny Transformer Results ---")
    print(f"Input Sentence: {sentence}")
    print("\nAttention Weights (How much each word looks at others):")
    for i, w1 in enumerate(sentence):
        row = [f"{w2}: {attn_weights[i, j]:.2f}" for j, w2 in enumerate(sentence)]
        print(f"{w1} looks at -> {' | '.join(row)}")
    
    print("\nOriginal 'cat' vector (first 3 dims):", [round(v, 3) for v in x[1][:3]])
    print("Contextual 'cat' vector (first 3 dims):", [round(v, 3) for v in context_vectors[1][:3]])
    print("\nNotice: The 'cat' vector has been UPDATED based on its neighbors!")

if __name__ == "__main__":
    demonstrate_transformer()
