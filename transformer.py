import numpy as np
import json

def softmax(x):
    # Numerical stability: subtract max
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.encoding = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = np.sin(position * div_term)
        self.encoding[:, 1::2] = np.cos(position * div_term)

    def add(self, x):
        # x shape: (seq_len, d_model)
        seq_len = x.shape[0]
        return x + self.encoding[:seq_len, :]

class TransformerBlock:
    def __init__(self, d_model=8, d_ff=32):
        self.d_model = d_model

        # 1. Self-Attention Weights (Identity initialization)
        # Using Identity matrix ensures input passes through without random noise initially
        self.W_q = np.eye(d_model)
        self.W_k = np.eye(d_model)
        self.W_v = np.eye(d_model)
        self.W_o = np.eye(d_model)

        # 2. Feed-Forward Network Weights (Identity initialization)
        self.W1 = np.eye(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.eye(d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def self_attention(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        d_k = Q.shape[-1]
        scores = (Q @ K.T) / np.sqrt(d_k)
        weights = softmax(scores)
        
        attention_output = weights @ V
        return attention_output @ self.W_o, weights

    def feed_forward(self, x):
        # Simple two-layer neural network
        return relu(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def forward(self, x):
        # 1. Attention + Residual Connection
        attn_out, weights = self.self_attention(x)
        x = x + attn_out # Skip connection
        
        # 2. Feed Forward + Residual Connection
        ff_out = self.feed_forward(x)
        x = x + ff_out # Skip connection
        
        return x, weights

def run_transformer_logic():
    # Load learned vectors
    try:
        with open("learned_vectors.json", "r") as f:
            vectors = json.load(f)
    except FileNotFoundError:
        print("Please run train.py first to generate vectors!")
        return

    # Define a sequence
    sentence = ["the", "cat", "sits", "on", "the", "mat"]
    x = np.array([vectors[w] for w in sentence])
    
    # 1. Add Positional Information
    pe = PositionalEncoding(d_model=8)
    x_pos = pe.add(x)
    
    # 2. Process through Transformer Block
    model = TransformerBlock(d_model=8)
    context_vectors, attn_weights = model.forward(x_pos)
    
    print(f"--- Complete Transformer Pipeline ---")
    print(f"Sequence: {' '.join(sentence)}")
    print(f"\nStep 1: Added Positional Encoding (Awareness of word order)")
    print(f"Step 2: Processed through Self-Attention (Context understanding)")
    print(f"Step 3: Processed through Feed-Forward Network (Feature refinement)")
    
    print("\nAttention Matrix Snippet (First 3 words):")
    print(attn_weights[:3, :3])
    
    print(f"\nResult: Word 'cat' has been transformed into a contextual vector.")
    print(f"Original: {np.round(x[1][:3], 3)}")
    print(f"Final:    {np.round(context_vectors[1][:3], 3)}")

if __name__ == "__main__":
    run_transformer_logic()
