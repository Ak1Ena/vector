import numpy as np
import json
import os
from train import train
from tokenizer import tokenizer
from transformer import TransformerBlock, PositionalEncoding, softmax
from plot import plot_embeddings

class SimpleLLMPipeline:
    """A unified pipeline representing the life cycle of an LLM."""
    
    def __init__(self, d_model=8, vocab_size=14):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pe = PositionalEncoding(d_model=d_model)
        self.transformer = TransformerBlock(d_model=d_model)
        self.static_brain = {}
        # New: The Prediction Head (Linear layer)
        self.W_pred = np.random.randn(d_model, vocab_size) * 0.1

    def stage_1_pretrain(self, force=False):
        """LEARNING: Reading corpus.txt and creating the static brain (SVD)."""
        print("\n[Stage 1: Pre-training] Building static knowledge from corpus.txt...")
        if force or not os.path.exists("learned_vectors.json"):
            train()
        
        with open("learned_vectors.json", "r") as f:
            self.static_brain = json.load(f)
        self.vocab = sorted(self.static_brain.keys())
        print(f"✅ Brain loaded with {len(self.static_brain)} unique word embeddings.")

    def stage_2_encode(self, sentence_list):
        """INPUT: Tokenizing words and adding positional information."""
        print(f"\n[Stage 2: Encoding] Processing input: '{' '.join(sentence_list)}'")
        
        # 1. Lookup static vectors (Dictionary meaning)
        x = np.array([self.static_brain.get(w, [0]*self.d_model) for w in sentence_list])
        
        # 2. Add word-order awareness (Positional Encoding)
        x_with_pos = self.pe.add(x)
        return x_with_pos

    def stage_3_transform(self, encoded_input):
        """THINKING: Using the Transformer to understand context."""
        print("[Stage 3: Transformation] Applying Self-Attention for context...")
        context_vectors, weights = self.transformer.forward(encoded_input)
        return context_vectors, weights

    def stage_5_train(self, encoded_context, target_word, learning_rate=0.1):
        """LEARNING: Training the head to predict the next word."""
        print(f"\n[Stage 5: Prediction Training] Learning to predict target word: '{target_word}'...")
        
        target_word_id = self.vocab.index(target_word)
        
        # Last word vector
        last_vec = encoded_context[-1]
        
        # Forward pass through head
        logits = last_vec @ self.W_pred
        probs = softmax(logits)
        
        # Backprop (Simple Gradient Descent)
        target = np.zeros(self.vocab_size)
        target[target_word_id] = 1
        
        error = probs - target
        self.W_pred -= learning_rate * np.outer(last_vec, error)
        print("✅ Head trained on this example.")

    def stage_5_predict(self, context_vectors):
        """SPEECH: Predicting the most likely next word."""
        print("\n[Stage 5: Prediction] Deciphering the next word...")
        
        last_word_vector = context_vectors[-1]
        
        # Pass through the prediction head
        logits = last_word_vector @ self.W_pred
        probs = softmax(logits)
        
        predicted_id = np.argmax(probs)
        predicted_word = self.vocab[predicted_id]
        
        print(f"🔮 Prediction: '{predicted_word}' (Prob: {probs[predicted_id]:.2f})")
        return predicted_word

    def stage_4_visualize(self, final_results):
        """OUTPUT: Plotting the final contextual understanding."""
        print("[Stage 4: Visualization] Generating semantic relationship map...")
        plot_embeddings(final_results)

def run_llm_pipeline():
    print("🤖 Starting Simple-LLM Pipeline 🤖")
    llm = SimpleLLMPipeline(d_model=8, vocab_size=14)

    # 1. PRE-TRAIN (Build the static knowledge)
    llm.stage_1_pretrain()

    # 2. TRAIN THE PREDICTION HEAD (Make it smart)
    # Give the model an example: "the cat" -> "sits"
    prompt = ["the", "cat"]
    target = "sits"
    
    encoded = llm.stage_2_encode(prompt)
    transformed, _ = llm.stage_3_transform(encoded)
    
    # Run training loop a few times so it actually learns
    for _ in range(50):
        llm.stage_5_train(transformed, target)

    # 3. GENERATE (Predicting the next word)
    next_word = llm.stage_5_predict(transformed)

    print(f"\n--- Full Generation Result ---")
    print(f"Input:  {' '.join(prompt)}")
    print(f"Output: {' '.join(prompt)} {next_word}")

    # 4. VISUALIZE
    final_knowledge = {
        "cat (static)": llm.static_brain["cat"],
        "cat (context)": transformed[1].tolist(),
    }
    llm.stage_4_visualize(final_knowledge)
    
    print("\n✨ LLM Pipeline Execution Finished! ✨")

if __name__ == "__main__":
    run_llm_pipeline()
