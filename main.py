import numpy as np
import json
import os
from train import train
from tokenizer import tokenizer
from transformer import TransformerStack, PositionalEncoding, softmax
from plot import plot_embeddings

class SimpleLLMPipeline:
    """A unified pipeline representing the life cycle of an LLM."""
    
    def __init__(self, d_model=8):
        self.d_model = d_model
        self.pe = PositionalEncoding(d_model=d_model)
        self.transformer = TransformerStack(d_model=d_model, num_layers=3)
        self.static_brain = {}
        self.vocab = []
        self.W_pred = None # Initialized after loading brain

    def stage_1_pretrain(self, force=False):
        """LEARNING: Reading corpus.txt and creating the static brain (SVD)."""
        print("\n[Stage 1: Pre-training] Building static knowledge from corpus.txt...")
        if force or not os.path.exists("learned_vectors.json"):
            train()
        
        with open("learned_vectors.json", "r") as f:
            self.static_brain = json.load(f)
        
        self.vocab = sorted(self.static_brain.keys())
        self.vocab_size = len(self.vocab)
        # Initialize W_pred now that we know vocab size
        self.W_pred = np.random.randn(self.d_model, self.vocab_size) * 0.1
        
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

    def stage_5_train_all(self, corpus_lines, epochs=100):
        """LEARNING: Training the head on the full corpus."""
        print(f"\n[Stage 5: Prediction Training] Training on {len(corpus_lines)} lines...")
        for epoch in range(epochs):
            for line in corpus_lines:
                words = line.strip().split()
                if len(words) < 2: continue
                
                # Train to predict next word for each word in sequence
                for i in range(len(words) - 1):
                    prompt = words[:i+1]
                    target = words[i+1]
                    
                    if target not in self.vocab: continue
                    
                    encoded = self.stage_2_encode(prompt)
                    transformed, _ = self.stage_3_transform(encoded)
                    self.stage_5_train(transformed, target)
        print("✅ Prediction head trained successfully.")

    def stage_5_train(self, encoded_context, target_word, learning_rate=0.1):
        """LEARNING: Training the head to predict the next word."""
        target_word_id = self.vocab.index(target_word)
        
        # Last word vector (the context of the prompt so far)
        last_vec = encoded_context[-1]
        
        # Forward pass through head
        logits = last_vec @ self.W_pred
        probs = softmax(logits)
        
        # Backprop (Simple Gradient Descent)
        target = np.zeros(self.vocab_size)
        target[target_word_id] = 1
        
        error = probs - target
        self.W_pred -= learning_rate * np.outer(last_vec, error)

    def stage_5_predict(self, context_vectors):
        """SPEECH: Predicting the most likely next word."""
        last_word_vector = context_vectors[-1]
        
        # Pass through the prediction head
        logits = last_word_vector @ self.W_pred
        probs = softmax(logits)
        
        predicted_id = np.argmax(probs)
        return self.vocab[predicted_id]

    def stage_6_generate(self, prompt, max_length=10):
        """GENERATION: A loop that predicts words until a stop condition."""
        print(f"\n[Stage 6: Generation] Generating (Max {max_length} words)...")
        current_seq = prompt.copy()
        
        for _ in range(max_length):
            encoded = self.stage_2_encode(current_seq)
            transformed, _ = self.stage_3_transform(encoded)
            next_word = self.stage_5_predict(transformed)
            
            # Stop if the model predicts a period or repeats the last word
            if next_word == "." or next_word == current_seq[-1]:
                print(f"🛑 Generation stopped: {'Period detected' if next_word == '.' else 'Repetition detected'}.")
                break
                
            print(f"🔮 Predicted: '{next_word}'")
            current_seq.append(next_word)
        
        return current_seq

    def stage_4_visualize(self, final_results):
        """OUTPUT: Plotting the final contextual understanding."""
        print("[Stage 4: Visualization] Generating semantic relationship map...")
        plot_embeddings(final_results)

def run_llm_pipeline(user_prompt="the cat"):
    print("🤖 Starting Simple-LLM Pipeline 🤖")
    llm = SimpleLLMPipeline(d_model=8)

    # 1. PRE-TRAIN
    llm.stage_1_pretrain()

    # Load corpus for training the head
    with open("corpus.txt", "r") as f:
        corpus = f.readlines()

    # 2. TRAIN THE PREDICTION HEAD
    llm.stage_5_train_all(corpus, epochs=5) # 5 epochs is enough for this tiny data

    # 3. GENERATE
    prompt = user_prompt.lower().split()
    generated_sequence = llm.stage_6_generate(prompt, length=3)


    print(f"\n--- Full Generation Result ---")
    print(f"Input:    {' '.join(prompt)}")
    print(f"Output:   {' '.join(generated_sequence)}")

    # 4. VISUALIZE (Plot only the prompt words + predicted words)
    final_knowledge = {w: llm.static_brain.get(w, [0]*8) for w in generated_sequence if w in llm.static_brain}
    # llm.stage_4_visualize(final_knowledge)
    
    print("\n✨ LLM Pipeline Execution Finished! ✨")

if __name__ == "__main__":
    # You can change the prompt here or take input from terminal
    user_input = input("Enter your prompt: ") or "i love playing"
    run_llm_pipeline(user_prompt=user_input)
