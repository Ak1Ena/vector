import numpy as np
import json
import os
from tqdm import tqdm
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
        self.W_pred = None 
        self.needs_training = False # Track if training is required

    def stage_1_pretrain(self, force=False):
        """LEARNING: Reading corpus.txt and creating/loading static brain and weights."""
        print("\n[Stage 1: Pre-training] Checking for existing knowledge...")
        
        vectors_path = "learned_vectors.json"
        weights_path = "prediction_weights.npy"
        
        self.needs_training = force or not os.path.exists(vectors_path) or not os.path.exists(weights_path)
        
        if not self.needs_training and os.path.exists(vectors_path):
            if os.path.getmtime("corpus.txt") > os.path.getmtime(vectors_path):
                print("⚠️ corpus.txt was modified. Re-training...")
                self.needs_training = True

        if self.needs_training:
            train()
        
        with open(vectors_path, "r") as f:
            self.static_brain = json.load(f)
        
        self.vocab = sorted(self.static_brain.keys())
        self.vocab_size = len(self.vocab)
        
        if self.needs_training:
            self.W_pred = np.random.randn(self.d_model, self.vocab_size) * 0.1
        else:
            self.W_pred = np.load(weights_path)
        
        print(f"✅ Brain loaded with {len(self.static_brain)} unique word embeddings.")

    def save_weights(self):
        """Save the prediction head weights."""
        np.save("prediction_weights.npy", self.W_pred)
        print("💾 Weights saved.")

    def stage_2_encode(self, sentence_list):
        """INPUT: Tokenizing words and adding positional information."""
        # print(f"\n[Stage 2: Encoding] Processing input: '{' '.join(sentence_list)}'")
        
        # 1. Lookup static vectors (Dictionary meaning)
        x = np.array([self.static_brain.get(w, [0]*self.d_model) for w in sentence_list])
        
        # 2. Add word-order awareness (Positional Encoding)
        x_with_pos = self.pe.add(x)
        return x_with_pos

    def stage_3_transform(self, encoded_input):
        """THINKING: Using the Transformer to understand context."""
        # print("[Stage 3: Transformation] Applying Self-Attention for context...")
        context_vectors, weights = self.transformer.forward(encoded_input)
        return context_vectors, weights

    def stage_5_train_all(self, corpus_lines, epochs=50):
        """LEARNING: Training the head on the full corpus."""
        print(f"\n[Stage 5: Prediction Training] Training on {len(corpus_lines)} lines...")
        
        for epoch in range(epochs):
            # Using tqdm for a progress bar per epoch
            with tqdm(total=len(corpus_lines), desc=f"Epoch {epoch+1}/{epochs}", unit="line") as pbar:
                for line in corpus_lines:
                    words = line.strip().split()
                    if len(words) < 2: 
                        pbar.update(1)
                        continue
                    
                    # Train to predict next word for each word in sequence
                    for i in range(len(words) - 1):
                        prompt = words[:i+1]
                        target = words[i+1]
                        
                        if target not in self.vocab: continue
                        
                        encoded = self.stage_2_encode(prompt)
                        transformed, _ = self.stage_3_transform(encoded)
                        self.stage_5_train(transformed, target)
                    pbar.update(1)
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

    def stage_5_predict(self, context_vectors, temperature=0.7, top_k=5):
        """SPEECH: Predicting the next word using Top-K sampling."""
        last_word_vector = context_vectors[-1]
        
        # Pass through the prediction head
        logits = last_word_vector @ self.W_pred
        
        # 1. Apply temperature
        logits = logits / temperature
        
        # 2. Top-K filtering: set all but the top K logits to -infinity
        if top_k > 0:
            top_k = min(top_k, self.vocab_size)
            indices_to_remove = logits < np.sort(logits)[-top_k]
            logits[indices_to_remove] = -float('inf')
            
        probs = softmax(logits)
        
        # 3. Sample
        predicted_id = np.random.choice(len(probs), p=probs)
        return self.vocab[predicted_id]

    def stage_6_generate(self, prompt, max_length=20):
        """GENERATION: Autoregressive dialogue generation with better stopping."""
        print(f"\n[Stage 6: Generation] Creating conversational response...")
        current_seq = prompt.copy()

        for _ in range(max_length):
            # 1. Encode
            encoded = self.stage_2_encode(current_seq)

            # 2. Transform
            transformed, _ = self.stage_3_transform(encoded)

            # 3. Predict
            next_word = self.stage_5_predict(transformed, temperature=0.7)

            # 4. Stopping conditions - be more flexible
            if next_word in [".", "!", "?"]:
                current_seq.append(next_word)
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

    # 2. TRAIN THE PREDICTION HEAD (Only if new or modified)
    if llm.needs_training:
        # Load corpus for training the head
        with open("corpus.txt", "r") as f:
            corpus = f.readlines()
        
        # Train
        llm.stage_5_train_all(corpus, epochs=20) # Increased to 50 epochs
        # Save weights
        llm.save_weights()
    else:
        print("\n[Stage 5: Prediction] Knowledge is up-to-date. Skipping training.")

    # 3. GENERATE
    prompt = user_prompt.lower().split()
    generated_sequence = llm.stage_6_generate(prompt, max_length=10) # Increased to 10 words


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
