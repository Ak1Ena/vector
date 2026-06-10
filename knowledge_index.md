# Knowledge Index

This file provides a quick summary of the detailed mathematical explanations found in `knowledge.html`.

## 1. Data Collection (Corpus)
- **Concept:** Converting raw sentences into word pairs.
- **Example:** "cat sits mat" → `(cat, sits)`, `(cat, mat)`.

## 2. Co-occurrence Matrix
- **Concept:** A table counting how often words appear together.
- **Math:** $C_{i,j} = \sum (\text{occurrences of } w_i, w_j)$.
- **Key Insight:** Related words (like cat/dog) will have similar rows in this matrix.

## 3. Dimensionality Reduction (SVD)
- **Concept:** Compressing the giant matrix into small 8D vectors.
- **Math:** $A = U \Sigma V^T$.
- **Analogy:** Like the 2D "shadow" of a 3D object.

## 4. Execution Pipeline
1. **Train:** `train.py` (Calculates SVD → `learned_vectors.json`).
2. **Tokenize:** `tokenizer.py` (Reads learned vectors).
3. **Plot:** `main.py` (Visualizes relationships).

---
*For the full Thai documentation with detailed examples and formulas, see `knowledge.html`.*
