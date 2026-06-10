# Word Embedding Learning Project

This project is a "from scratch" educational prototype for learning word embeddings using co-occurrence and SVD.

## Project Structure
- `train.py`: The trainer script. It reads `corpus.txt`, performs SVD-based dimensionality reduction, and saves `learned_vectors.json`.
- `tokenizer.py`: Contains the `tokenizer` function. It loads learned vectors from `learned_vectors.json`.
- `plot.py`: Handles the visualization logic using Matplotlib.
- `knowledge.html`: The primary knowledge base explaining the mathematical theory (Thai).
- `knowledge_index.md`: A concise English summary of the knowledge base.

## Building and Running

### Training
To learn vectors from your corpus:
```bash
python3 train.py
```

### Visualization
To generate the embedding plot:
```bash
python3 main.py
```

## Development Conventions
- **Knowledge Base:** Maintain `knowledge.html` as the primary reference for the mathematical evolution of the project. Always update it when new concepts are added.
- **Auto-Update:** After making changes to the learning logic or data, ensure `knowledge.html` is updated to reflect the new state.
