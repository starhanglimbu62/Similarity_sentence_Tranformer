# Sentence Similarity with Vector Embeddings

Compute semantic similarity between sentences using transformer-based vector embeddings and cosine similarity.

---

## What This Does

This project encodes sentences into high-dimensional vectors using a pre-trained transformer model, then measures how semantically close two sentences are using cosine similarity. The score ranges from `0` (completely unrelated) to `1` (identical meaning).

**Example:**
```
"He is an excellent software developer"
"He is good at coding"
Similarity Score: ~0.85
```

Same meaning. Different words. The model captures that.

---

## How It Works

| Step | What Happens |
|------|-------------|
| Load model | `all-MiniLM-L6-v2` encodes text into 384-dimensional vectors |
| Encode sentences | Each sentence becomes a dense float vector |
| Cosine similarity | Measures the angle between vectors via `util.cos_sim` |
| Similarity matrix | `model.similarity()` returns a full pairwise matrix |

---

## Project Structure

```
sentence-similarity/
├── similarity.py       # Main script
└── README.md
```

---

## Setup

**Requirements**

- Python 3.8+
- pip

**Install dependencies**

```bash
pip install sentence-transformers
```

---

## Usage

```bash
python similarity.py
```

**Expected Output**

```
Sentence embeddings:
(2, 384)
The similarity between the two sentences is: tensor([[0.8481]])
tensor([[1.0000, 0.8481],
        [0.8481, 1.0000]])
```

The diagonal values are always `1.0` (a sentence is identical to itself). The off-diagonal value is the semantic similarity between the two input sentences.

---

## Modify the Sentences

Open `similarity.py` and edit this line:

```python
sentences = ['Your first sentence here', 'Your second sentence here']
```

---

## Model

**`all-MiniLM-L6-v2`** from [sentence-transformers](https://www.sbert.net/)

- Lightweight and fast
- 384-dimensional embeddings
- Trained on 1B+ sentence pairs
- Strong performance on semantic similarity benchmarks

---

## Real-World Applications

- **Semantic Search** - Match queries to documents by meaning, not keywords
- **Recommendation Systems** - Surface similar content based on context
- **Duplicate Detection** - Flag near-identical text with different wording
- **NLP Pipelines** - Foundation layer for classification, clustering, and retrieval

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Model loading, encoding, and similarity utilities |
| `torch` | Tensor operations (installed automatically) |

---

## License

MIT
