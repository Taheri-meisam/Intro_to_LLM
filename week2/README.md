# Week 2: Working with Text

Welcome to Week 2! This week we learn how to turn text into numbers that neural networks can process.

## What You'll Learn

- **Tokenization**: Breaking text into pieces the model can understand
- **BPE Algorithm**: How GPT actually tokenizes text
- **Embeddings**: Representing tokens as meaningful vectors
- **Data Pipelines**: Preparing text for training

## Lessons

Run each lesson in order:

```bash
python 01_tokenization.py   # ~30 min - Why and how we tokenize
python 02_bpe.py            # ~45 min - Byte Pair Encoding algorithm
python 03_embeddings.py     # ~30 min - From IDs to vectors
python 04_data_pipeline.py  # ~45 min - Complete pipeline
```

## This Week's Project

Build a text preprocessing pipeline for Shakespeare:

```bash
python project_shakespeare.py
```

You'll create a complete pipeline that prepares text for a language model, which we'll use in Week 4 when we train GPT.

## Key Concepts

By the end of this week, you should understand:

1. Why we can't just use words directly (vocabulary explosion, OOV)
2. How subword tokenization solves these problems
3. The BPE algorithm that GPT uses
4. How embeddings capture meaning in vectors
5. The complete flow: Text → Tokens → IDs → Embeddings

## Quick Reference

```python
# Tokenization
tokens = text.split()  # Simple but limited
tokens = bpe_tokenize(text)  # What GPT uses

# Token to ID
token_to_id = {"hello": 0, "world": 1, ...}
ids = [token_to_id[t] for t in tokens]

# Embeddings
embedding = nn.Embedding(vocab_size, embed_dim)
vectors = embedding(torch.tensor(ids))

# Position embeddings
pos_embedding = nn.Embedding(max_length, embed_dim)
positions = torch.arange(len(ids))
pos_vectors = pos_embedding(positions)

# Combined
final = vectors + pos_vectors
```

## The Pipeline

```
"The cat sat" (raw text)
      ↓
["The", "cat", "sat"] (tokens)
      ↓
[45, 892, 1203] (token IDs)
      ↓
[[0.2, -0.5, ...], [0.8, 0.1, ...], ...] (embeddings)
      ↓
Ready for the transformer!
```

## Next Week Preview

In Week 3, we'll learn about attention mechanisms - the key innovation that makes transformers so powerful. You'll understand how the model decides which words to "pay attention to" when making predictions.
