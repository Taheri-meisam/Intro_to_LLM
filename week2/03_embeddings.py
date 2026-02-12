"""
Lesson 3: Embeddings
=====================

Now we have token IDs, but a neural network needs more than
single numbers. It needs rich representations that capture meaning.

This is where embeddings come in.

Usage: python 03_embeddings.py
"""

import torch
import torch.nn as nn


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 3: Embeddings")
print("  From Token IDs to Meaningful Vectors")
print("=" * 60)


# ---------------------------------------------------------------------------
# The Problem with Raw IDs
# ---------------------------------------------------------------------------

print("""
THE PROBLEM WITH RAW IDs

After tokenization, we have token IDs:
    "The cat sat" -> [45, 892, 1203]

But what can a neural network learn from these numbers?
Not much! They're arbitrary assignments.

Token 45 ("the") and token 46 could be completely unrelated,
or they could be similar words. The numbers don't tell us.

We need a representation where similar words are... similar.
""")

pause()


# ---------------------------------------------------------------------------
# What Are Embeddings?
# ---------------------------------------------------------------------------

print("WHAT ARE EMBEDDINGS?")
print("-" * 40)

print("""
An embedding converts each token ID into a VECTOR of numbers:

    Token 45 ("the") -> [0.2, -0.5, 0.8, 0.1, ...]
    Token 46 ("a")   -> [0.3, -0.4, 0.7, 0.2, ...]

Now we can measure similarity!
Similar words end up with similar vectors.

The embedding is just a lookup table:
    - Each row corresponds to one token
    - We look up the row for each token ID
    - The model learns useful values during training
""")

pause()

# Create an embedding table
vocab_size = 1000
embed_dim = 256

embedding = nn.Embedding(vocab_size, embed_dim)

print(f"Embedding table created:")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Embedding dimension: {embed_dim}")
print(f"  Table shape: {embedding.weight.shape}")
print(f"  Total parameters: {embedding.weight.numel():,}")

pause()


# ---------------------------------------------------------------------------
# Looking Up Embeddings
# ---------------------------------------------------------------------------

print("LOOKING UP EMBEDDINGS")
print("-" * 40)

# Some token IDs
token_ids = torch.tensor([45, 892, 1203])
print(f"Token IDs: {token_ids.tolist()}")

# Look them up
vectors = embedding(token_ids)
print(f"Embedding shape: {vectors.shape}")
print(f"  (3 tokens, each with {embed_dim} dimensions)")

print()
print("First 8 values of each embedding:")
for i, tid in enumerate(token_ids):
    vals = [f"{v:.3f}" for v in vectors[i, :8].tolist()]
    print(f"  Token {tid.item()}: [{', '.join(vals)}, ...]")

pause()


# ---------------------------------------------------------------------------
# Why Embeddings Work
# ---------------------------------------------------------------------------

print("WHY EMBEDDINGS WORK")
print("-" * 40)

print("""
During training, the model adjusts embeddings so that:
    - Similar words have similar vectors
    - Useful patterns emerge in the vector space

Famous example (Word2Vec):
    king - man + woman ≈ queen

The embedding space captures semantic relationships!

In a trained model:
    - "cat" and "dog" are close (both pets)
    - "cat" and "computer" are far apart
    - "run" and "running" are related

We start with random embeddings and let the model learn.
""")

pause()


# ---------------------------------------------------------------------------
# The Position Problem
# ---------------------------------------------------------------------------

print("THE POSITION PROBLEM")
print("-" * 40)

print("""
There's a catch. Consider these sentences:
    "The dog bit the man"
    "The man bit the dog"

Same words, VERY different meanings!
Position matters, but our embeddings don't know about position.

Let's verify - the same token gets the same embedding everywhere:
""")

pause()

# Same token at different positions
token_ids = torch.tensor([45, 892, 100, 45, 77])  # Token 45 appears twice

vectors = embedding(token_ids)

print(f"Token IDs: {token_ids.tolist()}")
print(f"Token 45 appears at position 0 and position 3")
print()
print(f"Are their embeddings identical?")
print(f"  {torch.equal(vectors[0], vectors[3])}")
print()
print("We need a way to encode position!")

pause()


# ---------------------------------------------------------------------------
# Positional Embeddings
# ---------------------------------------------------------------------------

print("POSITIONAL EMBEDDINGS")
print("-" * 40)

print("""
Solution: Add another embedding table for POSITIONS!

    Position 0 -> [0.1, 0.2, -0.3, ...]
    Position 1 -> [0.2, -0.1, 0.4, ...]
    ...

Then combine them:
    final = token_embedding + position_embedding

Now the same word at different positions has different representations!
""")

pause()

# Create both embedding tables
vocab_size = 1000
max_length = 512
embed_dim = 256

token_embedding = nn.Embedding(vocab_size, embed_dim)
position_embedding = nn.Embedding(max_length, embed_dim)

print(f"Token embedding: {token_embedding.weight.shape}")
print(f"Position embedding: {position_embedding.weight.shape}")

pause()


print("COMBINING TOKEN + POSITION")
print("-" * 40)


def get_embeddings(token_ids, token_emb, pos_emb):
    """Get combined token + position embeddings."""
    seq_length = token_ids.shape[0]
    
    # Get token embeddings
    tok = token_emb(token_ids)
    
    # Create position indices: [0, 1, 2, ...]
    positions = torch.arange(seq_length)
    
    # Get position embeddings
    pos = pos_emb(positions)
    
    # Add them together
    return tok + pos


# Test it
token_ids = torch.tensor([45, 892, 100, 45, 77])
combined = get_embeddings(token_ids, token_embedding, position_embedding)

print(f"Token IDs: {token_ids.tolist()}")
print(f"Combined embedding shape: {combined.shape}")
print()
print(f"Token 45 at position 0 vs position 3:")
print(f"  Are they identical now? {torch.equal(combined[0], combined[3])}")
print()
print("Now the model can distinguish same tokens at different positions!")

pause()


# ---------------------------------------------------------------------------
# Complete Embedding Module
# ---------------------------------------------------------------------------

print("COMPLETE GPT EMBEDDING MODULE")
print("-" * 40)

print("""
Let's build a complete embedding module like GPT uses.
It combines:
    - Token embeddings
    - Position embeddings
    - Dropout (for regularization)
""")

pause()


class GPTEmbedding(nn.Module):
    """Embedding module for GPT-style models."""
    
    def __init__(self, vocab_size, embed_dim, max_length, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.max_length = max_length
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
        
        Returns:
            Tensor of shape (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        tok_emb = self.token_embedding(input_ids)
        
        # Position indices: [0, 1, 2, ..., seq_length-1]
        positions = torch.arange(seq_length, device=input_ids.device)
        # Expand for batch: (seq_length,) -> (batch_size, seq_length)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Position embeddings
        pos_emb = self.position_embedding(positions)
        
        # Combine and apply dropout
        return self.dropout(tok_emb + pos_emb)


# Create the module
embed = GPTEmbedding(
    vocab_size=50000,
    embed_dim=768,
    max_length=1024,
    dropout=0.1
)

print("GPT Embedding module:")
print(f"  Vocabulary: 50,000 tokens")
print(f"  Embedding dimension: 768")
print(f"  Max sequence length: 1024")
print(f"  Total parameters: {sum(p.numel() for p in embed.parameters()):,}")

pause()

# Test with a batch
print("Testing with a batch:")
print()

batch = torch.randint(0, 50000, (2, 10))  # 2 sequences, 10 tokens each
print(f"Input shape: {batch.shape} (batch_size=2, seq_length=10)")

embed.eval()  # Turn off dropout for testing
with torch.no_grad():
    output = embed(batch)

print(f"Output shape: {output.shape}")
print(f"  (batch_size=2, seq_length=10, embed_dim=768)")

pause()


# ---------------------------------------------------------------------------
# The Complete Pipeline
# ---------------------------------------------------------------------------

print("THE COMPLETE PIPELINE")
print("-" * 40)

print("""
Let's trace through the full text-to-embedding pipeline:

    Text: "The cat sat"
      ↓
    Tokenize: ["the", "cat", "sat"]
      ↓
    To IDs: [45, 892, 1203]
      ↓
    Token embed: [[0.2, -0.5, ...], [0.8, 0.1, ...], [0.3, 0.7, ...]]
      ↓
    + Position embed: [[0.1, 0.2, ...], [0.2, -0.1, ...], [0.3, 0.0, ...]]
      ↓
    Final: [[0.3, -0.3, ...], [1.0, 0.0, ...], [0.6, 0.7, ...]]
      ↓
    Ready for transformer!
""")

pause()

# Simulate the pipeline
print("Simulating the pipeline:")
print()

# Pretend we have a vocabulary
vocab = {"the": 4, "cat": 15, "sat": 23}

text = "The cat sat"
print(f"1. Text: '{text}'")

tokens = text.lower().split()
print(f"2. Tokens: {tokens}")

ids = [vocab[t] for t in tokens]
print(f"3. IDs: {ids}")

# Create a small embedding for demo
small_embed = GPTEmbedding(vocab_size=100, embed_dim=64, max_length=512, dropout=0.0)
small_embed.eval()

input_ids = torch.tensor([ids])
print(f"4. Input tensor shape: {input_ids.shape}")

with torch.no_grad():
    embeddings = small_embed(input_ids)
print(f"5. Embedding shape: {embeddings.shape}")
print(f"   (batch=1, seq_len=3, embed_dim=64)")
print()
print("6. Ready for the transformer!")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. EMBEDDINGS convert token IDs to rich vectors
   - Lookup table: (vocab_size, embed_dim)
   - Similar tokens get similar vectors (after training)
   - Model learns useful representations

2. POSITIONAL EMBEDDINGS add position information
   - Transformers need explicit position encoding
   - Another lookup table: (max_length, embed_dim)
   - Same token at different positions -> different vectors

3. COMBINED: final = token_emb + position_emb

4. GPT EMBEDDING MODULE wraps it all:
   - Token embeddings
   - Position embeddings
   - Dropout for regularization

5. THE PIPELINE:
   Text -> Tokens -> IDs -> Embeddings -> Transformer

Next up: Building the complete data pipeline!
""")

print("=" * 60)
print("  End of Lesson 3")
print("=" * 60)
