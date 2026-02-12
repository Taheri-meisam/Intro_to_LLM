"""
Lesson 4: Building a Complete Data Pipeline
============================================

Time to put everything together! We'll build a complete pipeline
that takes raw text and prepares it for training a language model.

This is exactly what happens before training GPT.

Usage: python 04_data_pipeline.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 4: Complete Data Pipeline")
print("  From Raw Text to Training Batches")
print("=" * 60)


print("""
THE COMPLETE PIPELINE

    Raw Text
        ↓
    Tokenizer (text -> token IDs)
        ↓
    Dataset (creates input/target pairs)
        ↓
    DataLoader (batches for training)
        ↓
    Embedding Module (IDs -> vectors)
        ↓
    Ready for Transformer!

Let's build each component.
""")

pause()


# ---------------------------------------------------------------------------
# Component 1: Tokenizer
# ---------------------------------------------------------------------------

print("COMPONENT 1: THE TOKENIZER")
print("-" * 40)


class Tokenizer:
    """Simple word-level tokenizer with special tokens."""
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        
        for token in [self.pad_token, self.unk_token]:
            self._add_token(token)
    
    def _add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def _preprocess(self, text):
        text = text.lower()
        text = re.sub(r'([.,!?;:\'\"])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def train(self, texts):
        """Build vocabulary from texts."""
        for text in texts:
            for token in self._preprocess(text).split():
                self._add_token(token)
        return self
    
    def encode(self, text):
        """Convert text to token IDs."""
        tokens = self._preprocess(text).split()
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id[self.unk_token])
        return ids
    
    def decode(self, ids):
        """Convert token IDs back to text."""
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, self.unk_token)
            if token not in [self.pad_token, self.unk_token]:
                tokens.append(token)
        text = ' '.join(tokens)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text
    
    @property
    def vocab_size(self):
        return len(self.token_to_id)
    
    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]


# Train the tokenizer
corpus = [
    "The cat sat on the mat.",
    "The dog ran in the park.",
    "I love machine learning!",
    "Language models are fascinating.",
    "The quick brown fox jumps over the lazy dog.",
]

tokenizer = Tokenizer()
tokenizer.train(corpus)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print()

# Test it
text = "The cat sat on the mat."
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print(f"Text: '{text}'")
print(f"Encoded: {encoded}")
print(f"Decoded: '{decoded}'")

pause()


# ---------------------------------------------------------------------------
# Component 2: Dataset
# ---------------------------------------------------------------------------

print("COMPONENT 2: THE DATASET")
print("-" * 40)

print("""
For language modeling, we create input/target pairs:
    Input:  [The, cat, sat, on, the]
    Target: [cat, sat, on, the, mat]

The model learns to predict the next token at each position.
""")

pause()


class LanguageModelDataset(Dataset):
    """Dataset for language model training."""
    
    def __init__(self, texts, tokenizer, seq_length=32):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize all texts and concatenate
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(all_ids) - seq_length, seq_length):
            seq = all_ids[i:i + seq_length + 1]  # +1 for target
            if len(seq) == seq_length + 1:
                self.sequences.append(seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Input: all tokens except last
        input_ids = torch.tensor(seq[:-1])
        
        # Target: all tokens except first (shifted by 1)
        target_ids = torch.tensor(seq[1:])
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }


# Create dataset
dataset = LanguageModelDataset(corpus, tokenizer, seq_length=8)

print(f"Number of sequences: {len(dataset)}")
print()

# Look at one example
example = dataset[0]
print("Example from dataset:")
print(f"  Input IDs:  {example['input_ids'].tolist()}")
print(f"  Target IDs: {example['target_ids'].tolist()}")
print()

# Decode to see the text
input_text = tokenizer.decode(example['input_ids'].tolist())
target_text = tokenizer.decode(example['target_ids'].tolist())
print(f"  Input text:  '{input_text}'")
print(f"  Target text: '{target_text}'")

pause()


# ---------------------------------------------------------------------------
# Component 3: Collate Function
# ---------------------------------------------------------------------------

print("COMPONENT 3: COLLATE FUNCTION")
print("-" * 40)

print("""
When batching sequences of different lengths, we need padding.
The collate function:
    1. Pads all sequences to the same length
    2. Creates attention masks (1 for real tokens, 0 for padding)
""")

pause()


def collate_fn(batch, pad_id=0):
    """Pad sequences and create attention masks."""
    
    # Find max length in batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    target_ids = []
    attention_masks = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        padding_len = max_len - seq_len
        
        # Pad input and target
        padded_input = torch.cat([
            item['input_ids'],
            torch.full((padding_len,), pad_id)
        ])
        padded_target = torch.cat([
            item['target_ids'],
            torch.full((padding_len,), -100)  # -100 is ignored in loss
        ])
        
        # Create attention mask
        mask = torch.cat([
            torch.ones(seq_len),
            torch.zeros(padding_len)
        ])
        
        input_ids.append(padded_input)
        target_ids.append(padded_target)
        attention_masks.append(mask)
    
    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids),
        'attention_mask': torch.stack(attention_masks)
    }


# Test with varying length sequences
class VarLengthDataset(Dataset):
    def __init__(self):
        self.data = [
            {'input_ids': torch.tensor([1, 2, 3]), 'target_ids': torch.tensor([2, 3, 4])},
            {'input_ids': torch.tensor([5, 6, 7, 8, 9]), 'target_ids': torch.tensor([6, 7, 8, 9, 10])},
            {'input_ids': torch.tensor([11, 12]), 'target_ids': torch.tensor([12, 13])},
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

test_dataset = VarLengthDataset()
batch = collate_fn([test_dataset[i] for i in range(3)])

print("Collated batch:")
print(f"  Input IDs shape: {batch['input_ids'].shape}")
print(f"  Target IDs shape: {batch['target_ids'].shape}")
print(f"  Attention mask shape: {batch['attention_mask'].shape}")
print()
print(f"  Input IDs:\n{batch['input_ids']}")
print()
print(f"  Attention masks:\n{batch['attention_mask']}")

pause()


# ---------------------------------------------------------------------------
# Component 4: DataLoader
# ---------------------------------------------------------------------------

print("COMPONENT 4: DATALOADER")
print("-" * 40)

# Create a DataLoader with our collate function
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer.pad_id)
)

print(f"DataLoader created:")
print(f"  Batch size: 2")
print(f"  Number of batches: {len(loader)}")
print()

# Get one batch
batch = next(iter(loader))
print(f"Sample batch:")
print(f"  Input shape: {batch['input_ids'].shape}")
print(f"  Target shape: {batch['target_ids'].shape}")
print(f"  Mask shape: {batch['attention_mask'].shape}")

pause()


# ---------------------------------------------------------------------------
# Component 5: Embedding Module
# ---------------------------------------------------------------------------

print("COMPONENT 5: EMBEDDING MODULE")
print("-" * 40)


class GPTEmbedding(nn.Module):
    """Token + position embeddings for GPT."""
    
    def __init__(self, vocab_size, embed_dim, max_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        
        tok_emb = self.token_embedding(input_ids)
        
        positions = torch.arange(seq_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        return self.dropout(tok_emb + pos_emb)


# Create embedding module
embedding = GPTEmbedding(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    max_length=256,
    dropout=0.1
)

print(f"Embedding module created:")
print(f"  Vocab size: {tokenizer.vocab_size}")
print(f"  Embed dim: 128")
print(f"  Parameters: {sum(p.numel() for p in embedding.parameters()):,}")

pause()


# ---------------------------------------------------------------------------
# Complete Pipeline
# ---------------------------------------------------------------------------

print("THE COMPLETE PIPELINE IN ACTION")
print("-" * 40)


def trace_pipeline(text, tokenizer, embedding):
    """Trace text through the complete pipeline."""
    
    print(f"1. Raw text: '{text}'")
    
    # Tokenize
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_token[i] for i in token_ids]
    print(f"\n2. Tokens: {tokens}")
    print(f"   IDs: {token_ids}")
    
    # Create input/target split
    if len(token_ids) > 1:
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        print(f"\n3. Input IDs: {input_ids}")
        print(f"   Target IDs: {target_ids}")
    else:
        input_ids = token_ids
        print(f"\n3. Input IDs: {input_ids}")
    
    # To tensor with batch dimension
    input_tensor = torch.tensor([input_ids])
    print(f"\n4. Tensor shape: {input_tensor.shape}")
    
    # Get embeddings
    embedding.eval()
    with torch.no_grad():
        embedded = embedding(input_tensor)
    print(f"\n5. Embedded shape: {embedded.shape}")
    print(f"   (batch=1, seq_len={len(input_ids)}, embed_dim=128)")
    
    print(f"\n6. Ready for transformer!")


test_text = "The cat sat on the mat."
trace_pipeline(test_text, tokenizer, embedding)

pause()


# ---------------------------------------------------------------------------
# Training Loop Preview
# ---------------------------------------------------------------------------

print("PREVIEW: TRAINING LOOP")
print("-" * 40)

print("""
Here's what training would look like:
""")

code = '''
# Training loop
model = GPTModel(vocab_size, embed_dim, ...)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        attention_mask = batch['attention_mask']
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Compute loss
        loss = criterion(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
'''

print(code)
print()
print("We'll implement the full GPT model in Week 4!")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
THE COMPLETE DATA PIPELINE:

1. TOKENIZER
   - Converts text to token IDs
   - Handles special tokens (PAD, UNK)
   - encode() and decode() methods

2. DATASET
   - Creates input/target pairs
   - Input: tokens[:-1], Target: tokens[1:]
   - The model learns to predict next token

3. COLLATE FUNCTION
   - Pads variable-length sequences
   - Creates attention masks
   - Enables batching

4. DATALOADER
   - Batches data efficiently
   - Shuffles for randomness
   - Handles iteration

5. EMBEDDING MODULE
   - Token embeddings
   - Position embeddings
   - Combined with dropout

THE FLOW:
    Text -> Tokenize -> Dataset -> Batch -> Embed -> Model

This is EXACTLY what GPT uses!
Next week: Attention mechanisms - the heart of the transformer.
""")

print("=" * 60)
print("  End of Lesson 4")
print("=" * 60)
