"""
Week 2 Project: Shakespeare Text Processor
==========================================

Build a complete text processing pipeline for Shakespeare's works.
We'll use this pipeline in Week 4 to train a GPT model that
generates Shakespeare-like text.

Usage: python project_shakespeare.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import os


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Week 2 Project: Shakespeare Text Processor")
print("  Building a Pipeline for Text Generation")
print("=" * 60)


# ---------------------------------------------------------------------------
# The Data
# ---------------------------------------------------------------------------

print("""
THE PROJECT

We'll build a complete pipeline to process Shakespeare's text.
This same pipeline will be used in Week 4 to train a mini GPT.

Components to build:
  1. A tokenizer trained on Shakespeare
  2. A dataset class for language modeling
  3. A data loader with proper batching
  4. An embedding module
  
Let's start!
""")

pause()


# Shakespeare sample (a subset for this demo)
SHAKESPEARE = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to. 'Tis a consummation
Devoutly to be wished. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones.
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious.
If it were so, it was a grievous fault,
And grievously hath Caesar answered it.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that loured upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths,
Our bruised arms hung up for monuments,
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.

O Romeo, Romeo, wherefore art thou Romeo?
Deny thy father and refuse thy name.
Or if thou wilt not, be but sworn my love
And I'll no longer be a Capulet.
'Tis but thy name that is my enemy.
Thou art thyself, though not a Montague.
What's Montague? It is nor hand nor foot
Nor arm nor face nor any other part
Belonging to a man. O be some other name!
What's in a name? That which we call a rose
By any other name would smell as sweet.

If music be the food of love, play on,
Give me excess of it, that, surfeiting,
The appetite may sicken and so die.
That strain again, it had a dying fall.
O, it came o'er my ear like the sweet sound
That breathes upon a bank of violets,
Stealing and giving odour.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player,
That struts and frets his hour upon the stage,
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

Double, double toil and trouble;
Fire burn and cauldron bubble.
Fillet of a fenny snake,
In the cauldron boil and bake;
Eye of newt and toe of frog,
Wool of bat and tongue of dog.

The quality of mercy is not strained;
It droppeth as the gentle rain from heaven
Upon the place beneath. It is twice blest;
It blesseth him that gives and him that takes.
'Tis mightiest in the mightiest; it becomes
The throned monarch better than his crown.
His sceptre shows the force of temporal power,
The attribute to awe and majesty.

We are such stuff as dreams are made on,
And our little life is rounded with a sleep.
"""


# ---------------------------------------------------------------------------
# Component 1: Tokenizer
# ---------------------------------------------------------------------------

print("STEP 1: BUILDING THE TOKENIZER")
print("-" * 40)


class ShakespeareTokenizer:
    """
    A tokenizer trained on Shakespeare's text.
    Uses character-level tokenization for simplicity,
    which works well for generating Shakespeare-like text.
    """
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.pad_id = 0
        self.char_to_id[self.pad_token] = 0
        self.id_to_char[0] = self.pad_token
    
    def train(self, text):
        """Build vocabulary from text."""
        # Get unique characters
        chars = sorted(set(text))
        
        # Add to vocabulary (starting after special tokens)
        for char in chars:
            if char not in self.char_to_id:
                idx = len(self.char_to_id)
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
        
        return self
    
    def encode(self, text):
        """Convert text to token IDs."""
        return [self.char_to_id.get(c, self.pad_id) for c in text]
    
    def decode(self, ids):
        """Convert token IDs back to text."""
        chars = []
        for idx in ids:
            char = self.id_to_char.get(idx, '')
            if char != self.pad_token:
                chars.append(char)
        return ''.join(chars)
    
    @property
    def vocab_size(self):
        return len(self.char_to_id)


# Train the tokenizer
tokenizer = ShakespeareTokenizer()
tokenizer.train(SHAKESPEARE)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print()

# Show some vocabulary
print("Sample vocabulary:")
for i in range(min(20, tokenizer.vocab_size)):
    char = tokenizer.id_to_char[i]
    display = repr(char) if char in ['\n', ' ', '\t'] else f"'{char}'"
    print(f"  {i}: {display}")

pause()

# Test encoding/decoding
print("Testing tokenizer:")
print()

test = "To be, or not to be"
encoded = tokenizer.encode(test)
decoded = tokenizer.decode(encoded)

print(f"Original: '{test}'")
print(f"Encoded: {encoded[:20]}...")
print(f"Decoded: '{decoded}'")
print(f"Match: {test == decoded}")

pause()


# ---------------------------------------------------------------------------
# Component 2: Dataset
# ---------------------------------------------------------------------------

print("STEP 2: BUILDING THE DATASET")
print("-" * 40)


class ShakespeareDataset(Dataset):
    """
    Dataset for training a character-level language model on Shakespeare.
    Creates overlapping sequences for better training.
    """
    
    def __init__(self, text, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Encode entire text
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        print(f"  Total characters: {len(self.data):,}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Possible sequences: {len(self.data) - seq_length:,}")
    
    def __len__(self):
        # We can start a sequence at any position up to len - seq_length - 1
        return len(self.data) - self.seq_length - 1
    
    def __getitem__(self, idx):
        # Get a sequence of seq_length + 1 tokens
        chunk = self.data[idx:idx + self.seq_length + 1]
        
        # Input: first seq_length tokens
        # Target: last seq_length tokens (shifted by 1)
        return {
            'input_ids': chunk[:-1],
            'target_ids': chunk[1:]
        }


# Create dataset
print("Creating dataset...")
dataset = ShakespeareDataset(SHAKESPEARE, tokenizer, seq_length=64)
print(f"  Dataset size: {len(dataset)}")

pause()

# Look at an example
print("Example from dataset:")
print()

example = dataset[0]
input_ids = example['input_ids']
target_ids = example['target_ids']

print(f"Input shape: {input_ids.shape}")
print(f"Target shape: {target_ids.shape}")
print()

# Decode to see the text
input_text = tokenizer.decode(input_ids.tolist())
target_text = tokenizer.decode(target_ids.tolist())

print(f"Input text:\n'{input_text}'")
print()
print(f"Target text:\n'{target_text}'")
print()
print("Notice: target is shifted by 1 character from input!")

pause()


# ---------------------------------------------------------------------------
# Component 3: DataLoader
# ---------------------------------------------------------------------------

print("STEP 3: CREATING THE DATALOADER")
print("-" * 40)

# Split into train/val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Simple split (in practice, you'd want to split at sentence boundaries)
train_indices = range(train_size)
val_indices = range(train_size, len(dataset))

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    sampler=torch.utils.data.SubsetRandomSampler(train_indices)
)

val_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    sampler=torch.utils.data.SubsetRandomSampler(val_indices)
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print()

# Get a sample batch
batch = next(iter(train_loader))
print(f"Batch shapes:")
print(f"  Input: {batch['input_ids'].shape}")
print(f"  Target: {batch['target_ids'].shape}")

pause()


# ---------------------------------------------------------------------------
# Component 4: Embedding Module
# ---------------------------------------------------------------------------

print("STEP 4: BUILDING THE EMBEDDING MODULE")
print("-" * 40)


class GPTEmbedding(nn.Module):
    """Embedding module with token and position embeddings."""
    
    def __init__(self, vocab_size, embed_dim, max_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        tok_emb = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        return self.dropout(tok_emb + pos_emb)


# Create embedding module
embedding = GPTEmbedding(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    max_length=128,
    dropout=0.1
)

print(f"Embedding module created:")
print(f"  Vocab size: {tokenizer.vocab_size}")
print(f"  Embedding dimension: 64")
print(f"  Max sequence length: 128")
print(f"  Parameters: {sum(p.numel() for p in embedding.parameters()):,}")

pause()


# ---------------------------------------------------------------------------
# Complete Pipeline Test
# ---------------------------------------------------------------------------

print("TESTING THE COMPLETE PIPELINE")
print("-" * 40)

# Get a batch and run through embedding
embedding.eval()

batch = next(iter(train_loader))
input_ids = batch['input_ids']
target_ids = batch['target_ids']

print(f"1. Batch input shape: {input_ids.shape}")
print(f"   (batch_size=32, seq_length=64)")

with torch.no_grad():
    embeddings = embedding(input_ids)

print(f"\n2. Embedding shape: {embeddings.shape}")
print(f"   (batch_size=32, seq_length=64, embed_dim=64)")

print(f"\n3. Target shape: {target_ids.shape}")
print(f"   (batch_size=32, seq_length=64)")

print("\n4. Ready for transformer!")

pause()


# ---------------------------------------------------------------------------
# Save the Pipeline Components
# ---------------------------------------------------------------------------

print("SAVING PIPELINE COMPONENTS")
print("-" * 40)

# Save tokenizer vocabulary
vocab_data = {
    'char_to_id': tokenizer.char_to_id,
    'id_to_char': tokenizer.id_to_char
}
torch.save(vocab_data, 'shakespeare_vocab.pt')
print("Saved: shakespeare_vocab.pt")

# Save embedding weights (random for now, will be trained in Week 4)
torch.save(embedding.state_dict(), 'shakespeare_embedding.pt')
print("Saved: shakespeare_embedding.pt")

print()
print("These files will be used in Week 4 to train our GPT model!")

pause()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

print("PIPELINE STATISTICS")
print("-" * 40)

print(f"Text Statistics:")
print(f"  Total characters: {len(SHAKESPEARE):,}")
print(f"  Unique characters: {tokenizer.vocab_size}")
print()

print(f"Dataset Statistics:")
print(f"  Total sequences: {len(dataset):,}")
print(f"  Sequence length: 64")
print(f"  Train sequences: {train_size:,}")
print(f"  Validation sequences: {val_size:,}")
print()

print(f"Model Statistics:")
print(f"  Embedding dim: 64")
print(f"  Embedding parameters: {sum(p.numel() for p in embedding.parameters()):,}")

pause()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

# Clean up saved files
for f in ['shakespeare_vocab.pt', 'shakespeare_embedding.pt']:
    if os.path.exists(f):
        os.remove(f)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  WEEK 2 PROJECT COMPLETE!")
print("=" * 60)

print("""
WHAT YOU BUILT:

1. TOKENIZER
   - Character-level tokenization
   - Handles all of Shakespeare's text
   - encode() and decode() methods

2. DATASET
   - Creates input/target pairs
   - Overlapping sequences for training
   - Works with PyTorch DataLoader

3. DATALOADER
   - Batching (32 sequences per batch)
   - Train/validation split
   - Shuffling for training

4. EMBEDDING MODULE
   - Token embeddings
   - Position embeddings
   - Ready for the transformer

NEXT STEPS:

In Week 3, we'll learn about attention mechanisms.
In Week 4, we'll use this pipeline to train a GPT
model that generates Shakespeare-like text!

The pipeline you built today is EXACTLY what's used
to train real language models like GPT.
""")

print("=" * 60)
print("  End of Week 2")
print("=" * 60)
