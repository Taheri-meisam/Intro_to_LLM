"""
Lesson 2: Byte Pair Encoding (BPE)
===================================

Last lesson, you learned WHY we need subword tokenization.
Now you'll learn HOW it works by implementing the BPE algorithm
that GPT uses.

BPE is surprisingly simple but incredibly effective.

Usage: python 02_bpe.py
"""

from collections import Counter


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 2: Byte Pair Encoding (BPE)")
print("  The Algorithm Behind GPT's Tokenizer")
print("=" * 60)


# ---------------------------------------------------------------------------
# The Core Idea
# ---------------------------------------------------------------------------

print("""
THE BPE ALGORITHM

The idea is elegant:

    1. Start with characters as your vocabulary
    2. Count pairs of adjacent tokens
    3. Merge the most common pair into a new token
    4. Repeat until you reach desired vocabulary size

That's it! The algorithm learns from data what to merge.
Common sequences naturally become single tokens.
""")

pause()


# ---------------------------------------------------------------------------
# Step-by-Step Example
# ---------------------------------------------------------------------------

print("STEP-BY-STEP EXAMPLE")
print("-" * 40)

print("""
Let's trace through BPE with a simple corpus: "low lower lowest"

We'll use a special marker </w> to indicate word boundaries.
This helps us know where words end when decoding.
""")

pause()

print("STEP 1: Start with characters")
print("-" * 40)


def get_word_frequencies(corpus):
    """Split corpus into character sequences with word boundaries."""
    word_freqs = Counter(corpus.split())
    
    # Convert each word to space-separated characters + end marker
    char_vocab = {}
    for word, freq in word_freqs.items():
        # "low" -> "l o w </w>"
        chars = ' '.join(list(word)) + ' </w>'
        char_vocab[chars] = freq
    
    return char_vocab


corpus = "low lower lowest"
vocab = get_word_frequencies(corpus)

print(f"Corpus: '{corpus}'")
print()
print("Initial vocabulary (as characters):")
for word, freq in vocab.items():
    print(f"  '{word}': {freq}")

pause()


print("STEP 2: Count adjacent pairs")
print("-" * 40)


def count_pairs(vocab):
    """Count frequency of adjacent token pairs."""
    pairs = Counter()
    
    for word, freq in vocab.items():
        tokens = word.split()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += freq
    
    return pairs


pairs = count_pairs(vocab)

print("Pair frequencies:")
for pair, freq in pairs.most_common():
    print(f"  {pair}: {freq}")

print()
print(f"Most common pair: {pairs.most_common(1)[0]}")

pause()


print("STEP 3: Merge the most common pair")
print("-" * 40)


def merge_pair(vocab, pair):
    """Merge all occurrences of a pair in the vocabulary."""
    new_vocab = {}
    
    # The pair as a string: ('l', 'o') -> 'l o'
    bigram = ' '.join(pair)
    # The merged token: ('l', 'o') -> 'lo'
    merged = ''.join(pair)
    
    for word, freq in vocab.items():
        # Replace 'l o' with 'lo'
        new_word = word.replace(bigram, merged)
        new_vocab[new_word] = freq
    
    return new_vocab


# Merge the most common pair
best_pair = pairs.most_common(1)[0][0]
new_vocab = merge_pair(vocab, best_pair)

print(f"Merging {best_pair} -> {''.join(best_pair)}")
print()
print("Vocabulary after merge:")
for word, freq in new_vocab.items():
    print(f"  '{word}': {freq}")

pause()


print("STEP 4: Repeat!")
print("-" * 40)

print("""
We keep merging until we reach our desired vocabulary size.
Let's watch the full process:
""")

pause()


def train_bpe(corpus, num_merges):
    """Train BPE with a specified number of merges."""
    vocab = get_word_frequencies(corpus)
    merges = []
    
    print("Starting vocabulary:")
    for word, freq in vocab.items():
        print(f"  '{word}': {freq}")
    
    for i in range(num_merges):
        pairs = count_pairs(vocab)
        
        if not pairs:
            print(f"\nNo more pairs to merge!")
            break
        
        best_pair = pairs.most_common(1)[0][0]
        vocab = merge_pair(vocab, best_pair)
        merges.append(best_pair)
        
        print(f"\nMerge {i+1}: {best_pair} -> {''.join(best_pair)}")
        for word, freq in vocab.items():
            print(f"  '{word}': {freq}")
    
    return vocab, merges


corpus = "low lower lowest"
final_vocab, merges = train_bpe(corpus, num_merges=7)

pause()

print("Learned merge operations:")
for i, (a, b) in enumerate(merges):
    print(f"  {i+1}. {a} + {b} -> {a}{b}")

pause()


# ---------------------------------------------------------------------------
# Complete BPE Tokenizer
# ---------------------------------------------------------------------------

print("COMPLETE BPE TOKENIZER")
print("-" * 40)

print("""
Now let's build a complete tokenizer class that can:
  - Train on a corpus
  - Encode text to token IDs
  - Decode token IDs back to text
""")

pause()


class BPETokenizer:
    """A simple BPE tokenizer implementation."""
    
    def __init__(self):
        self.merges = []  # List of (token1, token2) merge operations
        self.vocab = {}   # token -> id mapping
        
        # Special tokens
        self.special_tokens = ['<PAD>', '<UNK>']
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
    
    def train(self, corpus, vocab_size):
        """Learn BPE merges from corpus."""
        # Get word frequencies as character sequences
        word_freqs = Counter(corpus.split())
        token_freqs = {}
        for word, freq in word_freqs.items():
            chars = ' '.join(list(word)) + ' </w>'
            token_freqs[chars] = freq
        
        # Add all characters to vocabulary
        for word in token_freqs.keys():
            for token in word.split():
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        # Perform merges until we reach vocab_size
        while len(self.vocab) < vocab_size:
            # Count pairs
            pairs = Counter()
            for word, freq in token_freqs.items():
                tokens = word.split()
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i + 1])] += freq
            
            if not pairs:
                break
            
            # Find most common pair
            best_pair = pairs.most_common(1)[0][0]
            
            # Merge it
            bigram = ' '.join(best_pair)
            merged = ''.join(best_pair)
            
            new_token_freqs = {}
            for word, freq in token_freqs.items():
                new_word = word.replace(bigram, merged)
                new_token_freqs[new_word] = freq
            
            token_freqs = new_token_freqs
            
            # Record the merge and add to vocabulary
            self.merges.append(best_pair)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
        
        # Build reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        return self
    
    def tokenize(self, text):
        """Convert text to list of tokens (strings)."""
        tokens = []
        
        for word in text.split():
            # Start with characters
            word_tokens = list(word) + ['</w>']
            
            # Apply merges in order learned
            for (a, b) in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == a and word_tokens[i + 1] == b:
                        word_tokens = word_tokens[:i] + [a + b] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(self, text):
        """Convert text to list of token IDs."""
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['<UNK>'])
        return ids
    
    def decode(self, ids):
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(i, '<UNK>') for i in ids]
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()


# Train on a larger corpus
corpus = """
the cat sat on the mat
the dog ran in the park
the cat and the dog played together
the quick brown fox jumps over the lazy dog
machine learning is fascinating
deep learning transforms everything
"""

tokenizer = BPETokenizer()
tokenizer.train(corpus, vocab_size=50)

print(f"Vocabulary size: {len(tokenizer.vocab)}")
print()
print("Learned merges (first 10):")
for i, (a, b) in enumerate(tokenizer.merges[:10]):
    print(f"  {a} + {b} -> {a}{b}")

pause()


# ---------------------------------------------------------------------------
# Testing the Tokenizer
# ---------------------------------------------------------------------------

print("TESTING THE TOKENIZER")
print("-" * 40)

test_sentences = [
    "the cat sat",
    "the dog ran",
    "machine learning",
]

print("Encoding test sentences:")
print()

for sentence in test_sentences:
    tokens = tokenizer.tokenize(sentence)
    ids = tokenizer.encode(sentence)
    decoded = tokenizer.decode(ids)
    
    print(f"Text: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    print(f"Decoded: '{decoded}'")
    print()

pause()


# ---------------------------------------------------------------------------
# Handling Unknown Text
# ---------------------------------------------------------------------------

print("HANDLING UNKNOWN TEXT")
print("-" * 40)

print("""
The magic of BPE: it can handle text it's never seen!
It just breaks unknown words into smaller known pieces.
""")

pause()

unknown_sentences = [
    "elephant",
    "transformers",
    "ChatGPT",
]

print("Tokenizing words NOT in training data:")
print()

for word in unknown_sentences:
    tokens = tokenizer.tokenize(word)
    print(f"'{word}' -> {tokens}")

print()
print("Even though these words were never seen, BPE handles them")
print("by breaking them into character pieces it knows!")

pause()


# ---------------------------------------------------------------------------
# BPE vs Other Algorithms
# ---------------------------------------------------------------------------

print("BPE VS OTHER TOKENIZATION ALGORITHMS")
print("-" * 40)

print("""
BPE isn't the only subword algorithm. Here are the main ones:

1. BPE (Byte Pair Encoding)
   Used by: GPT-2, GPT-3, GPT-4, RoBERTa
   Merges most frequent pairs
   
2. WordPiece
   Used by: BERT, DistilBERT
   Similar to BPE but uses likelihood instead of frequency
   Uses ## prefix for continuation: "playing" -> ["play", "##ing"]
   
3. Unigram
   Used by: T5, XLNet
   Starts with large vocab, prunes unlikely tokens
   
4. SentencePiece
   Used by: Many multilingual models
   Treats text as byte stream, handles any language

For most purposes, they perform similarly.
GPT uses BPE, so that's what we focus on.
""")

pause()


# ---------------------------------------------------------------------------
# Real-World Considerations
# ---------------------------------------------------------------------------

print("REAL-WORLD CONSIDERATIONS")
print("-" * 40)

print("""
When building production tokenizers:

1. VOCABULARY SIZE
   - GPT-2: 50,257 tokens
   - GPT-4: ~100,000 tokens
   - Larger vocab = shorter sequences but bigger embeddings
   
2. SPECIAL TOKENS
   - <PAD>: Padding for batching
   - <UNK>: Unknown tokens (rarely used with BPE)
   - <BOS>/<EOS>: Sequence boundaries
   - <SEP>: Segment separator (for BERT)
   
3. PREPROCESSING
   - Handle Unicode (emojis, accents)
   - Normalize whitespace
   - Decide on case sensitivity
   
4. TRAINING DATA
   - Should match what model will see
   - Multilingual models need diverse text
   - Code models need code in training

In practice, you'll use pretrained tokenizers.
For GPT, OpenAI provides the 'tiktoken' library.
""")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. BPE ALGORITHM
   - Start with characters
   - Count adjacent pairs
   - Merge most common pair
   - Repeat until desired vocab size

2. WHY IT WORKS
   - Common sequences become single tokens
   - Rare words break into known pieces
   - No true OOV problem
   - Learns from data, no linguistic rules needed

3. THE TOKENIZER COMPONENTS
   - train(): Learn merges from corpus
   - tokenize(): Text -> list of token strings
   - encode(): Text -> list of token IDs
   - decode(): Token IDs -> text

4. VOCABULARY COMPOSITION
   - Special tokens (<PAD>, <UNK>, etc.)
   - Single characters (fallback)
   - Subword pieces (from merges)
   - Complete words (frequent enough)

5. IN PRACTICE
   - Use pretrained tokenizers (tiktoken for GPT)
   - Vocab size ~50K-100K tokens
   - Different models use different tokenizers

Next up: Embeddings - converting token IDs to vectors!
""")

print("=" * 60)
print("  End of Lesson 2")
print("=" * 60)
