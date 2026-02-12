"""
Lesson 1: Tokenization
=======================

Neural networks only understand numbers, not text. So how do we
convert "Hello world" into something a model can process?

This is the tokenization problem, and getting it right matters
more than you might think.

Usage: python 01_tokenization.py
"""

import re
from collections import Counter


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 1: Tokenization")
print("  Turning Text into Numbers")
print("=" * 60)


# ---------------------------------------------------------------------------
# The Challenge
# ---------------------------------------------------------------------------

print("""
THE CHALLENGE

Neural networks work with numbers. Text is not numbers.
We need to convert text into numerical form.

The naive approach: assign each word a number.
    "The" -> 1
    "cat" -> 2
    "sat" -> 3
    ...

This seems reasonable, but it has serious problems.
Let's explore them.
""")

pause()


# ---------------------------------------------------------------------------
# Problem 1: Vocabulary Size
# ---------------------------------------------------------------------------

print("PROBLEM 1: VOCABULARY EXPLOSION")
print("-" * 40)

print("""
English has a LOT of words. Consider just one verb:

    run, runs, running, ran, runner, runners, rerun, overrun...

Each form needs its own entry in our vocabulary.
Multiply this across every word in the language.

A word-level vocabulary might need 500,000+ entries.
That means:
  - Huge embedding tables (500,000 x embedding_dim parameters)
  - Most words rarely appear in training
  - The model wastes capacity on rare words
""")

pause()

# Demonstrate the explosion
print("Let's see this concretely:")
print()

word_variants = [
    "run", "runs", "running", "ran", "runner", "runners",
    "play", "plays", "playing", "played", "player", "players", "replay",
    "compute", "computes", "computing", "computed", "computer", "computers",
]

print(f"Just 3 base words create {len(word_variants)} vocabulary entries.")
print(f"And we haven't even covered: recompute, precomputed, playable, etc.")
print()
print("Now imagine this for every word in the language...")

pause()


# ---------------------------------------------------------------------------
# Problem 2: Unknown Words
# ---------------------------------------------------------------------------

print("PROBLEM 2: UNKNOWN WORDS (OOV)")
print("-" * 40)

print("""
What happens when the model sees a word it's never seen before?

Training data: "The cat sat on the mat"
New input: "The cat sat on the mattress"

If "mattress" wasn't in training, we're stuck.
The usual solution: replace it with <UNK> (unknown).

But that throws away all information about the word!
""")

pause()

# Demonstrate OOV
training_vocab = {"the", "cat", "sat", "on", "mat", "dog", "ran"}

def tokenize_simple(text, vocab):
    words = text.lower().split()
    return [w if w in vocab else "<UNK>" for w in words]

print("Example:")
print(f"Vocabulary: {training_vocab}")
print()

text1 = "The cat sat on the mat"
text2 = "The cat sat on the mattress"

print(f"'{text1}'")
print(f"  -> {tokenize_simple(text1, training_vocab)}")
print()
print(f"'{text2}'")
print(f"  -> {tokenize_simple(text2, training_vocab)}")
print()
print("'mattress' became <UNK> - we lost important information!")

pause()


# ---------------------------------------------------------------------------
# Problem 3: Punctuation
# ---------------------------------------------------------------------------

print("PROBLEM 3: PUNCTUATION")
print("-" * 40)

print("""
Simple space-splitting handles punctuation poorly:

    "Hello, world!" -> ["Hello,", "world!"]

Now "Hello" and "Hello," are different vocabulary entries.
Same with "world" and "world!".

We could strip punctuation, but that loses information too.
"Let's eat, grandma" vs "Let's eat grandma" mean very different things!
""")

pause()


# ---------------------------------------------------------------------------
# The Character Approach
# ---------------------------------------------------------------------------

print("ATTEMPT 1: CHARACTER-LEVEL TOKENIZATION")
print("-" * 40)

print("""
One extreme: tokenize every character individually.

    "Hello" -> ["H", "e", "l", "l", "o"]

Advantages:
  - Tiny vocabulary (just ~100 characters)
  - Can handle ANY text, even typos
  - No unknown word problem ever!

Sounds perfect, right?
""")

pause()

text = "Hello world"
char_tokens = list(text)
print(f"Text: '{text}'")
print(f"Character tokens: {char_tokens}")
print(f"Number of tokens: {len(char_tokens)}")

pause()

print("THE CATCH")
print("-" * 40)

print("""
Character tokenization creates VERY long sequences.

Think about GPT's context window of 4096 tokens:
  - With words: ~4096 words ≈ 20,000 characters ≈ 15 pages
  - With characters: ~4096 characters ≈ 700 words ≈ 1.5 pages

That's a 10x difference in how much context the model can see!

Also, the model has to learn that "c" + "a" + "t" = cat.
This is much harder than just learning "cat" directly.
""")

pause()

# Compare sequence lengths
text = "The quick brown fox jumps over the lazy dog."
word_tokens = text.split()
char_tokens = list(text)

print("Comparison:")
print(f"  Text: '{text}'")
print(f"  Word tokens: {len(word_tokens)}")
print(f"  Char tokens: {len(char_tokens)}")
print(f"  Ratio: {len(char_tokens) / len(word_tokens):.1f}x longer with characters")

pause()


# ---------------------------------------------------------------------------
# The Solution: Subword Tokenization
# ---------------------------------------------------------------------------

print("THE SOLUTION: SUBWORD TOKENIZATION")
print("-" * 40)

print("""
The key insight: split text into SUBWORD pieces.

    "unfortunately" -> ["un", "fortun", "ately"]

This gives us the best of both worlds:
  - Moderate vocabulary size (~50,000 tokens)
  - Reasonable sequence lengths
  - Can handle any word by breaking it into pieces
  - Common words stay whole, rare words get split

This is what GPT, BERT, and all modern language models use.
""")

pause()

print("How subword tokenization handles different words:")
print()
print("  Common word:  'the'      -> ['the']           (stays whole)")
print("  Normal word:  'running'  -> ['run', 'ning']   (meaningful pieces)")
print("  Rare word:    'ChatGPT'  -> ['Chat', 'G', 'PT'] (broken into known pieces)")
print("  Misspelling:  'teh'      -> ['t', 'eh']       (still works!)")
print()
print("The algorithm learns which pieces are useful from the data.")

pause()


# ---------------------------------------------------------------------------
# Building a Simple Tokenizer
# ---------------------------------------------------------------------------

print("BUILDING A SIMPLE TOKENIZER")
print("-" * 40)


class SimpleTokenizer:
    """A basic word-level tokenizer to demonstrate the concepts."""
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Special tokens
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for token in self.special_tokens:
            self._add_token(token)
    
    def _add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def _preprocess(self, text):
        """Lowercase and separate punctuation."""
        text = text.lower()
        # Add spaces around punctuation
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def train(self, texts):
        """Build vocabulary from a list of texts."""
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
                ids.append(self.token_to_id['<UNK>'])
        return ids
    
    def decode(self, ids):
        """Convert token IDs back to text."""
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, '<UNK>')
            if token not in self.special_tokens:
                tokens.append(token)
        # Clean up spacing around punctuation
        text = ' '.join(tokens)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text
    
    @property
    def vocab_size(self):
        return len(self.token_to_id)


# Train and test
corpus = [
    "The cat sat on the mat.",
    "The dog ran in the park.",
    "I love machine learning!",
    "Language models are fascinating.",
]

tokenizer = SimpleTokenizer()
tokenizer.train(corpus)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print()
print("Vocabulary (first 15 tokens):")
for i in range(min(15, tokenizer.vocab_size)):
    print(f"  {i}: '{tokenizer.id_to_token[i]}'")

pause()

# Test encoding and decoding
print("Testing encode and decode:")
print()

test = "The cat sat on the mat."
encoded = tokenizer.encode(test)
decoded = tokenizer.decode(encoded)

print(f"Original: '{test}'")
print(f"Encoded:  {encoded}")
print(f"Decoded:  '{decoded}'")

pause()

# Show OOV handling
print("Handling unknown words:")
print()

test = "The elephant sat on the mat."
encoded = tokenizer.encode(test)
print(f"Text: '{test}'")
print(f"Encoded: {encoded}")
print(f"Note: 'elephant' became ID 1, which is <UNK>")

pause()


# ---------------------------------------------------------------------------
# Why This Matters for LLMs
# ---------------------------------------------------------------------------

print("WHY TOKENIZATION MATTERS FOR LLMs")
print("-" * 40)

print("""
Tokenization affects everything:

1. CONTEXT WINDOW
   GPT-4 has a 128K token context window.
   Efficient tokenization = more text fits in the window.

2. TRAINING EFFICIENCY
   The model sees each token once per example.
   If "unfortunately" is 1 token vs 4 tokens, it needs
   4x fewer examples to learn it equally well.

3. MULTILINGUAL CAPABILITY
   Different languages need different tokenization.
   Chinese, Japanese, Arabic all have unique challenges.

4. COST
   API pricing is per token.
   Better tokenization = cheaper API calls.

5. MODEL CAPABILITY
   How numbers are tokenized affects math ability!
   "380" as ["3", "80"] vs ["380"] makes a difference.
""")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)

print("""
1. Neural networks need numbers, not text

2. Word-level tokenization has problems:
   - Vocabulary too large (500K+ words)
   - Can't handle unknown words
   - Punctuation issues

3. Character-level has problems too:
   - Sequences become very long
   - Hard to learn meaning from characters

4. Subword tokenization is the solution:
   - Moderate vocabulary (~50K tokens)
   - Can handle any text
   - Used by GPT, BERT, and all modern LLMs

5. Common special tokens:
   - <PAD>: padding for batching
   - <UNK>: unknown words
   - <BOS>/<EOS>: beginning/end of sequence

Next up: The BPE algorithm that GPT actually uses!
""")

print("=" * 60)
print("  End of Lesson 1")
print("=" * 60)
