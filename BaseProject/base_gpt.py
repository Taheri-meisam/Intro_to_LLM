"""
Base Project: Fine-Tune a MiniGPT and Chat With It
====================================================

A minimal, self-contained project that brings together every
concept from the 7-week LLM course:

    Week 1: PyTorch training loop
    Week 2: Tokenizer and embeddings
    Week 3: Multi-head causal attention
    Week 4: Full GPT architecture
    Week 5: Evaluation and generation
    Week 6: Fine-tuning concepts
    Week 7: Instruction tuning + chat

What happens:
    1. Build a word-level tokenizer from hardcoded Q&A data
    2. Construct a small GPT (~120K parameters)
    3. Fine-tune it with loss masking (only learn responses)
    4. Drop into an interactive chat loop in the terminal

Usage:  python base_gpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import re
from collections import Counter


# ======================================================================
# DATASET — 80 instruction-response pairs across 4 topics
# ======================================================================

QA_DATA = [
    # --- Python Programming (20) ---
    ("What is a variable?",
     "A variable is a named container that stores a value. You create one with an equals sign, like x equals five."),
    ("What is a function?",
     "A function is a reusable block of code that performs a task. Define it with def, then call it by name."),
    ("What is a loop?",
     "A loop repeats code multiple times. A for loop iterates over a sequence, a while loop runs until a condition is false."),
    ("What is a list?",
     "A list is an ordered collection of items. You can add, remove, and access items by their index starting from zero."),
    ("What is a string?",
     "A string is a sequence of characters like text. You can slice, concatenate, and search within strings."),
    ("What is an if statement?",
     "An if statement lets your program make decisions. It checks a condition and runs code only when that condition is true."),
    ("What is a class?",
     "A class is a blueprint for creating objects. It bundles data and behavior together into a single reusable structure."),
    ("What is a dictionary?",
     "A dictionary stores key value pairs. You look up values by their key, like a real dictionary maps words to definitions."),
    ("What is recursion?",
     "Recursion is when a function calls itself. It needs a base case to stop and a recursive step that breaks the problem down."),
    ("What is an error?",
     "An error means something went wrong in your code. Read the traceback message carefully, it tells you the line and the problem."),
    ("What is a boolean?",
     "A boolean is a data type with two values, true or false. It is used in conditions and logical expressions."),
    ("What is an import?",
     "An import loads code from another file or library so you can use it. Python has many built in modules you can import."),
    ("What is a module?",
     "A module is a file containing Python code. You import it to reuse functions and classes across your programs."),
    ("What is indentation?",
     "Indentation defines code blocks in Python. Use four spaces to show which code belongs inside a function, loop, or condition."),
    ("What is a parameter?",
     "A parameter is a variable in a function definition. When you call the function, you pass an argument that fills that parameter."),
    ("What is a return statement?",
     "A return statement sends a value back from a function to the code that called it. Without return, the function gives None."),
    ("What is debugging?",
     "Debugging is finding and fixing errors in code. Use print statements, a debugger, or read the error message carefully."),
    ("What is a comment?",
     "A comment is text in your code that Python ignores. Use it to explain your logic. Start a comment with the hash symbol."),
    ("What is pip?",
     "Pip is the package installer for Python. Use pip install to download and install libraries from the internet."),
    ("What is a tuple?",
     "A tuple is like a list but immutable. Once created, you cannot change its elements. Use parentheses to create one."),

    # --- Machine Learning (20) ---
    ("What is machine learning?",
     "Machine learning is teaching computers to learn patterns from data instead of writing explicit rules for every case."),
    ("What is a neural network?",
     "A neural network is layers of connected nodes that learn to transform inputs into useful outputs through training."),
    ("What is a tensor?",
     "A tensor is a multi dimensional array used in deep learning. Scalars, vectors, and matrices are all types of tensors."),
    ("What is gradient descent?",
     "Gradient descent adjusts model weights to reduce the loss. It moves in the direction that decreases the error most."),
    ("What is a loss function?",
     "A loss function measures how wrong the model predictions are. Training tries to make this number as small as possible."),
    ("What is backpropagation?",
     "Backpropagation calculates how much each weight contributed to the error so gradient descent knows how to update them."),
    ("What is overfitting?",
     "Overfitting is when a model memorizes training data but fails on new data. Use dropout and early stopping to prevent it."),
    ("What is an embedding?",
     "An embedding maps discrete items like words to continuous vectors. Similar items end up with similar vector representations."),
    ("What is attention?",
     "Attention lets the model focus on relevant parts of the input. Each token decides how much to attend to every other token."),
    ("What is a transformer?",
     "A transformer is an architecture built on attention. It processes all tokens in parallel and captures long range dependencies."),
    ("What is fine tuning?",
     "Fine tuning takes a pretrained model and trains it further on a specific task. It adapts general knowledge to your needs."),
    ("What is tokenization?",
     "Tokenization splits text into small units called tokens. These can be words, subwords, or characters that the model processes."),
    ("What is a learning rate?",
     "The learning rate controls how big each weight update is. Too high and training is unstable, too low and it is very slow."),
    ("What is dropout?",
     "Dropout randomly turns off neurons during training. This prevents the model from relying too much on any single neuron."),
    ("What is cross entropy?",
     "Cross entropy measures how different two probability distributions are. It is the standard loss for classification tasks."),
    ("What is a batch?",
     "A batch is a group of training examples processed together. Batching makes training faster and more stable."),
    ("What is an epoch?",
     "An epoch is one complete pass through the entire training dataset. Models typically train for many epochs."),
    ("What is perplexity?",
     "Perplexity measures how surprised the model is by the data. Lower perplexity means the model predicts text better."),
    ("What is temperature?",
     "Temperature controls randomness in text generation. Low temperature gives focused text, high temperature gives creative text."),
    ("What is weight tying?",
     "Weight tying shares the embedding matrix with the output layer. It reduces parameters and often improves performance."),

    # --- Math Basics (20) ---
    ("What is addition?",
     "Addition combines numbers to get a total. Three plus four equals seven. It is the most fundamental math operation."),
    ("What is multiplication?",
     "Multiplication is repeated addition. Three times four means adding three four times, which gives twelve."),
    ("What is division?",
     "Division splits a number into equal parts. Twelve divided by three equals four, meaning twelve split into three groups."),
    ("What is a fraction?",
     "A fraction represents a part of a whole. The top number is the numerator and the bottom is the denominator."),
    ("What is algebra?",
     "Algebra uses letters to represent unknown numbers. You solve equations to find what value the letter stands for."),
    ("What is geometry?",
     "Geometry studies shapes, sizes, and space. It covers triangles, circles, angles, and three dimensional objects."),
    ("What is probability?",
     "Probability measures how likely something is to happen. It ranges from zero meaning impossible to one meaning certain."),
    ("What is a matrix?",
     "A matrix is a grid of numbers in rows and columns. Matrices are used in graphics, physics, and machine learning."),
    ("What is calculus?",
     "Calculus studies how things change. Derivatives measure rates of change, integrals measure accumulated quantities."),
    ("What is a prime number?",
     "A prime number is only divisible by one and itself. Examples are two, three, five, seven, and eleven."),
    ("What is the mean?",
     "The mean is the average. Add all numbers together and divide by how many there are. The mean of two and four is three."),
    ("What is pi?",
     "Pi is approximately three point one four. It is the ratio of a circle circumference to its diameter and it never ends."),
    ("What is a square root?",
     "The square root finds which number multiplied by itself gives the original. The square root of nine is three."),
    ("What is an exponent?",
     "An exponent says how many times to multiply a number by itself. Two to the power three equals eight."),
    ("What is a percentage?",
     "A percentage is a number out of one hundred. Fifty percent means fifty out of a hundred, the same as one half."),
    ("What is an equation?",
     "An equation says two expressions are equal. You solve it by finding the unknown value that makes both sides match."),
    ("What is logarithm?",
     "A logarithm is the inverse of an exponent. Log base two of eight equals three because two cubed is eight."),
    ("What is statistics?",
     "Statistics collects and analyzes data. It uses mean, median, mode, and standard deviation to find patterns."),
    ("What is a ratio?",
     "A ratio compares two quantities. A ratio of three to one means for every three of one thing there is one of another."),
    ("What is subtraction?",
     "Subtraction finds the difference between numbers. Ten minus four equals six. It is the opposite of addition."),

    # --- General Chat (20) ---
    ("Hello!",
     "Hello! I am your assistant. Ask me anything about programming, machine learning, or math."),
    ("Hi there!",
     "Hi! Welcome. What would you like to learn about today?"),
    ("How are you?",
     "I am doing great, thank you! I am ready to help you learn. What topic interests you?"),
    ("What can you do?",
     "I can answer questions about Python programming, machine learning, and math. Just ask me anything!"),
    ("Who are you?",
     "I am an AI assistant fine tuned to help with programming, machine learning, and math concepts."),
    ("Thank you!",
     "You are welcome! Happy to help. Keep asking questions, that is the best way to learn."),
    ("Goodbye!",
     "Goodbye! Keep learning and stay curious. Come back anytime you have questions."),
    ("Tell me something interesting.",
     "Here is a fun fact. The first computer bug was an actual moth found inside a computer in nineteen forty seven!"),
    ("I feel stuck.",
     "That is completely normal! Take a break, then approach the problem from a different angle. You will get through it."),
    ("This is too hard.",
     "Hard means you are learning. Break the problem into smaller steps and tackle them one at a time."),
    ("I need help.",
     "Of course! Tell me what topic or problem you are working on and I will do my best to explain it clearly."),
    ("Can you explain simply?",
     "Absolutely! Tell me the topic and I will break it down into the simplest terms I can."),
    ("I made a mistake.",
     "Mistakes are how we learn! Every expert made thousands of mistakes to get where they are. What happened?"),
    ("I want to learn coding.",
     "Great choice! Start with Python. Practice every day, build small projects, and do not be afraid of errors."),
    ("What should I learn first?",
     "Start with Python basics like variables, loops, and functions. Then move to data structures and simple projects."),
    ("I am confused.",
     "That is okay! Confusion means you are pushing your boundaries. Tell me what part is confusing and we will work through it."),
    ("Give me a tip.",
     "Write code every day, even just a little. Consistency beats intensity. Small daily practice builds strong skills."),
    ("I love learning!",
     "That is wonderful! Curiosity is the most powerful tool for learning. Keep that enthusiasm alive!"),
    ("Is AI hard to learn?",
     "It takes effort but it is very rewarding. Start with the basics and build up. This course is a great start!"),
    ("Tell me a joke.",
     "Why do programmers prefer dark mode? Because light attracts bugs!"),
]


# ======================================================================
# TOKENIZER  (Week 2 + Week 7)
# ======================================================================

class Tokenizer:
    """Word-level tokenizer with special tokens for instruction tuning."""

    SPECIAL = ['<PAD>', '<UNK>', '<USER>', '<ASST>', '<END>']

    def __init__(self, pairs, max_vocab=600):
        self.word2id = {t: i for i, t in enumerate(self.SPECIAL)}

        counts = Counter()
        for q, a in pairs:
            counts.update(self._split(q))
            counts.update(self._split(a))

        for word, _ in counts.most_common(max_vocab - len(self.SPECIAL)):
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)

        self.id2word = {i: w for w, i in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        self.pad_id = 0
        self.unk_id = 1
        self.end_id = 4

    @staticmethod
    def _split(text):
        text = text.lower()
        text = re.sub(r"([.,!?'])", r" \1 ", text)
        return text.split()

    def encode(self, question, answer=None, max_len=64):
        """Encode a Q&A pair with special tokens.

        Format: <USER> question tokens <ASST> answer tokens <END> <PAD>...
        Returns (token_ids, response_start_index).
        """
        tokens = ['<USER>'] + self._split(question) + ['<ASST>']
        resp_start = len(tokens)

        if answer:
            tokens += self._split(answer) + ['<END>']

        ids = [self.word2id.get(t, self.unk_id) for t in tokens]

        # Pad or truncate
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return ids, min(resp_start, max_len)

    def encode_prompt(self, question):
        """Encode just the prompt (for generation)."""
        tokens = ['<USER>'] + self._split(question) + ['<ASST>']
        return [self.word2id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids):
        """Decode token IDs back to readable text."""
        skip = set(self.SPECIAL)
        words = []
        for i in ids:
            if i == self.end_id:
                break
            w = self.id2word.get(i, '')
            if w and w not in skip:
                words.append(w)
        return ' '.join(words)


# ======================================================================
# DATASET WITH LOSS MASKING  (Week 7)
# ======================================================================

class QADataset(Dataset):
    """Wraps Q&A pairs with loss masking on prompt tokens."""

    def __init__(self, pairs, tokenizer, max_len=64):
        self.items = []
        for q, a in pairs:
            ids, resp_start = tokenizer.encode(q, a, max_len)
            labels = [-100] * resp_start + ids[resp_start:]
            labels = (labels + [-100] * max_len)[:max_len]
            self.items.append({
                'input_ids': torch.tensor(ids),
                'labels': torch.tensor(labels),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ======================================================================
# MINIGPT MODEL  (Weeks 2-4)
# ======================================================================

class MiniGPT(nn.Module):
    """A small GPT: 3 layers, 4 heads, d_model=80  (~120K params)."""

    def __init__(self, vocab, d_model=80, heads=4, layers=3, max_len=64, drop=0.1):
        super().__init__()
        self.max_len = max_len
        self.heads = heads
        self.head_dim = d_model // heads

        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
                'qkv': nn.Linear(d_model, 3 * d_model),
                'out': nn.Linear(d_model, d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(drop),
                ),
            }) for _ in range(layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))

    def _attn(self, x, block):
        B, T, C = x.shape
        qkv = block['qkv'](x).view(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return block['out']((weights @ V).transpose(1, 2).reshape(B, T, C))

    def forward(self, x, labels=None):
        B, T = x.shape
        h = self.drop(self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device)))
        for block in self.blocks:
            h = h + self._attn(block['ln1'](h), block)
            h = h + block['ffn'](block['ln2'](h))
        logits = self.lm_head(self.ln_f(h))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new=50, temperature=0.7, end_id=None):
        self.eval()
        for _ in range(max_new):
            x = ids[:, -self.max_len:]
            logits, _ = self(x)
            logits = logits[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
            if end_id is not None and nxt.item() == end_id:
                break
        return ids


# ======================================================================
# TRAINING  (Weeks 1, 5, 6)
# ======================================================================

def train(model, dataset, epochs=30, lr=1e-3, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\n  Training for {epochs} epochs ...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in loader:
            _, loss = model(batch['input_ids'], batch['labels'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/{epochs}  loss={avg:.4f}")

    print(f"\n  Done! Final loss: {avg:.4f}")
    return avg


# ======================================================================
# CHAT FUNCTION
# ======================================================================

def chat(model, tokenizer, temperature=0.7):
    """Ask the model a question and return the response text."""
    model.eval()

    def ask(question):
        prompt_ids = tokenizer.encode_prompt(question)
        ids = torch.tensor([prompt_ids])
        out = model.generate(ids, max_new=50, temperature=temperature,
                             end_id=tokenizer.end_id)
        return tokenizer.decode(out[0, len(prompt_ids):].tolist())

    return ask


# ======================================================================
# MAIN — run everything
# ======================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("  Base Project: Fine-Tune a MiniGPT & Chat")
    print("=" * 60)

    # --- 1. Tokenizer ---
    print("\n1. Building tokenizer ...")
    tok = Tokenizer(QA_DATA, max_vocab=600)
    print(f"   Vocabulary size: {tok.vocab_size}")

    # --- 2. Dataset ---
    print("\n2. Preparing dataset with loss masking ...")
    ds = QADataset(QA_DATA, tok, max_len=64)
    print(f"   {len(ds)} training pairs  |  max_len=64")

    # --- 3. Model ---
    print("\n3. Building MiniGPT ...")
    model = MiniGPT(tok.vocab_size, d_model=80, heads=4, layers=3, max_len=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"   Architecture: 3 layers, 4 heads, d_model=80")

    # --- 4. Train ---
    print("\n4. Fine-tuning on Q&A data ...")
    train(model, ds, epochs=30)

    # --- 5. Quick test ---
    print("\n5. Quick test on a few questions:\n")
    ask = chat(model, tok, temperature=0.5)
    for q in ["What is a variable?", "What is attention?", "Hello!"]:
        print(f"   You:  {q}")
        print(f"   Bot:  {ask(q)}\n")

    # --- 6. Interactive chat ---
    print("=" * 60)
    print("  INTERACTIVE CHAT  (type 'quit' to exit)")
    print("=" * 60)
    print()

    ask = chat(model, tok, temperature=0.7)
    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("  Bye!")
            break
        response = ask(user_input)
        print(f"  Bot: {response}\n")
