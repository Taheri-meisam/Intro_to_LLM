"""
Base Project: Fine-Tune a Real GPT-2 and Chat With It
======================================================

This script downloads a pretrained GPT-2 model from Hugging Face,
fine-tunes it on a small Q&A dataset, and drops you into an
interactive chat in the terminal.

Unlike base_gpt.py (which builds a tiny GPT from scratch), this
uses a REAL 124M-parameter GPT-2 that already understands English.
Fine-tuning teaches it to follow our Q&A format in just a few minutes.

Steps:
    1. Download GPT-2 (124M) tokenizer + model from Hugging Face
    2. Prepare instruction data in  User: ... Assistant: ...  format
    3. Fine-tune with loss masking (only learn the assistant responses)
    4. Chat interactively in the terminal

Requirements:
    pip install transformers accelerate

Usage:
    python base_gpt_fineTuned.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    print("\n  transformers is not installed. Run:")
    print("    pip install transformers accelerate\n")
    raise SystemExit(1)


# ======================================================================
# CONFIG
# ======================================================================

MODEL_NAME = "gpt2"           # 124M params — downloads ~500MB once
MAX_LEN    = 256               # max tokens per training example
EPOCHS     = 3                 # few epochs is enough for a real model
BATCH_SIZE = 2                 # small batch, fits any GPU
LR         = 5e-5              # standard fine-tuning learning rate
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# DATASET — 60 Q&A pairs for fine-tuning
# ======================================================================

QA_DATA = [
    # --- Python Programming ---
    ("What is a variable in Python?",
     "A variable is a name that refers to a value stored in memory. "
     "You create one with the assignment operator, like x = 5. "
     "Python figures out the type automatically."),
    ("What is a function in Python?",
     "A function is a reusable block of code defined with the def keyword. "
     "It takes parameters, does some work, and can return a result. "
     "Functions help you organize code and avoid repetition."),
    ("What is a list in Python?",
     "A list is an ordered, mutable collection created with square brackets. "
     "You can store any mix of types, access items by index, and use methods "
     "like append, pop, and sort to modify it."),
    ("What is a dictionary in Python?",
     "A dictionary maps keys to values using curly braces. "
     "Keys must be immutable (strings, numbers, tuples). "
     "Lookups by key are very fast, making dicts ideal for structured data."),
    ("What is a class in Python?",
     "A class is a blueprint for creating objects. It bundles data (attributes) "
     "and behavior (methods) together. You define it with the class keyword "
     "and create instances by calling it like a function."),
    ("How do I handle errors in Python?",
     "Use try-except blocks. Put risky code in the try block and handle "
     "specific exceptions in except blocks. You can also use finally for "
     "cleanup code that always runs."),
    ("What is a decorator in Python?",
     "A decorator is a function that wraps another function to modify its "
     "behavior. You apply it with the @ symbol above the function definition. "
     "Common examples include @property and @staticmethod."),
    ("What is a list comprehension?",
     "A list comprehension is a concise way to create lists. "
     "It looks like [expression for item in iterable if condition]. "
     "It is more readable and often faster than a regular for loop."),
    ("What is the difference between a tuple and a list?",
     "Lists are mutable — you can change their contents after creation. "
     "Tuples are immutable — once created they cannot be modified. "
     "Tuples are slightly faster and can be used as dictionary keys."),
    ("What is pip?",
     "Pip is Python's package manager. You use pip install to download and "
     "install third-party libraries from PyPI. It handles dependencies "
     "automatically and is included with modern Python."),

    # --- Machine Learning ---
    ("What is machine learning?",
     "Machine learning is a field where computers learn patterns from data "
     "instead of following hand-written rules. The model improves its "
     "predictions as it sees more examples during training."),
    ("What is a neural network?",
     "A neural network is a model made of layers of interconnected nodes. "
     "Each layer transforms its input, and the network learns by adjusting "
     "the connection weights to minimize prediction errors."),
    ("What is gradient descent?",
     "Gradient descent is the optimization algorithm that adjusts model "
     "weights to reduce the loss. It computes the gradient of the loss "
     "with respect to each weight and takes a step in the downhill direction."),
    ("What is overfitting?",
     "Overfitting happens when a model memorizes the training data instead "
     "of learning general patterns. It performs well on training data but "
     "poorly on new data. Regularization and more data help prevent it."),
    ("What is a transformer?",
     "A transformer is a neural network architecture built on self-attention. "
     "It processes all tokens in parallel, captures long-range dependencies, "
     "and is the foundation of GPT, BERT, and most modern language models."),
    ("What is attention in deep learning?",
     "Attention lets a model weigh the importance of different input tokens "
     "when producing each output. Self-attention computes relevance scores "
     "between all pairs of tokens in the sequence."),
    ("What is fine-tuning?",
     "Fine-tuning takes a pretrained model and trains it further on a "
     "smaller, task-specific dataset. The model already knows language, "
     "so it only needs to learn the new task, which requires much less data."),
    ("What is transfer learning?",
     "Transfer learning reuses knowledge from a model trained on one task "
     "to help with a different task. For example, GPT-2 learned general "
     "English, and we fine-tune it to follow our specific Q&A format."),
    ("What is tokenization?",
     "Tokenization splits text into smaller units called tokens that the "
     "model can process. GPT-2 uses byte-pair encoding, which breaks words "
     "into subword pieces based on frequency."),
    ("What is the difference between GPT and BERT?",
     "GPT is autoregressive — it predicts the next token and generates text "
     "left to right. BERT is bidirectional — it sees the whole sentence at "
     "once and is better for understanding tasks like classification."),

    # --- Math ---
    ("What is a derivative?",
     "A derivative measures how fast a function changes at a given point. "
     "It is the slope of the tangent line. In deep learning, derivatives "
     "tell us how to adjust weights to reduce the loss."),
    ("What is a matrix?",
     "A matrix is a rectangular grid of numbers arranged in rows and columns. "
     "Matrix multiplication is the core operation in neural networks, "
     "transforming inputs through layers of weights."),
    ("What is linear algebra?",
     "Linear algebra is the math of vectors, matrices, and linear "
     "transformations. It is fundamental to machine learning because "
     "neural networks are essentially chains of matrix operations."),
    ("What is probability?",
     "Probability measures how likely an event is to occur, ranging from "
     "0 (impossible) to 1 (certain). Machine learning models output "
     "probability distributions over possible predictions."),
    ("What is the chain rule in calculus?",
     "The chain rule lets you differentiate composite functions. If y depends "
     "on u and u depends on x, then dy/dx = dy/du * du/dx. "
     "Backpropagation is just the chain rule applied through the network."),
    ("What is a logarithm?",
     "A logarithm is the inverse of exponentiation. Log base b of x asks "
     "what power you raise b to in order to get x. The natural log (ln) "
     "uses base e and appears everywhere in machine learning."),
    ("What is cross-entropy?",
     "Cross-entropy measures how different two probability distributions are. "
     "In classification, it compares the model's predicted probabilities "
     "to the true labels. Lower cross-entropy means better predictions."),
    ("What is softmax?",
     "Softmax converts a vector of raw scores (logits) into a probability "
     "distribution. Each output is between 0 and 1, and they all sum to 1. "
     "It is used in the final layer of classifiers."),
    ("What is a dot product?",
     "The dot product multiplies corresponding elements of two vectors and "
     "sums the results. In attention, the dot product between query and key "
     "vectors measures how relevant two tokens are to each other."),
    ("What is backpropagation?",
     "Backpropagation computes gradients by applying the chain rule backward "
     "through the network. It tells each weight how much it contributed to "
     "the total error, so gradient descent can update it."),

    # --- General / Chat ---
    ("Hello!",
     "Hello! I'm your AI assistant, fine-tuned to help with Python, "
     "machine learning, and math. What would you like to learn about?"),
    ("Who are you?",
     "I'm a GPT-2 model that has been fine-tuned on educational Q&A data. "
     "I can help explain concepts in programming, machine learning, and "
     "math. Ask me anything!"),
    ("What can you help me with?",
     "I can explain programming concepts, machine learning fundamentals, "
     "and math topics. I'm especially good at Python, neural networks, "
     "and the math behind deep learning."),
    ("Thank you!",
     "You're welcome! I'm glad I could help. Don't hesitate to ask more "
     "questions — that's the best way to learn."),
    ("How do I get started with AI?",
     "Start by learning Python basics — variables, functions, loops. Then "
     "study the math essentials: linear algebra and calculus. After that, "
     "try building simple neural networks with PyTorch. Take it step by step."),
    ("What is GPT?",
     "GPT stands for Generative Pre-trained Transformer. It is a language "
     "model trained to predict the next token in a sequence. By pre-training "
     "on massive text data, it learns grammar, facts, and reasoning patterns."),
    ("Explain AI in simple terms.",
     "AI is about making computers perform tasks that normally require human "
     "intelligence — like understanding language, recognizing images, or making "
     "decisions. Machine learning is the most common approach today."),
    ("What is PyTorch?",
     "PyTorch is an open-source deep learning framework by Meta. "
     "It provides tensors (like NumPy arrays but with GPU support), "
     "automatic differentiation, and neural network building blocks."),
    ("What is the best way to learn programming?",
     "Write code every day, even if it's just for 15 minutes. Build small "
     "projects that interest you. Read error messages carefully — they're "
     "actually helpful. And don't be afraid to look things up."),
    ("I'm stuck on a problem.",
     "That's normal and it's actually a sign of learning! Try breaking the "
     "problem into smaller pieces. Explain the problem out loud (rubber duck "
     "debugging). Take a short break and come back with fresh eyes."),

    # --- Deeper ML Questions ---
    ("What is a loss function?",
     "A loss function measures how wrong the model's predictions are. "
     "During training, the optimizer adjusts weights to minimize this value. "
     "Common losses include cross-entropy for classification and MSE for regression."),
    ("What is a learning rate?",
     "The learning rate controls how big each weight update step is. "
     "Too high and training becomes unstable, too low and it takes forever. "
     "A good starting point is usually somewhere around 1e-4 to 1e-3."),
    ("What is dropout?",
     "Dropout randomly deactivates neurons during training with some "
     "probability. This prevents the model from relying too much on any "
     "single neuron and acts as regularization to reduce overfitting."),
    ("What is batch normalization?",
     "Batch normalization normalizes the inputs to each layer by adjusting "
     "the mean and variance across the batch. It stabilizes training, "
     "allows higher learning rates, and can speed up convergence."),
    ("What is an embedding?",
     "An embedding maps discrete items (like words or tokens) to dense "
     "continuous vectors. Similar items end up close together in the vector "
     "space. GPT-2 uses learned embeddings for its vocabulary."),
    ("What is the difference between training and inference?",
     "Training is when the model learns by adjusting its weights on data. "
     "Inference is when you use the trained model to make predictions on "
     "new inputs. Training is expensive, inference is cheap."),
    ("What is a GPU and why is it used for AI?",
     "A GPU (Graphics Processing Unit) has thousands of small cores that "
     "can do many calculations in parallel. Neural networks rely heavily on "
     "matrix math, which GPUs handle much faster than CPUs."),
    ("What is regularization?",
     "Regularization is any technique that prevents overfitting. Examples "
     "include dropout, weight decay, data augmentation, and early stopping. "
     "The goal is to make the model generalize better to new data."),
    ("What is an epoch?",
     "An epoch is one full pass through the entire training dataset. "
     "Models typically train for multiple epochs. More epochs means more "
     "learning, but too many can lead to overfitting."),
    ("What is perplexity?",
     "Perplexity measures how well a language model predicts text. "
     "It is 2 raised to the cross-entropy loss. Lower perplexity means "
     "the model is less surprised by the data and predicts it better."),

    # --- More Chat / Personality ---
    ("Tell me something interesting.",
     "GPT-2 was considered too dangerous to release in full when OpenAI "
     "first created it in 2019. Now it's freely available and we're "
     "fine-tuning it right here in this project!"),
    ("Can you write code?",
     "I can explain code concepts and help you understand how things work. "
     "For actual code generation, you'd want a model specifically trained "
     "on code, like Codex or a code-specialized version of GPT."),
    ("What makes a good programmer?",
     "Curiosity, patience, and persistence. Good programmers break problems "
     "down into smaller pieces, write clean readable code, test their work, "
     "and never stop learning. It's a skill that grows with practice."),
    ("I'm new to all of this.",
     "Welcome! Everyone starts somewhere. The fact that you're here and "
     "learning is already a great first step. Take your time, ask lots of "
     "questions, and don't compare yourself to others."),
    ("Goodbye!",
     "Goodbye! Keep learning and building things. Every expert was once a "
     "beginner. Come back anytime you have questions!"),

    # --- Extra pairs for robustness ---
    ("What is a tensor?",
     "A tensor is a multi-dimensional array — the fundamental data structure "
     "in deep learning. Scalars, vectors, and matrices are all tensors. "
     "PyTorch and TensorFlow operate primarily on tensors."),
    ("What is weight tying?",
     "Weight tying shares the same weight matrix between the input embedding "
     "layer and the output projection layer. This reduces the total parameter "
     "count and often improves language model performance."),
    ("What is layer normalization?",
     "Layer normalization normalizes activations across the feature dimension "
     "for each individual sample. It stabilizes training and is used in "
     "every transformer block, including GPT-2."),
    ("How does text generation work?",
     "The model predicts one token at a time. Given the tokens so far, "
     "it outputs a probability distribution over the vocabulary, samples "
     "the next token, appends it, and repeats. Temperature controls randomness."),
    ("What is temperature in text generation?",
     "Temperature scales the logits before softmax. Low temperature (like 0.3) "
     "makes output focused and deterministic. High temperature (like 1.5) "
     "makes it more random and creative. The default is usually around 0.7."),
]


# ======================================================================
# FORMAT DATA FOR INSTRUCTION TUNING
# ======================================================================

def format_conversation(question, answer):
    """Format a Q&A pair into the instruction-tuning template.

    Format:
        User: <question>
        Assistant: <answer><|endoftext|>
    """
    return f"User: {question}\nAssistant: {answer}{tokenizer.eos_token}"


# ======================================================================
# DATASET WITH LOSS MASKING
# ======================================================================

class FineTuneDataset(Dataset):
    """Tokenizes Q&A pairs and masks the prompt tokens in labels.

    Only the assistant's response tokens contribute to the loss.
    Prompt tokens (User: ... Assistant:) get label = -100.
    """

    def __init__(self, pairs, tokenizer, max_len=MAX_LEN):
        self.items = []

        for question, answer in pairs:
            # Encode the prompt part (everything up to and including "Assistant:")
            prompt_text = f"User: {question}\nAssistant:"
            full_text = f"{prompt_text} {answer}{tokenizer.eos_token}"

            prompt_ids = tokenizer.encode(prompt_text)
            full_ids = tokenizer.encode(full_text)

            # Truncate to max_len
            full_ids = full_ids[:max_len]
            prompt_len = min(len(prompt_ids), len(full_ids))

            # Labels: -100 on prompt tokens, real ids on response tokens
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[:max_len]

            # Pad to max_len
            pad_len = max_len - len(full_ids)
            input_ids = full_ids + [tokenizer.eos_token_id] * pad_len
            labels = labels + [-100] * pad_len

            attention_mask = [1] * len(full_ids) + [0] * pad_len

            self.items.append({
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'labels': torch.tensor(labels),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ======================================================================
# TRAINING
# ======================================================================

def train(model, dataset, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    print(f"\n  Training for {epochs} epochs on {DEVICE} ...\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for step, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"    Epoch {epoch}/{epochs}  loss={avg:.4f}")

    print(f"\n  Done! Final loss: {avg:.4f}")


# ======================================================================
# GENERATION
# ======================================================================

@torch.no_grad()
def generate_response(model, tokenizer, question, max_new_tokens=150,
                      temperature=0.7):
    """Generate a response to a user question."""
    model.eval()

    prompt = f"User: {question}\nAssistant:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the new tokens (skip the prompt)
    new_tokens = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up: stop at "User:" if the model tries to continue the conversation
    for stop in ["User:", "\nUser", "\n\nUser"]:
        if stop in response:
            response = response[:response.index(stop)]

    return response.strip()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("  Fine-Tune GPT-2 and Chat With It")
    print("=" * 60)

    # --- 1. Load pretrained model ---
    print(f"\n1. Loading pretrained {MODEL_NAME} from Hugging Face ...")
    print(f"   (first run downloads ~500MB, then it's cached)\n")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {MODEL_NAME}")
    print(f"   Parameters: {n_params:,}")
    print(f"   Device: {DEVICE}")

    # --- 2. Test BEFORE fine-tuning ---
    print("\n2. Testing BEFORE fine-tuning:\n")
    for q in ["What is a variable in Python?", "Hello!"]:
        r = generate_response(model, tokenizer, q)
        print(f"   You:  {q}")
        print(f"   Bot:  {r[:200]}")
        print()

    print("   (Notice: the base model doesn't follow our Q&A format)\n")
    input("   [Press Enter to start fine-tuning...]\n")

    # --- 3. Prepare dataset ---
    print("3. Preparing fine-tuning dataset ...")
    dataset = FineTuneDataset(QA_DATA, tokenizer, max_len=MAX_LEN)
    print(f"   {len(dataset)} training pairs  |  max_len={MAX_LEN}")

    # Show loss masking stats
    sample = dataset[0]
    masked = (sample['labels'] == -100).sum().item()
    total = sample['labels'].shape[0]
    print(f"   Loss masking: {masked}/{total} tokens masked in sample")

    # --- 4. Fine-tune ---
    print("\n4. Fine-tuning ...")
    train(model, dataset, epochs=EPOCHS)

    # --- 5. Test AFTER fine-tuning ---
    print("\n5. Testing AFTER fine-tuning:\n")
    test_questions = [
        "What is a variable in Python?",
        "What is a neural network?",
        "Hello!",
        "What is gradient descent?",
        "How does text generation work?",
    ]
    for q in test_questions:
        r = generate_response(model, tokenizer, q, temperature=0.5)
        print(f"   You:  {q}")
        print(f"   Bot:  {r}")
        print()

    print("   (The model now follows our Q&A format!)\n")
    input("   [Press Enter to start chatting...]\n")

    # --- 6. Interactive chat ---
    print("=" * 60)
    print("  INTERACTIVE CHAT  (type 'quit' to exit)")
    print("=" * 60)
    print()
    print("  Tips:")
    print("  - Ask about Python, ML, math, or general AI topics")
    print("  - Type 'temp 0.3' to change temperature (0.1 - 1.5)")
    print()

    temp = 0.7
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

        # Temperature control
        if user_input.lower().startswith('temp '):
            try:
                temp = float(user_input.split()[1])
                temp = max(0.1, min(1.5, temp))
                print(f"  Temperature set to {temp}\n")
            except (ValueError, IndexError):
                print("  Usage: temp 0.7\n")
            continue

        response = generate_response(model, tokenizer, user_input,
                                     temperature=temp)
        print(f"  Bot: {response}\n")
