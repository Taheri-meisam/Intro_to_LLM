"""
Lesson 3: Building Neural Networks
===================================

Now that you understand tensors and gradients, let's build actual
neural networks. PyTorch's nn.Module makes this surprisingly simple.

By the end of this lesson, you'll know how to build any neural network
architecture you can imagine.

Usage: python 03_neural_nets.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 3: Building Neural Networks")
print("  From Layers to Models")
print("=" * 60)


# ---------------------------------------------------------------------------
# What is a Neural Network?
# ---------------------------------------------------------------------------

print("""
WHAT IS A NEURAL NETWORK?

At its core, a neural network is just a function that transforms input
to output through a series of operations:

  Input → [Layer 1] → [Layer 2] → ... → [Layer N] → Output

Each layer typically does:
  output = activation(weights @ input + bias)

The "weights" and "bias" are learnable parameters.
The "activation" adds non-linearity so we can learn complex patterns.

Let's build this step by step.
""")

pause()


# ---------------------------------------------------------------------------
# Linear Layers: The Basic Building Block
# ---------------------------------------------------------------------------

print("LINEAR LAYERS")
print("-" * 40)

print("""
The most fundamental layer is the linear (or "fully connected") layer.
It computes: y = Wx + b

  - W is a weight matrix
  - b is a bias vector
  - x is your input
  - y is the output
""")

# Create a linear layer: 4 inputs → 3 outputs
layer = nn.Linear(in_features=4, out_features=3)

print(f"Linear layer: 4 inputs → 3 outputs")
print(f"Weight shape: {layer.weight.shape}")
print(f"Bias shape: {layer.bias.shape}")

# Pass some data through
x = torch.randn(2, 4)  # Batch of 2 samples, 4 features each
y = layer(x)

print(f"\nInput shape: {x.shape}")
print(f"Output shape: {y.shape}")

pause()


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------

print("ACTIVATION FUNCTIONS")
print("-" * 40)

print("""
Without activations, stacking linear layers is pointless—
you'd just get one big linear transformation.

Activations add non-linearity, letting the network learn
complex patterns. Here are the most common ones:
""")

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {x.tolist()}")
print()

# ReLU: max(0, x) - Most popular for hidden layers
print("ReLU (Rectified Linear Unit):")
print(f"  max(0, x) = {F.relu(x).tolist()}")
print("  Simple, effective, used almost everywhere")
print()

# Sigmoid: squashes to (0, 1) - Good for probabilities
print("Sigmoid:")
print(f"  1/(1+e^-x) = {[round(v, 3) for v in torch.sigmoid(x).tolist()]}")
print("  Outputs between 0 and 1, good for binary classification")
print()

# Tanh: squashes to (-1, 1)
print("Tanh:")
print(f"  {[round(v, 3) for v in torch.tanh(x).tolist()]}")
print("  Outputs between -1 and 1, centered around zero")
print()

# Softmax: converts to probabilities that sum to 1
print("Softmax (for classification):")
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(f"  Input (logits): {logits.tolist()}")
print(f"  Output (probs): {[round(v, 3) for v in probs.tolist()]}")
print(f"  Sum = {probs.sum().item():.1f} (always sums to 1!)")

pause()


# ---------------------------------------------------------------------------
# Building Models with nn.Module
# ---------------------------------------------------------------------------

print("BUILDING MODELS WITH nn.Module")
print("-" * 40)

print("""
nn.Module is the base class for all neural networks in PyTorch.

The pattern is always the same:
  1. Inherit from nn.Module
  2. Define your layers in __init__
  3. Define how data flows through them in forward()

PyTorch handles everything else automatically!
""")


class SimpleNetwork(nn.Module):
    """A basic 2-layer neural network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # Always call this first!
        
        # Define the layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Define how data flows through the network."""
        x = self.layer1(x)
        x = F.relu(x)  # Activation after first layer
        x = self.layer2(x)
        return x


# Create an instance
model = SimpleNetwork(input_size=10, hidden_size=5, output_size=2)
print("Our network:")
print(model)

pause()

# Test it
print("Testing our network:")
x = torch.randn(3, 10)  # Batch of 3 samples
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

pause()


# ---------------------------------------------------------------------------
# Viewing Parameters
# ---------------------------------------------------------------------------

print("VIEWING MODEL PARAMETERS")
print("-" * 40)

print("All the learnable parameters in our model:")
print()

for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

pause()


# ---------------------------------------------------------------------------
# Quick Building with nn.Sequential
# ---------------------------------------------------------------------------

print("QUICK BUILDING WITH nn.Sequential")
print("-" * 40)

print("""
For simple networks, nn.Sequential is even easier—
just list your layers in order:
""")

quick_model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)

print("Sequential model:")
print(quick_model)

# It works the same way
x = torch.randn(5, 10)
output = quick_model(x)
print(f"\nInput: {x.shape} → Output: {output.shape}")

pause()


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

print("LOSS FUNCTIONS")
print("-" * 40)

print("""
Loss functions measure how wrong the model is.
Lower loss = better model.

Different tasks need different loss functions:
""")

# For regression: Mean Squared Error
print("MSE Loss (for regression):")
predictions = torch.tensor([2.5, 0.0, 2.0])
targets = torch.tensor([3.0, 0.0, 2.0])
mse_loss = nn.MSELoss()
loss = mse_loss(predictions, targets)
print(f"  Predictions: {predictions.tolist()}")
print(f"  Targets:     {targets.tolist()}")
print(f"  MSE Loss:    {loss.item():.4f}")
print()

# For classification: Cross Entropy
print("Cross Entropy Loss (for classification):")
# Model outputs "logits" (raw scores)
logits = torch.tensor([[2.0, 0.5, 0.3],   # Sample 1: high score for class 0
                       [0.1, 2.5, 0.2]])  # Sample 2: high score for class 1
labels = torch.tensor([0, 1])  # True classes
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(logits, labels)
print(f"  Logits:\n{logits}")
print(f"  True labels: {labels.tolist()}")
print(f"  CE Loss: {loss.item():.4f}")

pause()


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

print("OPTIMIZERS")
print("-" * 40)

print("""
Optimizers update model parameters to reduce loss.
They implement gradient descent and its variants.

Adam is the most popular choice—it usually just works.
""")

model = SimpleNetwork(10, 5, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Adam optimizer created!")
print(f"Learning rate: 0.001")
print(f"Parameters being optimized: {sum(p.numel() for p in model.parameters())}")

pause()


# ---------------------------------------------------------------------------
# The Training Step Pattern
# ---------------------------------------------------------------------------

print("THE TRAINING STEP")
print("-" * 40)

print("""
Every training step follows this pattern:

  1. optimizer.zero_grad()   # Clear old gradients
  2. output = model(input)   # Forward pass
  3. loss = criterion(output, target)
  4. loss.backward()         # Compute gradients
  5. optimizer.step()        # Update weights

Let's see it in action:
""")

# Setup
model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Sample data
x = torch.randn(5, 10)
y = torch.randint(0, 2, (5,))

print(f"Before training: weight[0,0] = {model.weight[0,0].item():.6f}")

# One training step
optimizer.zero_grad()        # 1. Clear gradients
output = model(x)            # 2. Forward pass
loss = criterion(output, y)  # 3. Compute loss
loss.backward()              # 4. Compute gradients
optimizer.step()             # 5. Update weights

print(f"After training:  weight[0,0] = {model.weight[0,0].item():.6f}")
print(f"Loss: {loss.item():.4f}")

pause()


# ---------------------------------------------------------------------------
# Complete Example: Binary Classifier
# ---------------------------------------------------------------------------

print("COMPLETE EXAMPLE: BINARY CLASSIFIER")
print("-" * 40)


class BinaryClassifier(nn.Module):
    """Classifies inputs as class 0 or class 1."""
    
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Outputs probability between 0 and 1
        )
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        """Get class predictions (0 or 1)."""
        with torch.no_grad():
            probs = self.forward(x)
            return (probs > 0.5).float()


classifier = BinaryClassifier(input_size=4)
print("Binary Classifier:")
print(classifier)
print()

# Test it
x = torch.randn(5, 4)
probs = classifier(x)
predictions = classifier.predict(x)

print("Sample predictions:")
for i in range(5):
    print(f"  Sample {i}: prob = {probs[i].item():.3f}, class = {int(predictions[i].item())}")

pause()


# ---------------------------------------------------------------------------
# Practice Exercise
# ---------------------------------------------------------------------------

print("=" * 60)
print("  PRACTICE EXERCISE")
print("=" * 60)
print("""
Build a classifier for the Iris dataset:
  - Input: 4 features
  - Hidden layers: 16 → 8
  - Output: 3 classes (species of iris)
  - Use ReLU activations
  - Use CrossEntropyLoss

Try it yourself, then see the solution!
""")

pause()

print("SOLUTION")
print("-" * 40)


class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation! CrossEntropyLoss expects raw logits
        return x


model = IrisClassifier()
print("Iris Classifier:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# Quick test
x = torch.randn(10, 4)
output = model(x)
print(f"Input: {x.shape} → Output: {output.shape}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
1. nn.Linear(in, out) creates a linear layer: y = Wx + b

2. Activations (ReLU, Sigmoid, etc.) add non-linearity

3. The nn.Module pattern:
   - __init__: define your layers
   - forward: define how data flows through

4. nn.Sequential is great for simple stacked architectures

5. Loss functions:
   - MSELoss for regression
   - CrossEntropyLoss for classification

6. Optimizers (like Adam) update weights to minimize loss

7. Training step: zero_grad → forward → loss → backward → step

Next up: Putting it all together in a complete training loop!
""")

print("=" * 60)
print("  End of Lesson 3")
print("=" * 60)
