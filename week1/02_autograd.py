
import torch


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 2: Automatic Differentiation")
print("  How PyTorch Computes Gradients")
print("=" * 60)

print("""
Lesson 2: Automatic Differentiation
====================================

This is where PyTorch becomes magical. Instead of computing gradients
by hand (which gets messy fast), PyTorch tracks your operations and
computes derivatives automatically.

This is the foundation of how neural networks learn.

Usage: python 02_autograd.py
      """)
pause()
# ---------------------------------------------------------------------------
# Why Do We Need Gradients?
# ---------------------------------------------------------------------------

print("""
WHY GRADIENTS MATTER

Neural networks learn by adjusting their parameters to reduce errors.
But how do we know which direction to adjust them?

Gradients tell us:
  - Which direction makes the error smaller
  - How much each parameter contributes to the error

The learning algorithm is simple:
  1. Make a prediction
  2. Calculate the error (loss)
  3. Find the gradient of the loss with respect to each parameter
  4. Nudge parameters in the direction that reduces loss
  5. Repeat

Computing gradients by hand is pain in the neck and error-prone.
PyTorch does this automatically. That's what "autograd" means.
""")

pause()


# ---------------------------------------------------------------------------
# Basic Gradient Computation
# ---------------------------------------------------------------------------

print("BASIC GRADIENT COMPUTATION")
print("-" * 40)

print("""
Let's start simple. Consider: y = x^2

The derivative dy/dx = 2x

At x = 3, the gradient should be 2 * 3 = 6

Let's verify with PyTorch:
""")

# Create a tensor and tell PyTorch to track gradients
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")

# Do some computation
y = x ** 2
print(f"y = x^2 = {y}")

# Compute the gradient
y.backward()

print(f"dy/dx = {x.grad}")
print(f"Expected: 2 * 3 = 6 ✓")

pause()


# ---------------------------------------------------------------------------
# The Computational Graph
# ---------------------------------------------------------------------------

print("THE COMPUTATIONAL GRAPH")
print("-" * 40)

print("""
When you do operations on tensors with requires_grad=True,
PyTorch builds a "computational graph" behind the scenes.

Each operation is a node. When you call .backward(),
PyTorch walks backward through this graph applying the chain rule.

Let's see a more complex example: z = (x + 2)^2 * 3
""")

x = torch.tensor(2.0, requires_grad=True)

# Build the computation step by step
a = x + 2       # a = 4
b = a ** 2      # b = 16
z = b * 3       # z = 48

print(f"x = {x.item()}")
print(f"a = x + 2 = {a.item()}")
print(f"b = a^2 = {b.item()}")
print(f"z = b * 3 = {z.item()}")

# Backward pass
z.backward()

print(f"\ndz/dx = {x.grad.item()}")

print("""
Let's verify by hand:
  z = 3(x+2)^2
  dz/dx = 3 * 2(x+2) * 1 = 6(x+2)
  At x=2: dz/dx = 6(4) = 24 
""")

pause()


# ---------------------------------------------------------------------------
# Gradients with Vectors
# ---------------------------------------------------------------------------

print("GRADIENTS WITH VECTORS")
print("-" * 40)

print("""
In real neural networks, we work with vectors and matrices.
The gradient of a scalar with respect to a vector gives us
a gradient for each element.
""")

# Vector input
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"x = {x}")

# Compute y = x^2 (element-wise)
y = x ** 2
print(f"y = x^2 = {y}")

# To call backward(), we need a scalar. Sum gives us that.
loss = y.sum()
print(f"loss = sum(y) = {loss}")

# Now we can compute gradients
loss.backward()
print(f"\nd(loss)/dx = {x.grad}")
print("(This is 2x for each element: [2, 4, 6])")

pause()


# ---------------------------------------------------------------------------
# Important: Gradients Accumulate!
# ---------------------------------------------------------------------------

print("IMPORTANT: GRADIENTS ACCUMULATE")
print("-" * 40)

print("""
One gotcha in PyTorch: gradients add up by default!
If you call backward() twice, the gradients are summed.

This is sometimes useful, but usually you want to zero them first.
""")

x = torch.tensor(2.0, requires_grad=True)

# First computation
y = x ** 2
y.backward()
print(f"After first backward: x.grad = {x.grad}")

# Second computation (gradients accumulate!)
y = x ** 2
y.backward()
print(f"After second backward: x.grad = {x.grad} (accumulated!)")

# Reset gradients
x.grad.zero_()
print(f"After zero_(): x.grad = {x.grad}")

# Fresh computation
y = x ** 2
y.backward()
print(f"After fresh backward: x.grad = {x.grad}")

print("""
Remember: Always call optimizer.zero_grad() in training loops!
""")

pause()


# ---------------------------------------------------------------------------
# Controlling Gradient Tracking
# ---------------------------------------------------------------------------

print("CONTROLLING GRADIENT TRACKING")
print("-" * 40)

print("""
Sometimes you don't want to track gradients (like during evaluation).
Here's how to control it:
""")

# Method 1: torch.no_grad() context
x = torch.tensor(3.0, requires_grad=True)

print("With gradient tracking:")
y = x ** 2
print(f"  y.requires_grad = {y.requires_grad}")

print("\nWithout gradient tracking (using no_grad):")
with torch.no_grad():
    y = x ** 2
    print(f"  y.requires_grad = {y.requires_grad}")

pause()

# Method 2: detach()
print("Using detach():")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
z = y.detach()  # z is a copy that doesn't track gradients
print(f"y.requires_grad = {y.requires_grad}")
print(f"z.requires_grad = {z.requires_grad}")

pause()


# ---------------------------------------------------------------------------
# Gradient Descent in Action
# ---------------------------------------------------------------------------

print("GRADIENT DESCENT IN ACTION")
print("-" * 40)

print("""
Let's see gradient descent actually work.

We want to find the minimum of f(x) = (x - 3)^2

This is a parabola with minimum at x = 3.
Starting from x = 10, gradient descent should find it.
""")

# Start at x = 10
x = torch.tensor(10.0, requires_grad=True)
learning_rate = 0.1

print(f"Starting at x = {x.item()}")
print(f"Goal: Find x where f(x) = (x-3)^2 is minimum")
print(f"Learning rate: {learning_rate}")
print()

for step in range(20):
    # Forward: compute f(x)
    f = (x - 3) ** 2
    
    # Backward: compute gradient
    f.backward()
    
    # Update x (gradient descent step)
    with torch.no_grad():
        x -= learning_rate * x.grad
    
    # Zero gradient for next iteration
    x.grad.zero_()
    
    if step % 4 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, f(x) = {f.item():.4f}")

print()
print(f"Final x = {x.item():.4f}")
print(f"Target was x = 3.0")

pause()


# ---------------------------------------------------------------------------
# Preview: Gradients in Neural Networks
# ---------------------------------------------------------------------------

print("PREVIEW: GRADIENTS IN NEURAL NETWORKS")
print("-" * 40)

print("""
In neural networks, the same principle applies but with matrices.

A simple layer: y = Wx + b

We want gradients with respect to W and b to update them.
""")

# Weights and bias
W = torch.randn(3, 2, requires_grad=True)  # 3 outputs, 2 inputs
b = torch.randn(3, requires_grad=True)
x = torch.randn(2)  # Input

print(f"W shape: {W.shape}")
print(f"b shape: {b.shape}")
print(f"x shape: {x.shape}")

# Forward pass
y = W @ x + b
print(f"y = Wx + b, shape: {y.shape}")

# Loss (just sum for demo)
loss = y.sum()
print(f"loss = {loss.item():.4f}")

# Backward pass
loss.backward()

print(f"\nGradients computed:")
print(f"W.grad shape: {W.grad.shape}")
print(f"b.grad shape: {b.grad.shape}")

print("""
This is exactly what happens inside neural networks:
  1. Forward pass: compute output
  2. Compute loss
  3. Backward pass: compute gradients
  4. Update weights: W = W - lr * W.grad
""")

pause()


# ---------------------------------------------------------------------------
# Practice Exercises
# ---------------------------------------------------------------------------

print("=" * 60)
print("  PRACTICE EXERCISES")
print("=" * 60)
print("""
1. Compute the gradient of f(x) = x^3 + 2x^2 - 5x at x = 2
   (Expected: f'(x) = 3x^2 + 4x - 5, so f'(2) = 12 + 8 - 5 = 15)

2. Use gradient descent to find the minimum of:
   f(x, y) = (x - 1)^2 + (y + 2)^2
   (Minimum should be at x = 1, y = -2)

3. Why do we use torch.no_grad() during evaluation?
   What would happen if we didn't?
""")

pause()

print("SOLUTIONS")
print("-" * 40)

# 1. Gradient of polynomial
print("1. Gradient of f(x) = x^3 + 2x^2 - 5x at x = 2:")
x = torch.tensor(2.0, requires_grad=True)
f = x**3 + 2*x**2 - 5*x
f.backward()
print(f"   Computed: {x.grad.item()}")
print(f"   Expected: 15")
print()

# 2. Two-variable gradient descent
print("2. Minimizing f(x,y) = (x-1)^2 + (y+2)^2:")
x = torch.tensor(5.0, requires_grad=True)
y = torch.tensor(5.0, requires_grad=True)
lr = 0.1

for i in range(50):
    f = (x - 1)**2 + (y + 2)**2
    f.backward()
    
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
    
    x.grad.zero_()
    y.grad.zero_()

print(f"   Found: x = {x.item():.4f}, y = {y.item():.4f}")
print(f"   Expected: x = 1.0, y = -2.0")
print()

# 3. Explanation
print("""3. Why torch.no_grad() during evaluation?
   - Saves memory (no need to store computation graph)
   - Faster computation
   - Prevents accidentally updating weights
   - Good practice to separate training from evaluation
""")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
1. requires_grad=True tells PyTorch to track operations

2. Call .backward() to compute gradients automatically

3. Access gradients with .grad attribute

4. Gradients accumulate! Use .zero_() or optimizer.zero_grad()

5. Use torch.no_grad() when you don't need gradients

6. Gradient descent: x = x - learning_rate * gradient

7. This is the engine that powers neural network learning!

Next up: Building neural networks with nn.Module!
""")

print("=" * 60)
print("  End of Lesson 2")
print("=" * 60)
