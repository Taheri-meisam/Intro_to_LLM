

import torch
import numpy as np


# Helper to pause between sections
def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 1: Tensors")
print("  The Building Blocks of Deep Learning")
print("=" * 60)


# ---------------------------------------------------------------------------
# What is a Tensor?
# ---------------------------------------------------------------------------

print("""
Lesson 1: Tensors
=================

Tensors are the foundation of everything we'll build in this course.
Think of them as arrays that can run on GPUs and track
gradients (automatic differentiation) for learning. 
essentially, how much each parameter contributed to the error.

Tensors in frameworks like PyTorch can automatically keep track of every mathematical operation performed on them. 
Then, when you ask "how should I adjust these values to reduce the error?",
      
when a neural network learns, it needs to figure out how to adjust its internal parameters (weights) to reduce its errors. 
It does this through a process called backpropagation.
Run this file and follow along. Take your time with each section. run and understand the code...

Code : python 01_tensors.py
""" )
pause()


print("""
WHAT IS A TENSOR?

A tensor is just a multi-dimensional array. You might already know
these by other names (arrays , Numpy array, ...):

  - 0D tensor: a single number (scalar)        -> 5
  - 1D tensor: a list of numbers (vector)      -> [1, 2, 3]
  - 2D tensor: a table of numbers (matrix)     -> [[1, 2], [3, 4]]
  - 3D+ tensor: stacked tables                 -> images, video, etc.

PyTorch tensors are special because:
  1. They can run on GPUs for fast computation
  2. They can track operations for automatic differentiation
  3. They're optimized for the math we need in deep learning
""")

pause()


# -----------------------
# Creating Tensors
# -----------------------

print("CREATING TENSORS")
print("-" * 40)

# The most direct way: from a Python list
data = [1, 2, 3, 4, 5]
tensor = torch.tensor(data)
print(f"From a list: {tensor}")
print(f"Shape: {tensor.shape}")
print(f"Data type: {tensor.dtype}")

pause()

# 2D tensors (matrices)
print("2D Tensor (Matrix):")
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(matrix)
print(f"Shape: {matrix.shape}  (2 rows, 3 columns)")

pause()

# Common ways to create tensors
print("Common Creation Methods:")
print()

zeros = torch.zeros(3, 4)
print(f"torch.zeros(3, 4) - all zeros:\n{zeros}\n")

ones = torch.ones(2, 3)
print(f"torch.ones(2, 3) - all ones:\n{ones}\n")

random_uniform = torch.rand(2, 2)
print(f"torch.rand(2, 2) - random values between 0 and 1:\n{random_uniform}\n")

random_normal = torch.randn(2, 2)
print(f"torch.randn(2, 2) - random values from normal distribution:\n{random_normal}")

pause()

# From NumPy (common when working with existing data)
print("From NumPy:")
np_array = np.array([1.0, 2.0, 3.0])
from_numpy = torch.from_numpy(np_array)
print(f"NumPy array: {np_array}")
print(f"Torch tensor: {from_numpy}")

pause()


# ---------------------------------------------------------------------------
# Basic Operations
# ---------------------------------------------------------------------------

print("BASIC OPERATIONS")
print("-" * 40)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print()

# Arithmetic works element-by-element
print("Element-wise operations:")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")

pause()

# Aggregation operations
print("Aggregation operations:")
print(f"a.sum()  = {a.sum()}")
print(f"a.mean() = {a.mean()}")
print(f"a.max()  = {a.max()}")
print(f"a.min()  = {a.min()}")

pause()

# Matrix multiplication
print("Matrix multiplication:")
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

print(f"A:\n{A}\n")
print(f"B:\n{B}\n")
print(f"A @ B (matrix multiplication):\n{A @ B}")
print()
print(f"A.T (transpose):\n{A.T}")

pause()


# ---------------------------------------------------------------------------
# Indexing and Slicing
# ---------------------------------------------------------------------------

print("INDEXING AND SLICING")
print("-" * 40)

t = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(f"Our tensor:\n{t}\n")

print("Indexing (getting specific elements):")
print(f"t[0]       -> {t[0]}        (first row)")
print(f"t[0, 0]    -> {t[0, 0]}           (first element)")
print(f"t[-1]      -> {t[-1]}     (last row)")
print(f"t[:, 0]    -> {t[:, 0]}        (first column)")

pause()

print("Slicing (getting ranges):")
print(f"t[0:2] (rows 0-1):\n{t[0:2]}\n")
print(f"t[:, 1:3] (columns 1-2):\n{t[:, 1:3]}\n")
print(f"t[1:, 2:] (bottom-right corner):\n{t[1:, 2:]}")

pause()


# ---------------------------------------------------------------------------
# Reshaping
# ---------------------------------------------------------------------------

print("RESHAPING TENSORS")
print("-" * 40)

# Create a sequence of numbers
t = torch.arange(12)  # [0, 1, 2, ..., 11]
print(f"Original: {t}")
print(f"Shape: {t.shape}")
print()

# Reshape into different arrangements
print(f"Reshaped to (3, 4):\n{t.reshape(3, 4)}\n")
print(f"Reshaped to (4, 3):\n{t.reshape(4, 3)}\n")
print(f"Reshaped to (2, 2, 3):\n{t.reshape(2, 2, 3)}")

pause()

# Squeeze and unsqueeze
print("Squeeze and Unsqueeze:")
print("(These add or remove dimensions of size 1)")
print()

t = torch.tensor([[1, 2, 3]])  # Shape: (1, 3)
print(f"Original shape: {t.shape}")
print(f"After squeeze: {t.squeeze().shape}")  # Removes the dimension of size 1

t = torch.tensor([1, 2, 3])  # Shape: (3,)
print(f"\nOriginal shape: {t.shape}")
print(f"After unsqueeze(0): {t.unsqueeze(0).shape}")  # Adds dimension at position 0

pause()


# ---------------------------------------------------------------------------
# Device Management (CPU vs GPU)
# ---------------------------------------------------------------------------


print("""
A GPU can perform thousands of matrix operations simultaneously, 
      while a CPU handles them one at a time (roughly speaking). 
Training a neural network involves millions of matrix multiplications,
      so a GPU can turn hours of training into minutes.
The problem is that CPU and GPU have separate memory. 
      A tensor living on the CPU can't directly interact with a tensor on the GPU, you'd get an error.
So you need to be deliberate about where your tensors live.
 """)
print("DEVICE MANAGEMENT")
print("-" * 40)

print(f"CUDA (GPU) available: {torch.cuda.is_available()}")

# The standard pattern for device-agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a tensor on the appropriate device
tensor = torch.randn(3, 3).to(device)
print(f"Tensor device: {tensor.device}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Moving tensors between devices
    cpu_tensor = torch.randn(2, 2)
    gpu_tensor = cpu_tensor.to('cuda')
    back_to_cpu = gpu_tensor.to('cpu')
    
    print(f"Moved to GPU: {gpu_tensor.device}")
    print(f"Moved back to CPU: {back_to_cpu.device}")
else:
    print("(No GPU available - that's fine for this course!)")

pause()


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

print("DATA TYPES")
print("-" * 40)

print("""
Common data types:
  - torch.float32 (default for decimals)
  - torch.int64 (default for integers)
  - torch.bool (True/False)
""")

# PyTorch infers types from your data
float_tensor = torch.tensor([1.0, 2.0, 3.0])
int_tensor = torch.tensor([1, 2, 3])
bool_tensor = torch.tensor([True, False, True])

print(f"Float tensor: {float_tensor.dtype}")
print(f"Int tensor: {int_tensor.dtype}")
print(f"Bool tensor: {bool_tensor.dtype}")

pause()

# Converting between types
print("Type conversion:")
t = torch.tensor([1.5, 2.7, 3.9])
print(f"Original: {t} (dtype: {t.dtype})")
print(f"To int:   {t.int()} (dtype: {t.int().dtype})")
print(f"To long:  {t.long()} (dtype: {t.long().dtype})")

pause()


# ---------------------------------------------------------------------------
# Practice Exercises
# ---------------------------------------------------------------------------

print("=" * 60)
print("  PRACTICE EXERCISES")
print("=" * 60)
print("""
Try these on your own! Solutions are below.

1. Create a 4x4 identity matrix (hint: torch.eye())

2. Create 5 evenly spaced numbers between 0 and 1 (hint: torch.linspace())

3. Create a random 3x3 matrix and find:
   a) The sum of all elements
   b) The mean of each row
   c) The maximum value

4. Create a tensor [0, 1, 2, ..., 19] and reshape it to (4, 5)
   Then extract:
   a) The second row
   b) The last column
   c) The bottom-right 2x2 corner
""")

pause()

print("SOLUTIONS")
print("-" * 40)

# 1. Identity matrix
print("1. Identity matrix:")
print(torch.eye(4))
print()

# 2. Linspace
print("2. Evenly spaced numbers:")
print(torch.linspace(0, 1, 5))
print()

# 3. Random matrix operations
print("3. Random matrix operations:")
m = torch.rand(3, 3)
print(f"Matrix:\n{m}")
print(f"Sum: {m.sum()}")
print(f"Row means: {m.mean(dim=1)}")
print(f"Max value: {m.max()}")
print()

# 4. Reshape and slice
print("4. Reshape and slice:")
t = torch.arange(20).reshape(4, 5)
print(f"Tensor:\n{t}")
print(f"Second row: {t[1]}")
print(f"Last column: {t[:, -1]}")
print(f"Bottom-right 2x2:\n{t[-2:, -2:]}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
1. Tensors are multi-dimensional arrays (like numpy arrays but better)

2. Create tensors with:
   - torch.tensor() from data
   - torch.zeros(), torch.ones(), torch.rand(), torch.randn()
   - torch.arange(), torch.linspace()

3. Operations work element-wise: a + b, a * b, etc.

4. Matrix multiplication uses @: A @ B

5. Index like Python lists: t[0], t[1:3], t[:, 0]

6. Reshape with: t.reshape(), t.view(), t.squeeze(), t.unsqueeze()

7. Move to GPU with: tensor.to('cuda')

Next up: How PyTorch computes gradients automatically!
""")

print("=" * 60)
print("  End of Lesson 1")
print("=" * 60)
