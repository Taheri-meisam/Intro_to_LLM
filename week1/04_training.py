"""
Lesson 4: The Complete Training Loop
=====================================

This is where everything comes together. We'll build a complete
training pipeline from scratch: data loading, training, evaluation,
and saving your model.

After this lesson, you'll know the pattern used in virtually
every deep learning project.

Usage: python 04_training.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Lesson 4: The Complete Training Loop")
print("  From Data to Trained Model")
print("=" * 60)


# ---------------------------------------------------------------------------
# The Big Picture
# ---------------------------------------------------------------------------

print("""
THE TRAINING PIPELINE

Every machine learning project follows this pattern:

  1. LOAD DATA: Get your training and validation data ready
  2. CREATE MODEL: Build your neural network architecture
  3. TRAIN: Loop through the data, updating weights
  4. EVALUATE: Check how well the model performs
  5. SAVE: Store the trained model for later use

Let's build each piece.
""")

pause()


# ---------------------------------------------------------------------------
# Step 1: Creating a Dataset
# ---------------------------------------------------------------------------

print("STEP 1: CREATING A DATASET")
print("-" * 40)

print("""
PyTorch's Dataset class is how we organize our data.
We need to define:
  - __len__: how many samples we have
  - __getitem__: how to get a single sample

Let's create a synthetic dataset for classification.
""")


def generate_data(n_samples=1000):
    """Create a simple 2-class classification dataset."""
    # Class 0: points near (-1, -1)
    class0_x = torch.randn(n_samples // 2, 2) + torch.tensor([-1.0, -1.0])
    class0_y = torch.zeros(n_samples // 2)
    
    # Class 1: points near (1, 1)
    class1_x = torch.randn(n_samples // 2, 2) + torch.tensor([1.0, 1.0])
    class1_y = torch.ones(n_samples // 2)
    
    # Combine them
    X = torch.cat([class0_x, class1_x], dim=0)
    y = torch.cat([class0_y, class1_y], dim=0).long()
    
    return X, y


class SimpleDataset(Dataset):
    """A basic PyTorch Dataset."""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create our data
X, y = generate_data(1000)
dataset = SimpleDataset(X, y)

print(f"Dataset created!")
print(f"  Total samples: {len(dataset)}")
print(f"  Feature shape: {X.shape}")
print(f"  Class 0 samples: {(y == 0).sum().item()}")
print(f"  Class 1 samples: {(y == 1).sum().item()}")

pause()


# ---------------------------------------------------------------------------
# Step 2: Train/Validation Split
# ---------------------------------------------------------------------------

print("STEP 2: TRAIN/VALIDATION SPLIT")
print("-" * 40)

print("""
We always split data into training and validation sets:
  - Training set: used to update weights
  - Validation set: used to check if we're overfitting

Never train on your validation data!
""")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

pause()


# ---------------------------------------------------------------------------
# Step 3: DataLoaders
# ---------------------------------------------------------------------------

print("STEP 3: DATALOADERS")
print("-" * 40)

print("""
DataLoaders handle:
  - Batching: group samples together
  - Shuffling: randomize order (for training)
  - Parallel loading: load data efficiently

Think of them as iterators that give you batches of data.
""")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# Look at one batch
batch_x, batch_y = next(iter(train_loader))
print(f"\nOne batch:")
print(f"  X shape: {batch_x.shape}")
print(f"  y shape: {batch_y.shape}")

pause()


# ---------------------------------------------------------------------------
# Step 4: Create the Model
# ---------------------------------------------------------------------------

print("STEP 4: CREATE THE MODEL")
print("-" * 40)


class Classifier(nn.Module):
    """A simple classifier for our 2D data."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


model = Classifier()
print("Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

pause()


# ---------------------------------------------------------------------------
# Step 5: Loss and Optimizer
# ---------------------------------------------------------------------------

print("STEP 5: LOSS AND OPTIMIZER")
print("-" * 40)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Loss function: CrossEntropyLoss")
print("Optimizer: Adam with lr=0.01")

pause()


# ---------------------------------------------------------------------------
# Step 6: Training and Validation Functions
# ---------------------------------------------------------------------------

print("STEP 6: TRAINING AND VALIDATION FUNCTIONS")
print("-" * 40)


def train_one_epoch(model, dataloader, criterion, optimizer):
    """Train the model for one epoch."""
    model.train()  # Enable dropout, etc.
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion):
    """Evaluate the model on validation data."""
    model.eval()  # Disable dropout, etc.
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients needed
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


print("""
Two key functions:

train_one_epoch:
  - model.train() enables dropout
  - Loops through batches
  - Forward → Loss → Backward → Update
  - Returns loss and accuracy

validate:
  - model.eval() disables dropout
  - torch.no_grad() saves memory
  - Only forward pass (no weight updates)
  - Returns loss and accuracy
""")

pause()


# ---------------------------------------------------------------------------
# Step 7: The Training Loop
# ---------------------------------------------------------------------------

print("STEP 7: THE TRAINING LOOP")
print("-" * 40)

# Reset model
model = Classifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 20
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

print(f"Training for {num_epochs} epochs...")
print()

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # Record history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print progress
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2%}")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}")

print()
print(f"Training complete!")
print(f"Final validation accuracy: {history['val_acc'][-1]:.2%}")

pause()


# ---------------------------------------------------------------------------
# Step 8: Making Predictions
# ---------------------------------------------------------------------------

print("STEP 8: MAKING PREDICTIONS")
print("-" * 40)

model.eval()

# Create test points
test_points = torch.tensor([
    [-2.0, -2.0],  # Should be class 0
    [2.0, 2.0],    # Should be class 1
    [0.0, 0.0],    # On the boundary
    [-1.0, 1.0],   # Mixed
])

with torch.no_grad():
    logits = model(test_points)
    probs = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)

print("Test predictions:")
for i, point in enumerate(test_points):
    p0, p1 = probs[i].tolist()
    print(f"  Point {point.tolist()} → Class {predictions[i].item()}")
    print(f"    (P(class0)={p0:.2%}, P(class1)={p1:.2%})")

pause()


# ---------------------------------------------------------------------------
# Step 9: Saving and Loading
# ---------------------------------------------------------------------------

print("STEP 9: SAVING AND LOADING")
print("-" * 40)

print("""
Always save your trained models! Training takes time.

PyTorch saves the model's "state_dict" - a dictionary
containing all the learned parameters.
""")

# Save
torch.save(model.state_dict(), 'classifier.pth')
print("Model saved to 'classifier.pth'")

# Load
new_model = Classifier()  # Create fresh model
new_model.load_state_dict(torch.load('classifier.pth', weights_only=True))
new_model.eval()
print("Model loaded into new instance!")

# Verify they're the same
with torch.no_grad():
    original = model(test_points)
    loaded = new_model(test_points)
    print(f"Outputs match: {torch.allclose(original, loaded)}")

# Clean up
import os
os.remove('classifier.pth')

pause()


# ---------------------------------------------------------------------------
# Putting It All Together
# ---------------------------------------------------------------------------

print("=" * 60)
print("  THE COMPLETE PIPELINE")
print("=" * 60)
print("""
Here's the full pattern you'll use in every project:

# 1. Load/create data
dataset = YourDataset(data)
train_data, val_data = random_split(dataset, [train_size, val_size])

# 2. Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 3. Create model
model = YourModel()

# 4. Setup training
criterion = nn.CrossEntropyLoss()  # or MSELoss, etc.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
for epoch in range(num_epochs):
    # Train
    model.train()
    for x, y in train_loader:
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            ...

# 6. Save
torch.save(model.state_dict(), 'model.pth')

This pattern scales from toy examples to state-of-the-art models!
""")

pause()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
1. DATASET: Implement __len__ and __getitem__

2. DATALOADER: Handles batching and shuffling

3. TRAIN/VAL SPLIT: Always validate on unseen data

4. TRAINING MODE: model.train() enables dropout

5. EVAL MODE: model.eval() + torch.no_grad()

6. THE LOOP: zero_grad → forward → loss → backward → step

7. SAVE/LOAD: torch.save() and load_state_dict()

You now know the complete PyTorch training pattern!

Next up: Week 1 Project - MNIST Digit Classifier!
""")

print("=" * 60)
print("  End of Lesson 4")
print("=" * 60)
