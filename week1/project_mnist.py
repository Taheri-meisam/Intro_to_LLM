"""
Week 1 Project: MNIST Digit Classifier
========================================

This project ties together everything from Week 1.
You'll build a neural network that recognizes handwritten digits.

The MNIST dataset contains 70,000 images of handwritten digits (0-9).
Each image is 28x28 pixels in grayscale.

Your goal: Build a classifier that achieves >95% accuracy.

Code: python project.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def pause():
    input("\n[Press Enter to continue...]\n")


print("=" * 60)
print("  Week 1 Project: MNIST Digit Classifier")
print("  Recognizing Handwritten Digits")
print("=" * 60)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hyperparameters - feel free to experiment!
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
HIDDEN_SIZE = 128

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")


# ---------------------------------------------------------------------------
# Step 1: Load the Data
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  Step 1: Loading MNIST Dataset")
print("=" * 60)

# Transform: convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Download and load training data
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Download and load test data
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# Look at the data shape
sample_image, sample_label = train_dataset[0]
print(f"\nSample image shape: {sample_image.shape}")
print(f"Sample label: {sample_label}")

pause()


# ---------------------------------------------------------------------------
# Step 2: Visualize Some Samples
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Step 2: Exploring the Data")
print("=" * 60)

# Get a batch
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Show first 10 labels
print(f"\nFirst 10 labels: {labels[:10].tolist()}")

# Show a simple text visualization of one digit
print("\nText visualization of first digit:")
img = images[0].squeeze()  # Remove channel dimension
for row in range(0, 28, 2):  # Every other row for compact display
    line = ""
    for col in range(0, 28, 2):
        if img[row, col] > 0:
            line += "##"
        else:
            line += "  "
    print(line)
print(f"Label: {labels[0].item()}")

pause()


# ---------------------------------------------------------------------------
# Step 3: Build the Model
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Step 3: Building the Neural Network")
print("=" * 60)


class DigitClassifier(nn.Module):
    """
    A feedforward neural network for digit classification.
    
    Architecture:
        Input (784) -> Hidden (128) -> Hidden (64) -> Output (10)
    
    Each image is 28x28 = 784 pixels.
    We have 10 possible digits (0-9).
    """
    
    def __init__(self, hidden_size=128):
        super().__init__()
        
        # Layers
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten the image: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        
        # First hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation - CrossEntropyLoss handles it)
        x = self.fc3(x)
        
        return x


# Create model
model = DigitClassifier(hidden_size=HIDDEN_SIZE).to(device)

print("Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

pause()


# ---------------------------------------------------------------------------
# Step 4: Setup Training
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Step 4: Setting Up Training")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")

pause()


# ---------------------------------------------------------------------------
# Step 5: Training Functions
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Step 6: Train!
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Step 6: Training the Model")
print("=" * 60)
print()

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2%}")
    print(f"  Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2%}")
    print()

print(f"Final test accuracy: {test_acc:.2%}")

if test_acc > 0.95:
    print("Congratulations! You exceeded 95% accuracy!")
else:
    print("Try adjusting hyperparameters to reach 95%!")

pause()


# ---------------------------------------------------------------------------
# Step 7: Make Predictions
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Step 7: Making Predictions")
print("=" * 60)

model.eval()

# Get some test samples
test_images, test_labels = next(iter(test_loader))
test_images, test_labels = test_images.to(device), test_labels.to(device)

with torch.no_grad():
    outputs = model(test_images[:10])
    predictions = outputs.argmax(dim=1)

print("Predictions on 10 test samples:")
print(f"Predicted: {predictions.cpu().tolist()}")
print(f"Actual:    {test_labels[:10].cpu().tolist()}")
print(f"Correct:   {(predictions == test_labels[:10]).sum().item()}/10")

pause()


# ---------------------------------------------------------------------------
# Step 8: Analyze Errors
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Step 8: Analyzing Mistakes")
print("=" * 60)

# Find misclassified examples
model.eval()
mistakes = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                mistakes.append({
                    'image': images[i].cpu(),
                    'predicted': preds[i].item(),
                    'actual': labels[i].item()
                })

print(f"Total mistakes: {len(mistakes)} / {len(test_dataset)}")
print(f"Error rate: {len(mistakes) / len(test_dataset):.2%}")

if mistakes:
    print("\nFirst 5 mistakes:")
    for m in mistakes[:5]:
        print(f"  Predicted {m['predicted']}, was actually {m['actual']}")

pause()


# ---------------------------------------------------------------------------
# Step 9: Save the Model
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Step 9: Saving the Model")
print("=" * 60)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': test_acc,
    'hidden_size': HIDDEN_SIZE,
}, 'mnist_classifier.pth')

print("Model saved to 'mnist_classifier.pth'")

# Show how to load it
print("\nTo load this model later:")
print("""
checkpoint = torch.load('mnist_classifier.pth')
model = DigitClassifier(hidden_size=checkpoint['hidden_size'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
""")


# ---------------------------------------------------------------------------
# Challenges
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  CHALLENGES")
print("=" * 60)
print("""
Try these to deepen your understanding:

1. HYPERPARAMETER TUNING
   - Try different learning rates (0.01, 0.001, 0.0001)
   - Try different hidden sizes (64, 128, 256)
   - Try different batch sizes (32, 64, 128)
   What gives the best accuracy?

2. DEEPER NETWORK
   - Add a third hidden layer
   - Does it improve accuracy?
   - Does it train slower?

3. REGULARIZATION
   - Try different dropout rates (0.1, 0.3, 0.5)
   - Add L2 regularization (weight_decay in optimizer)
   - Does it help prevent overfitting?

4. LEARNING RATE SCHEDULE
   - Use torch.optim.lr_scheduler
   - Decrease learning rate over time
   - Does it improve final accuracy?

5. CONFUSION MATRIX
   - Which digits are most often confused?
   - 4 vs 9? 3 vs 8? 
   - Can you visualize this?
""")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  WEEK 1 COMPLETE!")
print("=" * 60)
print("""
You've completed Week 1.

What you learned:
  - Tensors: the foundation of PyTorch
  - Autograd: automatic gradient computation
  - Neural Networks: building with nn.Module
  - Training: the complete training loop
  - Project: a real classifier that works!

What you built:
  - A digit classifier with ~97% accuracy
  - Using everything from this week

Next week: Working with Text
  - How do we turn words into numbers?
  - Tokenization and embeddings
  - Building toward language models!


""")

print("=" * 60)
print("  End of Week 1")
print("=" * 60)
