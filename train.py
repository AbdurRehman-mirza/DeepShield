import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import random

#! ── Paths ──────────────────────────────────────────────
TRAIN_DIR        = "data/Dataset/Train"
VAL_DIR          = "data/Dataset/Validation"
MODEL_DIR        = "models"
CHECKPOINT_PATH  = "models/checkpoint.pth"
BEST_MODEL_PATH  = "models/best_model.pth"
os.makedirs(MODEL_DIR, exist_ok=True)

#! ── Settings ───────────────────────────────────────────
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 0.001
IMG_SIZE    = 224
TRAIN_SIZE  = 70000
VAL_SIZE    = 15000
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#! ── Data Transforms ────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

#! ── Load Datasets ──────────────────────────────────────
print("Loading dataset...")
full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
full_val   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)

random.seed(42)
train_indices = random.sample(range(len(full_train)), TRAIN_SIZE)
val_indices   = random.sample(range(len(full_val)),   VAL_SIZE)

train_dataset = Subset(full_train, train_indices)
val_dataset   = Subset(full_val,   val_indices)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes: {full_train.classes}")
print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

#! ── Model ──────────────────────────────────────────────
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

#! ── Loss & Optimizer ───────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#! ── Resume from Checkpoint if Exists ──────────────────
start_epoch    = 0
best_val_acc   = 0.0

if os.path.exists(CHECKPOINT_PATH):
    print("\n⚡ Checkpoint found! Resuming from last saved epoch...")
    checkpoint   = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch  = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]
    print(f"✅ Resuming from Epoch {start_epoch + 1} | Best Val Acc so far: {best_val_acc:.4f}\n")
else:
    print("\n🚀 Starting fresh training...\n")

#! ── Training Loop ──────────────────────────────────────
for epoch in range(start_epoch, EPOCHS):
    # Training
    model.train()
    train_loss, train_correct = 0.0, 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_acc = train_correct / len(train_dataset)

    #! Validation
    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss    += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)

    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss/len(val_loader):.4f} | Val Acc:   {val_acc:.4f}")

    #! ── Save Checkpoint After Every Epoch ──────────────
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc":    best_val_acc
    }, CHECKPOINT_PATH)
    print(f"  💾 Checkpoint saved! (Epoch {epoch+1})")

    #! ── Save Best Model ────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  ✅ Best model saved! Val Acc: {val_acc:.4f}\n")

print("\n🎉 Training Complete!")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")