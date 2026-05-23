import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# ── Paths ──────────────────────────────────────────────
TEST_DIR    = "data/Dataset/Test"
MODEL_PATH  = "models/best_model.pth"

# ── Settings ───────────────────────────────────────────
BATCH_SIZE  = 32
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Data Transform ─────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ── Load Test Dataset ──────────────────────────────────
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes: {test_dataset.classes}")
print(f"Test images: {len(test_dataset)}")

# ── Load Model ─────────────────────────────────────────
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# ── Evaluation ─────────────────────────────────────────
all_preds  = []
all_labels = []

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds   = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {(i+1)*BATCH_SIZE}/{len(test_dataset)} images...")

# ── Metrics ────────────────────────────────────────────
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall    = recall_score(all_labels, all_preds)
f1        = f1_score(all_labels, all_preds)
cm        = confusion_matrix(all_labels, all_preds)

print("\n" + "="*40)
print("       EVALUATION RESULTS")
print("="*40)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print("="*40)
print(f"\nConfusion Matrix:")
print(f"  Classes: {test_dataset.classes}")
print(f"  {cm}")
print("\nDone!")