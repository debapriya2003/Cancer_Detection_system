import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# --------------------------
# Device Check
# --------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
device = torch.device("cuda")
print(f"Using device: {device}")

# --------------------------
# Custom Loader (ensure RGB)
# --------------------------
def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGB")

# --------------------------
# Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --------------------------
# Dataset Paths
# --------------------------
train_dir = "dataset_sorted/train_dataset"
test_dir = "dataset_sorted/test_dataset"

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform, loader=custom_loader)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform, loader=custom_loader)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --------------------------
# CNN Model Definition
# --------------------------
class BreastCancerCNN(nn.Module):
    def __init__(self, num_classes):
        super(BreastCancerCNN, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Compute fully connected layer size dynamically
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1, self.relu, self.pool,
            self.conv2, self.relu, self.pool,
            self.conv3, self.relu, self.pool,
            self.conv4, self.relu, self.pool
        )
        self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def _get_conv_output(self):
        x = torch.randn(1, 3, 224, 224)
        x = self.convs(x)
        self._to_linear = x.numel()

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# --------------------------
# Initialize Model
# --------------------------
num_classes = len(train_dataset.classes)
model = BreastCancerCNN(num_classes).to(device)
print(f"Number of classes: {num_classes}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# Training Loop
# --------------------------
num_epochs = 30
print("🔥 Training started...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("✅ Training complete!")

# --------------------------
# Save Model
# --------------------------
torch.save(model.state_dict(), "breast_cancer_cnn.pth")
print("💾 Model saved as 'breast_cancer_cnn.pth'")
