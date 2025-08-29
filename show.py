import os
import argparse
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ----------------------------
# Model Definition (same as training)
# ----------------------------
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


# ----------------------------
# Utilities
# ----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes, device):
    model = BreastCancerCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    logging.info(f"✅ Model loaded from {model_path}")
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

def predict_single_image(image_path, model, transform, device, class_names):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
    return pred_class, probs, image

def risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    else:
        return "High"

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate_model(model, dataloader, device, class_names):
    y_true, y_pred, y_scores = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs[:, 1].cpu().numpy())  # Prob of class=1

    # Metrics
    logging.info("📊 Classification Report:\n" + classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    logging.info("✅ Confusion matrix saved as confusion_matrix.png")

    # ROC Curve
    if len(class_names) == 2:
        auc = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("roc_curve.png")
        logging.info(f"✅ ROC curve saved as roc_curve.png (AUC={auc:.3f})")

# ----------------------------
# Main
# ----------------------------
def main(args):
    device = get_device()
    transform = get_transforms()

    # Load datasets (for evaluation)
    test_dataset = datasets.ImageFolder(root=args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes
    num_classes = len(class_names)

    # Load model
    model = load_model(args.model_path, num_classes, device)

    # Inference on single image
    if args.image:
        pred_class, probs, img = predict_single_image(args.image, model, transform, device, class_names)
        risk = risk_level(probs[pred_class])
        logging.info(f"🔍 Prediction for {args.image}: {class_names[pred_class]} (Prob={probs[pred_class]:.4f}, Risk={risk})")
        plt.imshow(img)
        plt.title(f"Predicted: {class_names[pred_class]} ({risk})")
        plt.axis("off")
        plt.savefig("inference_result.png")
        logging.info("✅ Inference result saved as inference_result.png")

    # Evaluate on test dataset
    if args.evaluate:
        evaluate_model(model, test_loader, device, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breast Cancer Model Inference & Evaluation")
    parser.add_argument("--model_path", type=str, default="breast_cancer_cnn.pth", help="Path to trained model")
    parser.add_argument("--test_dir", type=str, default="dataset_sorted/test_dataset", help="Path to test dataset")
    parser.add_argument("--image", type=str, help="Run inference on a single image")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model on test dataset")
    args = parser.parse_args()
    main(args)
