

import torch
torch.cuda.empty_cache()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

"""# Check device (use GPU if available, otherwise use CPU)"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""# Define paths"""

input_dir = "/export/home/sachin/dataset/archive (1)/data"
output_dir = "/export/home/sachin/model3/model3_output"

training_dir = os.path.join(input_dir, "train")
validation_dir = os.path.join(input_dir, "val")
output_training_dir = os.path.join(output_dir, "training")

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_training_dir, exist_ok=True)

output_visualization_dir = os.path.join(output_dir, "visualize_predictions")

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_training_dir, exist_ok=True)
os.makedirs(output_visualization_dir, exist_ok=True)

"""# Custom Dataset Class for CityScapes"""

class CityScapesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.sample_images = sorted(os.listdir(os.path.join(root_dir, "image")))
        self.depth_images = sorted(os.listdir(os.path.join(root_dir, "depth")))
        self.label_images = sorted(os.listdir(os.path.join(root_dir, "label")))
        self.transform = transform

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, "image", self.sample_images[idx])
        depth_path = os.path.join(self.root_dir, "depth", self.depth_images[idx])
        label_path = os.path.join(self.root_dir, "label", self.label_images[idx])

        sample_image = np.load(sample_path).astype(np.float32)
        depth_image = np.load(depth_path).astype(np.float32)
        label_image = np.load(label_path).astype(np.float32)
        label_image[label_image == -1] = 0



        if self.transform:
            sample_image = self.transform(sample_image)
            depth_image = self.transform(depth_image)
            label_image = self.transform(label_image)

        return sample_image, depth_image, label_image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 256))
])

"""# Load Pretrained SegFormer Model with Few-Shot Learning and Enhancements"""

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

"""# Modify classifier for Few-Shot Learning"""

model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_sizes[-1], 512),  # Reduce feature size
    nn.ReLU(),  # Introduce non-linearity
    nn.Dropout(0.3),  # Prevent overfitting
    nn.Linear(512, 19)  # Output layer for 19 CityScapes classes
)

"""# Unfreeze only the classifier layers for training"""

for param in model.classifier.parameters():
    param.requires_grad = True  # Allow fine-tuning of the classifier

"""# Move model to device"""

model.to(device)

"""
# Feature Extractor"""

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

"""# Few-Shot Sampling Function"""

def select_few_shot_samples(dataset, num_samples_per_class=5):
    selected_indices = []
    class_counts = {i: 0 for i in range(19)}  # CityScapes has 19 classes

    for idx in range(len(dataset)):
        _, _, label = dataset[idx]  # Get label
        unique_classes = torch.unique(label).tolist()  # Find present classes

        for cls in unique_classes:
            if class_counts[cls] < num_samples_per_class:
                selected_indices.append(idx)
                class_counts[cls] += 1
                break  # Move to the next sample

        if all(count >= num_samples_per_class for count in class_counts.values()):
            break  # Stop when enough samples are collected

    return torch.utils.data.Subset(dataset, selected_indices)

"""# Apply few-shot selection to training dataset"""

train_dataset = CityScapesDataset(training_dir, transform=transform)
val_dataset = CityScapesDataset(validation_dir, transform=transform)

few_shot_train_dataset = select_few_shot_samples(train_dataset, num_samples_per_class=80)
train_loader = DataLoader(few_shot_train_dataset, batch_size=4, shuffle=True ,num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)

"""# Visualize a few mask samples from train and val"""

def visualize_samples(loader, title="Sample Images", save_path=None):
    images, depths, labels = next(iter(loader))
    fig, axs = plt.subplots(3, 3, figsize=(10, 8))
    for i in range(3):
        axs[i, 0].imshow(images[i].permute(1, 2, 0))
        axs[i, 0].set_title(f"Image {i}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(depths[i][0], cmap='gray')
        axs[i, 1].set_title(f"Depth {i}")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(labels[i][0], cmap='gray')
        axs[i, 2].set_title(f"Label {i}")
        axs[i, 2].axis("off")

    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

visualize_samples(train_loader, "Train Samples", os.path.join(output_dir, "train_samples.png"))

visualize_samples(val_loader, "Validation Samples", os.path.join(output_dir, "val_samples.png"))

print("Visualization complete.")

"""# Define Optimizer and Loss"""



optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=255)



torch.cuda.empty_cache()

"""
# Training Function"""

def train_model(model, train_loader, val_loader, epochs=20):
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, _, labels in train_loader:
            images, labels = images.to(device).float(), labels.to(device).squeeze(1).long()
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            outputs = torch.nn.functional.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _, labels in val_loader:
                images, labels = images.to(device).float(), labels.to(device).squeeze(1).long()
                outputs = model(pixel_values=images).logits
                outputs = torch.nn.functional.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_training_dir, "best_model.pth"))
            print(f"best model save {epochs}")

    return train_losses, val_losses

"""# Train the Model"""

train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10)

"""# Save Training Logs"""

np.save(os.path.join(output_training_dir, "train_losses.npy"), np.array(train_losses))
np.save(os.path.join(output_training_dir, "val_losses.npy"), np.array(val_losses))
with open(os.path.join(output_dir, "training_log.txt"), "w") as f:
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        f.write(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")

print("Training completed successfully.")

"""# Load Best Trained Model"""

best_model_path = os.path.join(output_training_dir, "best_model.pth")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Best trained model loaded successfully.")
else:
    print("Warning: No trained model found. Proceeding with the pre-trained model.")

"""# Evaluation Function"""

def evaluate_model(model, val_loader):
    model.eval()
    iou_per_class = {}
    pixel_accuracies = []
    num_classes = 19  # Adjust based on dataset classes
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, _, labels in val_loader:
            images, labels = images.to(device).float(), labels.to(device).long()

            # Ensure labels have the same shape as outputs (remove extra dimension)
            labels = labels.squeeze(1)  # Convert [batch_size, 1, H, W] → [batch_size, H, W]

            outputs = model(pixel_values=images).logits
            outputs = torch.argmax(outputs, dim=1)  # Convert to class predictions

            # Ensure outputs and labels have the same shape
            if outputs.shape != labels.shape:
                outputs = torch.nn.functional.interpolate(outputs.unsqueeze(1).float(), size=labels.shape[-2:], mode='nearest').squeeze(1).long()

            # Compute IoU per class
            intersection = (outputs == labels) & (labels >= 0)
            union = (outputs >= 0) | (labels >= 0)

            for class_id in range(num_classes):
                class_mask = labels == class_id  # Ensure shape matches
                class_intersection = intersection[class_mask].sum().item()
                class_union = union[class_mask].sum().item()

                if class_union > 0:
                    iou_per_class[class_id] = class_intersection / class_union
                else:
                    iou_per_class[class_id] = 0.0

            total_correct += (outputs == labels).sum().item()
            total_pixels += labels.numel()

        pixel_accuracy = total_correct / total_pixels
        mean_iou = np.mean(list(iou_per_class.values()))
        mean_pixel_accuracy = np.mean(list(pixel_accuracies)) if pixel_accuracies else pixel_accuracy

    print(f"Mean IoU: {mean_iou:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}, Mean Pixel Accuracy: {mean_pixel_accuracy:.4f}")
    print("IoU per class:")
    for class_id, iou in iou_per_class.items():
        print(f"Class {class_id}: IoU = {iou:.4f}")

    return mean_iou, pixel_accuracy, mean_pixel_accuracy, iou_per_class

"""# Evaluate Model"""

iou, pixel_acc, mean_pixel_acc, iou_per_class = evaluate_model(model, val_loader)

"""# Save Evaluation Results"""

with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
    f.write(f"Mean IoU: {iou:.4f}\n")
    f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n")
    f.write(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}\n")
    f.write("IoU per class:\n")
    for class_id, class_iou in iou_per_class.items():
        f.write(f"Class {class_id}: IoU = {class_iou:.4f}\n")

print("Evaluation completed successfully and results saved.")

"""# Plot Training Loss"""

train_losses = np.load(os.path.join(output_training_dir, "train_losses.npy"))
val_losses = np.load(os.path.join(output_training_dir, "val_losses.npy"))

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "loss_vs_epochs.png"))
plt.show()

print("Loss plot saved successfully.")

"""# Prediction Visualization Function"""

import random

def visualize_predictions(model, val_loader, num_samples=5):
    model.eval()
    batch = random.choice(list(val_loader))  # Randomly select a batch
    images, _, labels = batch
    images, labels = images.to(device).float(), labels.to(device).long()

    batch_size = len(images)
    print(f"Batch size: {batch_size}, Requested num_samples: {num_samples}")

    # Ensure num_samples does not exceed batch size
    num_samples = min(num_samples, batch_size)

    with torch.no_grad():
        outputs = model(pixel_values=images).logits
        predictions = torch.argmax(outputs, dim=1)

    for i in range(num_samples):
        if i >= batch_size:
            print(f"Error: Index {i} is out of bounds for batch size {batch_size}")
            continue  # Skip invalid indices

        image_id = f"image_{i}"  # Assuming IDs are sequential, modify if necessary
        combined_path = os.path.join(output_visualization_dir, f"{image_id}_combined.png")

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(images[i].cpu().permute(1, 2, 0).numpy().squeeze())
        axs[0].set_title(f"Input Image (ID: {image_id})")
        axs[0].axis("off")

        axs[1].imshow(labels[i].cpu().numpy().squeeze(), cmap="gray")
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(predictions[i].cpu().numpy().squeeze(), cmap="gray")
        axs[2].set_title("Predicted Segmentation")
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(combined_path)
        # plt.close()
        print(f"Saved: {combined_path}")

    print("All predictions saved successfully.")

visualize_predictions(model, val_loader, num_samples=4)

