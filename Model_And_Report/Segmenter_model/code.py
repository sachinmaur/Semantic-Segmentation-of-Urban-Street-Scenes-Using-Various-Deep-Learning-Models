
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import fnmatch
# ----------------------------
# 1. Define Paths and Directories
# ----------------------------
dataset_path = "/export/home/sachin/dataset/archive (1)/data"
output_path = "/export/home/sachin/Out_put_MODEL_1"
train_image_path = os.path.join(dataset_path, "train", "image")
train_depth_path = os.path.join(dataset_path, "train", "depth")
train_label_path = os.path.join(dataset_path, "train", "label")
os.makedirs(output_path, exist_ok=True)
predictions_output_path = os.path.join(output_path, "predictions")
os.makedirs(predictions_output_path, exist_ok=True)
logs_path = os.path.join(output_path, "logs.txt")
data_visualization_path = os.path.join(output_path, "data_visualization")
os.makedirs(data_visualization_path, exist_ok=True)


training_output_path = os.path.join(output_path, "training_output")
log_path = os.path.join(training_output_path, "training_logs.txt")
loss_plot_path = os.path.join(training_output_path, "loss_plot.png")
model_save_path = os.path.join(training_output_path, "trained_model.pth")
evaluation_output_path = os.path.join(output_path, "evaluation")

# Create directories if they do not exist

os.makedirs(training_output_path, exist_ok=True)
os.makedirs(evaluation_output_path, exist_ok=True)





# ----------------------------
# 2. Dataset Class
# ----------------------------
class CityScapes(Dataset):
    def __init__(self, root, train=True):
        self.train = train
        self.root = os.path.expanduser(root)
        self.data_path = os.path.join(root, 'train' if train else 'val')
        self.data_len = len([f for f in os.listdir(os.path.join(self.data_path, 'image')) if f.endswith('.npy')])
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # Resize images to 224x224
        ])

        self.transform_label = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __getitem__(self, index):
    # Load the data
        image = np.load(os.path.join(self.data_path, 'image', f'{index}.npy')).astype(np.float32)
        label = np.load(os.path.join(self.data_path, 'label', f'{index}.npy')).astype(np.int64)  # Load as int64
        depth = np.load(os.path.join(self.data_path, 'depth', f'{index}.npy')).astype(np.float32)

        # Convert the image to uint8 for PIL compatibility
        image = (image * 255).astype(np.uint8)

        # Ensure label is in int64 before modifying values
        label = label.astype(np.int64)

        # Replace 255 (ignore regions) with -1 for CrossEntropyLoss
        label[label == 255] = -1  # Ensures ignore regions are properly marked

        # Convert label to uint8 before using PIL.Image (PIL does not support int64)
        label = label.astype(np.uint8)

        # Apply transformations
        image = self.transform_image(Image.fromarray(image))

        # Resize label manually and preserve class values
        label = Image.fromarray(label)
        label = label.resize((224, 224), resample=Image.NEAREST)
        label = np.array(label).astype(np.int64)  # Convert back to int64 for PyTorch compatibility

        # Ensure label values are within valid class range
        label[label > num_classes - 1] = -1  # Map out-of-range values to -1

        # Convert label to PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)

        return {
            'image': image,
            'semantic': label,  # Properly mapped to valid class range
            'depth': torch.tensor(depth, dtype=torch.float32)  # Depth remains float32
        }


    def __len__(self):
        return self.data_len



# ----------------------------
# 3. DataLoader
# ----------------------------
train_dataset = CityScapes(dataset_path, train=True)
val_dataset = CityScapes(dataset_path, train=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# ----------------------------
# 4. Segmenter Model Definition
# ----------------------------
class Segmenter(nn.Module):
    def __init__(self, num_classes):
        super(Segmenter, self).__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder.forward_features(x)
        features = features[:, 1:, :].permute(0, 2, 1).reshape(x.size(0), 768, 14, 14)
        segmentation_mask = self.decoder(features)
        return nn.functional.interpolate(segmentation_mask, scale_factor=16, mode='bilinear', align_corners=False)

num_classes = 19
model = Segmenter(num_classes)

# ----------------------------
# 5. Loss and Optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Reduce learning rate


# ----------------------------
# 6. Visualization Functions
# ----------------------------
def visualize_sample(image_path, depth_path, label_path, index, output_dir):
    img_files = sorted(os.listdir(image_path))
    depth_files = sorted(os.listdir(depth_path))
    label_files = sorted(os.listdir(label_path))

    index = min(index, len(img_files) - 1, len(depth_files) - 1, len(label_files) - 1)

    img = np.load(os.path.join(image_path, img_files[index]))
    depth = np.load(os.path.join(depth_path, depth_files[index])).squeeze()
    label = np.load(os.path.join(label_path, label_files[index])).squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("RGB Image")

    axes[1].imshow(depth, cmap='gray')
    axes[1].set_title("Depth Map")

    axes[2].imshow(label, cmap='jet')
    axes[2].set_title("Segmentation Label")

    save_path = os.path.join(output_dir, f"sample_visualization_{index}.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

# Visualize a few samples from train and val datasets
with open(os.path.join(data_visualization_path, "data_visualization_log.txt"), "w") as log_file:
    for i in range(3):
        train_sample_path = visualize_sample(train_image_path, train_depth_path, train_label_path, i, data_visualization_path)
        # val_sample_path = visualize_sample(val_image_path, val_depth_path, val_label_path, i, data_visualization_path)
        log_file.write(f"Train Sample {i} saved at: {train_sample_path}\n")
        # log_file.write(f"Val Sample {i} saved at: {val_sample_path}\n")



# ----------------------------




# 7. Training Function
# ----------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cuda'):
    model.to(device)
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    with open(logs_path, "w") as log_file:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                images = batch['image'].to(device)
                labels = batch['semantic'].to(device)
                optimizer.zero_grad()
                outputs = model(images)
                labels = nn.functional.interpolate(labels.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_loader: # Iterate through the validation dataloader (which returns dictionaries)
                    images = batch['image'].to(device) # Access the 'image' tensor from the dictionary
                    labels = batch['semantic'].to(device) # Access the 'semantic' tensor (labels) from the dictionary
                    outputs = model(images)
                    loss = criterion(outputs, labels.long()) # Calculate the loss
                    val_loss += loss.item() # Accumulate the validation loss

            train_losses.append(running_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
           
            log_file.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}\n")
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
            if val_losses[-1]<best_val_loss :
               best_val_loss=val_losses[-1]
               torch.save(model.state_dict(), model_save_path)
               print(f"✅ Best Model Saved at Epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")

    # Save loss curve
     # ✅ Save loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid()

    plt.savefig(loss_plot_path)  # ✅ Save the plot
    plt.close()

    print(f"Loss plot saved at: {loss_plot_path}")
 
# ----------------------------
# 8. Visualization Predictions
# ----------------------------
def visualize_predictions(model, val_loader, device='cuda'):
    model.to(device)
    model.eval()
    
    total_images = 0  
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            depths = batch['depth']
            labels = batch['semantic']
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            if total_images >= 10:
            	return 

            for i in range(len(images)):
                image = images[i].permute(1, 2, 0).cpu().numpy() * 255
                depth = depths[i].numpy()
                label = labels[i].numpy()
                pred = preds[i]

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(image.astype(np.uint8))
                axes[0].set_title("RGB Image")
                axes[1].imshow(depth, cmap='gray')
                axes[1].set_title("Depth Map")
                axes[2].imshow(label, cmap='jet')
                axes[2].set_title("Ground Truth")
                axes[3].imshow(pred, cmap='jet')
                axes[3].set_title("Predicted Segmentation")

                save_path = os.path.join(predictions_output_path, f"visualization_{idx}_{i}.png")
                plt.savefig(save_path)
                plt.close()
                total_images += 1 
         

# ----------------------------
# 9. Run Training and Visualization
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
visualize_predictions(model, val_loader, device=device)

import torch
import numpy as np

def evaluate_model(model, dataloader, num_classes, device='cuda'):
    model.to(device)
    model.eval()

    intersection = np.zeros(num_classes)  # True Positives (TP)
    union = np.zeros(num_classes)         # TP + FP + FN
    pixel_acc_total = 0
    pixel_acc_count = 0
    mean_acc_total = np.zeros(num_classes)
    mean_acc_count = np.zeros(num_classes)

    with torch.no_grad():
        for batch in dataloader:  # FIXED: Extract data correctly
            images = batch['image'].to(device)
            labels = batch['semantic'].to(device)  # Semantic segmentation labels

            # Ensure label dimensions are correct
            if labels.dim() == 3:  # Expecting [batch, H, W]
                labels = labels.unsqueeze(1)  # Add channel dimension -> [batch, 1, H, W]

            labels = torch.nn.functional.interpolate(labels.float(), size=(224, 224), mode='nearest').squeeze(1).to(torch.long)

            # Get predictions
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Convert logits to class predictions

            for i in range(num_classes):
                pred_mask = preds == i
                true_mask = labels == i

                intersection[i] += torch.sum(pred_mask & true_mask).item()  # True Positives
                union[i] += torch.sum(pred_mask | true_mask).item()         # True Positives + False Positives + False Negatives

                mean_acc_total[i] += torch.sum(true_mask & pred_mask).item()
                mean_acc_count[i] += torch.sum(true_mask).item()

            # Compute pixel accuracy
            correct_pixels = (preds == labels).sum().item()
            total_pixels = labels.numel()

            pixel_acc_total += correct_pixels
            pixel_acc_count += total_pixels

    # Compute IoU per class
    iou_per_class = intersection / (union + 1e-8)  # Avoid division by zero
    mean_iou = np.mean(iou_per_class)  # Compute mean IoU

    # Compute mean pixel accuracy per class
    mean_pixel_acc_per_class = mean_acc_total / (mean_acc_count + 1e-8)

    # Compute overall pixel accuracy
    pixel_accuracy = pixel_acc_total / pixel_acc_count

    # Logging results

# Logging results
    eval_file_path = os.path.join(output_path, "evaluation_results.txt")

    with open(eval_file_path, "a") as log_file:  # Fix: Append mode
        log_file.write("\n===== Evaluation Results =====\n")
        log_file.write(f"Pixel Accuracy: {pixel_accuracy:.4f}\n")
        log_file.write(f"Mean Pixel Accuracy: {np.mean(mean_pixel_acc_per_class):.4f}\n")
        log_file.write(f"Mean IoU (mIoU): {mean_iou:.4f}\n")
        log_file.write(f"IoU per class:\n")
        for i, iou in enumerate(iou_per_class):
            log_file.write(f"Class {i}: IoU {iou:.4f}\n")

    # Debugging: Check if results are written correctly
    print(f"Evaluation results saved at: {eval_file_path}")

    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean Pixel Accuracy: {np.mean(mean_pixel_acc_per_class):.4f}")
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")
    print("IoU per class:", iou_per_class)

    return pixel_accuracy, np.mean(mean_pixel_acc_per_class), mean_iou, iou_per_class

evaluate_model(model, val_loader, num_classes=19, device='cuda' if torch.cuda.is_available() else 'cpu')

