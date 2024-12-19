# fine tuning the model
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import get_scheduler

data_dir = "D:\\fer2013\\train"
# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load the dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Create a DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Print class-to-index mapping
print("Class-to-Index Mapping:", train_dataset.class_to_idx)

processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Scheduler for learning rate adjustment
num_training_steps = len(train_loader) * 5  # Assuming 5 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
epochs = 5  # Number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"Epoch {epoch+1}/{epochs}")
    for images, labels in tqdm(train_loader, desc="Training"):
        # Move data to device
        images, labels = images.to(device), labels.to(device)

        # Convert images to PIL and process
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
        inputs = processor(images=pil_images, return_tensors="pt").to(device)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        new_order = [2, 1, 4, 6, 3, 0, 5]  # New order of classes
        logits = logits[:, new_order]

        # Calculate loss
        loss = criterion(logits, labels)
        running_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update accuracy
        predicted = torch.argmax(logits, dim=-1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Scheduler step
    lr_scheduler.step()

    # Print epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total * 100
    print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
processor.save_pretrained("fine_tuned_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Validate the model
model.eval()
correct = 0
total = 0

test_dataset = datasets.ImageFolder(root='D:\\fer2013\\test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize per-class statistics
num_classes = len(test_dataset.classes)
class_correct = [0] * num_classes
class_total = [0] * num_classes

# Initialize lists for confusion matrix
all_labels = []
all_predictions = []
batch_index = 0
# Disable gradient computation for evaluation
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        # Move images and labels to device
        images, labels = images.to(device), labels.to(device)
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
        # Process images through the model
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        # Process images through the model
        outputs = model(**inputs)
        logits = outputs.logits
        # Define the new order of classes
        new_order = [2, 1, 4, 6, 3, 0, 5]  # New order of classes
        logits = logits[:, new_order]
        if batch_index < 3:  # 只打印前三个 batch
            print(f"Batch {batch_index + 1}:")
            print("Logits sample (first image in batch):", logits[0].cpu().numpy())
            print("Predicted class for first image:", torch.argmax(logits[0]).item())
            print("Actual label for first image:", labels[0].item())
        # Get predicted classes
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted = torch.argmax(logits, dim=-1)
        batch_index = batch_index+1
        # Update overall accuracy
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update per-class accuracy
        for label, prediction in zip(labels, predicted):
            class_total[label.item()] += 1
            if label == prediction:
                class_correct[label.item()] += 1

        # Collect labels and predictions for confusion matrix
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate total accuracy
total_accuracy = correct / total * 100
print(f"Total Accuracy: {total_accuracy:.2f}%")
#print(f"logits: {logits:.2f}%")

# Calculate per-class accuracy
print("\nPer-Class Accuracy:")
for i, class_name in enumerate(test_dataset.classes):
    if class_total[i] > 0:
        accuracy = class_correct[i] / class_total[i] * 100
        print(f"{class_name}: {accuracy:.2f}%")
    else:
        print(f"{class_name}: No samples.")

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.title("Confusion Matrix")
plt.show()


##################

model.to(device)

# Validate the model
model.eval()
correct = 0
total = 0

test_dataset = datasets.ImageFolder(root='D:\\fer2013\\filtered_test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize per-class statistics
num_classes = len(test_dataset.classes)
class_correct = [0] * num_classes
class_total = [0] * num_classes

# Initialize lists for confusion matrix
all_labels = []
all_predictions = []
batch_index = 0
# Disable gradient computation for evaluation
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        # Move images and labels to device
        images, labels = images.to(device), labels.to(device)
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
        # Process images through the model
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        # Process images through the model
        outputs = model(**inputs)
        logits = outputs.logits
        # Define the new order of classes
        new_order = [2, 1, 4, 6, 3, 0, 5]  # New order of classes
        logits = logits[:, new_order]
        if batch_index < 3:  # 只打印前三个 batch
            print(f"Batch {batch_index + 1}:")
            print("Logits sample (first image in batch):", logits[0].cpu().numpy())
            print("Predicted class for first image:", torch.argmax(logits[0]).item())
            print("Actual label for first image:", labels[0].item())
        # Get predicted classes
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted = torch.argmax(logits, dim=-1)
        batch_index = batch_index+1
        # Update overall accuracy
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update per-class accuracy
        for label, prediction in zip(labels, predicted):
            class_total[label.item()] += 1
            if label == prediction:
                class_correct[label.item()] += 1

        # Collect labels and predictions for confusion matrix
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate total accuracy
total_accuracy = correct / total * 100
print(f"Total Accuracy: {total_accuracy:.2f}%")
#print(f"logits: {logits:.2f}%")

# Calculate per-class accuracy
print("\nPer-Class Accuracy:")
for i, class_name in enumerate(test_dataset.classes):
    if class_total[i] > 0:
        accuracy = class_correct[i] / class_total[i] * 100
        print(f"{class_name}: {accuracy:.2f}%")
    else:
        print(f"{class_name}: No samples.")

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.title("Confusion Matrix")
plt.show()