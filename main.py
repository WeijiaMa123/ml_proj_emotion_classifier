import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Check if GPU is available
'''if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("GPU is not available. Using CPU.")'''

data_dir = "D:\\fer2013"
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

# Example: Iterate through the DataLoader
'''for images, labels in train_loader:
    print("Batch of images shape:", images.shape)  # Shape: [batch_size, channels, height, width]
    print("Batch of labels:", labels)  # Labels corresponding to the classes
    break'''

'''# plot the distribution of images
num_of_images = []
name_of_classes=[]

# Iterate over subfolders (classes)
for a_class in os.listdir(data_dir + '/train'):
    class_path = os.path.join(data_dir, 'train', a_class)
    if os.path.isdir(class_path):  # Check if it's a directory
        num_of_images.append(len(os.listdir(class_path)))
        name_of_classes.append(a_class)
plt.figure()
plt.bar(name_of_classes,num_of_images,color = "purple")
plt.title("Distribution of different emotions")
plt.xlabel("class_names")
plt.show()'''


#load model
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
#print(processor)

id2label = model.config.id2label
label2id = {label: idx for idx, label in id2label.items()}


# Load test dataset
#test_dataset = datasets.ImageFolder(root='D:\\fer2013\\test', transform=transform)
test_dataset = datasets.ImageFolder(root='D:\\fer2013\\filtered_test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Ensure model is in evaluation mode and moved to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Initialize variables for accuracy calculation
correct = 0
total = 0

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