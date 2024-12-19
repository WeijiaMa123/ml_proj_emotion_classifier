import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

# Display the id-to-label mapping
print(f"Model ID-to-Label Mapping: {model.config.id2label}")
