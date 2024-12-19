# testing the pre trained model for sinlge images
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Define transformations for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),          # Convert image to PyTorch tensor
])

# Load model and processor
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
id2label = model.config.id2label

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load a single image
image_path = "D:\\fer2013\\test\\fear\\PrivateTest_91810341.jpg"
image = Image.open(image_path).convert("RGB")

# Apply transformations
transformed_image = transform(image).unsqueeze(0)  # Add batch dimension
transformed_image = transformed_image.to(device)

# Process image through the model
with torch.no_grad():
    pil_image = transforms.ToPILImage()(transformed_image.squeeze(0).cpu())
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    new_order = [2, 1, 4, 6, 3, 0, 5]  # New order of classes
    logits = logits[:, new_order]
    new_id2label = {i: id2label[idx] for i, idx in enumerate(new_order)}
# Get predicted class
probs = torch.nn.functional.softmax(logits, dim=-1)
predicted_class_idx = torch.argmax(probs, dim=-1).item()
predicted_class = new_id2label[predicted_class_idx]

# Display the image and prediction
plt.imshow(image)
plt.title(f"Predicted Emotion: {predicted_class}")
plt.axis("off")
plt.show()

# Print logits and probabilities
print("Logits:", logits.cpu().numpy())
print("Probabilities:", probs.cpu().numpy())
