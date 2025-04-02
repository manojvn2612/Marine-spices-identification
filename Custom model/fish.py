import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import PIL.Image as Image
import torch_directml as dml 

device = dml.device()
print(f"Using device: {device}")

class FishDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, -1]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load Dataset
data = pd.read_csv("fathomnet_images.csv")
unique_labels = data["label"].unique()
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}  
data["label"] = data["label"].map(label_to_idx)
num_classes = len(unique_labels)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

dataset = FishDataset(data, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train Model
epochs = 10
for epoch in range(1):
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "resnet_fish_model1.pth")
print("Model saved successfully")

# Load Model for Testing
model.load_state_dict(torch.load("resnet_fish_model.pth", map_location=device))
model.eval()
print("Model loaded successfully")

# Inference Function
def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict(image_path):
    image = preprocess(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.argmax(outputs, dim=1).item()
        print(f"Predicted Class: {idx_to_label[predicted]}")
        return idx_to_label[predicted]

# Test Images
test_images = [
    "fathomnet_data/Lampocteis_cruentiventer/images/Lampocteis_cruentiventer_31.jpg",
    "fathomnet_data/Nanomia/images/Nanomia_29.jpg",
    "fathomnet_data/Bathochordaeus/images/Bathochordaeus_3.jpg"
]

for img in test_images:
    if os.path.exists(img):
        predict(img)
    else:
        print(f"Test image not found: {img}")
