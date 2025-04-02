import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms  
from torch.utils.data import DataLoader
import torch_directml as dml
import pandas as pd
import PIL.Image as Image

device = dml.device()
print(f"Using device: {device}")

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0)

        # Fully Connected Layers (Dynamically Initialized)
        self.fc1 = None  # Will be initialized dynamically
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.fc4 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 512).to(device)
            self.add_module("fc1", self.fc1)  

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x

# Dataset class
class FishImage():
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]  
        label = self.data.iloc[idx, -1]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    epochs = 20
    model = CNNModel().to(device)
    print(model)

    # Load dataset
    data = pd.read_csv("fathomnet_images.csv")
    unique_labels = data["label"].unique()
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    data["label"] = data["label"].map(label_to_idx)

    transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor()])
    dataset = FishImage(data, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}] = {total_loss}")

    torch.save(model.state_dict(), "fish_species_model.pth")
    print("Model saved successfully")

    model.load_state_dict(torch.load("fish_species_model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully")

    def preprocess(image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor()])
        image = transform(image)
        return image.unsqueeze(0)

    def predict(image_path):
        image = preprocess(image_path).to(device)
        with torch.no_grad():
            outputs = model(image)
            predicted = torch.argmax(outputs, dim=1)
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
            print(f"Predicted Class: {idx_to_label[predicted.item()]}")
            return idx_to_label[predicted.item()]

    test_image = "fathomnet_data/Lampocteis_cruentiventer/images/Lampocteis_cruentiventer_0.jpg"
    predict(test_image)
    test_image = "fathomnet_data/Nanomia/images/Nanomia_0.jpg"
    predict(test_image)
    test_image = "fathomnet_data/Bathochordaeus/images/Bathochordaeus_0.jpg"
    predict(test_image)
