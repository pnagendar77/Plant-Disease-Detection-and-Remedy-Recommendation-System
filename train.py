import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)

def get_model(num_classes, weights=ResNet50_Weights.DEFAULT, device=None):
    model = resnet50(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    if device:
        model = model.to(device)
    return model



data_dir = "C:/Users/Nagendar Goud/OneDrive/Desktop/Datasets/PlantDoc"

def train_model(data_dir, batch_size=32, epochs=10, lr=1e-4, save_path='plantdoc_resnet50.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=transform)
    test_dataset = datasets.ImageFolder(f'{data_dir}/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    print(f"Classes: {train_dataset.classes}")

if __name__ == '__main__':
    train_model(data_dir)