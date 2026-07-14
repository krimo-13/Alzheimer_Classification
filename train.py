from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import AlzheimerCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder("dataset/train/train", transform=transform)
val_dataset = ImageFolder("dataset/val/val",transform=transform)
test_dataset = ImageFolder("dataset/test/test",transform=transform)


train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32,shuffle=False)

test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)


print("Classes :", train_dataset.classes)
print("Nombre de classes :", len(train_dataset.classes))
print("Nombre d'images d'entraînement :", len(train_dataset))
print("Nombre d'images de validation :", len(val_dataset))
print("Nombre d'images de test :", len(test_dataset))

model = AlzheimerCNN(num_classes=len(train_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
running_loss=0
num_epochs=10
for epoch in range(num_epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )
