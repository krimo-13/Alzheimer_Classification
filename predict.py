import torch
from torchvision import transforms
from PIL import Image
from model import AlzheimerCNN

# Charger le modèle
model = AlzheimerCNN(num_classes=3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Classes dans le même ordre que ImageFolder
classes = ["Mild_Demented", "Non_Demented", "Very_Mild_Demented"]

# Transformations pour l'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Charger et préparer l'image
image = Image.open("test_image.jpg")
image = transform(image)
image = image.unsqueeze(0)  # Ajouter dimension batch

# Prédiction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    label = classes[predicted.item()]

print(f"Classe prédite : {label}")
