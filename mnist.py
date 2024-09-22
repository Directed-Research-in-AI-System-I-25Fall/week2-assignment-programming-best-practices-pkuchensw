# Step 2: Download and preprocess the MNIST dataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ResNetModel, ResNetConfig, ResNetForImageClassification
from PIL import Image
class GrayToRGB(object):
    def __call__(self, img):
        return img.convert("RGB")    
transform = transforms.Compose([
    GrayToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:     
        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += ((predicted %10)== labels).sum().item()
        if total % 100==0:
            print(f'Accuracy: {correct / total * 100:.2f}%')
        if total>=300:
            break

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

