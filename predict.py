import os
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt

def imshow(img, idx, root, target_label=None, pred_label=None):
    path = root + f'/example_{idx}.png'
    img = img.numpy() # img: tensor(CxHxW)
    plt.imshow(np.transpose(img, (1,2,0)))
    
    title = ''
    if target_label is not None:
        title += f'Target: {target_label}'
    if pred_label is not None:
        title += f'|Pred: {pred_label}'
    plt.title(title)
    plt.savefig(path)
    plt.show()


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    to_tensor = transforms.Compose([transforms.ToTensor()])
    testset = CIFAR10('./', False, transform=to_tensor, download=False)
    classes = testset.classes

    classifier = resnet18(num_classes=10).to(device)
    classifier.load_state_dict(torch.load('./runs/cifar10-baseline.pth', map_location=device))
    classifier.eval()
    
    root = './images'
    os.makedirs(root, exist_ok=True)
    
    for i, (image, label) in enumerate(iter(testset)):
        image = image.to(device)
        pred_labels = classifier(image.unsqueeze(0))
        _, predicted = pred_labels.max(dim=1)
        imshow(image.cpu(), i, root, classes[label], classes[predicted.item()])
        if i>2:
            break