import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from dataclasses import dataclass

@dataclass
class Config:
    epochs: int=200
    batch_size: int = 128
    lr: float = 1e-3

def get_data_loader(is_train, config):
    if is_train:
        to_tensor = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    dataset = CIFAR10('./', is_train, transform=to_tensor, download=True)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

def train(device, model, criterion, optimizer, scheduler, train_loader, writer, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss/len(train_loader)
    val_acc = correct/total
    print(f'>>> Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Acc/train', val_acc, epoch)

def test(device, model, criterion, test_loader, writer, epoch):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(dim=1)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss/len(test_loader)
    val_acc = correct/total
    print(f'>>> Epoch: {epoch}, Test Loss: {avg_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    writer.add_scalar('Loss/test', avg_loss, epoch)
    writer.add_scalar('Acc/test', val_acc, epoch)

def main():
    config = Config()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader = get_data_loader(True, config)
    val_loader = get_data_loader(False, config)

    classifier = resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.lr)
    steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    log_path = './runs/cifar10-baseline'
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    for epoch in range(1, config.epochs+1):
        train(device, classifier, criterion, optimizer, scheduler, train_loader, writer, epoch)
        test(device, classifier, criterion, val_loader, writer, epoch)
    torch.save(classifier.state_dict(), os.path.join(log_path, 'resnet18_cifar10.pth'))
    print('>>> Training Finished.')

if __name__ == '__main__':
    main()