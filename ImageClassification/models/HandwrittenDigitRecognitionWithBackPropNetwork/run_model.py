import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from src.model import LeNet1
from src.data_loader import MNISTDataLoader
from src.train import Train


def main():
    batch_size = 256
    epochs = 10
    lr = 0.01
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    data_loader = MNISTDataLoader(transform, batch_size=batch_size)
    train_loader, test_loader = data_loader.get_loaders()
    
    # Model
    model = LeNet1()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Trainer
    trainer = Train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_one_epoch(epoch)
        test_acc = trainer.evaluate(epoch)
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Test Acc={test_acc:.2f}%")
    
    trainer.save_checkpoint("lenet1_checkpoint.pth")


if __name__ == "__main__":
    
    main()
