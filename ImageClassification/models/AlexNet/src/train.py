from math import inf

import torch
from torch import nn
from tqdm import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from src.model import AlexNet
from src.data_loader import DataLoaderCIFAR10

class Metrics:
    @classmethod
    def Accuracy(cls, y_true: torch.Tensor,  outputs: torch.Tensor) -> float:
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_true).sum().item()
        return correct / y_true.size(0)


class TrainModel:
    OUT_MODEL = "./chkpt/model.pth"
    def __init__(self, model: AlexNet, 
                    epochs: int,
                    lr: float,
                    momentum: float,
                    w_decay: float,
                    device: str = "cuda"):
        
        self.model = model
        self.model.to(device)

        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = SGD(model.parameters(), lr, momentum, weight_decay = w_decay)

        self.device= device
    
    def train(self, trainset: DataLoader):
        p_bar = tqdm(range(self.epochs), total = self.epochs)
        best_loss = inf

        for epoch in p_bar:
            epoch_losses, epoch_accuracies = self._train_epoch(trainset)

            epoch_loss = sum(epoch_losses)/len(epoch_losses)
            epoch_accuracy = sum(epoch_accuracies)/len(epoch_accuracies)

            p_bar.set_description(f"Epoch: {epoch}, Loss: {epoch_loss:0.4f}, Accuracy: {epoch_accuracy:0.4f}")
            p_bar.update(1)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._save_model(f"./chkpt/{epoch}_{epoch_loss:0.4f}.pth")

        self._save_model()

    def _train_epoch(self, trainset: DataLoader):
        
        losses = []
        accuracies = []

        p_bar = tqdm(trainset, total = len(trainset))
        for inputs, labels in p_bar:
            inputs, labels = inputs.to(self.device, non_blocking = True), labels.to(self.device, non_blocking = True)

            self.optimizer.zero_grad()

            output = self.model(inputs)
            loss = self.criterion(output, labels)

            loss.backward()
            self.optimizer.step()

            accuracy = Metrics.Accuracy(labels, output)

            p_bar.set_description(f"Loss: {loss:0.4f}, Accuracy: {accuracy:0.4f}")
            p_bar.update(1)

            losses.append(loss.item())
            accuracies.append(accuracy)
        return losses, accuracies

    def _save_model(self, path: str = OUT_MODEL) -> None:
        torch.save(self.model.state_dict(), path)

    def eval(self, testset: Dataset, load: bool):
        if load:
            self.model.load_state_dict(torch.load(self.OUT_MODEL, weights_only=True))

        self.model.eval()
        results = []

        with torch.inference_mode():
            for inputs, labels in testset:
                inputs, labels = inputs.to(self.device, non_blocking = True), labels.to(self.device, non_blocking = True)

                output = self.model(inputs)
                _, predicted = torch.max(output, 1)

                results.append((inputs, labels, output))

                del inputs, labels, output, predicted

        return results