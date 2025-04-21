from math import inf
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from utils import Parameters, Metrics


class TrainModel:
    def __init__(self, config: Parameters, out_dir: str, device: str = "cuda"):
        self.model = config.model
        self.device = device
        self.optimizer = config.optimizer
        self.criterion = config.loss_function

        self.out_dir = out_dir        
        self._move_to_device()

    def _move_to_device(self) -> None:
        if torch.cuda.is_available() and self.device == "cuda":
            self.model.to(self.device)
            self.criterion.to(self.device)
    
    def train(self, trainset: DataLoader, valset: Optional[DataLoader]) -> None:
        best_loss = inf
        p_bar = tqdm(range(self.hyperparams.epochs), total = self.hyperparams.epochs)

        for epoch in p_bar:
            epoch_losses, epoch_accuracies = self._train_epoch(trainset)
            epoch_loss, epoch_accuracy = self._compute_avg_metrics(epoch_losses, epoch_accuracies)

            p_bar.set_description(f"Epoch: {epoch}, Loss: {epoch_loss:0.4f}, Accuracy: {epoch_accuracy:0.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._save_model(f"./chkpt/{epoch}_{epoch_loss:0.4f}.pth")
            p_bar.update(1)
        self._save_model()

    def _compute_avg_metrics(self, loss: list, accuracy: list) -> tuple[float, float]:
        return sum(loss)/len(loss), sum(accuracy)/len(accuracy)

    def _train_epoch(self, trainset: DataLoader) -> tuple[list, list]:
        
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
            losses.append(loss.item())
            accuracies.append(accuracy)

            p_bar.set_description(f"Loss: {loss:0.4f}, Accuracy: {accuracy:0.4f}")
            p_bar.update(1)
        return losses, accuracies

    def _save_model(self) -> None:
        torch.save(self.model.state_dict(), f"{self.out_dir}/last_model.pth")

    def eval(self, testset: DataLoader, load: bool = False):
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