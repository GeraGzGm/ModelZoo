from math import inf
from typing import Optional

import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from ..base_train import BaseTraining, TrainRegistry
from ..inference import Results
from utils import Metrics

@TrainRegistry.register("Classification")
class Classification(BaseTraining):
    def __init__(self, config, out_dir = None, model_path = None, device = "cuda", *kwargs):
        super().__init__(config, out_dir, model_path, device)

    def __call__(self, inference_transforms: Optional[list] = None, classes: Optional[list] = None, mode: str = "train"):
        match mode:
            case "train":
                self.board = SummaryWriter(self.out_dir + "logs/")
                self._init_tensorboard(self.out_dir + "logs/")
                self.train()
            case "inference":
                results = self.eval(self.testset, self.model_path)
                Results.display_results(results, inference_transforms, classes)
            case _:
                raise ValueError("Wrong mode type.")

    def train(self):
        self._load_model(self.model_path)

        best_val_accuracy = -inf
        pbar = tqdm(range(self.config.epochs), total = self.config.epochs)

        for epoch in pbar:
            loss, accuracy = self._train_epoch(self.trainset)

            best_val_accuracy, val_accuracy, val_loss = self._eval_valset(epoch, best_val_accuracy)            
            self.step_scheduler(val_loss)

            self._tensorboard_log((loss, accuracy), (val_loss, val_accuracy), epoch)
            pbar.set_description(f"Loss: {loss:5f}, Val_Loss: {val_loss:5f}, Acc: {accuracy:5f}, Val_Acc: {val_accuracy:5f}")

        self._save_model(f"{self.out_dir}/last_epoch.pth")

    def _train_epoch(self, trainset) -> tuple[np.mean]:
        self.model.train()

        losses = []
        accuracies = []

        for inputs, labels in trainset:
            inputs, labels = inputs.to(self.device, non_blocking = True), labels.to(self.device, non_blocking = True)
            self.optimizer.zero_grad()

            output = self.model(inputs)
            loss = self.model.compute_loss(output, labels, self.criterion)

            loss.backward()
            self.optimizer.step()

            output = self.model.get_main_output(output)
            accuracy = Metrics.Accuracy(labels, output)

            losses.append(loss.item())
            accuracies.append(accuracy)
        return np.mean(losses), np.mean(accuracy)

    def eval(self, testset: DataLoader, load_path: Optional[str] = None) -> list[tuple[Tensor, Tensor, float, float], float]:
        self._load_model(load_path)

        self.model.eval()
        predictions = []
        val_losses = []

        with torch.inference_mode():
            for inputs, labels in testset:
                inputs, labels = inputs.to(self.device, non_blocking = True), labels.to(self.device, non_blocking = True)

                output = self.model(inputs)
                loss = self.criterion(output, labels)

                accuracy = Metrics.Accuracy(labels, output)
                predictions.append((inputs.cpu(), labels.cpu(), output.cpu(), accuracy))
                val_losses.append(loss.cpu())
        return predictions, np.mean(val_losses)

    def _eval_valset(self, epoch: int, best_val_accuracy: float) -> tuple[float, float]:
        val_loss = 0.0

        if self.valset:
            val_output, val_loss = self.eval(self.valset, False)
            val_accuracy = np.mean([output[3] for output in val_output])

            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_model(f"{self.out_dir}/{epoch}_{val_accuracy:0.4f}.pth")

        return best_val_accuracy, val_accuracy, val_loss

    def _tensorboard_log(self, train: tuple, val: tuple, epoch: int) -> None:
        self.board.add_scalar("Loss/Train", train[0], epoch)
        self.board.add_scalar("Accuracy/Train", train[1], epoch)
        self.board.add_scalar("Loss/Val", val[0], epoch)
        self.board.add_scalar("Accuracy/Val", val[1], epoch)