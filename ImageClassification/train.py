import os
import random
from math import inf
from enum import Enum
from typing import Optional
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from utils import Parameters, Metrics


class TrainModel:
    def __init__(self, config: Parameters, out_dir: Optional[str] = None, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model = config.model
        self.optimizer = config.optimizer
        self.criterion = config.loss_function

        self.config = config

        self.trainset = config.datasets[0]
        self.valset = config.datasets[1]
        self.testset = config.datasets[2]

        self.out_dir = out_dir
        self.model_path = model_path
        self._move_to_device()

    def _move_to_device(self) -> None:
        if torch.cuda.is_available() and self.device == "cuda":
            self.model.to(self.device)
            self.criterion.to(self.device)
    
    def __call__(self, inference_transforms: Optional[list] = None, classes: Optional[list] = None, mode: str = "train"):
        match mode:
            case "train":
                self.train()
            case "inference":
                results = self.eval(self.testset, self.model_path)
                Results.display_results(results, inference_transforms, classes)
            case _:
                raise ValueError("Wrong mode type.")

    def train(self) -> None:
        best_val_accuracy = -inf
        p_bar_train = tqdm(range(self.config.epochs), total = self.config.epochs)

        for epoch in p_bar_train:
            epoch_losses, epoch_accuracies = self._train_epoch(self.trainset)
            epoch_loss, epoch_accuracy = self._compute_avg_metrics(epoch_losses, epoch_accuracies)

            description = f"Epoch: {epoch}, Loss: {epoch_loss:0.4f}, Accuracy: {epoch_accuracy:0.4f}"

            accuracy, best_val_accuracy = self._eval_valset(epoch, best_val_accuracy)

            p_bar_train.set_description(description + accuracy)
            p_bar_train.update(1)
        p_bar_train.clear()
        self._save_model(f"{self.out_dir}/last_epoch.pth")

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

    def _compute_avg_metrics(self, loss: list, accuracy: list) -> tuple[float, float]:
        """Returns loss average and accuracy."""
        loss_avg = sum(loss)/len(loss) if loss else None
        accuracy_avg = sum(accuracy)/len(accuracy) if accuracy else None
        return loss_avg, accuracy_avg

    def _eval_valset(self, epoch: int, best_val_accuracy: float) -> str:
        description = f""

        if self.valset:
            val_output = self.eval(self.valset, False)
            _, val_accuracy = self._compute_avg_metrics([], [data[3] for data in val_output])

            description = f" Val_Accuracy: {val_accuracy: 0.4f}"
            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_model(f"{self.out_dir}/{epoch}_{val_accuracy:0.4f}.pth")

        return description, best_val_accuracy

    def eval(self, testset: DataLoader, load_path: Optional[str] = None) -> list[tuple[Tensor, Tensor, float, float]]:
        if load_path:
            self.model.load_state_dict(torch.load(load_path, weights_only=True))

        self.model.eval()
        predictions = []

        p_bar_eval = tqdm(testset, total = len(testset))

        with torch.inference_mode():
            for inputs, labels in p_bar_eval:
                inputs, labels = inputs.to(self.device, non_blocking = True), labels.to(self.device, non_blocking = True)

                output = self.model(inputs)

                accuracy = Metrics.Accuracy(labels, output)
                predictions.append((inputs.cpu(), labels.cpu(), output.cpu(), accuracy))
        return predictions

    def _save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok = True)
        torch.save(self.model.state_dict(), path)

class Results:
    def __init__(self):
        pass

    @classmethod
    def display_results(cls, results: list, transforms: Optional[list], classes: Optional[Enum]):
        accuracy = cls.compute_accuracy([result[3] for result in results])
        print(f"TestSet Accuracy: {accuracy}")

        batch = random.choice(results)
        _, axes = cls.create_subplots(nrows = len(batch))

        mean, std = cls.extract_mean_std(transforms)
        for i in range(len(batch)):
            img = cls.denormalize_img(batch[0][i], mean, std)
            y_true = batch[1][i]
            pred_scores = batch[2][i]

            img = (img.permute(1,2,0) * 255).type(torch.uint8)
            y_pred = torch.argmax(pred_scores)
            pred_scores, preds_class = cls.get_classes(pred_scores, classes)

            cls._plot_image(axes[i, 0], img, y_true, y_pred, classes)
            cls._plot_bar(axes[i, 1], preds_class)

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compute_accuracy(accuracy: list) -> float:
        return sum(accuracy)/len(accuracy)
    
    @staticmethod
    def create_subplots(nrows: int, figsize: tuple = (10,12)):
        return plt.subplots(nrows = nrows, ncols = 2, figsize = figsize)

    @staticmethod
    def extract_mean_std(transforms: list[dict]) -> tuple[list, list]:
        for transform in transforms:
            if (mean := transform.get("mean", None)) and (std := transform.get("std", None)):
                return mean, std
        return None, None

    @staticmethod
    def denormalize_img(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        mean_tensor = torch.tensor(mean).view(3, 1, 1)
        std_tensor = torch.tensor(std).view(3, 1, 1)
        return  (img * std_tensor) + mean_tensor
    
    @staticmethod
    def get_classes(scores: torch.Tensor, classes: Enum) -> tuple[torch.Tensor, dict]:
        scores = torch.softmax(scores, 0)
        preds_class = {classes.get_key(pred): float(scores[pred]) for pred in range(len(scores))}
        preds_class = dict(sorted(preds_class.items(), key = lambda x: x[1], reverse = True))
        return scores, preds_class
    
    @staticmethod
    def _plot_image(axes, img: np.ndarray, y_true, y_pred, classes) -> None:
        axes.imshow(img)
        axes.axis("off")
        axes.set_title(f"{ classes.get_key(int(y_true)) } \n Pred: { classes.get_key(int(y_pred)) }")

    @staticmethod
    def _plot_bar(axes, preds_class) -> None:
        axes.barh(list(preds_class.keys())[:5], list(preds_class.values())[:5])
        axes.set_xlim(0, 1)

