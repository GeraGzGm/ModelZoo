import os
import random
import subprocess
from math import inf
from enum import Enum
from typing import Optional


import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from utils import Parameters, Metrics


class Trainer:
    BOARD_PORT = "6006"

    def __init__(self, config: Parameters, out_dir: Optional[str] = None, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model = config.model
        self.optimizer = config.optimizer
        self.criterion = config.loss_function
        self.scheduler = config.scheduler

        self.config = config

        self.trainset = config.datasets[0]
        self.valset = config.datasets[1]
        self.testset = config.datasets[2]
        self.out_dir = out_dir
        self.model_path = model_path

        self.board = SummaryWriter(out_dir + "logs/")
        self._init_tensorboard(out_dir + "logs/")
        self._move_to_device()

    def _move_to_device(self) -> None:
        if torch.cuda.is_available() and self.device == "cuda":
            self.model.to(self.device)
            self.criterion.to(self.device)
    
    def _init_tensorboard(self, out_dir: str):
        subprocess.Popen(["tensorboard", f"--logdir={out_dir}", "--port", self.BOARD_PORT])
    
    def __call__(self, inference_transforms: Optional[list] = None, classes: Optional[list] = None, mode: str = "train"):
        match mode:
            case "train":
                self.train()
            case "inference":
                results = self.eval(self.testset, self.model_path)
                Results.display_results(results, inference_transforms, classes)
            case _:
                raise ValueError("Wrong mode type.")

    def train(self):
        best_val_accuracy = -inf

        self._load_model(self.model_path)

        pbar = tqdm(range(self.config.epochs), total = self.config.epochs)

        for epoch in pbar:
            self.model.train()
            loss, accuracy = self._train_epoch(self.trainset)

            best_val_accuracy, val_accuracy, val_loss = self._eval_valset(epoch, best_val_accuracy)            
            self.step_scheduler(val_loss)

            self._tensorboard_log((loss, accuracy), (val_loss, val_accuracy), epoch)
            pbar.set_description(f"Loss: {loss:5f}, Val_Loss: {val_loss:5f}, Acc: {accuracy:5f}, Val_Acc: {val_accuracy:5f}")

        self._save_model(f"{self.out_dir}/last_epoch.pth")
    
    def _train_epoch(self, trainset: DataLoader) -> tuple[list, list]:
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

    def _eval_valset(self, epoch: int, best_val_accuracy: float) -> tuple[float, float]:
        val_loss = 0.0

        if self.valset:
            val_output, val_loss = self.eval(self.valset, False)
            val_accuracy = np.mean([output[3] for output in val_output])

            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_model(f"{self.out_dir}/{epoch}_{val_accuracy:0.4f}.pth")

        return best_val_accuracy, val_accuracy, val_loss

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

    def step_scheduler(self, val_loss: float) -> None:
        if self.scheduler:
            self.scheduler.step(val_loss)

    def _tensorboard_log(self, train: tuple, val: tuple, epoch: int) -> None:
        self.board.add_scalar("Loss/Train", train[0], epoch)
        self.board.add_scalar("Accuracy/Train", train[1], epoch)
        self.board.add_scalar("Loss/Val", val[0], epoch)
        self.board.add_scalar("Accuracy/Val", val[1], epoch)

    def _save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok = True)
        torch.save(self.model.state_dict(), path)

    def _load_model(self, load_path: str | None):
        
        if load_path:
            try:
                state_dict = torch.load(load_path, map_location = self.device)
                self.model.load_state_dict(state_dict)
            except:
                pass

class Results:

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

