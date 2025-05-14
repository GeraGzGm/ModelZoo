from math import inf

import numpy as np
from tqdm import tqdm

from ..base_train import BaseTraining, TrainRegistry
from ...utils import Metrics

@TrainRegistry.register("Classification")
class Classification(BaseTraining):
    def __init__(self, config, out_dir = None, model_path = None, device = "cuda", *kwargs):
        super().__init__(config, out_dir, model_path, device)

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

    def _train_epoch(self, trainset):
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