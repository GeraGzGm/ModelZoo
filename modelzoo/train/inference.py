import random
from enum import Enum
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt



class Results:

    @classmethod
    def display_results(cls, results: list, transforms: Optional[list], classes: Optional[Enum]):
        results, _ = results
        accuracy = cls.compute_accuracy([result[3] for result in results])

        batch = random.choice(results)

        fig, axes = cls.create_subplots(nrows = len(batch))
        fig.suptitle(f"TestSet Accuracy: {accuracy:0.5f}")

        mean, std = cls.extract_mean_std(transforms)
        for i in range(len(batch)):
            img = cls.denormalize_img(batch[0][i], mean, std)
            y_true = batch[1][i]
            pred_scores = torch.softmax(batch[2][i], dim = -1)

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
