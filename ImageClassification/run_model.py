import argparse

import torch
from torch import nn

from build_config import ModelConfigs
from train import TrainModel

def create_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog = "run_model.py",
        description = "Run any classification model."
    )
    parser.add_argument("--config_file", help = "Path where the model config file is.")
    parser.add_argument("--run_type", help = "Train or Inference")
    parser.add_argument("--out_dir", help = "Directory were the checkpoints will be stored.", default = None)
    parser.add_argument("--model_path", help = ".pth file path." , default = None)
    return parser.parse_args()

if __name__ == "__main__":
    args = create_argparse()

    config = ModelConfigs(args.config_file).get_model_configs()
    run_type = args.run_type.lower()
    out_dir = args.out_dir
    model_path = args.model_path

    run = TrainModel(config, out_dir, model_path, "cuda")

    match run_type:
        case "train":
            run()
        case "inference":
            run(mode = run_type, inference_transforms = config.inferece_transforms, classes = config.labels)
        case _:
            raise ValueError("Wrong run type")
