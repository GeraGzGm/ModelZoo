import argparse

import torch
from torch import nn

from build_config import ModelConfigs


def create_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog = "run_model.py",
        description = "Run any classification model."
    )
    parser.add_argument("config_file", help = "Path where the model config file is.")
    parser.add_argument("out_dir", help = "Path were the checkpoints will be stored.")
    return parser.parse_args()

if __name__ == "__main__":
    args = create_argparse()

    config = ModelConfigs(args.config_file).get_model_configs()
    out_dir = args.out_dir

    match config.run_type.lower():
        case "train":
            pass
        case "inference":
            pass
        case _:
            raise ValueError("Wrong run type")


    







#self.criterion = nn.CrossEntropyLoss().to(device)
#self.optimizer = SGD(model.parameters(), lr, momentum, weight_decay = w_decay)