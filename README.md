## Key Features

- **Standardized Training** for multiple architectures:
  - AlexNet
  - GoogLeNet (InceptionV1)
  - ResNet
  - VGG
  - MobileNet-V1
- **Configuration-Driven** via JSON files
```json
{
    "model": "VGG16",
    "dataset": "cifar10",
    "epochs": 30,
    "batch_size": 32,
    "inference_batch_size": 4,

    "optimizer": "sgd",
    "optimizer_kwargs": { 
        "lr": 1e-2,
        "momentum": 0.9,
        "weight_decay": 5e-4
    },
    "loss_function": "cross_entropy",
    "train_transforms": [
        {"type": "Resize", "p": 1, "height": 256, "width": 256, "interpolation": "INTER_LINEAR"},
        {"type": "HorizontalFlip", "p": 0.5},
        {"type": "RandomCrop", "height": 224, "width": 224, "p": 1},
        {"type": "Normalize", "mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616], "p": 1}
    ],
    "inference_transforms":[
        {"type": "Resize", "p": 1, "height": 224, "width": 224, "interpolation": "INTER_LINEAR"},
        {"type": "Normalize", "mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616], "p": 1}
    ],
    "train_ratio": [0.99, 0.01]
}
```


- **Extensible Design** for adding new models
- **Reproducible Experiments** with checkpointing

## Available datasets
- Cifar-10 (cifar10)

## Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/readpapers.git
cd readpapers
pip install -r requirements.txt
```

### 2. Training a Model
```bash
python ImageClassification/src/train.py \
  --run_type train \
  --config_file ImageClassification/_configs/inceptionv1.json \
  --out_dir ImageClassification/models/GoogLeNet/
```

### 3. Evaluation
```bash
python ImageClassification/src/train.py \
  --run_type inference \
  --model_path ImageClassification/models/GoogLeNet/best.pth


```
