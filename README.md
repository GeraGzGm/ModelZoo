## Key Features

- **Standardized Training** for multiple architectures:
  - AlexNet
  - GoogLeNet (InceptionV1)
  - ResNet
  - VGG
  - MobileNet-V1
- **Configuration-Driven** via JSON files
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
