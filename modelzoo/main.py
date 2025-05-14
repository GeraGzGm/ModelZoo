from typing import Optional
from .build_config import ModelConfigs

def real_main(run_type: str, config_file: str, model_path: Optional[str], out_dir: Optional[str]):
    config = ModelConfigs(config_file).get_model_configs()
    trainer_cls = ModelConfigs.get_trainer(config.train_type)

    trainer = trainer_cls(config, out_dir, model_path)
    trainer(
        inference_transforms=config.inferece_transforms,
        classes=config.labels,
        mode=run_type.lower()
    )