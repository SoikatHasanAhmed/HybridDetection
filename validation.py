"""
Inference and validation script for the Hybrid detection model.

This script loads a pretrained checkpoint and runs validation or testing on a dataset.
Usage:
    python validation.py dataset=gen4 dataset.path=/path/to/data checkpoint=/path/to/checkpoint.ckpt
    python validation.py dataset=gen1 dataset.path=/path/to/data checkpoint=/path/to/checkpoint.ckpt use_test_set=True
"""

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path

import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module


def _remap_legacy_backbone_keys(state_dict: dict) -> dict:
    """Map older backbone checkpoint keys to current names."""
    remapped = dict(state_dict)

    # Legacy checkpoints can store lstm_3 where the current model expects lstm_1.
    for suffix in ('conv3x3_dws.weight', 'conv3x3_dws.bias', 'conv1x1.weight', 'conv1x1.bias'):
        old_k = f'mdl.backbone.lstm_3.{suffix}'
        new_k = f'mdl.backbone.lstm_1.{suffix}'
        if new_k not in remapped and old_k in remapped:
            remapped[new_k] = remapped[old_k]

    # Legacy checkpoints can store ann_features_{1,2}_2.0 while current uses ann_features_{1,2}.1.
    for block in ('1', '2'):
        for suffix in (
            'conv.conv.weight',
            'conv.norm.weight',
            'conv.norm.bias',
            'conv.norm.running_mean',
            'conv.norm.running_var',
            'conv.norm.num_batches_tracked',
        ):
            old_k = f'mdl.backbone.ann_features_{block}_2.0.{suffix}'
            new_k = f'mdl.backbone.ann_features_{block}.1.{suffix}'
            if new_k not in remapped and old_k in remapped:
                remapped[new_k] = remapped[old_k]

    return remapped


@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------

    module = fetch_model_module(config=config)

    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = _remap_legacy_backbone_keys(state_dict)

    model_keys = set(module.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    module.load_state_dict(filtered_state, strict=config.checkpoint_load_strict)

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = [ModelSummary(max_depth=2)]

    # ---------------------
    # Validation
    # ---------------------

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=config.training.precision,
        move_metrics_to_cpu=False,
    )
    with torch.inference_mode():
        if config.use_test_set:
            trainer.test(model=module, datamodule=data_module)
        else:
            trainer.validate(model=module, datamodule=data_module)


if __name__ == '__main__':
    main()
