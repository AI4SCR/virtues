import os
from loguru import logger
from omegaconf import OmegaConf
from jsonargparse import CLI
from pathlib import Path

from utils.utils import setup_wandb_and_config, set_seed
from dataset.imc_dataset import get_imc_dataset, get_union_imc_datasets

def main(config_path: Path = Path("configs/base_config.yaml")):
    conf = OmegaConf.load(config_path)

    cli_conf = OmegaConf.from_cli()

    if hasattr(cli_conf, 'base') and hasattr(cli_conf.base,
                                             'additional_config') and cli_conf.base.additional_config is not None:
        additional_conf = OmegaConf.load(cli_conf.base.additional_config)
        conf = OmegaConf.merge(conf, additional_conf)

    conf = OmegaConf.merge(conf, cli_conf)

    if conf.dataset.union_list is not None and conf.dataset.filter_channels is not None:
        raise ValueError("Cannot use multiple datasets with filter channels at the same time")

    os.makedirs(conf.experiment.dir, exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}', exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints', exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}/wandb', exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}/logs', exist_ok=True)

    logger.add(f'{conf.experiment.dir}/{conf.experiment.name}/logs/train.log')

    logger.info(f"Starting experiment {conf.experiment.name}")

    conf = setup_wandb_and_config(conf)

    if conf.dataset.filter_channels is not None and isinstance(conf.dataset.filter_channels, str):
        conf.dataset.filter_channels = [conf.dataset.filter_channels]
    logger.info(OmegaConf.to_yaml(conf))
    if conf.dataset.filter_channels is not None:
        logger.info(f'Using {len(conf.dataset.filter_channels)} genes')
        logger.info(f'Gene set: {conf.dataset.filter_channels}')

    set_seed(conf.training.seed)

    if conf.dataset.union_list is None:
        imc_dataset = get_imc_dataset(conf, filter_channel_names=conf.dataset.filter_channels)
    else:
        imc_dataset = get_union_imc_datasets(conf, conf.dataset.union_list, filter_channel_names=None)
    imc_dataset.create_random_crops_dir()


if __name__ == "__main__":
    CLI(main, as_positional=False)