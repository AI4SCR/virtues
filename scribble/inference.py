from pathlib import Path

from omegaconf import OmegaConf

from external.virtues.utils.downstream_utils import dump_train_test_tokens
from external.virtues.utils.downstream_utils import load_virtues
from external.virtues.dataset.imc_dataset import get_imc_dataset

# %%
config_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/projects/virtues/external/virtues/configs/base_config.yaml')
assert config_path.exists()
conf = OmegaConf.load(config_path)

# %%
imc_dataset = get_imc_dataset(conf, filter_channel_names=conf.dataset.filter_channels)
mae_model = load_virtues(conf, run_name=conf.experiment.name)

save_dir = Path(f'{conf.experiment.dir}/{conf.experiment.name}/{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}')
save_dir.mkdir(parents=True, exist_ok=True)
sv_dict = dump_train_test_tokens(conf, imc_dataset, mae_model, channel_names=None,
                       save_path=save_dir,
                       mode='cls', task_level=conf.downstream.task_level)
