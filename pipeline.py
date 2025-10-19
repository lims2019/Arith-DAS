import matplotlib

matplotlib.use("Agg")

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import os
import logging
import numpy as np
import random
import torch
from typing import Type

import utils
import trainer
from trainer import BaseTrainer

log_level_dict = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="configs")
def main(cfg: DictConfig) -> None:
    cfg = cfg.config_groups

    log_level = log_level_dict[cfg.experiment.kwargs.log_level]
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if cfg.experiment.kwargs.seed is not None:
        set_seed(cfg.experiment.kwargs.seed)
        logging.info(f"Set seed to {cfg.experiment.kwargs.seed}")

    experiment_prefix = cfg.experiment.kwargs.experiment_prefix
    working_dir = os.path.join(
        os.path.abspath(HydraConfig.get().runtime.output_dir),
        experiment_prefix,
    )
    os.makedirs(working_dir, exist_ok=True)

    os.chdir(working_dir)
    logging.info(f"Change working dir to {working_dir}")

    trainer_cls: Type[BaseTrainer] = getattr(trainer, cfg.trainer.name)

    cfg = OmegaConf.to_container(cfg, resolve=True)

    cfg["trainer"]["kwargs"].update(cfg["experiment"]["kwargs"])

    trainer_experiment = trainer_cls(
        **cfg["trainer"]["kwargs"],
    )

    trainer_experiment.run_experiment()


if __name__ == "__main__":
    main()
