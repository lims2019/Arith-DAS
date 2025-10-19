from abc import ABC, abstractmethod
from torch import nn
import numpy as np
import json
import os
import logging
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
from paretoset import paretoset
from pygmo import hypervolume


class BaseTrainer(ABC):
    def __init__(
        self,
        bit_width: int,
        encode_type: str,
        num_episodes: int,
        log_dir: str,
        build_dir: str,
        device,
        log_freq: int,
        save_freq: int,
        n_full_target_delay_processing: int,
        pareto_target: List[str],
        reference_point: list,
        **kwargs,
    ):
        self.bit_width = bit_width
        self.encode_type = encode_type
        self.num_episodes = num_episodes
        self.log_dir = log_dir
        self.build_dir = build_dir
        self.device = device
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.n_full_target_delay_processing = n_full_target_delay_processing
        self.reference_point = reference_point
        self.pareto_target = pareto_target
        self.kwargs = kwargs
        self.tb_logger = SummaryWriter(self.log_dir)

        self._cache = {}

    @abstractmethod
    def run_experiment(self):
        pass
