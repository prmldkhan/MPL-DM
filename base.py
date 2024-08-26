from re import I
import torch
import random
import numpy as np
from torch.cuda import init
from utils import scan_checkpoint, load_checkpoint, save_checkpoint

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda')
        self.init_seed()

    def init_seed(self):
        seed = self.config.seed
        random.seed(seed)
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def set_checkpoint_model(self, cp_path, prefix):
        self.cp = scan_checkpoint(self.config.checkpoint_path + f'/model_{self.subj_idx}', prefix=prefix)
        self.state_dict= load_checkpoint(self.cp, self.device)
        self.model.load_state_dict(self.state_dict['model'])
        self.last_epoch = self.state_dict['epoch']
        self.init_optimizer()
