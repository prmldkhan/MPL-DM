import numpy as np
import sys
import torch
import glob
import os



def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '??')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]



