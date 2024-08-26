import os
import shutil

def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))

class AttrDict():
    def __init__(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)

def class_to_dict(c):
    return vars(c)

def merge_config_class(c1,c2):
    c1_dict = class_to_dict(c1)
    c2_dict = class_to_dict(c2)
    c1_dict.update(c2_dict)
    return AttrDict(c1_dict)





