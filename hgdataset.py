from re import I
import numpy as np
import pandas as pd
import bisect
import torch
from torch.cuda import device
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, random_split
import random
import glob
import os

from torch.utils.data.dataset import T

class HGDataset(Dataset):
    def __init__(self, x_data, y_data, sid):

        self.x_data = torch.Tensor(x_data).unsqueeze(1)
        self.y_data = torch.Tensor(y_data).long()
        self.sid = sid
        self.in_chans = self.x_data.shape[2]
        self.input_time_length = 1125


    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        X = self.x_data[idx, 0,:,0:1125]
        y = self.y_data[idx]

        X = X.unsqueeze(0)

        return X, y

    def __Filter_class__(self, known):
        targets = self.y_data.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.y_data = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.x_data = torch.index_select(self.x_data, 0, mask)

    def __Set_class_zero__(self):
        self.y_data = torch.zeros_like(self.y_data)


class HGD_OSR(object):
    INPUT_SHAPE = (1, 22, 1125)
    environments = np.r_[0:14]
    ENVIRONMENTS = list(map(str, environments.tolist()))

    def __init__(self, root, test_envs, known, is_inference, fine=False):
        self.test_envs = test_envs
        self.train_envs = list(set(self.environments) - set(test_envs))
        self.known = known
        self.unknown = list(set(list(range(0, 4))) - set(known))

        train_set_list = []
        valid_set_list = []
        torch.manual_seed(0)
        if fine == False:
            if not is_inference:
                for i in self.train_envs:
                    train_set_x = np.load(root+f'/train_set_x_{i+1:02d}.npy')
                    test_set_x = np.load(root+f'/test_set_x{i+1:02d}.npy')
                    valid_set_x = np.load(root+f'/valid_set_x{i+1:02d}.npy')

                    train_set_y = np.load(root+f'/train_set_y{i+1:02d}.npy')
                    test_set_y = np.load(root+f'/test_set_y{i+1:02d}.npy')
                    valid_set_y = np.load(root+f'/valid_set_y{i+1:02d}.npy')


                    x_temp = np.concatenate([train_set_x,test_set_x])
                    y_temp = np.concatenate([train_set_y,test_set_y])

                    train_set_temp = HGDataset(x_data=x_temp,y_data=y_temp,sid=i)
                    train_set_temp.__Filter_class__(known=self.known)

                    valid_set_temp = HGDataset(x_data=valid_set_x,y_data=valid_set_y,sid=i)
                    valid_set_temp.__Filter_class__(known=self.known)

                    train_set_list.append(train_set_temp)
                    valid_set_list.append(valid_set_temp)


                self.train_set = torch.utils.data.ConcatDataset(train_set_list)
                self.valid_set = torch.utils.data.ConcatDataset(valid_set_list)

            test_x_list = []    
            test_y_list = []
            for i in test_envs:
                train_set_x = np.load(root+f'/train_set_x_{i+1:02d}.npy')
                test_set_x = np.load(root+f'/test_set_x{i+1:02d}.npy')
                valid_set_x = np.load(root+f'/valid_set_x{i+1:02d}.npy')

                train_set_y = np.load(root+f'/train_set_y{i+1:02d}.npy')
                test_set_y = np.load(root+f'/test_set_y{i+1:02d}.npy')
                valid_set_y = np.load(root+f'/valid_set_y{i+1:02d}.npy')

                x_temp = np.concatenate([train_set_x,valid_set_x,test_set_x])
                y_temp = np.concatenate([train_set_y,valid_set_y,test_set_y])

                test_x_list.append(x_temp)
                test_y_list.append(y_temp)

            x_temp = np.concatenate(test_x_list)
            y_temp = np.concatenate(test_y_list)
            

            self.test_set = HGDataset(x_data=x_temp,y_data=y_temp,sid=test_envs)
            self.test_set.__Filter_class__(known=self.known)

            self.outset = HGDataset(x_data=x_temp,y_data=y_temp,sid=test_envs)
            self.outset.__Filter_class__(known=self.unknown)
        else:
            train_x_list = []
            test_x_list = []
            valid_x_list = []

            train_y_list = []
            test_y_list = []
            valid_y_list = []

            for i in test_envs:
                train_set_x = np.load(root+f'/train_set_x_{i+1:02d}.npy')
                test_set_x = np.load(root+f'/test_set_x{i+1:02d}.npy')
                valid_set_x = np.load(root+f'/valid_set_x{i+1:02d}.npy')

                train_set_y = np.load(root+f'/train_set_y{i+1:02d}.npy')
                test_set_y = np.load(root+f'/test_set_y{i+1:02d}.npy')
                valid_set_y = np.load(root+f'/valid_set_y{i+1:02d}.npy')


                train_x_list.append(train_set_x)
                test_x_list.append(test_set_x)
                valid_x_list.append(valid_set_x)

                train_y_list.append(train_set_y)
                test_y_list.append(test_set_y)
                valid_y_list.append(valid_set_y)

            x_train = np.concatenate(train_x_list)
            y_train = np.concatenate(train_y_list)
            
            x_valid = np.concatenate(valid_x_list)
            y_valid = np.concatenate(valid_y_list)

            x_test = np.concatenate(test_x_list)
            y_test = np.concatenate(test_y_list)


            self.train_set = HGDataset(x_data=x_train,y_data=y_train,sid=test_envs)
            self.train_set.__Filter_class__(known=self.known)

            self.valid_set = HGDataset(x_data=x_valid,y_data=y_valid,sid=test_envs)
            self.valid_set.__Filter_class__(known=self.known)

            self.test_set = HGDataset(x_data=x_test,y_data=y_test,sid=test_envs)
            self.test_set.__Filter_class__(known=self.known)

            self.outset = HGDataset(x_data=x_test,y_data=y_test,sid=test_envs)
            self.outset.__Filter_class__(known=self.unknown)



