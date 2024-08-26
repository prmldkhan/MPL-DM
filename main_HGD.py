import os
from resnet_1d_18 import Resnet1d
from hgdataset import HGD_OSR
import random
import numpy as np
import argparse
import models
from train_eval import *
import train_eval
import utils
import json
import pandas as pd
from env import AttrDict, build_env
from utils import scan_checkpoint, load_checkpoint, save_checkpoint
from base import BaseTrainer

from ARPL.core import evaluation
from metric import calculate_nll, compute_aupr, compute_oscr, eval_ece


class MPLDM_Trainer(BaseTrainer):
    def __init__(self, config, subj_idx):
        #init model
        self.config = config
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda')
        self.init_seed()
        
        
        self.subj_idx = subj_idx
        self.is_inference = False

        seed = self.config.seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.init_checkpoint()
        self.init_dataset()

        self.init_seed()
        
        embedding_net = Resnet1d(self.config.n_class, self.config.n_ch, self.config.n_time)
        self.model = models.FcClfNet_MPLDM(embedding_net, config)

        mb_params = utils.param_size(self.model)

        print(f"Model size = {mb_params:.4f} MB")
        self.model.cuda(device=self.device)
       
        self.train_func = train_eval.train_mpldm
        self.eval_func = train_eval.eval_mpldm

        if self.cp is None:
            self.state_dict = None
            self.last_epoch = -1
        else:
            self.state_dict= load_checkpoint(self.cp, self.device)
            self.model.load_state_dict(self.state_dict['model'])
            self.last_epoch = self.state_dict['epoch']
        self.init_optimizer()

    def init_checkpoint(self):
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        os.makedirs(self.config.checkpoint_path + f'/model_{self.subj_idx}', exist_ok=True)
        print("checkpoints directory : ", self.config.checkpoint_path)
        if os.path.isdir(self.config.checkpoint_path) and self.config.use_checkpoint:
            self.cp = scan_checkpoint(self.config.checkpoint_path + f'/model_{self.subj_idx}', 'cp_acc_')
        else:
            self.cp = None
    
    def set_checkpoint_model(self, cp_path, prefix):
        self.cp = scan_checkpoint(self.config.checkpoint_path + f'/model_{self.subj_idx}', prefix=prefix)
        self.state_dict= load_checkpoint(self.cp, self.device)
        self.model.load_state_dict(self.state_dict['model'])
        self.last_epoch = self.state_dict['epoch']
        self.init_optimizer()

    def init_dataset(self):
        test_envs = np.r_[self.subj_idx].tolist()
        unknown = self.config.unknown 
        known = list(set(list(range(0, 4))) - set(unknown))
        Data = HGD_OSR(self.config.data_root,test_envs,known,self.is_inference,self.config.fine)

        if self.is_inference:
            if self.config.fine == False:
                self.test_loader = torch.utils.data.DataLoader(Data.test_set,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)
                self.out_loader = torch.utils.data.DataLoader(Data.outset,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)
            else:
                self.test_loader = torch.utils.data.DataLoader(Data.test_set,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)
                self.out_loader = torch.utils.data.DataLoader(Data.outset,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)
        else:
            if self.config.fine == False:
                self.train_loader = torch.utils.data.DataLoader(Data.train_set,batch_size=self.config.batch_size, shuffle=True, pin_memory=True, num_workers=self.config.num_workers)
                self.valid_loader = torch.utils.data.DataLoader(Data.valid_set,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)
                self.test_loader = torch.utils.data.DataLoader(Data.test_set,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)
                self.out_loader = torch.utils.data.DataLoader(Data.outset,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)
            else:
                self.train_loader = torch.utils.data.DataLoader(Data.train_set,batch_size=32, shuffle=True, pin_memory=True, num_workers=self.config.num_workers)
                self.valid_loader = torch.utils.data.DataLoader(Data.valid_set,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)
                self.test_loader = torch.utils.data.DataLoader(Data.test_set,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)
                self.out_loader = torch.utils.data.DataLoader(Data.outset,batch_size=self.config.test_batch_size, shuffle=False, pin_memory=True, num_workers=self.config.num_workers)

    def init_optimizer(self):
        if self.config.fine:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
            self.scheduler = None
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)

    
    def training(self):
        results_columns = [f'valid_loss', f'test_loss', f'valid_accuracy', f'test_accuracy']
        df = pd.DataFrame(columns=results_columns)


        valid_min_acc = -1*float('inf')
        for epoch in range(max(0, self.last_epoch), self.config.epochs):
            print(epoch)
            self.train_func(10, self.model, self.device, self.train_loader, self.optimizer, self.scheduler, self.cuda, gpuidx=None, epoch=epoch)
            valid_loss, valid_score, targets, preds = self.eval_func(self.model, self.device, self.valid_loader)
            results_dict = self.get_metrics()
            results = {f'valid_loss': valid_loss, f'test_loss': results_dict['loss'], f'valid_accuracy': valid_score,
                    f'test_accuracy': results_dict['acc']}
            df = df.append(results, ignore_index=True)
            print(results)
            lr = self.scheduler.get_last_lr()[0]
            print(f'LR : {lr}')

            self.scheduler.step()

            if valid_score >= valid_min_acc:  
                valid_min_acc = valid_score
                best_acc_by_acc = results_dict['acc']
                checkpoint_path = "{}/{}/cp_acc_{:02d}".format(self.config.checkpoint_path, f'model_{self.subj_idx}', self.subj_idx)
                save_checkpoint(checkpoint_path,
                {'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'epoch': epoch})
                best_acc_epoch = epoch
            print(f'current best(loss) acc : {best_acc_by_acc:.4f} at epoch {best_acc_epoch}')


        print(f"subject:{self.subj_idx}, acc:{best_acc_by_acc}")

        df = pd.DataFrame(np.array(results_dict['acc']).reshape(-1, 1), columns=['sess2-on']) 
        test_loss, test_score, targets, preds = self.eval_func(self.model, self.device, self.test_loader)
        print(f"all acc: {np.mean(results_dict['acc']):.4f}")
        return df
    
        
    def finetuning(self):
        results_columns = [f'valid_loss', f'test_loss', f'valid_accuracy', f'test_accuracy']
        df = pd.DataFrame(columns=results_columns)
        valid_min_acc = -1*float('inf')
        for epoch in range(0,100):
            print(epoch)
            self.train_func(10, self.model, self.device, self.train_loader, self.optimizer, self.scheduler, self.cuda, gpuidx=None, epoch=epoch)
            valid_loss, valid_score,targets, preds = self.eval_func(self.model, self.device, self.valid_loader)

            results_dict = self.get_metrics()
            test_loss = results_dict['loss']
            test_score = results_dict['acc']
            if valid_score >= valid_min_acc:  
                valid_min_acc = valid_score
                best_acc_by_acc = results_dict['acc']
                checkpoint_path = "{}/{}/cp_fine_acc_{:02d}".format(self.config.checkpoint_path, f'model_{self.subj_idx}', self.subj_idx)
                save_checkpoint(checkpoint_path,
                {'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'epoch': epoch})
                best_acc_epoch = epoch
            print(f'current best(loss) acc : {best_acc_by_acc:.4f} at epoch {best_acc_epoch}')

        checkpoint_path = "{}/{}/cp_fine_last_acc_{:02d}".format(self.config.checkpoint_path, f'model_{self.subj_idx}', self.subj_idx)
        save_checkpoint(checkpoint_path,{'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'epoch': epoch})
        
        df = pd.DataFrame(np.array(test_score).reshape(-1, 1), columns=['sess2-on'])
        print(f"all acc: {np.mean(test_score):.4f}")

        return df
    
        
    def get_metrics(self):
        test_loss, test_score, targets_k, _pred_k  = self.eval_func(self.model, self.device, self.test_loader)
        out_loss, out_score, out_targets, _pred_u  = self.eval_func(self.model, self.device, self.out_loader)

        evidence = torch.exp(torch.Tensor(_pred_k))
        alpha = evidence + 1
        uncertainty_known = self.config.n_class / torch.sum(alpha, dim=1)
        evidence = torch.exp(torch.Tensor(_pred_u))
        alpha = evidence + 1
        uncertainty_unknown = self.config.n_class / torch.sum(alpha, dim=1)

        # Use predictive confidence for uncertainty
        confidence_known = 1-uncertainty_known.numpy()
        confidence_unknown = 1-uncertainty_unknown.numpy()
        
        uncertainty_known = uncertainty_known.numpy()
        uncertainty_unknown = uncertainty_unknown.numpy()

        df = dict()
        df['acc'] = test_score
        df['loss'] = test_loss
        df['uncertainty_k'] = uncertainty_known.mean()
        df['uncertainty_u'] = uncertainty_unknown.mean()

        # closed-set metic
        # Predictive mean calculation
        y_pred = _pred_k
        y_true_close = targets_k
    
        # ECE
        pred_np = np.argmax(y_pred,axis=1)
        ece = eval_ece(confidence_known,pred_np,y_true_close,15)
        df['ece']  = ece

        # NLL
        nll_sum = calculate_nll(y_true_close,y_pred,num_classes=3)
        nll = nll_sum/len(y_true_close)
        df['nll']  = nll.item()

        x1 = confidence_known #confidence of known
        x2 = confidence_unknown #confidence of unknown

        # AUROC
        results = evaluation.metric_ood(x1.copy(), x2.copy())['Bas']
        df['auroc']  = results['AUROC']
        
        #AUPR
        open_set_labels = np.concatenate([np.zeros_like(x1),np.ones_like(x2)])
        open_set_preds = np.concatenate([-x1,-x2]) #uncertainty 
        aupr = compute_aupr(open_set_preds.copy(), open_set_labels.copy(), normalised_ap=False)
        df['aupr']  = aupr

        # OSCR 
        oscr , roc = compute_oscr(_pred_k, _pred_u, targets_k, x1,x2)
        df['oscr']  = oscr
        df['open_set_preds'] = np.concatenate([confidence_known,confidence_unknown])
        df['roc'] = roc

        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openbmi_gigadb')
    parser.add_argument('--config', default='config_HGD_closed.json')
    parser.add_argument('--subject_group',type=int, default=0)
    parser.add_argument('--remove_class',type=int, default=2)
    parser.add_argument('--method', default='ours')
    parser.add_argument('--save_num', type=int, default=2024)
    parser.add_argument('--fine', action='store_true')
    parser.add_argument("--gpu_device", type=int, help="GPU device number to use", default=2)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    
    with open(args.config) as f:
        data = f.read()

    json_config = json.loads(data)
    config = AttrDict(json_config)
    config.fine = args.fine
    config.method = args.method
    config.checkpoint_path = f"../Result_HGD{args.save_num}/{config.model}_{config.method}_rm{args.remove_class}"

    config.remove_class = args.remove_class
    config.unknown = [args.remove_class]
    build_env(args.config, 'config.json', config.checkpoint_path)
    with open(__file__, 'r') as f:
        with open(config.checkpoint_path+'/backup_main.py', 'w') as out:
            for line in (f.readlines()):
                print(line, end='', file=out)

    df_all = pd.DataFrame()
    
    target = np.r_[args.subject_group]
    for fold_idx in target:
        trainer = MPLDM_Trainer(config, fold_idx)
        if config.fine:
            trainer.set_checkpoint_model(config.checkpoint_path, 'cp_acc_') 
            trainer.finetuning()
        else:
            trainer.training()

