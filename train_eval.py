import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from dirmixup import DirichletMixupTorch
import utils
import time
import numpy as np
from torch.nn.parallel.data_parallel import DataParallel
import torch

def compute_annealing_coef(epoch, total_epoch, method='step', annealing_step=10, annealing_start=0.1):
    if method == 'step':
        annealing_coef = min(1.0, epoch / annealing_step)
    elif method == 'exp':
        annealing_coef = annealing_start * torch.exp(-torch.log(torch.tensor(annealing_start)) / total_epoch * epoch)
    else:
        raise NotImplementedError
    return annealing_coef

def distinct_random_prototype_indices(num_classes, num_centers):
    indices = [base + np.random.randint(0, num_centers) for base in range(0, num_classes * num_centers, num_centers)]
    return indices

def cosine_annealing_coef(epoch, maxepoch=100):
    epoch = torch.tensor(epoch, dtype=torch.float32)
    maxepoch = torch.tensor(maxepoch, dtype=torch.float32)
    pi = torch.tensor(np.pi, dtype=torch.float32)
    return 0.5 * (1 + torch.cos(pi * (1 + epoch / maxepoch)))

def train_mpldm(log_interval, model, device, train_loader, optimizer, scheduler, cuda, gpuidx, epoch=1, config=None):
    correct = []
    start = time.time()

    model.train()
    t_data = []
    t_model = []

    t3 = time.time()
    preds = []
    
    train_loader_is_list = False
    if type(train_loader) is list:
        train_loader_is_list = True
        train_loader = zip(*train_loader)

    dirichlet_mixup_torch = DirichletMixupTorch(n_classes=3)
    
    for batch_idx, datas in enumerate(train_loader):
        if train_loader_is_list:
            minibatches_device = [(x.to(device), y.to(device))
                for x,y in datas]
            data = torch.cat([x for x,y in minibatches_device])
            target = torch.cat([y for x,y in minibatches_device])
        else:
            if len(datas[1].shape) == 2:
                data, target = datas[0].to(device), datas[1].to(device)
            else:
                data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)

        t2 = time.time()
        t_data.append(t2 - t3)

        optimizer.zero_grad()

        if scheduler is None:
            annealing_coef=cosine_annealing_coef(epoch)
        else:
            max_lr = scheduler.base_lrs[0]
            annealing_coef = 1- scheduler.get_last_lr()[0] / max_lr
        
        embeddings = model.embedding_net.get_embeddings(data)
        embeddings = embeddings[1:]
        logits1, loss1 = model.criterion1(model.clf1(embeddings[0]), target, target, annealing_coef)
        logits2, loss2 = model.criterion2(model.clf2(embeddings[1]), target, target, annealing_coef)
        logits3, loss3 = model.criterion3(model.clf3(embeddings[2]), target, target, annealing_coef)

        proto_indices = distinct_random_prototype_indices(num_classes=3, num_centers=model.criterion1.Dist.num_centers)
        mixup_output1_ce, mixed_y1 = dirichlet_mixup_torch.mixup_samples(centers = model.criterion1.points[proto_indices, :], alpha= 5.0, k_samples=20) # numclass, numproto(mean), 000,111,222
        mixup_output2_ce, mixed_y2 = dirichlet_mixup_torch.mixup_samples(centers = model.criterion2.points[proto_indices, :], alpha= 5.0, k_samples=20)
        
        embeddings_fake1_1, embeddings_fake1_2 = model.embedding_net.get_embeddings(mixup_output1_ce,layer_num=3)
        embeddings_fake1_2 = model.clf3(embeddings_fake1_2)

        embeddings_fake2 = model.embedding_net.get_embeddings(mixup_output2_ce,layer_num=4)
        embeddings_fake2 = model.clf3(embeddings_fake2[0])

        loss_fake1_1_ce = model.criterion2.fake_loss(model.clf2(embeddings_fake1_1), mixed_y1)
        loss_fake1_2_ce = model.criterion3.fake_loss(embeddings_fake1_2, mixed_y1)
        loss_fake2_ce = model.criterion3.fake_loss(embeddings_fake2, mixed_y2)

        loss = loss1+ loss2 +loss3 + annealing_coef*(loss_fake1_1_ce + loss_fake1_2_ce + loss_fake2_ce)
        logits = (logits1+logits2+logits3)/3

        pred = logits.argmax(dim=1, keepdim=True) 

        correct.append(pred.eq((target).view_as(pred)).sum().item())

        loss.backward()
        optimizer.step()


        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} \tLoss: {loss.item():.6f}')


        t3 = time.time()
        t_model.append(t3 - t2)
        preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds)

    print("time :", time.time() - start)
    print(f"t_data : {sum(t_data)} , t_model : {sum(t_model)}")
    print(f'Train set: Accuracy: {sum(correct)}/{len(preds)} ({100. * sum(correct) / len(preds):.4f}%)')

def eval_mpldm(model, device, test_loader):
    model.eval()

    test_loss = []
    correct = []
    preds = []
    targets = []

    list(test_loader)
    with torch.no_grad():
        for datas in test_loader:
            if len(datas) > 4:
                minibatches_device = [(x.to(device), y.to(device))
                    for x,y in datas]
                data = torch.cat([x for x,y in minibatches_device])
                target = torch.cat([y for x,y in minibatches_device])
            else:
                data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)
            embeddings = model(data)
            
            logits1, loss1 = model.criterion1(embeddings[0], target, target)
            logits2, loss2 = model.criterion2(embeddings[1], target, target)
            logits3, loss3 = model.criterion3(embeddings[2], target, target)
            loss = loss1 + loss2 + loss3
            logits = (logits1+logits2+logits3)/3
            test_loss.append(loss.item()) 

            preds.append(logits.cpu().numpy())
            pred = logits.argmax(dim=1, keepdim=True) 
            correct.append(pred.eq(target.view_as(pred)).sum().item())
            targets.append(target.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    loss = sum(test_loss) / len(preds)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(preds),
        100. * sum(correct) / len(preds)))

    return loss, 100. * sum(correct) / len(preds), targets, preds