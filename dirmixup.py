from itertools import count
import torch
import torch.nn as nn
import random
import numpy as np


class Dirchlet_Mixup(nn.Module):
    def __init__(self, p=0.5, alpha=5.0, eps=1e-6, batch_size=1, num_domain=3, style="original", shuffle_num=1):
        super().__init__()
        self.p = p
        self.eps = eps
        self.alpha = alpha
        self._activated = True

        self.batch_size=batch_size
        self.num_domain=3
        
        self.style=style
        
        ## Original Mixup
        self.beta=torch.distributions.Beta(self.alpha, self.alpha)

        ## shuffle_domain_num 섞을 domain 갯수
        # shuffle_domain_num=random.randint(2,self.num_domain-1) # include both ending points
        # random.randint(2, self.num_domain//2)
        self.shuffle_domain_num=3
        self.diri = torch.distributions.dirichlet.Dirichlet(torch.tensor(np.repeat(self.alpha, self.num_domain)))
    
    def __repr__(self):
        return f'Mixup(prob={self.p}, distrib_alpha={self.alpha}, eps={self.eps}, style={self.style})'

    def set_activation_status(self, status=True):
        self._activated = status
    
    def forward(self, x, y=None, **options):
        # if random.random() > self.p:
        #     return x, y, 0
        if 'alpha' in options:
            self.alpha = options.get("alpha")
            self.diri = torch.distributions.dirichlet.Dirichlet(torch.tensor(np.repeat(self.alpha, self.num_domain)))

        if self.style=="ori":
            B = x.size(0)
            lmda = self.beta.sample((B, 1, 1))
            lmda = lmda.to(x.device)
            
            shift_num=random.randint(1,self.num_domain-1)
            perm=torch.arange(0,B,dtype=torch.long)
            perm_a=perm[(-1)*self.batch_size*shift_num:]
            perm_b=perm[:(-1)*self.batch_size*shift_num]
            perm=torch.cat([perm_a,perm_b],0)
                
        elif self.style=="dir":
            output, counts = torch.unique(y, sorted=True, return_counts=True)
            B = x.size(0)
            C = x.size(1)

            if 'centers' in options:
                # print("!!")
                centers = options.get("centers")
                centers #shape nclass*npoint => N,3,16
                centers_expand = centers.expand(B,self.num_domain,C)

                target = torch.tensor([0,1,2])
                target = torch.nn.functional.one_hot(target)
                target_expand = target.expand(B,self.num_domain,3).to(x.device)

            lmdas = self.diri.sample((B, 1, 1))
            
            lmdas = lmdas.to(x.device).to(torch.float32)
            lmdas = lmdas.squeeze(dim=1)
            lmdas = lmdas.squeeze(dim=1)
            lmdas = lmdas.unsqueeze(dim=-1)

            mixed_x = centers_expand.mul(lmdas).sum(dim=1)
            mixed_y = target_expand.mul(lmdas).sum(dim=1)

            # # shift_num=random.sample(range(1,self.num_domain-1), self.shuffle_domain_num-1) # choose number of random shift  

            # indices1 = torch.randperm(B, device=x.device, dtype=torch.long)
            # indices2 = torch.randperm(B, device=x.device, dtype=torch.long)
            
            # addition_indexs = [indices1,indices2]
            # # num_list = list(range(self.num_domain))

            # # idx = []
            # # for d in num_list:
            # #     idx_list_per_class = []
            # #     y_d = torch.where(y==d)[0]
            # #     idx_list_per_class.append(y_d)

            # #     num_list_remain = num_list.copy()
            # #     num_list_remain.pop(d)
            # #     for _d in num_list_remain:
            # #         y_add = torch.where(y==_d)[0]
            # #         if len(y_add) >= len(y_d):
            # #             y_add = y_add[0:len(y_d)]
            # #         else:
            # #             y_add = torch.cat(y_add, 

            # #         idx_list_per_class.append(y_add)
                
            # #     idx_temp = torch.stack(idx_list_per_class,dim=1)
            # #     idx.append(idx_temp)

        else:
                raise NotImplementedError



        # xs=[x]
        # y = torch.nn.functional.one_hot(y)
        # ys=[y]
        # for j in range(len(addition_indexs)):
        #     xs.append(x[addition_indexs[j]])
        #     ys.append(y[addition_indexs[j]])


        # mixed_x = sum([xs[j].unsqueeze(dim=2)*lmdas[j] for j in range(self.num_domain)]).squeeze()
        # mixed_y = sum([ys[j]*lmdas[j].view(-1,1) for j in range(self.num_domain)])
        
        return mixed_x, mixed_y, 1 
          
          
          
            
        # index = torch.randperm(B).to(x.device)
        
        # mixed_x = lmda * x + (1 - lmda) * x[index, :] # input x와 순서 바꾼 x 섞어서 새로운 mixed_x 생성, lam: 섞는 정도 (alpha에 따라 결정)
        # y_a, y_b = y, y[index] # y_a: target, y_b: 순서 바뀐 y
        # return mixed_x, y_a, y_b, lmda # mixed_x, target, 순서 섞인 y, 섞인 정도

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class DirichletMixupTorch:
    def __init__(self, n_classes):
        # Number of classes
        self.n_classes = n_classes

    def mixup_samples(self, centers, alpha=1.0, k_samples=1):
        # Check if the centers shape is consistent with n_classes
        assert centers.shape[0] == self.n_classes, "Number of centers should match the number of classes."

        # Dirichlet distribution
        device = centers.device
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha] * self.n_classes).to(device))
        
        # Generate mixing coefficients from the Dirichlet distribution
        mix_coeffs = dirichlet.sample((k_samples,))

        # Depending on the dimensionality of centers, apply appropriate multiplication
        if len(centers.shape) == 2:  # For 2D centers [n_classes, dim]
            mixed_samples = torch.matmul(mix_coeffs, centers)
        elif len(centers.shape) == 3:  # For 3D centers [n_classes, dim, len]
            feat_dim = centers.shape[1]
            feat_len = centers.shape[2]
            mixed_samples = torch.matmul(mix_coeffs, centers.reshape(self.n_classes,-1))
            mixed_samples = mixed_samples.reshape(k_samples,feat_dim,feat_len)
        else:
            raise ValueError("Centers should be 2D or 3D.")

        return mixed_samples, mix_coeffs




def mixup_data_dirichlet(x, y, alpha, device="cpu", k_samples=10):
    '''Compute the mixup data for each unique class in y using Dirichlet distribution.
    Return mixed inputs and targets.'''
    
    n_classes = len(torch.unique(y))
    
    # Create a Dirichlet distribution with the given alpha value
    dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha] * n_classes).to(device))
    
    # Generate mixing coefficients from the Dirichlet distribution for each sample in the batch
    mix_coeffs = dirichlet.sample((x.size(0),))

    mixed_x = torch.zeros_like(x).to(device)

    for cls_idx, cls in enumerate(torch.unique(y)):
        # Get indices of samples with the current class
        indices = (y == cls).nonzero(as_tuple=True)[0]
        
        # Randomly select indices for the entire batch from the current class
        random_indices = indices[torch.randint(0, len(indices), (x.size(0),))]
        
        # Update the mixed_x tensor with data from the current class
        mixed_x += mix_coeffs[:, cls_idx].unsqueeze(1).unsqueeze(2) * x[random_indices]
    

    # for i, sample in enumerate(x):
    #     for cls_idx, cls in enumerate(torch.unique(y)):
    #         # Get indices of samples with the current class
    #         indices = (y == cls).nonzero(as_tuple=True)[0]
            
    #         # Use a random index from the current class for mixing
    #         idx = indices[torch.randint(0, len(indices), (1,)).item()]
            
    #         mixed_x[i] += mix_coeffs[i, cls_idx] * x[idx]

    return mixed_x, mix_coeffs




# Example usage:
n_classes = 3
centers_torch = torch.tensor([[1.0, 2.0], 
                              [3.0, 4.0], 
                              [5.0, 6.0]])  # Example centers for 3 classes with 2D features

dirichlet_mixup_torch = DirichletMixupTorch(n_classes=n_classes)
k_samples = 5
alpha_value = 2.0
mixed_torch, labels_torch = dirichlet_mixup_torch.mixup_samples(centers_torch, alpha=alpha_value, k_samples=k_samples)

mixed_torch, labels_torch




if __name__ == '__main__':
    # Mixup = Dirchlet_Mixup(num_domain=3, style='ori', shuffle_num=2)
    # x = torch.rand(64,16,4)
    # y1 = torch.ones(32)
    # y2 = torch.zeros(32)
    # y = torch.cat([y1,y2])
    # y = y[torch.randperm(64)]

    # m_x,m_y = Mixup(x,y)

    # Example usage:
    n_classes = 3
    centers_torch = torch.tensor([[1.0, 2.0], 
                                [3.0, 4.0], 
                                [5.0, 6.0]])  # Example centers for 3 classes with 2D features

    dirichlet_mixup_torch = DirichletMixupTorch(n_classes=n_classes)
    k_samples = 5
    alpha_value = 2.0
    mixed_torch, labels_torch = dirichlet_mixup_torch.mixup_samples(centers_torch, alpha=alpha_value, k_samples=k_samples)


    print(mixed_torch,labels_torch)