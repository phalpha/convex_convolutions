#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
from keras.datasets import cifar10
import numpy as np

USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)


# In[3]:


NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)


# In[ ]:


# parameters, K = number of patches, P = pooling size, C = output dimension, f = filter size




class custom_multiclass_torch(torch.nn.Module):
    def __init__(self, d):
        super(custom_multiclass_torch, self).__init__()

        #initialize tensor array variables

        self.Z_1_arr_train = [[None for i in range(K/P)] for j in range(C)]
        self.Z_1arr_prime_train = [[None for i in range(K/P)] for j in range(C)]
        self.Z_2_arr_train = [[None for i in range(K/P)] for j in range(C)]
        self.Z_2_arr_prime_train = [[None for i in range(K/P)] for j in range(C)]
        self.Z_4_arr_train = [[None for i in range(K/P)] for j in range(C)]
        self.Z_4_arr_prime_train = [[None for i in range(K/P)] for j in range(C)]

        self.Z_arr_train = [[None for i in range(K/P)] for j in range(C)]
        self._arr_prime_train = [[None for i in range(K/P)] for j in range(C)]

        # set random weights using kaising normalization
        for i in range(K/P):
            for j in range(C):
                fan_in = f+1
                #w = torch.randn((f+1,f+1), device=device, dtype=dtype) * np.sqrt(2. / fan_in)
                #w.requires_grad = True
                #Z_arr[i][j] = w
                #w2 = torch.randn((f+1,f+1), device=device, dtype=dtype) * np.sqrt(2. / fan_in)
                #w2.requires_grad = True
                #Z_arr_prime_train[i][j] = w2 

                w1 = torch.randn((f,f), device=device, dtype=dtype, required_grads = True) 
                w2 = torch.randn((f,1), device=device, dtype=dtype,  required_grads = True) 
                w3 = torch.randn((1,1), device=device, dtype=dtype,  required_grads = True) 
                self.Z_1_arr_train[i][j] = w1
                self.Z_2_arr_train[i][j] = w2
                self.Z_4_arr_train[i][j] = w3
                v1 = torch.randn((f,f), device=device, dtype=dtype,  required_grads = True) 
                v2 = torch.randn((f,1), device=device, dtype=dtype,  required_grads = True) 
                v3 = torch.randn((1,1), device=device, dtype=dtype,  required_grads = True) 
                self.Z_1_arr_prime_train[i][j] = v1
                self.Z_2_arr_prime_train[i][j] = v2
                self.Z_4_arr_prime_train[i][j] = v3
    
        self.Z_arr_train = torch.vstack(torch.hstack(z_1_arr_train, z_2_arr_train), torch.hstack(torch.transpose(z_2_arr_train),z_4_arr_train))
        self.Z_arr_prime_train = torch.vstack(torch.hstack(z_1_arr_prime_train, z_2_arr_prime_train), torch.hstack(torch.transpose(z_2_arr_prime_train).z_4_prime_arr_train))

    def forward(self, x_patches):
        ypred = torch.zeros((N,C))
        for i in range(N):
            for t in range(C):
                constant_part = 0
                for k in range(K/P):
                    constant_part += self.Z_4_matrix[k][t] - self.Z_4_prime_matrix[k][t]
                constant_part *= c

                linear_part = 0
                for k in range(K/P):
                    for l in range(P):
                        linear_part += torch.transpose(x_patches[i][(k-1)*P+ l]) * (self.Z_2_matrix[k][t] - self.Z_2_prime_matrix[k][t])
                linear_part *= b/P

                quadratic_part = 0
                for k in range(K/P):
                    for l in range(P):
                        quadratic_part += torch.transpose(x_patches[i][(k-1)*P+ l]) * (self.Z_1_matrix[k][t] - self.Z_1_prime_matrix[k][t]) * x_patches[i][(k-1)*P+ l] 
                quadratic_part *= a/P


                ypred[i][t] = quadratic_part + linear_part + constant_part

        return ypred

    
    
    def customloss(Yhat, Y):
        #loss = nn.MSELoss()
        #objective1 = loss(Yhat, Y)
        objective1 = 0.5 * torch.norm(yhat - y)**2 *n/y.shape[0]

        objective2 = 0 

        for i in range(K/P):
        for j in range(C):
            objective2 += self.Z_4_arr[i][j] + self.Z_4arr_prime[i][j]
            
        objective = objective1+objective2

        return objective
            
                  
def train_project(model, loss_fn, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            # print(scores.shape)
            # input()

            #loss = F.cross_entropy(scores, y)
            loss = loss_fn(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()
                    
        

X_train = None
X_test = None

Y_train = None
Y_test = None

f = 4
P = 2
C = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


train_images = X_train.astype(np.float64)
train_labels = y_train.astype(np.float64)
test_images = X_test.astype(np.float64)
test_labels = y_test.astype(np.float64)

patches_train = torch.nn.functional.unfold(torch.tensor(train_images), kernel_size=(f,f), stride=stride, padding=0)
patches_test = torch.nn.functional.unfold(torch.tensor(test_images), kernel_size=(f,f), stride=stride, padding=0)


K = patches_train.shape[-1]


Yhat_train =  None
Yhat_test = None


a,b,c = 1,2,3
beta = 1



yhat_train = forward(patches_train)



