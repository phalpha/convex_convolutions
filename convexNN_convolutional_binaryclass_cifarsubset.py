#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import packages
import numpy as np
import time
import cvxpy as cp
import torch
import sys
import keras
import tensorflow
import cv2


# In[ ]:


# This notebook contains the code for the experiment in figure 11 - plots c, d of the paper
# https://arxiv.org/pdf/2101.02429.pdf
# The code implements the convex formulation for the two-layer convolutional neural network with global
# average pooling for "binary classification" (hence, scalar output). The dataset is the fashion-mnist dataset.
# For other details, read the explanation for figure 11 - plots c, d of the paper.


# In[ ]:


# load the fashion mnist dataset
# change this directory so it points to the folder where you downloaded the fashion-mnist dataset.
directory = '/Users/phalpha/Desktop/Stanford/project/fashion_mnist/data/fashion/'
sys.path.insert(1, directory)
from fashion_mnist.utils import mnist_reader
from keras.datasets import cifar10

#X_train, y_train = mnist_reader.load_mnist(directory, kind='train')
#X_test, y_test = mnist_reader.load_mnist(directory, kind='t10k')

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
#X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
train_images = X_train.astype(np.float64)
train_labels = y_train.astype(np.float64)
A = train_images.copy()
y = train_labels.copy()

inds = np.argwhere(y <= 1)[:,0] # get the first two classes
A = A[inds, :]
y = y[inds].reshape((inds.shape[0], 1))
#print(A.shape)
n, d = A.shape[0], A.shape[1]*A.shape[1]
print(n,d)
# test set
test_images = X_test.astype(np.float64)
test_labels = y_test.astype(np.float64)
A_test = test_images.copy()
y_test = test_labels.copy()

inds_test = np.argwhere(y_test <= 1)[:,0] # get the first two classes
A_test = A_test[inds_test, :]
y_test = y_test[inds_test].reshape((inds_test.shape[0], 1))

A = (A-128)/ 255
A_test = (A_test-128) / 255


print(A.shape, y.shape, A_test.shape, y_test.shape)


# In[ ]:


# CNN 2D - precompute
A_v2 = np.swapaxes(A.reshape(A.shape[0], 1, 32, 96), 2, 3)
A_test_v2 = np.swapaxes(A_test.reshape(A_test.shape[0], 1, 32, 96), 2, 3)

print(A_v2.shape, "A_v2")
print(A_test_v2.shape, "A_test_v2")
stride = 4
f = 4 # filter dimension


patches_train = torch.nn.functional.unfold(torch.tensor(A_v2), kernel_size=(f, f), stride=stride, padding=0)
patches_test = torch.nn.functional.unfold(torch.tensor(A_test_v2), kernel_size=(f, f), stride=stride, padding=0)
patches_train = patches_train.numpy()
patches_test = patches_test.numpy()

K = patches_train.shape[-1]

# start pre-computing
ff = 1*f**2
X_K = np.zeros((n, ff**2+ff+1))
for i in range(n):
    if i % 100 == 0:
        print(i, end=", ")
    for k in range(K):
        x_ik = patches_train[i,:,k:k+1]
        X_K[i, 0:ff**2] += (1/K * np.matmul(x_ik, x_ik.T).reshape((ff**2)))
        X_K[i, ff**2:ff**2+ff] += (1/K * x_ik.reshape((ff)))
    X_K[i, ff**2+ff] += 1

X_K_save = X_K.copy()


# In[8]:


# parameters
a, b, c = 0.09, 0.5, 0.47

beta = 10**(-6)

scs_max_iters = 200000
tol = 10**(-6)


# In[9]:


# finish pre-computing
X_K = X_K_save.copy()
X_K[:, 0:ff**2] = a * X_K[:, 0:ff**2]
X_K[:, ff**2:ff**2+ff] = b * X_K[:, ff**2:ff**2+ff]
X_K[:, ff**2+ff] = c

X_KTX_K = np.matmul(X_K.T, X_K)
X_KTy = np.matmul(X_K.T, y)
y_normsq = np.sum(y**2)


# In[ ]:


# create and solve the optimization problem
Z1 = cp.Variable((1*f**2, 1*f**2), symmetric=True)
Z2 = cp.Variable((1*f**2, 1))
Z4 = cp.Variable((1,1))

Z1_prime = cp.Variable((1*f**2, 1*f**2), symmetric=True)
Z2_prime = cp.Variable((1*f**2,1))
Z4_prime = cp.Variable((1,1))


z = cp.vstack((cp.reshape((Z1-Z1_prime), (ff**2,1)), (Z2-Z2_prime), (Z4-Z4_prime)))
objective = cp.quad_form(z, X_KTX_K) - 2 * z.T @ X_KTy + y_normsq

objective *= 0.5
#objective += (beta*(Z4 + Z4_prime))
objective += (beta*(cp.trace(Z1) + cp.trace(Z1_prime)))


Z = cp.vstack((cp.hstack((Z1, Z2)), cp.hstack((Z2.T, Z4))))
Z_prime = cp.vstack((cp.hstack((Z1_prime, Z2_prime)), cp.hstack((Z2_prime.T, Z4_prime))))
constraints = []
constraints = [cp.trace(Z1) == Z4]
constraints += [cp.trace(Z1_prime) == Z4_prime]
constraints += [Z >> 0] + [Z_prime >> 0]

prob = cp.Problem(cp.Minimize(objective), constraints)
start_time = time.time()
print("started..")
prob.solve(max_iters=scs_max_iters)#cp.CVXOPT)#solver=cp.SCS) #25000
print(prob.solver_stats.extra_stats)
end_time = time.time()
time_elapsed_cvx = end_time - start_time
print("time elapsed: " + str(time_elapsed_cvx))

# Print result.
print(prob.status)
print("The optimal value is", prob.value)
print("The optimal value is", objective.value)


# In[ ]:


# compute accuracies
y_hat = np.zeros((A_v2.shape[0], 1))
for i in range(A_v2.shape[0]):
    patches_ = patches_train[i, :]
    Zp = np.matmul((Z1.value-Z1_prime.value), patches_)
    quad_term = a * np.sum(np.multiply(patches_, Zp)) / K
    lin_term = b * np.sum(np.matmul((Z2.value-Z2_prime.value).T, patches_)) / K
    y_hat[i,0] = quad_term + lin_term + c*(Z4.value-Z4_prime.value)
y_pred = y_hat > 0.5

y_hat_test = np.zeros((A_test_v2.shape[0], 1))
for i in range(A_test_v2.shape[0]):
    patches_ = patches_test[i, :]
    Zp = np.matmul((Z1.value-Z1_prime.value), patches_)
    quad_term = a * np.sum(np.multiply(patches_, Zp)) / K
    lin_term = b * np.sum(np.matmul((Z2.value-Z2_prime.value).T, patches_)) / K
    y_hat_test[i,0] = quad_term + lin_term + c*(Z4.value-Z4_prime.value)
y_pred_test = y_hat_test > 0.5


noncvx_cost = 0.5*np.sum((y-y_hat)**2) + beta*(Z[-1,-1]+Z_prime[-1,-1]).value
noncvx_cost_test = 0.5*np.sum((y_test-y_hat_test)**2) + beta*(Z[-1,-1]+Z_prime[-1,-1]).value


training_acc = np.sum(y == y_pred) / y.shape[0]
test_acc = np.sum(y_test == y_pred_test) / y_test.shape[0]

print("costs:", noncvx_cost, noncvx_cost_test)
print("accuracies:", training_acc, test_acc)


num_neurons_cvx = np.sum(np.linalg.eig(Z.value)[0] > tol) + np.sum(np.linalg.eig(Z_prime.value)[0] > tol)
print("num_neurons_cvx: " + str(num_neurons_cvx))


# In[ ]:





# In[ ]:





# In[ ]:
