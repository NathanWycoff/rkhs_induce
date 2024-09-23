#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  vax_vigp_bakeoff.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.24.2023

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from tqdm import tqdm
from time import time
import sys
import pandas as pd
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP, GP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

#M = 2
#M = 5
M = 1
mb_size = 256
seed = 0
lr = 1e-2
epochs = 100

rkhs = False

np.random.seed(123)

#D = get_D(M)

P = 1
N = 5
NN = 500

X = np.random.uniform(size=[N,P])
#XX = np.random.uniform(size=[NN,P])
XX = np.linspace(0,1,num=NN).reshape([NN,P])
y = np.cos(4*np.pi*np.sum(X, axis = 1))
yy = np.cos(4*np.pi*np.sum(XX, axis = 1))

## Rescaling
mu_y = np.mean(y)
sig_y = np.std(y)
y = (y-mu_y) / sig_y
yy = (yy-mu_y) / sig_y

## Next thing to try: no scaling of X.
min_X = np.min(X, axis = 0)
max_X = np.max(X, axis = 0)
X = (X-min_X[np.newaxis,:]) / (max_X-min_X)[np.newaxis,:]
XX = (XX-min_X[np.newaxis,:]) / (max_X-min_X)[np.newaxis,:]

class GPModel(ApproximateGP):
    def __init__(self, inducing_points, rkhs):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True, rkhs = rkhs)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

train_x = torch.tensor(X)
train_y = torch.tensor(y)
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=mb_size, shuffle=True)
test_x = torch.tensor(XX)
test_y = torch.tensor(yy)

#inducing_points = train_x[:M, :]
if rkhs:
    basis_vectors = torch.rand([D,M,P])
    basis_coefs = torch.rand([D,M,1])
    inducing_points = torch.concat([basis_vectors,basis_coefs], axis = 2)
else:
    inducing_points = train_x[np.random.choice(train_x.shape[0],M), :]

init_points = inducing_points.clone()
print(init_points)

model = GPModel(inducing_points=inducing_points, rkhs = rkhs)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model.double()
likelihood.double()

ls_init = -1
model.covar_module.base_kernel.raw_lengthscale = torch.nn.Parameter(ls_init*torch.ones_like(model.covar_module.base_kernel.raw_lengthscale))
likelihood.raw_noise = -6*torch.ones_like(likelihood.raw_noise)

model.train()
likelihood.train()

#optimizer = torch.optim.Adam([
#    {'params': model.parameters()},
#    {'params': likelihood.parameters()},
#], lr=lr)

optimizer = torch.optim.Adam([
    {'params': model.variational_strategy._variational_distribution.variational_mean},
], lr=lr)


# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
mse = lambda output, y_batch: torch.sum(torch.square(output.mean-y_batch))


costs = []
mb_per_epoch = np.ceil(N/mb_size)
epochs_iter = tqdm(range(epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        optimizer.zero_grad()
        output = model(x_batch)
        #loss = -mll(output, y_batch)
        loss = mse(output, y_batch)
        minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        costs.append(loss.detach().numpy())

fig = plt.figure()
plt.plot(costs)
plt.savefig("costs.pdf")
plt.close()

model.eval()
likelihood.eval()

model(train_x).mean

yy_hat = model(test_x).mean.detach().numpy()
yy_u = model(test_x).mean.detach().numpy() + 2*np.sqrt(model(test_x).variance.detach().numpy())
yy_l = model(test_x).mean.detach().numpy() - 2*np.sqrt(model(test_x).variance.detach().numpy())
y_hat = model(train_x).mean.detach().numpy()

print(np.mean(np.square(y-y_hat)))
print(np.mean(np.square(yy-yy_hat)))

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

ex_model = ExactGPModel(train_x, train_y, likelihood)
ex_model.double()
ex_model.mean_module.raw_constant = model.mean_module.raw_constant
ex_model.covar_module.raw_outputscale = model.covar_module.raw_outputscale
ex_model.covar_module.base_kernel.raw_lengthscale = model.covar_module.base_kernel.raw_lengthscale

ex_model.eval()
yy_ex = ex_model(test_x).mean.detach().numpy()
yy_eu = ex_model(test_x).mean.detach().numpy() + 2*np.sqrt(ex_model(test_x).variance.detach().numpy())
yy_el = ex_model(test_x).mean.detach().numpy() - 2*np.sqrt(ex_model(test_x).variance.detach().numpy())

fig = plt.figure()
plt.scatter(X.flatten(), y)

plt.plot(XX.flatten(), yy_hat, label = 'var', color = 'blue')
plt.plot(XX.flatten(), yy_l, color = 'blue', linestyle='--')
plt.plot(XX.flatten(), yy_u, color = 'blue', linestyle='--')

plt.plot(XX.flatten(), yy_ex, label = 'ex', color = 'orange')
plt.plot(XX.flatten(), yy_el, color = 'orange', linestyle='--')
plt.plot(XX.flatten(), yy_eu, color = 'orange', linestyle='--')

plt.legend()

l,u = plt.gca().get_ylim()
plt.vlines(model.variational_strategy.inducing_points.detach().numpy(), l, u)
plt.vlines(inducing_points, l, u, color = 'orange')
plt.savefig("temp.pdf")
plt.close()

for i in model.named_parameters():
    print(i)
