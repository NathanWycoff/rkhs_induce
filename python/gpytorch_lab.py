
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  vax_vigp_bakeoff.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.24.2023

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from tensorflow_probability.substrates import jax as tfp
from jax import config
from tqdm import tqdm
import optax
from time import time
import sys
import pandas as pd
import pickle

from gpfy.likelihoods import Gaussian
from gpfy.model import GP
from gpfy.optimization import create_training_step
from gpfy.spherical import NTK
from gpfy.spherical_harmonics import SphericalHarmonics
from gpfy.variational import VariationalDistributionTriL

import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

config.update("jax_enable_x64", True)

#exec(open("python/jax_vsgp_lib.py").read())
exec(open("python/sim_settings.py").read())

for i in range(10):
    print("manual!")
M = 128
#max_iters = 100
max_iters = 4000
#max_iters = 10000
seed = 0
#seed = 5
#methods = ['hens']
#methods = ['sphere']
lr = 1e-2
#lr = 1e-3
debug = True
jit = True
track = True
es_patience = np.inf

rkhs = False

np.random.seed(123)

D = get_D(M)

if problem=='syn_sine':
    P = 2
    N = 2000
    NN = 500

    X = np.random.uniform(size=[N,P])
    XX = np.random.uniform(size=[NN,P])
    y = np.cos(4*np.pi*np.sum(X, axis = 1))
    yy = np.cos(4*np.pi*np.sum(XX, axis = 1))
elif problem in ['kin40k']:
    with open(datdir+'/'+problem+'.pkl','rb') as f:
        X, y, XX, yy = pickle.load(f)
    N,P = X.shape
    assert N == len(y)
    NN = XX.shape[0]
    assert NN == len(yy)
    assert P==XX.shape[1]
elif problem in ['year','keggu']:
    with open(datdir+'/'+problem+'.pkl','rb') as f:
        X_all, y_all = pickle.load(f)
    p_out = 0.1
    NN = int(np.ceil(X_all.shape[0]*p_out))
    N = X_all.shape[0] - NN
    P = X_all.shape[1]
    ind_train = np.random.choice(X_all.shape[0],N,replace=False)
    ind_test = np.setdiff1d(np.arange(X_all.shape[0]), ind_train)
    X = X_all[ind_train,:]
    y = y_all[ind_train]
    XX = X_all[ind_test,:]
    yy = y_all[ind_test]
else:
    raise Exception("Unknown problem.")


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
test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=mb_size, shuffle=True)

#inducing_points = train_x[:M, :]
if rkhs:
    basis_vectors = torch.rand([D,M,P])
    basis_coefs = torch.rand([D,M,1])
    inducing_points = torch.concat([basis_vectors,basis_coefs], axis = 2)
else:
    inducing_points = train_x[np.random.choice(train_x.shape[0],M), :]


model = GPModel(inducing_points=inducing_points, rkhs = rkhs)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model.double()
likelihood.double()

#
ls_init = -P
model.covar_module.base_kernel.raw_lengthscale = torch.nn.Parameter(ls_init*torch.ones_like(model.covar_module.base_kernel.raw_lengthscale))
#

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=lr)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

XX_torch = torch.tensor(XX)
yy_torch = torch.tensor(yy)

costs = []
lengthscales = []
variances = []
oos_mse = []
oos_nll = []
mb_per_epoch = np.ceil(N/mb_size)
epochs_iter = tqdm(range(int(np.ceil(max_iters/mb_per_epoch))), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        costs.append(loss.detach().numpy())
        lengthscales.append(float(model.covar_module.base_kernel.raw_lengthscale.detach().numpy()))
        variances.append(float(model.covar_module.raw_outputscale.detach().numpy()))

        yy_hat = model(XX_torch).mean.detach().numpy()

        #oos_mse.append(float(np.mean(np.square(yy-yy_hat))))
        #oos_nll.append(model(XX_torch).log_prob(yy_torch))


#costs = np.array(costs)
#lengthscales = np.array(lengthscales).flatten()
#variances = np.array(variances)

model.eval()
likelihood.eval()
yy_hat = model(test_x).mean.detach().numpy()

print(np.mean(np.square(yy-yy_hat)))

fig = plt.figure(figsize=[4,12])
plt.subplot(4,1,1)
plt.plot(costs)
#plt.subplot(4,1,2)
#plt.plot(oos_mse)
plt.subplot(4,1,3)
plt.plot(lengthscales)
plt.subplot(4,1,4)
plt.plot(variances)
plt.savefig("costs.pdf")
plt.close()

print("mean params:")
for mm in model.mean_module.parameters():
    print(mm)

print("covar params:")
for mm in model.covar_module.parameters():
    print(mm)

for i in model.named_hyperparameters():
    print(i)

