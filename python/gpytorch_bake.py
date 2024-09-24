#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  vax_vigp_bakeoff.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.24.2023

print("Preload")

import numpy as np
#import jax.numpy as jnp
#import jax
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
#from tensorflow_probability.substrates import jax as tfp
#from jax import config
from tqdm import tqdm
#import optax
from time import time
import sys
import pandas as pd
import pickle

#from gpfy.likelihoods import Gaussian
#from gpfy.model import GP
#from gpfy.optimization import create_training_step
#from gpfy.spherical import NTK
#from gpfy.spherical_harmonics import SphericalHarmonics
#from gpfy.variational import VariationalDistributionTriL

import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

print("Post import")

#config.update("jax_enable_x64", True)
exec(open("python/sim_settings.py").read())

manual = False
#manual = True

if precision=='64':
    torch_dt = torch.float64
    adam_eps = 1e-8
elif precision=='32':
    torch_dt = torch.float32
    adam_eps = 1e-6
elif precision=='16':
    torch_dt = torch.float16
    adam_eps = 1e-4
else:
    raise Exception("Precision not supported.")
torch.set_default_dtype(torch_dt)

if manual:
    if len(sys.argv)>1:
        for i in range(5):
            print('!!!!!!')
            print("Manual mode engaged but command line arguments passed; exiting.")
            print('!!!!!!')
        quit()
    for i in range(10):
        print("manual!")
    M = 100
    seed = 0
else:
    M = int(sys.argv[1])
    seed = int(sys.argv[2])

sim_id = str(M)+'_'+str(seed)

np.random.seed(seed)

#if seed % 2 == 0:
#    methods = list(reversed(methods))

print('-----')
print(methods)
print('-----')

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
    X_all = np.concatenate([X,XX], axis = 0)
    y_all = np.concatenate([y,yy], axis = 0)
    #N,P = X.shape
    #assert N == len(y)
    #NN = XX.shape[0]
    #assert NN == len(yy)
    #assert P==XX.shape[1]
elif problem in ['year','keggu']:
    with open(datdir+'/'+problem+'.pkl','rb') as f:
        X_all, y_all = pickle.load(f)
else:
    raise Exception("Unknown problem.")

if problem in ['kin40k','year','keggu']:
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

first_iv = np.random.choice(N,M,replace=False)

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

mses = []
nlls = []
tds = []
tpis = []
costs = []
for method in methods:
    print(method)
    if False:
        pass
    elif 'torch' in method:
        if method=='torch_vanil':
            hetero = False
        elif 'hetero' in method:
            hetero = True
        else:
            raise Exception("Unknown torch method '" + method + "'")

        class GPModel(ApproximateGP):
            def __init__(self, inducing_points, hetero):
                variational_distribution = CholeskyVariationalDistribution(M)
                variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True, hetero = hetero)
                super(GPModel, self).__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=P))

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        train_x = torch.tensor(X, dtype = torch_dt)
        train_y = torch.tensor(y, dtype = torch_dt)
        test_x = torch.tensor(XX, dtype = torch_dt)
        test_y = torch.tensor(yy, dtype = torch_dt)

        inducing_points = train_x[first_iv, :]
        if hetero:
            # Under mapping x->(1/2+exp(x))*ls this initializes to the vanilla boi.
            ls_scale = torch.ones([M,P])*np.log(1/2)
            inducing_points = torch.stack([inducing_points,ls_scale], dim=-1)

        model = GPModel(inducing_points=inducing_points, hetero = hetero)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if precision=='64':
            model.double()
            likelihood.double()
        #elif precision=='32':
        #    model.single()
        #    likelihood.single()
        elif precision=='16':
            model.half()
            likelihood.half()

        if problem=='syn_sine':
            ls_init = -P
        else:
            ls_init = np.log(1.)
        if ls_init!=0. and init_style=='samp_orth':
            assert False
        model.covar_module.base_kernel.raw_lengthscale = torch.nn.Parameter(ls_init*torch.ones_like(model.covar_module.base_kernel.raw_lengthscale))

        if torch.cuda.is_available():
            print("Cuda detected")
            model = model.cuda()
            likelihood = likelihood.cuda()
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            test_x = test_x.cuda()
            test_y = test_y.cuda()

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=mb_size, shuffle=True)
        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=mb_size, shuffle=True)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=lr, eps = 1e-2)
        #], lr=lr, eps = adam_eps)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

        tt = time()

        mb_per_epoch = np.ceil(N/mb_size)
        epochs_iter = tqdm(range(int(np.ceil(max_iters/mb_per_epoch))), desc="Epoch")
        costs_it = np.nan*np.zeros(max_iters)
        ii = 0
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
                if ii < max_iters:
                    #costs_it[ii] = loss.detach().numpy()
                    costs_it[ii] = loss.cpu().detach().numpy()
                    ii += 1

        td = time()-tt

        model.eval()
        likelihood.eval()
        #yy_hat = model(test_x).mean.detach().numpy()
        yy_hat = model(test_x).mean.cpu().detach().numpy()
        if precision != '16':
            nll = -model(test_x).log_prob(test_y).cpu().detach().numpy()
        else:
            nll = np.nan
    else:
        raise Exception("Unknown method!")

    tpi = td / max_iters
    mse = np.mean(np.square(yy_hat-yy))

    mses.append(mse)
    nlls.append(nll)
    tds.append(td)
    tpis.append(tpi)

    costs.append(costs_it)

## For debugging, show optimization cost.
fname = 'temp.png' if manual else figsdir+"/"+sim_id+".png"
fig = plt.figure(figsize=[5,3])
for mi,method in enumerate(methods):
    plt.subplot(1,len(methods),1+mi)
    plt.plot(costs[mi])
    plt.title(method)
    #ax = plt.gca()
    #ax1 = ax.twinx()
    #ax1.plot(100*np.arange(len(mse_ess[mi])),mse_ess[mi], color = 'green')

plt.tight_layout()
plt.savefig(fname)
plt.close()

fig = plt.figure()
plt.plot()

##for p in model.named_parameters():
##    print(p)
#
#G = model.variational_strategy.inducing_points[:,:,:-1]
#A = model.variational_strategy.inducing_points[:,:,-1]
#
### Is the first row of A still big relative to the other rows?
#fig = plt.figure()
#plt.title("Coefficient vectors.")
#plt.boxplot([A[0,:].detach().cpu().numpy(), A[1:,:].flatten().detach().cpu().numpy()])
#plt.savefig("temp.png")
#plt.close()

## Save simulation results.
print(mses)
print(nlls)
if manual:
    for i in range(10):
        print("manual!")
else:
    #df = pd.DataFrame([['hen',mse, td, tpi, M, seed], ['m2',mse2, td2, tpi2, M, seed]])
    dat = []
    for mi,meth in enumerate(methods):
        dat.append([meth,mses[mi], nlls[mi], tds[mi], tpis[mi], M, seed])
    df = pd.DataFrame(dat)
    df.columns = ['Method','MSE','NLL','Time', 'TPI', 'M', 'seed']
    df.to_csv(simdir+'/'+sim_id+'.csv')


