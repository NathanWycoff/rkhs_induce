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

exec(open("python/jax_vsgp_lib.py").read())
exec(open("python/sim_settings.py").read())

#manual = False
manual = True

if manual:
    for i in range(10):
        print("manual!")
    M = 128
    #max_iters = 100
    max_iters = 4000
    #max_iters = 15000
    seed = 0
    #seed = 5
    #methods = ['hens']
    #methods = ['sphere']
    lr = 1e-2
    debug = True
    jit = True
    track = True
    es_patience = np.inf
else:
    M = int(sys.argv[1])
    seed = int(sys.argv[2])
    debug = False
    jit = True
    track = False
    es_patience = None

sim_id = str(M)+'_'+str(seed)

np.random.seed(seed)

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

## Run a small model to compile stuff before anything is timed.
mod = HensmanGP(X, y, M, jit = jit)
mod.fit(verbose=verbose, lr=lr, iters=20, debug = debug, mb_size = mb_size, track = track)

mses = []
tds = []
tpis = []
costs = []
mse_ess = []
for method in methods:
    print(method)
    tt = time()
    if method=='hens':
        mod = HensmanGP(X, y, M, jit = jit, es_patience = es_patience)
    elif method=='four':
        #mod = FFGP(X, y, M, jit = jit, es_patience = 10000)
        #print("Biggest boi")
        mod = FFGP(X, y, M, jit = jit, es_patience = es_patience)
        Knm = mod.get_Knm(X, mod.params)
        print(np.max(np.abs(Knm)))
    elif method=='m2':
        mod = M2GP(X, y, M, D=D, jit = jit, es_patience = es_patience)
    elif method=='sphere':
        key = jax.random.PRNGKey(42)

        NUM_FREQUENCIES = 7
        PHASE_TRUNCATION = 30
        k = NTK(depth=5)
        sh = SphericalHarmonics(num_frequencies=NUM_FREQUENCIES, phase_truncation=PHASE_TRUNCATION)
        print("Sphere is using fixed M.")
        lik = Gaussian()
        q = VariationalDistributionTriL()
        m = GP(k)
        m_new = m.conditional(sh, q)
        data_dict = {"x": X, "y": y}
        from datasets import Dataset
        dataset = Dataset.from_dict(data_dict).with_format("jax", dtype=jnp.float64)
        param = m_new.init(
            key,
            input_dim=P,
            num_independent_processes=1,
            likelihood=lik,
            sh_features=sh,
            variational_dist=q,
        )
        param = param.set_trainable(collection=k.name, variance=True)
        train_step = create_training_step(m_new, dataset, ("x", "y"), q, lik)
        param_new, state, elbos = m_new.fit(param, train_step, optax.adam(lr), max_iters)
        pred_mu, pred_var = m_new.predict_diag(param_new, XX)
        yy_hat = pred_mu
    elif method=='gpytorch':

        class GPModel(ApproximateGP):
            def __init__(self, inducing_points):
                variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
                variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
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

        inducing_points = train_x[:M, :]
        model = GPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.double()
        likelihood.double()

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

        model.eval()
        likelihood.eval()
        yy_hat = model(test_x).mean.detach().numpy()
        #means = torch.tensor([0.])
        #with torch.no_grad():
        #    for x_batch, y_batch in test_loader:
        #        preds = model(x_batch)
        #        means = torch.cat([means, preds.mean.cpu()])
        #yy_hat = means[1:].detach().numpy()
    else:
        raise Exception("Unknown method!")
    if method in my_methods:
        mod.fit(verbose=verbose, lr=lr, iters=max_iters, debug = debug, mb_size = mb_size, track = track)
        yy_hat = mod.pred(XX)

    td = time()-tt
    tpi = td / max_iters
    mse = jnp.mean(jnp.square(yy_hat-yy))

    mses.append(mse)
    tds.append(td)
    tpis.append(tpi)

    costs.append(mod.costs)
    mse_ess.append(mod.mse_es)
    if not manual:
        del mod

## For debugging, show optimization cost.
fname = 'temp.png' if manual else figsdir+"/"+sim_id+".png"
fig = plt.figure(figsize=[5,3])
for mi,method in enumerate(methods):
    plt.subplot(1,len(methods),1+mi)
    plt.plot(costs[mi])
    plt.title(method)
    ax = plt.gca()
    ax1 = ax.twinx()
    ax1.plot(100*np.arange(len(mse_ess[mi])),mse_ess[mi], color = 'green')
plt.tight_layout()
plt.savefig(fname)
plt.close()

## Save simulation results.
if manual:
    for i in range(10):
        print("manual!")
    print(mses)
else:
    #df = pd.DataFrame([['hen',mse, td, tpi, M, seed], ['m2',mse2, td2, tpi2, M, seed]])
    dat = []
    for mi,meth in enumerate(methods):
        dat.append([meth,mses[mi], tds[mi], tpis[mi], M, seed])
    df = pd.DataFrame(dat)
    df.columns = ['Method','MSE','Time', 'TPI', 'M', 'seed']
    df.to_csv(simdir+'/'+sim_id+'.csv')

### manual sigma2 est.
#####
mod.plot()
#ell = np.exp(mod.params['ell'])
#sss = 3000
#subset = np.random.choice(mod.X.shape[0], sss, replace = False)
#Xs = mod.X[subset,:]
#ys = mod.y[subset]
#C = mod.get_K(Xs, Xs, ell, 1.)
#gI = mod.eps_nug * np.eye(sss)
#sigma2_est = np.sum(ys*jnp.linalg.solve(C+gI,ys)) / sss
#print(sigma2_est)
#print(np.exp(mod.params['sigma2']))
