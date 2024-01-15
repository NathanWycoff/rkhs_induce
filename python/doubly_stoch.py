#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  doubly_stoch.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.13.2024

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from tensorflow_probability.substrates import jax as tfp
from jax import config
from tqdm import tqdm
import warnings
import optax

exec(open("python/jax_vsgp_lib.py").read())

np.random.seed(123)

#N = 5
N = 2000
NN = 500
#M = 100
#M = 20
M = 10
P = 1
iters = 2000
#iters = 500

verbose = True
debug = True
#ls = 'backtracking'
#ls = 'lipschitz'
#ls = 'armijo'
ls = 'fixed_lr'
#jit = True
jit = True

func = lambda X: np.cos(4*np.pi*np.sum(X, axis = 1))
#func = lambda X: np.ones([X.shape[0]])

X = np.random.uniform(size=[N,P])
XX = np.random.uniform(size=[NN,P])
y = func(X)
yy = func(XX)

mu_y = np.mean(y)
sig_y = np.std(y)
y = (y-mu_y) / (sig_y+1e-8)
yy = (yy-mu_y) / (sig_y+1e-8)

mod = HensmanGP(X, y, M=M, jit = jit, natural = False)
fit_pre = mod.pred(X)
pred_pre = mod.pred(XX)
mod.fit(verbose=verbose, lr=5e-2, iters=iters, debug = debug)
fit_post = mod.pred(X)
pred_post = mod.pred(XX)

print(np.sum(np.square(fit_pre-y)))
print(np.sum(np.square(fit_post-y)))

#mod.params
#np.linalg.eigh(mod.params['S'])
#S = mod.params['S']
#np.max(np.abs(S - S.T))

fig = plt.figure(figsize=[8,3])
if P==1:
    plt.subplot(1,2,1)
    plt.scatter(XX[:,0], yy, label = 'True')
    plt.scatter(XX[:,0], pred_pre, label = 'pre')
    plt.scatter(XX[:,0], pred_post, label = 'post')
    plt.scatter(X[:,0], y, label = 'Train')
    plt.legend()

plt.subplot(1,2,2)
plt.plot(mod.costs)

plt.savefig("temp.pdf")
plt.close()
