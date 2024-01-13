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

N = 2000
NN = 500
M = 100
P = 1

verbose = True
debug = True
ls = 'backtracking'

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

mod = HensmanGP(X, y, M=M, jit = False)
pred_pre = mod.pred(XX)
mod.fit(verbose=verbose, ls=ls, iters=200, debug = debug)
pred_post = mod.pred(XX)

#mod.params
#np.linalg.eigh(mod.params['S'])
#S = mod.params['S']
#np.max(np.abs(S - S.T))

fig = plt.figure(figsize=[10,3])
plt.subplot(1,3,1)
plt.scatter(XX[:,0], yy, label = 'True')
plt.scatter(XX[:,0], pred_pre, label = 'pre')
plt.scatter(XX[:,0], pred_post, label = 'post')
plt.legend()

plt.subplot(1,3,2)
plt.scatter(XX[:,0], yy, label = 'True')
plt.scatter(XX[:,0], pred_post, label = 'post')
plt.legend()

plt.subplot(1,3,3)
plt.plot(mod.costs)

plt.savefig("temp.pdf")
plt.close()
