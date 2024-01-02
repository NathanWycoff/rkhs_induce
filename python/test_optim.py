#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  test_optim.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.31.2023

exec(open("python/jax_vigp_class.py").read())

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from tensorflow_probability.substrates import jax as tfp
import jaxopt
from jax import config
from tqdm import tqdm
import optax

config.update("jax_enable_x64", True)

exec(open("python/jax_vigp_class.py").read())

np.random.seed(123)

N = 200
NN = 500
M = 4
P = 1
lr = 1e0

func = lambda X: np.cos(4*np.pi*np.sum(X, axis = 1))

X = np.random.uniform(size=[N,P])
XX = np.random.uniform(size=[NN,P])
y = func(X)
yy = func(XX)

mu_y = np.mean(y)
sig_y = np.std(y)
y = (y-mu_y) / (sig_y+1e-8)
yy = (yy-mu_y) / (sig_y+1e-8)

### Basis independence of prediction.
modf = TGP(X,y,M=M)
modb = TGP(X,y,M=M)
moda = TGP(X,y,M=M)

modf.fit(iters=500,ls='fixed_lr',ls_params={'lr':1e-6})
modb.fit(iters=500,ls='backtracking')
moda.fit(iters=500,ls='backtracking',pc='exp_ada',pc_params={'beta2':0.0})
#moda.fit(iters=100,ls='backtracking',pc='exp_ada')

print(np.nanmin(modf.costs))
print(np.nanmin(modb.costs))
print(np.nanmin(moda.costs))

moda.opt.pc_vars

fig = plt.figure()
plt.scatter(XX[:,0],yy, label = 'True')
plt.scatter(XX[:,0], modf.pred(XX), label = 'fixed_lr')
plt.scatter(XX[:,0], modb.pred(XX), label = 'backtracking')
plt.scatter(XX[:,0], moda.pred(XX), label = 'ada_backtracking')
plt.legend()
plt.savefig("post.pdf")
plt.close()

