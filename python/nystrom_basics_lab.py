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
import jaxopt
from jax import config
from tqdm import tqdm
import optax

config.update("jax_enable_x64", True)

exec(open("python/jax_vigp_class.py").read())

np.random.seed(123)

N = 20
NN = 500
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
mod1 = M2GP(X,y,M=2,D=2)
mod1.params['A'] = jnp.array([[1.,0.],[0.,1.]])
mod1.params['Z'] = jnp.array([[[0.2],[0.8]],[[0.2],[0.8]]])

mod2 = M2GP(X,y,M=2,D=2)
mod2.params['A'] = jnp.array([[1.,1.],[1.,-1.]])
mod2.params['Z'] = jnp.array([[[0.2],[0.8]],[[0.2],[0.8]]])

mod1.get_Kmm(mod1.params)
mod2.get_Kmm(mod2.params)

print(mod1.pred(XX)[:10])
print(mod2.pred(XX)[:10])

fig = plt.figure()
plt.scatter(XX[:,0],yy, label = 'True')
plt.scatter(XX[:,0], mod1.pred(XX), label = '1')
plt.scatter(XX[:,0], mod2.pred(XX), label = '2')
ax = plt.gca()
mm, ma = ax.get_ylim()
plt.vlines(X[:,0],ymin=mm, ymax=ma)
plt.legend()
plt.savefig("basis_independence.pdf")
plt.close()

### Use projected <x~, X> instead of exact <x~,X>?
np.random.seed(123)
fig = plt.figure(figsize=[12,12])
iters = 9
ncols = 3
nrows = int(np.ceil(iters/ncols))
for pi in range(iters):
    mod = SGGP(X,y,M=2)

    ell = jnp.exp(mod.params['ell'])
    sigma2 = jnp.exp(mod.params['sigma2'])

    Knm = mod.get_Knm(mod.params)
    Kmm = mod.get_Kmm(mod.params)
    kstar = mod.get_K(XX, mod.X, ell)

    k_proj_star = mod.get_K(XX, mod.params['Z'], ell) @ np.linalg.solve(Kmm, Knm.T)

    preds0 = mod.pred(XX)
    preds1 = kstar @ np.linalg.solve(Knm @ np.linalg.solve(Kmm, Knm.T) + sigma2*np.eye(N), y)
    preds2 = k_proj_star @ np.linalg.solve(Knm @ np.linalg.solve(Kmm, Knm.T) + sigma2*np.eye(N), y)

    print(preds0[:10])
    print(preds1[:10])
    print(preds2[:10])

    plt.subplot(nrows,ncols,pi+1)
    plt.scatter(XX[:,0],yy, label = 'True')
    plt.scatter(XX[:,0], preds1, label = 'exact')
    plt.scatter(XX[:,0], preds2, label = 'projected')
    ax = plt.gca()
    mm, ma = ax.get_ylim()
    plt.vlines(X[:,0],ymin=mm, ymax=ma)
    plt.legend()
plt.tight_layout()
plt.savefig("proj_kstar.pdf")
plt.close()

