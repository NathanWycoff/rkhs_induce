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

N = 2000
NN = 500
#M = 16
M = 50
#P = 1
P = 2

## Optimization params
ls = 'backtracking'
#pc = 'id'
pc = 'exp_ada'
max_iters = 250

verbose = True

it = 0

np.random.seed(it)
X = np.random.uniform(size=[N,P])
XX = np.random.uniform(size=[NN,P])
y = np.cos(4*np.pi*np.sum(X, axis = 1))
yy = np.cos(4*np.pi*np.sum(XX, axis = 1))

mu_y = np.mean(y)
sig_y = np.std(y)
y = (y-mu_y) / sig_y
yy = (yy-mu_y) / sig_y

np.random.seed(it+1)
mod = TGP(X, y, M)
#mod.fit(verbose=verbose)
mod.fit(verbose=verbose, ls=ls, pc=pc, iters=max_iters)
yy_hat = mod.pred(XX)
print(jnp.mean(jnp.square(yy_hat-yy)))

#mod.params
