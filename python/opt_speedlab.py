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
from time import time
import sys
import pandas as pd

exec(open("python/jax_vigp_class.py").read())
exec(open("python/sim_settings.py").read())

manual = True

if manual:
    for i in range(10):
        print("Manual settings.")
    M = 50
    seed = 1
    verbose = True
    #max_iters = 5

    ## Optimization params
    ls = 'backtracking'
    #ls = 'fixed_lr'
    #pc = 'id'
    pc = 'exp_ada'
    max_iters = 250

else:
    M = int(sys.argv[1])
    seed = int(sys.argv[2])

sim_id = str(M)+'_'+str(seed)

np.random.seed(seed)

P = 2
N = 2000
NN = 500
D = get_D(M)

X = np.random.uniform(size=[N,P])
XX = np.random.uniform(size=[NN,P])
y = np.cos(4*np.pi*np.sum(X, axis = 1))
yy = np.cos(4*np.pi*np.sum(XX, axis = 1))

mu_y = np.mean(y)
sig_y = np.std(y)
y = (y-mu_y) / sig_y
yy = (yy-mu_y) / sig_y

np.random.seed(1)
tt = time()
mod = TGP(X, y, M)
print("Object created after %f seconds"%(time()-tt))
mod.fit(verbose=verbose, ls=ls, iters=max_iters)

print(mod.get_elbo(mod.params))
mod.params['Z'] = mod.params['Z']*0+100.
mod.params['sigma2'] = 0*mod.params['sigma2']-1.25
print(mod.get_elbo(mod.params))
