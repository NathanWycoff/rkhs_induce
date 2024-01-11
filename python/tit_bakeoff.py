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

config.update("jax_enable_x64", True)

exec(open("python/jax_vigp_class.py").read())
exec(open("python/sim_settings.py").read())

manual = False
#manual = True

if manual:
    for i in range(10):
        print("manual!")
    M = 10
    seed = 0
    debug = True
else:
    M = int(sys.argv[1])
    seed = int(sys.argv[2])
    debug = False

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
else:
    raise Exception("Unknown problem.")

## Rescaling
mu_y = np.mean(y)
sig_y = np.std(y)
y = (y-mu_y) / sig_y
yy = (yy-mu_y) / sig_y

min_X = np.min(X, axis = 0)
max_X = np.max(X, axis = 0)
X = (X-min_X[np.newaxis,:]) / (max_X-min_X)[np.newaxis,:]
XX = (XX-min_X[np.newaxis,:]) / (max_X-min_X)[np.newaxis,:]

## Run a short one to compile things
mod = TGP(X, y, M)
mod.fit(verbose=verbose, ls=ls, iters=20, debug = debug)

## Run vanilla method.
tt = time()
mod = TGP(X, y, M)
mod.fit(verbose=verbose, ls=ls, iters=max_iters, debug = debug)
yy_hat = mod.pred(XX)

td = time()-tt
tpi = td / mod.opt.out_it
mse = jnp.mean(jnp.square(yy_hat-yy))

## Run new method.
tt = time()
mod2 = M2GP(X, y, M,D=D)
mod2.fit(verbose=verbose, ls=ls, iters=max_iters, debug = debug)
yy_hat_2 = mod2.pred(XX)

td2 = time()-tt
tpi2 = td2 / mod2.opt.out_it
mse2 = jnp.mean(jnp.square(yy_hat_2-yy))

## For debugging, show optimization cost.
fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(mod.costs)
ax = plt.gca()
axv = ax.twinx()
axv.plot(np.log10(mod.ss), color = 'green')
plt.title("Tit-")
plt.subplot(1,2,2)
plt.plot(mod2.costs)
ax = plt.gca()
axv = ax.twinx()
axv.plot(np.log10(mod2.ss), color = 'green')
plt.title("M2-")
plt.tight_layout()
plt.savefig(figsdir+"/"+sim_id+".pdf")
plt.close()

## Save simulation results.
df = pd.DataFrame([['tit',mse, td, tpi, M, seed], ['m2',mse2, td2, tpi2, M, seed]])
df.columns = ['Method','MSE','Time', 'TPI', 'M', 'seed']
df.to_csv(simdir+'/'+sim_id+'.csv')

if manual:
    for i in range(10):
        print("manual!")
