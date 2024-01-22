#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  vax_vigp_bakeoff.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.24.2023

# was 45 iters a second

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

exec(open("python/jax_vsgp_lib.py").read())
exec(open("python/sim_settings.py").read())

manual = False
#manual = True

if manual:
    for i in range(10):
        print("manual!")
    #M = 1000 #0.18213301
    #M = 600 # 0.24
    #M = 320 # 0.34
    #M = 128  # 0.4
    M = 128
    max_iters = 30000 
    #max_iters = 2000
    #lr = 1e-3
    seed = 0
    #seed = 5
    debug = True
    jit = True
    track = False
    #methods = ['four']
    #methods = ['hens']
else:
    M = int(sys.argv[1])
    seed = int(sys.argv[2])
    debug = False
    jit = True
    track = False

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
    tt = time()
    if method=='hens':
        mod = HensmanGP(X, y, M, jit = jit)
    elif method=='four':
        #mod = FFGP(X, y, M, jit = jit, es_patience = 10000)
        #print("Biggest boi")
        mod = FFGP(X, y, M, jit = jit)
        Knm = mod.get_Knm(X, mod.params)
        print(np.max(np.abs(Knm)))
    elif method=='m2':
        mod = M2GP(X, y, M, D=D, jit = jit)
    else:
        raise Exception("Unknown method!")
    mod.fit(verbose=verbose, lr=lr, iters=max_iters, debug = debug, mb_size = mb_size, track = track)

    td = time()-tt
    tpi = td / mod.last_it

    yy_hat = mod.pred(XX)
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
