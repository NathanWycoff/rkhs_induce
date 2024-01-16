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

for i in range(10):
    print("Early stopping?")

config.update("jax_enable_x64", True)

exec(open("python/jax_vsgp_lib.py").read())
exec(open("python/sim_settings.py").read())

#manual = False
manual = True

if manual:
    for i in range(10):
        print("manual!")
    #M = 10
    M = 120
    #max_iters = 2000
    #max_iters = 6000
    #max_iters = 5000 # 0.109 0.074
    #max_iters = 10000 #Errors are 0.084 and 0.041
    #max_iters = 15000 # Errors are 0.085 and 0.053
    max_iters = 30000 # Errors are 0.103 and 0.089
    lr = 1e-3
    #seed = 0
    seed = 5
    debug = True
    jit = True
    track = False
else:
    M = int(sys.argv[1])
    seed = int(sys.argv[2])
    debug = False
    jit = True

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
mod = HensmanGP(X, y, M, jit = jit)
mod.fit(verbose=verbose, lr=lr, iters=20, debug = debug, mb_size = mb_size, track = track)

## Run vanilla method.
tt = time()
mod = HensmanGP(X, y, M, jit = jit)
mod.fit(verbose=verbose, lr=lr, iters=max_iters, debug = debug, mb_size = mb_size, track=track)
td = time()-tt
tpi = td / mod.last_it

yy_hat = mod.pred(XX)
mse = jnp.mean(jnp.square(yy_hat-yy))

fig = plt.figure()
plt.plot(mod.mse_es)
plt.savefig("es.png")
plt.close()

#print(mse)
#
#fig = plt.figure()
#plt.plot(np.sqrt(np.sum(np.square(mod.tracking['theta1']), axis = 0)))
#ax = plt.gca()
#ax1 = ax.twinx()
#ax1.plot(mod.costs)
#plt.savefig("yeboi.png")
#plt.close()
#
#fig = plt.figure()
#plt.plot(mod.costs, label = 'cost')
#plt.tight_layout()
#plt.savefig("cost.png")
#plt.close()
#
#fig = plt.figure()
##plt.plot(mod.costs, label = 'cost')
#plt.subplot(2,2,1)
#plt.plot(mod.nll, label = 'nll')
#plt.title('nll')
#plt.subplot(2,2,2)
#plt.plot(mod.tr1, label = 'tr1')
#plt.title('tr1')
#plt.subplot(2,2,3)
#plt.plot(mod.tr2, label = 'tr2')
#plt.title('tr2')
#plt.subplot(2,2,4)
#plt.plot(mod.kl, label = 'kl')
#plt.title('kl')
#plt.tight_layout()
#plt.savefig("costs.png")
#plt.close()

## Run new method.
tt = time()
mod2 = M2GP(X, y, M,D=D, jit = jit)
mod2.fit(verbose=verbose, lr=lr, iters=max_iters, debug = debug, mb_size = mb_size)
td2 = time()-tt
tpi2 = td2 / mod2.last_it

yy_hat_2 = mod2.pred(XX)
mse2 = jnp.mean(jnp.square(yy_hat_2-yy))

## For debugging, show optimization cost.
fname = 'temp.png' if manual else figsdir+"/"+sim_id+".png"
fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(mod.costs)
plt.title("Tit-")
plt.subplot(1,2,2)
plt.plot(mod2.costs)
plt.title("M2-")
plt.tight_layout()
plt.savefig(fname)
plt.close()

## Save simulation results.
if manual:
    for i in range(10):
        print("manual!")
    print(mse)
    print(mse2)

else:
    df = pd.DataFrame([['hen',mse, td, tpi, M, seed], ['m2',mse2, td2, tpi, M, seed]])
    df.columns = ['Method','MSE','Time', 'TPI', 'M', 'seed']
    df.to_csv(simdir+'/'+sim_id+'.csv')

