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
from time import time

config.update("jax_enable_x64", True)

exec(open("python/jax_vigp_class.py").read())

np.random.seed(123)

N = 2000
NN = 500
#M = 5
M = 25
#P = 1
P = 2

#reps = 30
reps = 10
#reps = 2

do1 = False

## Optimization params
ls = 'backtracking'
#pc = 'id'
pc = 'exp_ada'
max_iters = 250

##  What should D be set to?
#D = 5
D = int(np.ceil(np.sqrt(M)))

mse = np.zeros(reps)
times = np.zeros(reps)
msem2 = np.zeros(reps)
timesm2 = np.zeros(reps)

verbose = False

fig = plt.figure()
for it in tqdm(range(reps)):
    np.random.seed(it)
    X = np.random.uniform(size=[N,P])
    XX = np.random.uniform(size=[NN,P])
    y = np.cos(4*np.pi*np.sum(X, axis = 1))
    yy = np.cos(4*np.pi*np.sum(XX, axis = 1))

    mu_y = np.mean(y)
    sig_y = np.std(y)
    y = (y-mu_y) / sig_y
    yy = (yy-mu_y) / sig_y

    np.random.seed(it+reps)
    tt = time()
    mod = TGP(X, y, M)
    #mod.fit(verbose=verbose)
    mod.fit(verbose=verbose, ls=ls, pc=pc, iters=max_iters)
    yy_hat = mod.pred(XX)
    td = time()-tt

    mse[it] = jnp.mean(jnp.square(yy_hat-yy))
    times[it] = td

    np.random.seed(it+2*reps)
    tt = time()
    mod2 = M2GP(X, y, M,D=D)
    #mod2.fit(verbose=verbose)
    mod2.fit(verbose=verbose, ls=ls, pc=pc, iters=max_iters)
    yy_hat_2 = mod2.pred(XX)
    td = time()-tt

    msem2[it] = jnp.mean(jnp.square(yy_hat_2-yy))
    timesm2[it] = td

    plt.subplot(reps,2,2*it+1)
    plt.plot(mod.costs)
    ax = plt.gca()
    axv = ax.twinx()
    axv.plot(np.log10(mod.ss), color = 'green')
    plt.title("Tit-"+str(it))
    plt.subplot(reps,2,2*it+2)
    plt.plot(mod2.costs)
    ax = plt.gca()
    axv = ax.twinx()
    axv.plot(np.log10(mod2.ss), color = 'green')
    plt.title("M2-"+str(it))

    if np.isnan(msem2[it]):
        print("ye boi")
        break

    if P==1:
        fig = plt.figure()
        plt.scatter(XX[:,0],yy, label = 'True')
        plt.scatter(XX[:,0], mod.pred(XX), label = 'titsias')
        plt.scatter(XX[:,0], mod2.pred(XX), label = 'yaboinate')
        plt.legend()
        plt.savefig("post.pdf")
        plt.close()
plt.tight_layout()
plt.savefig("bakeoff_costs.pdf")
plt.close()

fig = plt.figure()
plt.subplot(2,1,1)
trans = np.log10
plt.title("P=%d"%P)
if do1:
    meths = [trans(mse),trans(msem1),trans(msem2)]
else:
    #meths = [trans(mse),trans(msem2)]
    meths = mse-msem2
plt.boxplot(meths)

plt.subplot(2,1,2)
plt.scatter(np.arange(reps),trans(mse), label = 'titsias')
plt.scatter(np.arange(reps),trans(msem2), label = 'yaboi')

plt.savefig("bakeoff.pdf")
plt.close()
