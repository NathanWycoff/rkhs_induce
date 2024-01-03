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
import sys
import pandas as pd

config.update("jax_enable_x64", True)

exec(open("python/jax_vigp_class.py").read())
exec(open("python/sim_settings.py").read())

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

tt = time()
mod = TGP(X, y, M)
mod.fit(verbose=verbose, ls=ls, pc=pc, iters=max_iters)
yy_hat = mod.pred(XX)

td = time()-tt
mse = jnp.mean(jnp.square(yy_hat-yy))

tt = time()
mod2 = M2GP(X, y, M,D=D)
mod2.fit(verbose=verbose, ls=ls, pc=pc, iters=max_iters)
yy_hat_2 = mod2.pred(XX)

td2 = time()-tt
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
df = pd.DataFrame([['tit',mse, td, M, seed], ['m2',mse2, td2, M, seed]])
df.columns = ['Method','MSE','Time', 'M', 'seed']
df.to_csv(simdir+'/'+sim_id+'.csv')
