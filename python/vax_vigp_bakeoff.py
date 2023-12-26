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

#np.random.seed(123)

N = 2000
NN = 500
M = 10
P = 2

g_nug = 1e-6

iters = 1000

mse = np.zeros(iters)
mseme = np.zeros(iters)

for it in tqdm(range(iters)):
    X = np.random.uniform(size=[N,P])
    XX = np.random.uniform(size=[NN,P])
    y = np.cos(4*np.pi*np.sum(X, axis = 1))
    yy = np.cos(4*np.pi*np.sum(XX, axis = 1))

    mu_y = np.mean(y)
    sig_y = np.std(y)
    y = (y-mu_y) / sig_y
    yy = (yy-mu_y) / sig_y

    mod = vargp(X, y, M)
    modme = vargp_rkhs(X, y, M)

    mod.fit(verbose=False)
    modme.fit(verbose=False)

    mod.params['Z']

    yy_hat = mod.pred(XX)
    yy_hat_me = modme.pred(XX)

    #fig = plt.figure()
    #plt.scatter(X[:,0], y)
    #plt.scatter(XX[:,0], yy, label = 'test')
    #plt.scatter(XX[:,0], yy_hat, label = 'std')
    #plt.scatter(XX[:,0], yy_hat_me, label = 'rkhs')
    #plt.legend()
    #plt.savefig("pred.pdf")
    #plt.close()

    mse[it] = jnp.mean(jnp.square(yy_hat-yy))
    mseme[it] = jnp.mean(jnp.square(yy_hat_me-yy))

fig = plt.figure()
trans = np.log10
plt.title("P=%d"%P)
plt.boxplot([trans(mse),trans(mseme)])
plt.savefig("bakeoff.pdf")
plt.close()
