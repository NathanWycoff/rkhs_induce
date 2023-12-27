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
M = 10
P = 1

reps = 10

do1 = False

mse = np.zeros(reps)
if do1:
    msem1 = np.zeros(reps)
msem2 = np.zeros(reps)

verbose = True

for it in tqdm(range(reps)):
    X = np.random.uniform(size=[N,P])
    XX = np.random.uniform(size=[NN,P])
    y = np.cos(4*np.pi*np.sum(X, axis = 1))
    yy = np.cos(4*np.pi*np.sum(XX, axis = 1))

    mu_y = np.mean(y)
    sig_y = np.std(y)
    y = (y-mu_y) / sig_y
    yy = (yy-mu_y) / sig_y

    mod = SGGP(X, y, M)
    mod.fit(verbose=verbose)
    yy_hat = mod.pred(XX)
    mse[it] = jnp.mean(jnp.square(yy_hat-yy))

    if do1:
        mod1 = M1GP(X, y, M)
        mod1.fit(verbose=verbose)
        yy_hat_1 = mod1.pred(XX)
        msem1[it] = jnp.mean(jnp.square(yy_hat_1-yy))

    mod2 = M2GP(X, y, M,D=5)
    mod2.fit(verbose=verbose)
    yy_hat_2 = mod2.pred(XX)
    msem2[it] = jnp.mean(jnp.square(yy_hat_2-yy))


fig = plt.figure()
trans = np.log10
plt.title("P=%d"%P)
if do1:
    meths = [trans(mse),trans(msem1),trans(msem2)]
else:
    meths = [trans(mse),trans(msem2)]
plt.boxplot(meths)
plt.savefig("bakeoff.pdf")
plt.close()
