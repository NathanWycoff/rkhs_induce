#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  laz_lab.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.19.2024

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

np.random.seed(123)

debug = True
max_iters = 2000
#lr = 1e-4
lr = 1e-3

M = 50
P = 1
N = 2000
NN = 500

X = np.random.uniform(size=[N,P])
XX = np.random.uniform(size=[NN,P])
y = np.cos(4*np.pi*np.sum(X, axis = 1))
yy = np.cos(4*np.pi*np.sum(XX, axis = 1))

#mod = FFGP(X, y, M=M)
mod = M2GP(X, y, M=M)
pred_pre = mod.pred(XX)
mod.fit(verbose=verbose, lr=lr, iters=max_iters, debug = debug, mb_size = mb_size)
pred_post = mod.pred(XX)

fig = plt.figure(figsize=[8,3])
if P==1:
    plt.subplot(1,2,1)
    plt.scatter(XX[:,0], yy, label = 'True')
    plt.scatter(XX[:,0], pred_pre, label = 'pre')
    plt.scatter(XX[:,0], pred_post, label = 'post')
    plt.scatter(X[:,0], y, label = 'Train')
    plt.legend()

plt.subplot(1,2,2)
plt.plot(mod.costs)

plt.savefig("temp.png")
plt.close()
