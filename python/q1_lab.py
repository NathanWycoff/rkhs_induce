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
#M = 10
M = 2
P = 1
lr = 1e0

#func = lambda X: np.cos(4*np.pi*np.sum(X, axis = 1))
func = lambda X: np.ones([X.shape[0]])

X = np.random.uniform(size=[N,P])
XX = np.random.uniform(size=[NN,P])
y = func(X)
yy = func(XX)

mu_y = np.mean(y)
sig_y = np.std(y)
y = (y-mu_y) / (sig_y+1e-8)
yy = (yy-mu_y) / (sig_y+1e-8)

modb = SGGP(X,y,M=M)
mod2 = M2GP(X,y,M=M,D=100)

modb.fit(learning_rate=lr)
mod2.fit(learning_rate=lr)

fig = plt.figure()
plt.scatter(XX[:,0],yy, label = 'True')
plt.scatter(XX[:,0], modb.pred(XX), label = 'Snelson-Ghahramani')
plt.scatter(XX[:,0], mod2.pred(XX), label = 'M2')
plt.legend()
plt.savefig("temp.pdf")
plt.close()


# M2 debugging
np.random.seed(123)

N = 2
NN = 500
M = 1
P = 1
lr = 1e0

func = lambda X: np.cos(4*np.pi*np.sum(X, axis = 1))
#func = lambda X: np.ones([X.shape[0]])

X = np.random.uniform(size=[N,P])
XX = np.random.uniform(size=[NN,P])
y = func(X)
yy = func(XX)

mu_y = np.mean(y)
sig_y = np.std(y)
y = (y-mu_y) / (sig_y+1e-8)
yy = (yy-mu_y) / (sig_y+1e-8)

modb = SGGP(X,y,M=M)
#modb.params['Z'] = jnp.array(0.5*np.ones([M,P]))
modb.params['Z'] = jnp.array(0.25*np.ones([M,P]))
modb.g_nug = 1e-2

#mod2 = M2GP(X,y,M=M,D=1)
#mod2.params['A'] = jnp.array([[1.]])
#mod2.params['Z'] = jnp.array([[0.5]])

fig = plt.figure()
plt.scatter(XX[:,0],yy, label = 'True')
plt.scatter(XX[:,0], modb.pred(XX), label = 'Snelson-Ghahramani')
#plt.scatter(XX[:,0], mod2.pred(XX), label = 'M2')
ax = plt.gca()
mm, ma = ax.get_ylim()
plt.vlines(X[:,0],ymin=mm, ymax=ma)
plt.legend()
plt.savefig("temp.pdf")
plt.close()

