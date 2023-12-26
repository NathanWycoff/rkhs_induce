#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  inducing_bg.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.19.2023

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

N = 500
P = 2

g_nug = 1e-6

#ell = jnp.repeat(1e-1,P)
#ell_init = jnp.repeat(jnp.array(0., dtype = np.float64),P)
ell_init = jnp.repeat(jnp.array(np.log(1e-1), dtype = np.float64),P)
#ell_init = jnp.repeat(np.log(1e-1),P)

X = np.random.uniform(size=[N,P])
y = np.cos(4*np.pi*np.sum(X, axis = 1))
y = (y-np.mean(y)) / np.std(y)

#D = np.zeros([N,N])
#for i in range(N):
#    for j in range(N):
#        D[i,j] = np.square(np.sum(X[i,:]-X[j,:]))

kernel = lambda x,y, ell: jnp.exp(-jnp.sum(jnp.square(x-y)/ell))
get_K = lambda X1, X2, ell: jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y, ell))(X2))(X1)

def nll_pre(params):
    ell = jnp.exp(params['ell'])
    K = get_K(X,X,ell) + g_nug*jnp.eye(N)
    Kiy = jnp.linalg.solve(K, y)
    ldetK = jnp.linalg.slogdet(K)[1]
    yKiy = y@Kiy
    #tau2_hat = yKiy / N
    ll = -0.5*ldetK - N/2*jnp.log(yKiy)
    return -ll

get_nll = jax.jit(nll_pre)
grad_nll = jax.grad(nll_pre)
vng_nll = jax.value_and_grad(get_nll)

#optax
#learning_rate = 5e-3
learning_rate = 1e-1
optimizer = optax.adam(learning_rate)
# Obtain the `opt_state` that contains statistics for the optimizer.
params = {'ell': ell_init}
opt_state = optimizer.init(params)

iters = 100
costs = np.zeros(iters)
for i in tqdm(range(iters)):
    cost, grad = vng_nll(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    costs[i] = cost

fig = plt.figure()
plt.plot(costs)
plt.savefig("temp.pdf")
plt.close()

## Jaxopt
#solver = jaxopt.LBFGS(fun=nll_pre, maxiter=10000)
#res = solver.run(ell_init)
#res.params
#nll_pre(res.params)

nll_pre({'ell':ell_init})
nll_pre({'ell':ell_init+np.log(1e-1)})
nll_pre({'ell':ell_init+np.log(1e-2)})

gsize = 20
#lg = np.linspace(-2,0,num=gsize)
lg = np.linspace(np.log(1e-3),2,num=gsize)
C = np.zeros([gsize,gsize])
for i in tqdm(range(gsize)):
    for j in range(gsize):
        ellij = jnp.array([lg[i],lg[j]])
        C[i,j] = get_nll({'ell':ellij})
        if np.isnan(C[i,j]):
            raise Exception()

fig = plt.figure()
# from https://stackoverflow.com/questions/44260491/matplotlib-how-to-make-imshow-read-x-y-coordinates-from-other-numpy-arrays
dx = (lg[1]-lg[0])/2.
extent = [lg[0]-dx, lg[-1]+dx, lg[0]-dx, lg[-1]+dx]
# from https://stackoverflow.com/questions/44260491/matplotlib-how-to-make-imshow-read-x-y-coordinates-from-other-numpy-arrays
plt.imshow(C, extent = extent, origin="upper")
plt.colorbar()
plt.savefig("surf.pdf")
plt.close()

#iters = 100
#costs = np.zeros(iters)
#
#lr = 1e-7
#
#for i in range(iters):
#    nll, grad = vng_nll(ell)
#    ell = ell - lr * grad
#    costs[i] = nll
#
#fig = plt.figure()
#plt.plot(costs)
#plt.savefig("costs.pdf")
#plt.close()

get_K(X, X, ell)

### Plot surface
triang = mtri.Triangulation(X[:,0], X[:,1])
refiner = mtri.UniformTriRefiner(triang)
tri_refi, z_test_refi = refiner.refine_field(y, subdiv=4)
plt.figure()
plt.gca().set_aspect('equal')
cmap = cm.get_cmap(name='terrain')
plt.tricontourf(tri_refi, z_test_refi, cmap=cmap)
plt.colorbar()
plt.savefig("temp.pdf")
plt.close()
