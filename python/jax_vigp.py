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

#np.random.seed(123)

N = 2000
NN = 500
M = 100
P = 1

g_nug = 1e-6

ell_init = jnp.repeat(jnp.array(np.log(1e-1), dtype = np.float64),P)
sigma2_init = jnp.array(np.log(1e-5), dtype = np.float64)

X = np.random.uniform(size=[N,P])
XX = np.random.uniform(size=[NN,P])
Z_init = jnp.array(np.random.uniform(size=[M,P]))
y = np.cos(4*np.pi*np.sum(X, axis = 1))
y = (y-np.mean(y)) / np.std(y)

kernel = lambda x,y, ell: jnp.exp(-jnp.sum(jnp.square(x-y)/ell))
get_K = lambda X1, X2, ell: jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y, ell))(X2))(X1)

def elbo_pre(params):
    ell = jnp.exp(params['ell'])
    sigma2 = jnp.exp(params['sigma2'])
    Z = params['Z']

    Knm = get_K(X, Z, ell)
    Kmm = get_K(Z, Z, ell)
    Knn = get_K(X, X, ell)

    ## Direct Ktilde computation
    ##Kmmi = jnp.linalg.inv(Kmm + g_nug*jnp.eye(M))
    #Qnn = Knm @ jnp.linalg.solve(Kmm + g_nug*jnp.eye(M), Knm.T)
    #Ktilde = Knn - Qnn
    #ktilde = jnp.diag(Ktilde)

    # Compute diag of Ktilde directly.
    Kmmi = jnp.linalg.inv(Kmm+g_nug*jnp.eye(M)) #Even faster with chol
    q_diag = jax.vmap(lambda k: k.T @ Kmmi @ k)(Knm)
    # Warning: assumes that correlation matrix diagonal is 1.
    ktilde = jnp.ones(N) - q_diag

    ### Manual likelihood calc
    #QI = Qnn+sigma2*jnp.eye(N)
    #ldetK = jnp.linalg.slogdet(QI)[1]
    #yKiy = y@jnp.linalg.solve(QI, y)
    ##tau2_hat = yKiy / N
    #ll = -0.5*ldetK - N/2*jnp.log(yKiy)

    ## Using TFP full covar
    #QI = Qnn+sigma2*jnp.eye(N)
    #dist = tfp.distributions.MultivariateNormalFullCovariance(covariance_matrix = QI)
    #ll = dist.log_prob(y)

    ## Using TFP diag + LR
    ed = jnp.linalg.eigh(Kmm+g_nug*jnp.eye(M))
    U = Knm @ ed[1] @ jnp.diag(jnp.sqrt(1/ed[0]))
    #U @ U.T - Qnn = 0.
    dist = tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(cov_diag_factor = sigma2*jnp.ones(N), cov_perturb_factor = U)
    ll = dist.log_prob(y)

    reg = 1./(2.*sigma2)*jnp.sum(ktilde)

    return -ll + reg

def pred(params, XX, X, y):
    ell = jnp.exp(params['ell'])
    sigma2 = jnp.exp(params['sigma2'])
    Z = params['Z']

    Knm = get_K(X, Z, ell)
    Kmm = get_K(Z, Z, ell)
    Knn = get_K(X, X, ell)

    Qnn = Knm @ jnp.linalg.solve(Kmm + g_nug*jnp.eye(M), Knm.T)

    kstar = get_K(XX, X, ell)
    QI = Qnn+sigma2*jnp.eye(N)
    ret = kstar @ np.linalg.solve(QI, y)
    return ret

get_elbo = jax.jit(elbo_pre)
grad_elbo = jax.grad(elbo_pre)
vng_elbo = jax.value_and_grad(get_elbo)

#learning_rate = 1e-8
learning_rate = 1e-1
#learning_rate = 1e-4
optimizer = optax.adam(learning_rate)
params = {'ell': ell_init, 'sigma2' : sigma2_init, 'Z' : Z_init}
opt_state = optimizer.init(params)

iters = 100
Zs = np.zeros([M,iters])
costs = np.zeros(iters)
for i in tqdm(range(iters)):
    cost, grad = vng_elbo(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    costs[i] = cost
    Zs[:,i] = np.array(params['Z'].copy()).flatten()

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(costs)
plt.subplot(2,1,2)
plt.plot(Zs.T)
plt.savefig("cost.pdf")
plt.close()

yy_hat = pred(params, XX, X, y)

fig = plt.figure()
plt.scatter(X[:,0], y)
plt.scatter(XX[:,0], yy_hat)
plt.savefig("pred.pdf")
plt.close()
