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
from jax import config
from tqdm import tqdm
import warnings
import optax

PRECISION = '64'

if PRECISION=='64':
    config.update("jax_enable_x64", True)
    npdtype = np.float64
else:
    npdtype = np.float32

class VSGP(object):
    def __init__(self, X, y, M = 10, jit = True, natural = True, N_es = 512):
        self.meth_name = 'Variational Stochastic GP.'

        # Split off a chunk of data for early stopping.
        N,self.P = X.shape
        #N_es = int(np.ceil(prop_es * N))
        ind_es = np.random.choice(N, N_es, replace = False)
        ind_train = np.setdiff1d(np.arange(N), ind_es)
        X_es = X[ind_es,:]
        y_es = y[ind_es]
        X_train = X[ind_train,:]
        y_train = y[ind_train]
        self.X = X_train
        self.y = y_train
        self.X_es = X_es
        self.y_es = y_es
        self.N = self.X.shape[0]

        self.es_patience = 25 # Patience in epochs.
        #self.es_patience = 10 # Patience in epochs.

        self.M = M
        ell_init = jnp.repeat(jnp.array(np.log(1e-1), dtype = npdtype),P)
        # COVAR =  sigma2*K + (gamma2+g_nug)*I
        sigma2_init = jnp.array(np.log(1.), dtype = npdtype) # Scale Parameter.
        gamma2_init = jnp.array(np.log(1e-4), dtype = npdtype) # Error Variance.
        self.params = {'ell': ell_init, 'sigma2' : sigma2_init, 'gamma2' : gamma2_init}


        print("TODO: jit to 1e-8?")
        self.g_nug = 1e-6
        self.natural = natural
        self.tracking = {}

        self.kernel = lambda x,y, ell, sigma2: sigma2*jnp.exp(-jnp.sum(jnp.square(x-y)/ell))
        self.get_K = lambda X1, X2, ell, sigma2: jax.vmap(lambda x: jax.vmap(lambda y: self.kernel(x, y, ell, sigma2))(X2))(X1)

        self.jit = jit
        self.compile(jit)

        #DRY3
        m_init = jnp.zeros(self.M, dtype = npdtype)
        S_init = jnp.eye(self.M, dtype = npdtype)
        if self.natural:
            theta1_init = jnp.linalg.solve(S_init, m_init)
            theta2_init = -jnp.linalg.inv(S_init)/2
            self.params['theta1'] = theta1_init
            self.params['theta2'] = theta2_init
        else:
            self.params['m'] = m_init
            self.params['S'] = S_init
        #DRY3


    def get_Knm(self,params):
        raise NotImplementedError()

    def get_Kmm(self,params):
        raise NotImplementedError()

    def elbo_pre_individ(self, params, X, y):
        mb_size = X.shape[0]

        ## PRECOMPUTING THINGS
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        gamma2 = jnp.exp(params['gamma2'])
        # DRY1
        if self.natural:
            theta1 = params['theta1']
            theta2 = params['theta2']
            S = -1/2*jnp.linalg.inv(theta2)
            S = S.T @ S
            m = S@theta1
        else:
            m = params['m']
            S = params['S']
        # DRY1

        #trust_tfp = True
        trust_tfp = False
        assert not trust_tfp

        gI = self.g_nug*jnp.eye(self.M, dtype=npdtype)

        Knm = self.get_Knm(X, params)
        Kmm = self.get_Kmm(params)

        # Compute diag of Ktilde directly.
        Kmmi = jnp.linalg.inv(Kmm+self.g_nug*jnp.eye(self.M, dtype=npdtype)) #Even faster with chol
        q_diag = jax.vmap(lambda k: k.T @ Kmmi @ k)(Knm)
        ktilde = sigma2*jnp.ones(mb_size) - q_diag

        ## NOW COMPUTE TERMS OF LOSS FUNCTION
        mb_rescale = self.N/mb_size
        mu_y = Knm @ jnp.linalg.solve(Kmm+gI, m)
        if trust_tfp:
            dist_y = tfp.distributions.Normal(loc=mu_y, scale = jnp.sqrt(gamma2))
            nll = -jnp.sum(dist_y.log_prob(self.y))
        else:
            #nll = self.N/2.*jnp.log(gamma2) + 1./(2.*gamma2) * jnp.sum(jnp.square(self.y-mu_y))
            nll = self.N/2.*jnp.log(gamma2) + 1./(2.*gamma2) * mb_rescale*jnp.sum(jnp.square(y-mu_y))

        tr1 = mb_rescale * 1./(2.*gamma2) * jnp.sum(ktilde)

        KiSKi = Kmmi @ S @ Kmmi
        tr2 = mb_rescale * jnp.sum(Knm.T * (KiSKi @ Knm.T)) / gamma2

        ### KL via TFP
        if trust_tfp:
            dist_q = tfp.distributions.MultivariateNormalFullCovariance(loc=m, covariance_matrix=S)
            dist_p = tfp.distributions.MultivariateNormalFullCovariance(loc=jnp.zeros(self.M,dtype=npdtype), covariance_matrix=Kmm+gI)
            kl = dist_q.kl_divergence(dist_p)
        else:
            S_slogdet = jnp.linalg.slogdet(S)
            kl = jnp.linalg.slogdet(Kmm+gI)[1] - S_slogdet[1] + jnp.sum(jnp.diag(Kmmi@S)) + m.T @ Kmmi @ m - self.P 
            lam1 = np.min(jnp.linalg.eigh(S)[0])
            print("Todo: think. Jit?")
            kl = jax.lax.cond(lam1<-1e-10, lambda: np.inf, lambda: kl)

        return nll, tr1, tr2, kl

    def elbo_pre(self, params, X, y):
        a,b,c,d = self.elbo_pre_individ(params, X, y)
        loss = a+b+c+d
        return loss

    def pred(self, XX):
        gI = self.g_nug*jnp.eye(self.M, dtype=npdtype)
        # DRY1
        if self.natural:
            theta1 = self.params['theta1']
            theta2 = self.params['theta2']
            S = -1/2*jnp.linalg.inv(theta2)
            #TODO: since this aint the natural param anymore...
            S = S.T @ S
            m = S@theta1
        else:
            m = self.params['m']
            S = self.params['S']
        # DRY1

        Kmm = self.get_Kmm(self.params)
        kstar = self.get_Knm(XX, self.params)
        ret = kstar @ jnp.linalg.solve(Kmm+gI, m)

        return ret


    def compile(self, jit):
        if jit:
            self.get_elbo = jax.jit(self.elbo_pre)
            self.get_guys = jax.jit(self.elbo_pre_individ)
            self.grad_elbo = jax.jit(jax.grad(self.elbo_pre))
            self.vng_elbo = jax.jit(jax.value_and_grad(self.elbo_pre))
        else:
            print("not jitting!")
            self.get_elbo = self.elbo_pre
            self.get_guys = self.elbo_pre_individ
            self.grad_elbo = jax.grad(self.elbo_pre)
            self.vng_elbo = jax.value_and_grad(self.elbo_pre)

    def fit(self, iters = 100, lr = 5e-3, mb_size = 256, ls_params = {}, verbose = True, debug = False, track = False, es_every=100):
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(self.params)

        es_iters = int(np.ceil(iters/es_every))

        self.costs = np.nan*np.zeros(iters)
        self.mse_es = np.nan*np.zeros(es_iters)
        if track:
            self.nll = np.nan*np.zeros(iters)
            self.tr1 = np.nan*np.zeros(iters)
            self.tr2 = np.nan*np.zeros(iters)
            self.kl = np.nan*np.zeros(iters)
        if track:
            for v in self.params:
                self.tracking[v] = np.zeros(self.params[v].flatten().shape+(iters,))
        for i in tqdm(range(iters), disable = not verbose):
            batch = np.random.choice(self.N,mb_size,replace=False)
            cost, grad = self.vng_elbo(self.params, X[batch,:], y[batch])
            updates, opt_state = optimizer.update(grad, opt_state)
            self.params = optax.apply_updates(self.params, updates)
            self.costs[i] = cost

            if i%es_every==0:
                es_pred = self.pred(self.X_es)
                self.mse_es[i//es_every] = jnp.mean(jnp.square(es_pred-self.y_es))
                opt_iter = np.nanargmin(self.mse_es)*es_every
                its_since_opt = i-opt_iter
                eps_since_opt = its_since_opt * mb_size / self.N
                if eps_since_opt > self.es_patience:
                    print("ES Break!")
                    break

            if track:
                for v in self.params:
                    self.tracking[v][:,i] = jnp.copy(self.params[v].flatten())
                print("unnecessary eval!")
                nll, tr1, tr2, kl =  self.get_guys(self.params, X, y)
                self.nll[i] = nll
                self.tr1[i] = tr1
                self.tr2[i] = tr2
                self.kl[i] = kl

        self.last_it = i

    def plot(self, fname = 'track.png'):
        if len(self.tracking)==0:
            print("Skipping plotting of empty tracking")
        else:
            nparams = len(self.tracking)
            ncols = 2
            nrows = int(np.ceil(nparams/ncols))

            fig = plt.figure()
            for vi,v in enumerate(self.params):
                plt.subplot(nrows,ncols,vi+1)
                #plt.plot(self.tracking[v].T)
                plt.plot(self.tracking[v][:25,:].T)
                plt.title(v)
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

class HensmanGP(VSGP):
    def __init__(self, X, y, M = 10, jit = True, natural = True):
        print("new m/Z init.")
        VSGP.__init__(self, X, y, M, jit, natural)

        init_samp = np.random.choice(self.N,self.M,replace=self.N<self.M)
        Z_init = jnp.array(self.X[init_samp,:])

        #DRY3
        #m_init = jnp.zeros(self.M, dtype = npdtype)
        m_init = jnp.array(self.y[init_samp])
        S_init = jnp.eye(self.M, dtype = npdtype)
        if self.natural:
            theta1_init = jnp.linalg.solve(S_init, m_init)
            theta2_init = -jnp.linalg.inv(S_init)/2
            self.params['theta1'] = theta1_init
            self.params['theta2'] = theta2_init
        else:
            self.params['m'] = m_init
            self.params['S'] = S_init
        #DRY3

        self.params['Z'] = Z_init

        self.meth_name = 'Hensman_et_al'

    def get_Knm(self,X,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        Z = params['Z']
        return self.get_K(X, Z, ell, sigma2)

    def get_Kmm(self,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        Z = params['Z']
        return self.get_K(Z, Z, ell, sigma2)


# Method 2 Arbitrary RKHS Inducing functions
class M2GP(VSGP):
    def __init__(self, X, y, M = 10, D = 5, jit = True, natural = True):
        VSGP.__init__(self, X, y, M, jit, natural)
        self.D = D
        A_init = jnp.array(jnp.ones([self.M,D])) / D

        init_samps = [np.random.choice(self.N,self.D,replace=self.N<self.D) for _ in range(self.M)]
        Z_init = jnp.stack([jnp.array(self.X[init_samp,:]) for init_samp in init_samps], axis = 0)

        self.params['A'] = A_init
        self.params['Z'] = Z_init
        self.meth_name = 'M2GP'

        #DRY3
        m_init = jnp.array([np.mean(self.y[init_samp]) for init_samp in init_samps])
        S_init = jnp.eye(self.M, dtype = npdtype)
        if self.natural:
            theta1_init = jnp.linalg.solve(S_init, m_init)
            theta2_init = -jnp.linalg.inv(S_init)/2
            self.params['theta1'] = theta1_init
            self.params['theta2'] = theta2_init
        else:
            self.params['m'] = m_init
            self.params['S'] = S_init
        #DRY3

    def get_Knm(self,X,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        A = params['A']
        Z = params['Z'] 

        #K_big = np.zeros([N,M,D])
        #for m in range(M):
        #    K = self.get_K(X, Z[m,:,:], ell)
        #    K_big[:,m,:] = K
        K_big = jnp.transpose(jax.vmap(lambda z: self.get_K(X, z, ell, sigma2))(Z), [1, 0, 2])

        #Kxz = np.zeros([N,M])
        #for m in range(M):
        #    Kxz[:,m] = K_big[:,m,:] @ A[m,:]
        Kxz = jnp.einsum('ijk,jk->ij', K_big, A)

        return Kxz

    def get_Kmm(self,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        A = params['A']
        Z = params['Z'] 

        #K_big = np.zeros([M,M,D,D])
        #for m1 in range(M):
        #    for m2 in range(M):
        #        K = self.get_K(Z[m1,:,:], Z[m2,:,:], ell)
        #        K_big[m1,m2,:,:] = K
        K_big = jax.vmap(lambda x: jax.vmap(lambda y: self.get_K(x, y, ell, sigma2))(Z))(Z)

        #Kzz = np.zeros([M,M])
        #for m1 in range(M):
        #    for m2 in range(M):
        #        Kzz[m1,m2] = A[m1,:] @ K_big[m1,m2,:,:] @ A[m2,:]
        Kzz = jnp.einsum('ik,ijkl,jl->ij', A, K_big, A)

        return Kzz

