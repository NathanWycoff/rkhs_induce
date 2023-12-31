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
import warnings

config.update("jax_enable_x64", True)

class YAJO(object):
    def __init__(self, vng, params, ls = 'backtrack', ls_params = {}, pc = 'id', pc_params = {}):
        self.vng = vng
        assert ls in ['fixed_lr','backtracking']
        self.ls = ls
        assert pc in ['id','exp_ada']
        self.pc = pc
        self.ls_params = ls_params
        self.pc_params = pc_params

        self.debug = False

        self._init_pc(params)
        self._init_ls()


    def init_state(self):
        optstate = {'it':0,'done':False,'message':'inprog'}
        return optstate

    def _init_ls(self):
        if self.ls=='backtracking':
            defaults = {
                    'lr_init' : 1.,
                    'shrink' : 0.2,
                    #'grow' : 2,
                    'max_iter' : 15,
                    }
        elif self.ls=='fixed_lr':
            defaults = {'lr' : 5e-3}
        else:
            raise Exception("Unknown ls")

        for s in defaults:
            if s not in self.ls_params:
                self.ls_params[s] = defaults[s]

    def _init_pc(self, params):
        assert type(params)==dict
        self.pc_vars = {}

        if self.pc=='id':
            defaults = {}
        elif self.pc=='exp_ada':
            self.pc_vars['v_ada'] = {}
            self.pc_vars['vhat'] = {}
            for v in params:
                self.pc_vars['v_ada'][v] = jnp.zeros_like(params[v])
                self.pc_vars['vhat'][v] = jnp.zeros_like(params[v])
            defaults = {
                    'beta2' : 0.999,
                    'eps' : 1e-8,
                    }
        else:
            raise Exception("Unknown pc")

        for s in defaults:
            if s not in self.ls_params:
                self.pc_params[s] = defaults[s]

    def step(self, params, optstate):
        assert type(params)==dict
        val, grad = self.vng(params)

        if not np.isfinite(val):
            print("Nonfinite initial cost.")
            import IPython; IPython.embed()

        sd = {}
        if self.pc=='id':
            for v in grad:
                sd[v] = -grad[v]
        elif self.pc=='exp_ada':
            beta2 = self.pc_params['beta2']
            vhat = {}
            for v in params:
                self.pc_vars['v_ada'][v] = beta2*self.pc_vars['v_ada'][v] + (1-beta2) * jnp.square(grad[v])
                self.pc_vars['vhat'][v] = jnp.sqrt(self.pc_vars['v_ada'][v] / (1-jnp.power(beta2, optstate['it'])))
                sd[v] = -grad[v] / (self.pc_vars['vhat'][v] + self.pc_params['eps'])
        else:
            raise Exception("Bad pc.")

        if self.ls=='backtracking':
            lr = self.ls_params['lr_init']
            candval = np.inf
            candgrad_finite = False
            it = 0
            while ((not candgrad_finite) or (np.isnan(candval)) or candval > val) and it<self.ls_params['max_iter']:
                it += 1
                cand_params = {}
                for v in params:
                    cand_params[v] = params[v] + lr * sd[v]
                lr *= self.ls_params['shrink']
                candval, candgrad = self.vng(cand_params)

                candgrad_finite = True
                for v in candgrad:
                    if np.any(~np.isfinite(candgrad[v])):
                        candgrad_finite = False
            if self.debug:
                print("ls took %d iters"%it)
                print("old cost:%f"%val)
                print("new cost:%f"%candval)
            if it==0:
                import IPython; IPython.embed()
            ls_failed = candval > val or (not np.isfinite(candval))
            params = cand_params
        elif self.ls=='fixed_lr':
            ls_failed = False
            for v in params:
                params[v] += self.ls_params['lr'] * sd[v]
        else:
            raise Exception("Bad ls.")

        if ls_failed:
            print("Line search failed!")
            optstate['done'] = True
            optstate['message'] = 'ls_failure'

        optstate['it'] += 1

        return params, optstate, val, grad




class VIGP(object):
    def __init__(self, X, y, M = 10):
        self.meth_name = 'Var Induc GP'
        self.N,self.P = X.shape
        self.M = M
        ell_init = jnp.repeat(jnp.array(np.log(1e-1), dtype = np.float64),P)
        sigma2_init = jnp.array(np.log(1e-5), dtype = np.float64)
        self.params = {'ell': ell_init, 'sigma2' : sigma2_init}
        self.X = X
        self.y = y
        self.g_nug = 1e-6

        self.kernel = lambda x,y, ell: jnp.exp(-jnp.sum(jnp.square(x-y)/ell))
        self.get_K = lambda X1, X2, ell: jax.vmap(lambda x: jax.vmap(lambda y: self.kernel(x, y, ell))(X2))(X1)

        self.compile()

    def get_Knm(self,params):
        raise NotImplementedError()

    def get_Kmm(self,params):
        raise NotImplementedError()

    def elbo_pre(self, params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])

        Knm = self.get_Knm(params)
        Kmm = self.get_Kmm(params)
        #Knn = self.get_K(self.X, self.X, ell)

        # Compute diag of Ktilde directly.
        Kmmi = jnp.linalg.inv(Kmm+self.g_nug*jnp.eye(self.M)) #Even faster with chol
        q_diag = jax.vmap(lambda k: k.T @ Kmmi @ k)(Knm)
        # Warning: assumes that correlation matrix diagonal is 1.
        ktilde = jnp.ones(self.N) - q_diag

        ## CODE BLOCK A
        ## Using TFP diag + LR
        ed = jnp.linalg.eigh(Kmm+self.g_nug*jnp.eye(self.M))
        U = Knm @ ed[1] @ jnp.diag(jnp.sqrt(1/ed[0]))
        #U @ U.T - Qnn = 0.
        dist = tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(cov_diag_factor = sigma2*jnp.ones(self.N), cov_perturb_factor = U)
        ## CODE BLOCK A
        ll = dist.log_prob(y)

        reg = 1./(2.*sigma2)*jnp.sum(ktilde)

        return -ll + reg

    def pred(self, XX):
        ell = jnp.exp(self.params['ell'])
        sigma2 = jnp.exp(self.params['sigma2'])

        Knm = self.get_Knm(self.params)
        Kmm = self.get_Kmm(self.params)
        #Knn = self.get_K(self.X, self.X, ell)

        kstar = self.get_K(XX, self.X, ell)

        # Naive inversion
        #Qnn = Knm @ jnp.linalg.solve(Kmm + self.g_nug*jnp.eye(self.M), Knm.T)
        #QI = Qnn+sigma2*jnp.eye(self.N)
        #ret = kstar @ np.linalg.solve(QI, self.y)

        ## CODE BLOCK A
        ## Using TFP diag + LR
        ed = jnp.linalg.eigh(Kmm+self.g_nug*jnp.eye(self.M))
        U = Knm @ ed[1] @ jnp.diag(jnp.sqrt(1/ed[0]))
        #U @ U.T - Qnn = 0.
        dist = tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(cov_diag_factor = sigma2*jnp.ones(self.N), cov_perturb_factor = U)
        ## CODE BLOCK A
        ret = kstar @ dist.cov_operator.solve(self.y.reshape((-1,1))).flatten()

        return ret

    def compile(self):
        self.get_elbo = jax.jit(self.elbo_pre)
        self.grad_elbo = jax.jit(jax.grad(self.elbo_pre))
        self.vng_elbo = jax.jit(jax.value_and_grad(self.elbo_pre))

    def fit(self, iters = 100, ls = 'fixed_lr', ls_params = {}, verbose = True, pc = 'id'):
        self.opt = YAJO(self.vng_elbo, self.params, ls=ls, ls_params=ls_params, pc = pc)
        optstate = self.opt.init_state()

        self.costs = np.nan*np.zeros(iters)
        for i in tqdm(range(iters), disable = not verbose):
            #cost, grad = self.vng_elbo(self.params)
            self.params, optstate, cost, grad = self.opt.step(self.params, optstate)
            self.costs[i] = cost
            if optstate['done']:
                print("Optim exit with message "+optstate['message'])
                break

        fig = plt.figure()
        plt.plot(self.costs)
        #plt.yscale('log')
        plt.savefig(self.meth_name+"_cost.pdf")
        plt.close()


# Oh this is actually Titsias'
class TGP(VIGP):
    def __init__(self, X, y, M = 10):
        VIGP.__init__(self, X, y, M)
        Z_init = jnp.array(np.random.uniform(size=[self.M,self.P]))
        self.params['Z'] = Z_init
        self.meth_name = 'SGGP'

    def get_Knm(self,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        Z = params['Z']
        return self.get_K(self.X, Z, ell)

    def get_Kmm(self,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        Z = params['Z']
        return self.get_K(Z, Z, ell)

# Method 1 RKHS Inducing GP: Inducing functions assumed to lie in span of data and parameterized by coefficients
class M1GP(VIGP):
    def __init__(self, X, y, M = 10):
        VIGP.__init__(self, X, y, M)
        #A_init = jnp.array(np.random.normal(size=[self.M,self.N])) / jnp.sqrt(self.N)
        A_init = jnp.array(np.eye(self.N)[np.random.choice(self.N,self.M,replace=False),:]) 
        self.params['A'] = A_init
        self.meth_name = 'M1GP'

    def get_Knm(self,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        A = params['A']
        Kxx = self.get_K(self.X, self.X, ell)
        return Kxx @ A.T

    def get_Kmm(self,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        A = params['A']
        Kxx = self.get_K(self.X, self.X, ell)
        return A @ Kxx @ A.T

# Method 2 Arbitrary RKHS Inducing functions
class M2GP(VIGP):
    def __init__(self, X, y, M = 10, D = 5):
        VIGP.__init__(self, X, y, M)
        self.D = D
        A_init = jnp.array(np.random.normal(size=[self.M,D])) / jnp.sqrt(D)
        #A_init = jnp.array(np.eye(self.D)[np.random.choice(self.D,self.M,replace=False),:]) 
        Z_init = jnp.array(np.random.uniform(size=[self.M,self.D,self.P]))
        self.params['A'] = A_init
        self.params['Z'] = Z_init
        self.meth_name = 'M2GP'

    def get_Knm(self,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        A = params['A']
        Z = params['Z'] 

        #K_big = np.zeros([N,M,D])
        #for m in range(M):
        #    K = self.get_K(self.X, Z[m,:,:], ell)
        #    K_big[:,m,:] = K
        K_big = jnp.transpose(jax.vmap(lambda z: self.get_K(self.X, z, ell))(Z), [1, 0, 2])

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
        K_big = jax.vmap(lambda x: jax.vmap(lambda y: self.get_K(x, y, ell))(Z))(Z)

        #Kzz = np.zeros([M,M])
        #for m1 in range(M):
        #    for m2 in range(M):
        #        Kzz[m1,m2] = A[m1,:] @ K_big[m1,m2,:,:] @ A[m2,:]
        Kzz = jnp.einsum('ik,ijkl,jl->ij', A, K_big, A)

        return Kzz

