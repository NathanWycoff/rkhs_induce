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

#class YAJO(object):
#    def __init__(self, vng, params, steps_per = 1, ls = 'backtrack', ls_params = {}, pc = 'id', pc_params = {}):
#        self.vng = vng
#        assert ls in ['fixed_lr','backtracking']
#        self.ls = ls
#        assert pc in ['id','exp_ada']
#        self.pc = pc
#        self.ls_params = ls_params
#        self.pc_params = pc_params
#
#        self.steps_per = steps_per
#
#        self.debug = False
#
#        self._init_pc(params)
#        self._init_ls()
#
#        #self._update_params = lambda x,d,ss: dict([(v,x[v]+ss*d[v]) for v in x])
#        #def _update_params(params, pc_vars, curval, curgrad, ss, it):
#        def _update_params(params, curgrad, ss, v_ada, it):
#            cand_params = {}
#            for v in params:
#                cand_params[v] = params[v]
#            candgrad = curgrad
#
#            beta2 = self.pc_params['beta2']
#
#            for step in range(self.steps_per):
#                # Update precond
#                sd = {}
#                if self.pc=='id':
#                    for v in candgrad:
#                        sd[v] = -candgrad[v]
#                elif self.pc=='exp_ada':
#                    for v in params:
#                        v_ada[v] = beta2*v_ada[v] + (1-beta2) * jnp.square(candgrad[v])
#                        vhat = jnp.sqrt(v_ada[v] / (1-jnp.power(beta2, it)))
#                        sd[v] = -candgrad[v] / (vhat + self.pc_params['eps'])
#                else:
#                    raise Exception("Bad pc.")
#
#                # Get new params
#                for v in params:
#                    cand_params[v] = cand_params[v] + ss * sd[v]
#
#                # Evaluate and increment counter.
#                candval, candgrad = self.vng(cand_params)
#                it += 1
#
#            return cand_params, v_ada, candval, candgrad 
#        self.update_params = jax.jit(_update_params)
#
#
#    def init_state(self):
#        optstate = {'it':0,'done':False,'message':'inprog'}
#        return optstate
#
#    def _init_ls(self):
#        if self.ls=='backtracking':
#            defaults = {
#                    'lr_init' : 1.,
#                    'shrink' : 0.2,
#                    'grow' : 2,
#                    'max_iter' : 15,
#                    }
#        elif self.ls=='fixed_lr':
#            defaults = {'ss' : 5e-3}
#        else:
#            raise Exception("Unknown ls")
#
#        for s in defaults:
#            if s not in self.ls_params:
#                self.ls_params[s] = defaults[s]
#
#    def _init_pc(self, params):
#        assert type(params)==dict
#        self.pc_vars = {}
#
#        if self.pc=='id':
#            defaults = {}
#        elif self.pc=='exp_ada':
#            self.pc_vars['v_ada'] = {}
#            self.pc_vars['vhat'] = {}
#            for v in params:
#                self.pc_vars['v_ada'][v] = jnp.zeros_like(params[v])
#                self.pc_vars['vhat'][v] = jnp.zeros_like(params[v])
#            defaults = {
#                    'beta2' : 0.999,
#                    'eps' : 1e-8,
#                    }
#        else:
#            raise Exception("Unknown pc")
#
#        for s in defaults:
#            if s not in self.pc_params:
#                self.pc_params[s] = defaults[s]
#
#    def step(self, params, optstate):
#        assert type(params)==dict
#        val, grad = self.vng(params)
#
#        if not np.isfinite(val):
#            print("Nonfinite initial cost.")
#            import IPython; IPython.embed()
#
#        if self.ls=='backtracking':
#            #ss = self.ls_params['lr_init']
#            if 'last_ls_ss' in optstate:
#                ss = optstate['last_ls_ss'] * self.ls_params['grow']
#            else:
#                ss = self.ls_params['lr_init']
#            candval = np.inf
#            candgrad_finite = False
#            it = 0
#            while ((not candgrad_finite) or (np.isnan(candval)) or candval > val) and it<self.ls_params['max_iter']:
#                it += 1
#                #cand_params = {}
#                #for v in params:
#                #    cand_params[v] = params[v] + ss * sd[v]
#                cand_params, self.pc_vars['v_ada'], candval, candgrad = self.update_params(params, grad, ss, self.pc_vars['v_ada'], optstate['it'])
#                optstate['it'] += self.steps_per
#                ss *= self.ls_params['shrink']
#                #candval, candgrad = self.vng(cand_params)
#
#                candgrad_finite = True
#                for v in candgrad:
#                    if np.any(~np.isfinite(candgrad[v])):
#                        candgrad_finite = False
#            # Undo last shrink if ss was ok.
#            ss /= self.ls_params['shrink']
#            if self.debug:
#                print("ls took %d iters"%it)
#                print("old cost:%f"%val)
#                print("new cost:%f"%candval)
#            if it==0:
#                print("it should never be 0.")
#                import IPython; IPython.embed()
#            ls_failed = candval > val or (not np.isfinite(candval))
#            params = cand_params
#        elif self.ls=='fixed_lr':
#            ls_failed = False
#            for v in params:
#                ss = self.ls_params['ss']
#                params[v] += ss * sd[v]
#            optstate['it'] += 1 #TODO: having two distinct "it"s floating around is very confusing.
#        else:
#            raise Exception("Bad ls.")
#
#        if ls_failed:
#            if self.debug:
#                print("Line search failed!")
#            optstate['done'] = True
#            optstate['message'] = 'ls_failure'
#
#        if self.ls in ['backtracking']:
#            optstate['last_ls_it'] = it
#        optstate['last_ls_ss'] = ss
#
#        return params, optstate, val, grad

class YAJO(object):
    def __init__(self, vng, params, steps_per = 1, ls = 'backtrack', ls_params = {}, debug = False):
        self.vng = vng
        assert ls in ['fixed_lr','backtracking']
        self.ls = ls
        self.ls_params = ls_params

        self.steps_per = steps_per

        self.debug = debug

        self._init_ls()

        self.scale_updates = jax.jit(lambda u, ss: dict([(v,ss*u[v]) for v in u]))

        self.optimizer = optax.adam(1.)
        self.opt_state = self.optimizer.init(params)
        self.reset_after = 3

        self.out_it = 0
        self.done = False

    def _init_ls(self):
        if self.ls=='backtracking':
            defaults = {
                    'lr_init' : 1.,
                    'lr_max' : 1.,
                    'shrink' : 0.2,
                    'grow' : 2,
                    'max_iter' : 15,
                    }
        elif self.ls=='fixed_lr':
            defaults = {'ss' : 5e-3}
        else:
            raise Exception("Unknown ls")

        for s in defaults:
            if s not in self.ls_params:
                self.ls_params[s] = defaults[s]

    def step(self, params):
        assert type(params)==dict
        assert not self.done
        val, grad = self.vng(params)

        if not np.isfinite(val):
            print("Nonfinite initial cost.")
            import IPython; IPython.embed()

        if self.ls=='backtracking':
            #ss = self.ls_params['lr_init']
            if self.out_it==0:
                ss = self.ls_params['lr_init']
            else:
                ss = jnp.minimum(self.last_ls_ss * self.ls_params['grow'], self.ls_params['lr_max'])
            candval = np.inf
            candgrad_finite = False
            it = 0

            while ((not candgrad_finite) or np.isnan(candval) or candval > val) and it<self.ls_params['max_iter']:
                it += 1
                candparams = {}
                for v in params:
                    candparams[v] = jnp.copy(params[v])
                candgrad = grad
                if it>self.reset_after:
                    self.opt_state = self.optimizer.init(params)

                for i in range(self.steps_per):
                    updates, self.opt_state = self.optimizer.update(candgrad, self.opt_state)
                    updates = self.scale_updates(updates, ss)
                    candparams = optax.apply_updates(candparams, updates)
                    candval, candgrad = self.vng(candparams)
                    if np.isnan(candval) or candval > val:
                        break

                candgrad_finite = True
                for v in candgrad:
                    if np.any(~np.isfinite(candgrad[v])):
                        candgrad_finite = False

                ss *= self.ls_params['shrink']

            # Undo last shrink if ss was ok.
            ss /= self.ls_params['shrink']
            if self.debug:
                print("ls took %d iters"%it)
                print("old cost:%f"%val)
                print("new cost:%f"%candval)
            if it==0:
                print("it should never be 0.")
                if self.debug:
                    import IPython; IPython.embed()
            ls_failed = candval > val or (not np.isfinite(candval))
            if ls_failed and self.debug:
                print('ls failed!')
                import IPython; IPython.embed()
            params = candparams
        elif self.ls=='fixed_lr':
            ls_failed = False
            for v in params:
                ss = self.ls_params['ss']
                params[v] += ss * sd[v]
        else:
            raise Exception("Bad ls.")

        if ls_failed:
            if self.debug:
                print("Line search failed!")
            self.done = True
            self.message = 'ls_failure'

        if self.ls in ['backtracking']:
            self.last_ls_it = it
        self.last_ls_ss = ss

        self.out_it += 1

        return params, val, grad




class VIGP(object):
    def __init__(self, X, y, M = 10):
        self.meth_name = 'Var Induc GP'
        self.N,self.P = X.shape
        self.M = M
        ell_init = jnp.repeat(jnp.array(np.log(1e-1), dtype = npdtype),P)
        #sigma2_init = jnp.array(np.log(1e-5), dtype = npdtype)
        sigma2_init = jnp.array(np.log(1e-8), dtype = npdtype)
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

        Knm = self.get_Knm(self.X, params)
        Kmm = self.get_Kmm(params)
        #Knn = self.get_K(self.X, self.X, ell)

        # Compute diag of Ktilde directly.
        Kmmi = jnp.linalg.inv(Kmm+self.g_nug*jnp.eye(self.M)) #Even faster with chol
        q_diag = jax.vmap(lambda k: k.T @ Kmmi @ k)(Knm)
        # Warning: assumes that correlation matrix diagonal is 1.
        ktilde = jnp.ones(self.N) - q_diag

        ed = jnp.linalg.eigh(Kmm+self.g_nug*jnp.eye(self.M))
        U = Knm @ ed[1] @ jnp.diag(jnp.sqrt(1/ed[0]))
        #U @ U.T is Qnn in the notation of Titsias09.
        dist = tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(cov_diag_factor = (sigma2+self.g_nug)*jnp.ones(self.N), cov_perturb_factor = U)
        ll = dist.log_prob(y)

        reg = 1./(2.*(sigma2+self.g_nug))*jnp.sum(ktilde)

        return -ll + reg

    def pred(self, XX):
        ell = jnp.exp(self.params['ell'])
        sigma2 = jnp.exp(self.params['sigma2'])

        Knm = self.get_Knm(self.X, self.params)
        Kmm = self.get_Kmm(self.params)

        #kstar = self.get_K(XX, self.X, ell)
        #kstar = self.get_K(XX, self.X, ell)
        kstar = self.get_Knm(XX, self.params)

        ret = kstar@jnp.linalg.solve(Kmm+ Knm.T @ Knm / sigma2, Knm.T @ y)/sigma2
        return ret


    def compile(self):
        self.get_elbo = jax.jit(self.elbo_pre)
        self.grad_elbo = jax.jit(jax.grad(self.elbo_pre))
        self.vng_elbo = jax.jit(jax.value_and_grad(self.elbo_pre))

    def fit(self, iters = 100, ls = 'fixed_lr', ls_params = {}, verbose = True, debug = False):
        steps_per = 20
        eiters = int(np.ceil(iters/steps_per))

        self.opt = YAJO(self.vng_elbo, self.params, steps_per = steps_per, ls=ls, ls_params=ls_params, debug = debug)

        self.costs = np.nan*np.zeros(eiters)
        self.ls_its = np.nan*np.zeros(eiters)
        self.ss = np.nan*np.zeros(eiters)
        for i in tqdm(range(eiters), disable = not verbose):
            #cost, grad = self.vng_elbo(self.params)
            self.params, cost, grad = self.opt.step(self.params)
            self.costs[i] = cost
            if ls in ['backtracking']:
                self.ls_its[i] = self.opt.last_ls_it
            self.ss[i] = self.opt.last_ls_ss
            if self.opt.done:
                if verbose:
                    print("Optim exit with message "+self.opt.message+" after "+str(i)+" outer its.")
                break



# Oh this is actually Titsias'
class TGP(VIGP):
    def __init__(self, X, y, M = 10):
        VIGP.__init__(self, X, y, M)
        Z_init = jnp.array(np.random.uniform(size=[self.M,self.P]))
        self.params['Z'] = Z_init
        self.meth_name = 'SGGP'

    def get_Knm(self,X,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        Z = params['Z']
        return self.get_K(X, Z, ell)

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

    def get_Knm(self,X,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        A = params['A']
        Kxx = self.get_K(X, self.X, ell)
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

    def get_Knm(self,X,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        A = params['A']
        Z = params['Z'] 

        #K_big = np.zeros([N,M,D])
        #for m in range(M):
        #    K = self.get_K(X, Z[m,:,:], ell)
        #    K_big[:,m,:] = K
        K_big = jnp.transpose(jax.vmap(lambda z: self.get_K(X, z, ell))(Z), [1, 0, 2])

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
