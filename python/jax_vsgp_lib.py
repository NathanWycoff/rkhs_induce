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
        #print("scale_updates is evil!")
        #self.scale_updates = jax.jit(lambda u, ss: dict([(v,ss*u[v]) if v=='theta1' else (v,0.*u[v])for v in u]))

        #self.optimizer = optax.adam(1.)
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
                    #updates = self.scale_updates(candgrad, ss)
                    #print("ignoring adam.")
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
            #ls_failed = False
            ls_failed = False
            #for v in params:
            #    ss = self.ls_params['ss']
            #    params[v] += ss * grad[v]
            ss = self.ls_params['ss']
            for i in range(self.steps_per):
                updates, self.opt_state = self.optimizer.update(grad, self.opt_state)
                updates = self.scale_updates(updates, ss)
                params = optax.apply_updates(params, updates)
                val, grad = self.vng(params)

            if not np.isfinite(val):
                ls_failed = True

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


class VSGP(object):
    def __init__(self, X, y, M = 10, jit = True, natural = True):
        self.meth_name = 'Var Induc GP'
        self.N,self.P = X.shape
        self.M = M
        ell_init = jnp.repeat(jnp.array(np.log(1e-1), dtype = npdtype),P)
        #gamma2_init = jnp.array(np.log(1e-5), dtype = npdtype)
        # COVAR =  sigma2*K + (gamma2+g_nug)*I
        sigma2_init = jnp.array(np.log(1.), dtype = npdtype) # Scale Parameter.
        gamma2_init = jnp.array(np.log(1e-8), dtype = npdtype) # Error Variance.
        #gamma2_init = jnp.array(np.log(1.), dtype = npdtype) # Error Variance.
        self.params = {'ell': ell_init, 'sigma2' : sigma2_init, 'gamma2' : gamma2_init}
        self.X = X
        self.y = y
        print("TODO: jit to 1e-8?")
        self.g_nug = 1e-6
        self.natural = natural

        self.kernel = lambda x,y, ell, sigma2: sigma2*jnp.exp(-jnp.sum(jnp.square(x-y)/ell))
        self.get_K = lambda X1, X2, ell, sigma2: jax.vmap(lambda x: jax.vmap(lambda y: self.kernel(x, y, ell, sigma2))(X2))(X1)

        self.jit = jit
        self.compile(jit)

        #m_init = jnp.array(np.random.normal(size=[self.M]))*1e-6
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


    def get_Knm(self,params):
        raise NotImplementedError()

    def get_Kmm(self,params):
        raise NotImplementedError()

    def elbo_pre(self, params):
        ## PRECOMPUTING THINGS
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        gamma2 = jnp.exp(params['gamma2'])
        # DRY1
        if self.natural:
            theta1 = params['theta1']
            theta2 = params['theta2']
            S = -1/2*jnp.linalg.inv(theta2)
            print("Sus")
            S = S.T @ S
            print("Sus")
            m = S@theta1
        else:
            m = params['m']
            S = params['S']
        # DRY1

        #trust_tfp = True
        trust_tfp = False

        gI = self.g_nug*jnp.eye(self.M, dtype=npdtype)

        Knm = self.get_Knm(self.X, params)
        Kmm = self.get_Kmm(params)
        #Knn = self.get_K(self.X, self.X, ell)

        # Compute diag of Ktilde directly.
        Kmmi = jnp.linalg.inv(Kmm+self.g_nug*jnp.eye(self.M, dtype=npdtype)) #Even faster with chol
        q_diag = jax.vmap(lambda k: k.T @ Kmmi @ k)(Knm)
        # Warning: assumes that data covariance matrix diagonal is sigma2.
        ktilde = sigma2*jnp.ones(self.N) - q_diag

        ## NOW COMPUTE TERMS OF LOSS FUNCTION
        mu_y = Knm @ jnp.linalg.solve(Kmm+gI, m)
        if trust_tfp:
            dist_y = tfp.distributions.Normal(loc=mu_y, scale = jnp.sqrt(gamma2))
            nll = -jnp.sum(dist_y.log_prob(self.y))
        else:
            nll = self.N/2.*jnp.log(gamma2) + 1./(2.*gamma2) * jnp.sum(jnp.square(self.y-mu_y))
        #import IPython; IPython.embed()

        tr1 = 1./(2.*gamma2) * jnp.sum(ktilde)

        KiSKi = Kmmi @ S @ Kmmi
        tr2 = jnp.sum(Knm.T * (KiSKi @ Knm.T)) / gamma2

        ##import IPython; IPython.embed()
        #print('ye')
        #import IPython; IPython.embed()
        #tr2 = 0
        #for n in range(self.N):
        #    kk = (Kmmi @ Knm[n:(n+1),:].T)
        #    LAMBDAn =  kk @ kk.T 
        #    A = S @ LAMBDAn
        #    tr2 += jnp.sum(jnp.diag(A))
        #tr2 /= gamma2
        #print('boi')

        #tr23 = 0
        #KiSKi = Kmmi @ S @ Kmmi
        #for n in range(N):
        #    kk = Knm[n:(n+1),:].T
        #    tr23 += (kk.T @ KiSKi @ kk)/gamma2


        ### KL via TFP
        if trust_tfp:
            dist_q = tfp.distributions.MultivariateNormalFullCovariance(loc=m, covariance_matrix=S)
            dist_p = tfp.distributions.MultivariateNormalFullCovariance(loc=jnp.zeros(self.M,dtype=npdtype), covariance_matrix=Kmm+gI)
            kl = dist_q.kl_divergence(dist_p)
        else:
            S_slogdet = jnp.linalg.slogdet(S)
            kl = jnp.linalg.slogdet(Kmm+gI)[1] - S_slogdet[1] + jnp.sum(jnp.diag(Kmmi@S)) + m.T @ Kmmi @ m - self.P 
            lam1 = np.min(jnp.linalg.eigh(S)[0])
            #kl = jax.lax.cond(S_slogdet[0]<0, lambda: np.inf, lambda: kl)
            print("Todo: think. Jit?")
            kl = jax.lax.cond(lam1<-1e-10, lambda: np.inf, lambda: kl)
            #kl = jax.lax.cond(S_slogdet[0]<0, lambda: np.inf, lambda: kl)
            #Other way KL just for fun:
            #Si = jnp.linalg.inv(S)
            #kl = - jnp.linalg.slogdet(Kmm+gI)[1] + S_slogdet[1] + jnp.sum(jnp.diag(Si@(Kmm+gI))) + m.T @ Si @ m - self.P 

        loss = nll + tr1 + tr2 + kl
        #loss = nll  + tr1 + kl
        #loss = nll + tr2

        return loss

    def pred(self, XX):
        gI = self.g_nug*jnp.eye(self.M, dtype=npdtype)
        # DRY1
        if self.natural:
            theta1 = self.params['theta1']
            theta2 = self.params['theta2']
            S = -1/2*jnp.linalg.inv(theta2)
            print("Sus")
            S = S.T @ S
            print("Sus")
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
            self.grad_elbo = jax.jit(jax.grad(self.elbo_pre))
            self.vng_elbo = jax.jit(jax.value_and_grad(self.elbo_pre))
        else:
            print("not jitting!")
            self.get_elbo = self.elbo_pre
            self.grad_elbo = jax.grad(self.elbo_pre)
            self.vng_elbo = jax.value_and_grad(self.elbo_pre)

    def fit(self, iters = 100, ls = 'fixed_lr', ls_params = {}, verbose = True, debug = False):
        #steps_per = 20
        steps_per = 1
        eiters = int(np.ceil(iters/steps_per))

        self.opt = YAJO(self.vng_elbo, self.params, steps_per = steps_per, ls=ls, ls_params=ls_params, debug = debug)

        self.costs = np.nan*np.zeros(eiters)
        self.ls_its = np.nan*np.zeros(eiters)
        self.ss = np.nan*np.zeros(eiters)
        for i in tqdm(range(eiters), disable = not verbose):
            if not self.jit:
                print("not nec")
                self.get_elbo(self.params)
                print("not nec")
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
class HensmanGP(VSGP):
    def __init__(self, X, y, M = 10, jit = True, natural = True):
        VSGP.__init__(self, X, y, M, jit)
        Z_init = jnp.array(np.random.uniform(size=[self.M,self.P]))
        self.params['Z'] = Z_init
        self.meth_name = 'Hensman_et_al'

        #Kmm = self.get_Kmm(self.params)
        #gI = self.g_nug*jnp.eye(self.M,dtype=npdtype)
        #L = np.linalg.cholesky(Kmm+gI)
        #self.params['m'] = jnp.array(L @ np.random.normal(size=[self.M]))
        #self.params['S'] = jnp.array(Kmm) + gI

    def get_Knm(self,X,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        gamma2 = jnp.exp(params['gamma2'])
        Z = params['Z']
        return self.get_K(X, Z, ell, sigma2)

    def get_Kmm(self,params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        gamma2 = jnp.exp(params['gamma2'])
        Z = params['Z']
        return self.get_K(Z, Z, ell, sigma2)
