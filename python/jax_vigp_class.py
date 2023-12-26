#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  inducing_bg.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.19.2023

class vargp(object):
    def __init__(self, X, y, M = 10):
        N,P = X.shape
        ell_init = jnp.repeat(jnp.array(np.log(1e-1), dtype = np.float64),P)
        sigma2_init = jnp.array(np.log(1e-5), dtype = np.float64)
        Z_init = jnp.array(np.random.uniform(size=[M,P]))
        self.params = {'ell': ell_init, 'sigma2' : sigma2_init, 'Z' : Z_init}
        self.X = X
        self.y = y

        self.kernel = lambda x,y, ell: jnp.exp(-jnp.sum(jnp.square(x-y)/ell))
        self.get_K = lambda X1, X2, ell: jax.vmap(lambda x: jax.vmap(lambda y: self.kernel(x, y, ell))(X2))(X1)

        self.compile()

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

    def elbo_pre(self, params):
        ell = jnp.exp(params['ell'])
        sigma2 = jnp.exp(params['sigma2'])
        #Z = self.params['Z']

        #Knm = self.get_K(self.X, Z, ell)
        Knm = self.get_Knm(params)
        #Kmm = self.get_K(Z, Z, ell)
        Kmm = self.get_Kmm(params)
        Knn = self.get_K(self.X, self.X, ell)

        # Compute diag of Ktilde directly.
        Kmmi = jnp.linalg.inv(Kmm+g_nug*jnp.eye(M)) #Even faster with chol
        q_diag = jax.vmap(lambda k: k.T @ Kmmi @ k)(Knm)
        # Warning: assumes that correlation matrix diagonal is 1.
        ktilde = jnp.ones(N) - q_diag

        ## Using TFP diag + LR
        ed = jnp.linalg.eigh(Kmm+g_nug*jnp.eye(M))
        U = Knm @ ed[1] @ jnp.diag(jnp.sqrt(1/ed[0]))
        #U @ U.T - Qnn = 0.
        dist = tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(cov_diag_factor = sigma2*jnp.ones(N), cov_perturb_factor = U)
        ll = dist.log_prob(y)

        reg = 1./(2.*sigma2)*jnp.sum(ktilde)

        return -ll + reg

    def pred(self, XX):
        ell = jnp.exp(self.params['ell'])
        sigma2 = jnp.exp(self.params['sigma2'])
        #Z = self.params['Z']

        #Knm = self.get_K(self.X, Z, ell)
        Knm = self.get_Knm(self.params)
        #Kmm = self.get_K(Z, Z, ell)
        Kmm = self.get_Kmm(self.params)
        Knn = self.get_K(self.X, self.X, ell)

        Qnn = Knm @ jnp.linalg.solve(Kmm + g_nug*jnp.eye(M), Knm.T)

        kstar = self.get_K(XX, self.X, ell)
        QI = Qnn+sigma2*jnp.eye(N)
        ret = kstar @ np.linalg.solve(QI, self.y)
        return ret

    def compile(self):
        self.get_elbo = jax.jit(self.elbo_pre)
        self.grad_elbo = jax.jit(jax.grad(self.elbo_pre))
        self.vng_elbo = jax.jit(jax.value_and_grad(self.elbo_pre))

    def fit(self, iters = 100, learning_rate = 1e-1, verbose = True):
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.params)

        #Zs = np.zeros([M,iters])
        costs = np.zeros(iters)
        for i in tqdm(range(iters), disable = not verbose):
            cost, grad = self.vng_elbo(self.params)
            updates, opt_state = optimizer.update(grad, opt_state)
            self.params = optax.apply_updates(self.params, updates)
            costs[i] = cost
            #Zs[:,i] = np.array(self.params['Z'].copy()).flatten()

        fig = plt.figure()
        #plt.subplot(2,1,1)
        plt.plot(costs)
        #plt.subplot(2,1,2)
        #plt.plot(Zs.T)
        plt.savefig("cost.pdf")
        plt.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  inducing_bg.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.19.2023

class vargp_rkhs(vargp):
    def __init__(self, X, y, M = 10):
        vargp.__init__(self, X, y, M)
        #A_init = jnp.array(np.random.normal(size=[M,N])) / jnp.sqrt(N)
        A_init = jnp.array(np.eye(N)[np.random.choice(N,M,replace=False),:]) 
        #self.params = {'ell': ell_init, 'sigma2' : sigma2_init, 'A' : A_init}
        self.params['A'] = A_init
        del self.params['Z']

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

