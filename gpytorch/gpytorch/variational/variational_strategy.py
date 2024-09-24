#!/usr/bin/env python3

import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

from gpytorch.variational._variational_strategy import _VariationalStrategy
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution

from ..distributions import MultivariateNormal
from ..models import ApproximateGP
from ..settings import _linalg_dtype_cholesky, trace_mode
from ..utils.errors import CachingError
from ..utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from ..utils.warnings import OldVersionWarning
from . import _VariationalDistribution

def _ensure_updated_strategy_flag_set(
    state_dict: Dict[str, Tensor],
    prefix: str,
    local_metadata: Dict[str, Any],
    strict: bool,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    error_msgs: Iterable[str],
):
    device = state_dict[list(state_dict.keys())[0]].device
    if prefix + "updated_strategy" not in state_dict:
        state_dict[prefix + "updated_strategy"] = torch.tensor(False, device=device)
        warnings.warn(
            "You have loaded a variational GP model (using `VariationalStrategy`) from a previous version of "
            "GPyTorch. We have updated the parameters of your model to work with the new version of "
            "`VariationalStrategy` that uses whitened parameters.\nYour model will work as expected, but we "
            "recommend that you re-save your model.",
            OldVersionWarning,
        )


class VariationalStrategy(_VariationalStrategy):
    r"""
    The standard variational strategy, as defined by `Hensman et al. (2015)`_.
    This strategy takes a set of :math:`m \ll n` inducing points :math:`\mathbf Z`
    and applies an approximate distribution :math:`q( \mathbf u)` over their function values.
    (Here, we use the common notation :math:`\mathbf u = f(\mathbf Z)`.
    The approximate function distribution for any abitrary input :math:`\mathbf X` is given by:

    .. math::

        q( f(\mathbf X) ) = \int p( f(\mathbf X) \mid \mathbf u) q(\mathbf u) \: d\mathbf u

    This variational strategy uses "whitening" to accelerate the optimization of the variational
    parameters. See `Matthews (2017)`_ for more info.

    :param model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """

    def __init__(
        self,
        model: ApproximateGP,
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        learn_inducing_locations: bool = True,
        jitter_val: Optional[float] = None,
        hetero: bool = False,
        aniso: bool = False,
    ):
        super().__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations, jitter_val=jitter_val, hetero=hetero, aniso = aniso
        )
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.has_fantasy_strategy = True

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLinearOperator(ones))
        return res

    @property
    @cached(name="pseudo_points_memo")
    def pseudo_points(self) -> Tuple[Tensor, Tensor]:
        # TODO: have var_mean, var_cov come from a method of _variational_distribution
        # while having Kmm_root be a root decomposition to enable CIQVariationalDistribution support.

        raise Exception('Yeeeee bwoiiiii')

        # retrieve the variational mean, m and covariance matrix, S.
        if not isinstance(self._variational_distribution, CholeskyVariationalDistribution):
            raise NotImplementedError(
                "Only CholeskyVariationalDistribution has pseudo-point support currently, ",
                "but your _variational_distribution is a ",
                self._variational_distribution.__name__,
            )

        var_cov_root = TriangularLinearOperator(self._variational_distribution.chol_variational_covar)
        var_cov = CholLinearOperator(var_cov_root)
        var_mean = self.variational_distribution.mean
        if var_mean.shape[-1] != 1:
            var_mean = var_mean.unsqueeze(-1)

        # compute R = I - S
        cov_diff = var_cov.add_jitter(-1.0)
        cov_diff = -1.0 * cov_diff

        # K^{1/2}
        Kmm = self.model.covar_module(self.inducing_points)
        Kmm_root = Kmm.cholesky()

        # D_a = (S^{-1} - K^{-1})^{-1} = S + S R^{-1} S
        # note that in the whitened case R = I - S, unwhitened R = K - S
        # we compute (R R^{T})^{-1} R^T S for stability reasons as R is probably not PSD.
        eval_var_cov = var_cov.to_dense()
        eval_rhs = cov_diff.transpose(-1, -2).matmul(eval_var_cov)
        inner_term = cov_diff.matmul(cov_diff.transpose(-1, -2))
        # TODO: flag the jitter here
        inner_solve = inner_term.add_jitter(self.jitter_val).solve(eval_rhs, eval_var_cov.transpose(-1, -2))
        inducing_covar = var_cov + inner_solve

        inducing_covar = Kmm_root.matmul(inducing_covar).matmul(Kmm_root.transpose(-1, -2))

        # mean term: D_a S^{-1} m
        # unwhitened: (S - S R^{-1} S) S^{-1} m = (I - S R^{-1}) m
        rhs = cov_diff.transpose(-1, -2).matmul(var_mean)
        # TODO: this jitter too
        inner_rhs_mean_solve = inner_term.add_jitter(self.jitter_val).solve(rhs)
        pseudo_target_mean = Kmm_root.matmul(inner_rhs_mean_solve)

        # ensure inducing covar is psd
        # TODO: make this be an explicit root decomposition
        try:
            pseudo_target_covar = CholLinearOperator(inducing_covar.add_jitter(self.jitter_val).cholesky()).to_dense()
        except NotPSDError:
            from linear_operator.operators import DiagLinearOperator

            evals, evecs = torch.linalg.eigh(inducing_covar)
            pseudo_target_covar = (
                evecs.matmul(DiagLinearOperator(evals + self.jitter_val)).matmul(evecs.transpose(-1, -2)).to_dense()
            )

        return pseudo_target_covar, pseudo_target_mean

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: Optional[LinearOperator] = None,
        **kwargs,
    ) -> MultivariateNormal:
        # Compute full prior distribution
        if self.hetero:
            a_0 = 1/torch.square(self.model.covar_module.base_kernel.lengthscale)
            sigma2 = self.model.covar_module.outputscale

            if self.aniso:
                ls_scale = inducing_points[:,:,1:]
                Z = inducing_points[:,:,0]
                M,P = Z.shape

                B = ls_scale.transpose(1,2) @ ls_scale
                li_0 = 1/torch.sqrt(a_0)
                A_zi = li_0[:,:,torch.newaxis]*(0.5*torch.eye(P)[torch.newaxis,:,:]+B)*li_0[:,torch.newaxis,:]
                A_z = torch.linalg.inv(A_zi)
                L_z = torch.linalg.cholesky(A_z).transpose(1,2)
                #torch.max(torch.abs(A_z - L_z.transpose(1,2) @ L_z))
                #D = torch.diag(l_0.squeeze())
                #A_z[0,:,:] - (D @ (0.5*torch.eye(P)[torch.newaxis,:,:]+B[0,:,:]) @ D)

                # XZ corr.
                D_xz = x[torch.newaxis,:,:]-Z[:,torch.newaxis,:]
                D_xz = L_z[:,torch.newaxis,:,:] @ D_xz[:,:,:,torch.newaxis]
                D_xz = D_xz.squeeze()
                D_xz = torch.square(D_xz)
                K_zx = sigma2*torch.exp(-0.5*torch.sum(D_xz, axis = -1))

                # ZZ Corr
                Dai = torch.diag(1/a_0.squeeze())
                DELTA = (A_zi[torch.newaxis,:,:,:]+A_zi[:,torch.newaxis,:,:]-Dai[torch.newaxis,torch.newaxis,:,:])
                DELTAi = torch.linalg.inv(DELTA)
                L_d = torch.linalg.cholesky(DELTAi).transpose(-1,-2)

                D_zz = Z[:,torch.newaxis,:]-Z[torch.newaxis,:,:]
                D_zz = L_d @ D_zz[:,:,:,torch.newaxis]
                D_zz = D_zz.squeeze()
                D_zz = torch.square(D_zz)
                #D_zzD = (DELTAi@D_zz[:,:,:,torch.newaxis]).squeeze()

                A_zldet = torch.linalg.slogdet(A_z)[1]
                A_0ldet = torch.sum(torch.log(a_0))
                DELTA_ldet = torch.linalg.slogdet(DELTA)[1]
                lconst = A_0ldet[torch.newaxis,torch.newaxis] -A_zldet[torch.newaxis,:]-A_zldet[:,torch.newaxis]-DELTA_ldet
                const = torch.exp(0.5*lconst)

                K_zz = sigma2*const*torch.exp(-0.5*torch.sum(D_zz, axis = -1))

                #D_xx = a_0[torch.newaxis,:,:]*torch.square(x[torch.newaxis,:,:]-x[:,torch.newaxis,:])
                #K_xx = sigma2*torch.exp(-0.5*torch.sum(D_xx, axis = -1))
                #K_top = torch.concat([K_xx,K_zx.T], axis = 1)
                #K_bot = torch.concat([K_zx,K_zz], axis = 1)
                #K = torch.concat([K_top, K_bot], axis = 0)
                #torch.linalg.eigh(K)[0]

            else:
                M,P,_ = inducing_points.shape
                Z = inducing_points[:,:,0]
                lss = inducing_points[:,:,1]

                a_z = (0.5+torch.exp(lss))*a_0

                # XZ corr.
                D_xz = a_z[:,torch.newaxis,:]*torch.square(x[torch.newaxis,:,:]-Z[:,torch.newaxis,:])
                K_zx = sigma2*torch.exp(-0.5*torch.sum(D_xz, axis = -1))

                # ZZ Corr
                mults = 1/(1/a_z[torch.newaxis,:,:] + 1/a_z[:,torch.newaxis,:] - 1/a_0[torch.newaxis,:,:])
                D_zz = mults*torch.square(Z[:,torch.newaxis,:]-Z[torch.newaxis,:,:])
                const = torch.sqrt(torch.prod(mults/a_z[torch.newaxis,:,:]/a_z[:,torch.newaxis,:]*a_0[torch.newaxis,:,:], axis = -1))
                K_zz = sigma2*const*torch.exp(-0.5*torch.sum(D_zz, axis = -1))


            #D_xx = a_0[torch.newaxis,:,:]*torch.square(x[torch.newaxis,:,:]-x[:,torch.newaxis,:])
            #K_xx = torch.exp(-0.5*torch.sum(D_xx, axis = -1))
            #K_xx *= self.model.covar_module.outputscale
            #torch.sum(torch.square(data_data_covar.to_dense() - K_xx))
            #K_top = torch.concat([K_xx,K_zx.T], axis = 1)
            #K_bot = torch.concat([K_zx,K_zz], axis = 1)
            #K = torch.concat([K_top, K_bot], axis = 0)

            #self.model.forward(x).covariance_matrix/0.6931

            #torch.min(torch.linalg.eigh(K)[0])
            #torch.min(torch.linalg.eigh(K_zz)[0])
            # Not pd!

            ## Package for rest of function.
            test_mean = self.model.mean_module(x)
            #induc_induc_covar = K_zz.add_jitter(self.jitter_val)
            induc_induc_covar = K_zz + self.jitter_val * torch.eye(M)
            induc_data_covar = K_zx.to_dense()
            data_data_covar = self.model.covar_module(x)

        else:
            full_inputs = torch.cat([inducing_points, x], dim=-2)
            full_output = self.model.forward(full_inputs, **kwargs)
            full_covar = full_output.lazy_covariance_matrix

            # Covariance terms
            num_induc = inducing_points.size(-2)
            induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
            induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
            data_data_covar = full_covar[..., num_induc:, num_induc:]
            test_mean = full_output.mean[..., num_induc:]


        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        #import IPython; IPython.embed()
        #import IPython; IPython.embed()
        try:
            L = self._cholesky_factor(induc_induc_covar)
        except Exception as e:
            import IPython; IPython.embed()
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(x.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        #import IPython; IPython.embed()
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(self.jitter_val).to_dense()
                + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x: Tensor, prior: bool = False, **kwargs) -> MultivariateNormal:
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter(self.jitter_val))

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                if isinstance(variational_dist, MultivariateNormal):
                    mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                    whitened_mean = L.solve(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                    covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.to_dense()
                    covar_root = covar_root.type(_linalg_dtype_cholesky.value())
                    whitened_covar = RootLinearOperator(L.solve(covar_root).to(variational_dist.loc.dtype))
                    whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                    self._variational_distribution.initialize_variational_distribution(
                        whitened_variational_distribution
                    )

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)
