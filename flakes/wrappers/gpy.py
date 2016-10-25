from flakes.string import StringKernel
from paramz.transformations import Logexp
from GPy.kern import Kern
from GPy.core.parameterization import Param
import numpy as np
import GPy
from paramz.caching import Cache_this


class GPyStringKernel(StringKernel, Kern):
    """
    Flakes string kernel wrapped in a GPy API.
    """
    def __init__(self, gap_decay=1.0, match_decay=1.0,
                 order_coefs=[1.0], variance=1.0, 
                 mode='tf-batch',
                 sim='dot', wrapper='none',
                 active_dims=None, name='string',
                 embs=None, alphabet=None,
                 device='/cpu:0', batch_size=1000,
                 config=None, index=None):
        Kern.__init__(self, 1, active_dims, name)
        StringKernel.__init__(self, gap_decay, match_decay,
                              order_coefs, variance, mode, 
                              sim=sim, wrapper=wrapper, embs=embs,
                              alphabet=alphabet, device=device,
                              batch_size=batch_size, config=config,
                              index=index)
        self.gap_decay = Param('gap_decay', gap_decay, Logexp())
        self.match_decay = Param('match_decay', match_decay, Logexp())
        self.order_coefs = Param('coefs', order_coefs, Logexp())
        self.graph = None
        self.link_parameter(self.gap_decay)
        self.link_parameter(self.match_decay)
        self.link_parameter(self.order_coefs)
        self.wrapper = wrapper
        if wrapper != 'none':
            self.variance = Param('variance', variance, Logexp())
            self.link_parameter(self.variance)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: 
            dL_dK = (dL_dK + dL_dK.T) / 2
        # Workaround to enable kronecker product gradient
        # We need to update gradients to reflect a tiled gram matrix
        if dL_dK.shape[0] != self.gap_grads.shape[0]:
            n_out = dL_dK.shape[0] / self.gap_grads.shape[0]
            self.gap_decay.gradient = np.sum(np.tile(self.gap_grads, (n_out, n_out)) * dL_dK)
            self.match_decay.gradient = np.sum(np.tile(self.match_grads, (n_out, n_out)) * dL_dK)
            for i in xrange(self.order):
                self.order_coefs.gradient[i] = np.sum(np.tile(self.coef_grads[:, :, i], (n_out, n_out)) * dL_dK)
            if self.wrapper != 'none':
                self.variance.gradient = np.sum(np.tile(self.var_grads, (n_out, n_out)) * dL_dK)
        else:
            self.gap_decay.gradient = np.sum(self.gap_grads * dL_dK)
            self.match_decay.gradient = np.sum(self.match_grads * dL_dK)
            for i in xrange(self.order):
                self.order_coefs.gradient[i] = np.sum(self.coef_grads[:, :, i] * dL_dK)
            if self.wrapper != 'none':
                self.variance.gradient = np.sum(self.var_grads * dL_dK)

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None):
        #result = self.K(X, X, diag=True)
        #print "NOT CACHED"
        result = StringKernel.K(self, X, X2)
        return result[:len(X)]

    @Cache_this(limit=3, ignore_args=())
    def Kdiag(self, X):
        result = StringKernel.K(self, X, X, diag=True)
        #result = self.K(X, X, diag=True)
        return result[:len(X)]

    def _get_params(self):
        """
        Overriding this because of the way GPy handles parameters.
        """
        if self.wrapper == 'none':
            return [self.gap_decay[0], self.match_decay[0],
                    self.order_coefs]
        else:
            return [self.gap_decay[0], self.match_decay[0],
                    self.order_coefs, self.variance[0]]
        

class RBFStringKernel(StringKernel, GPy.kern.RBF):
    """
    String kernel with an RBF wrapper
    """

    def __init__(self, gap_decay=1.0, match_decay=1.0,
                 order_coefs=[1.0], variance=1.0, 
                 mode='tf-batch',
                 sim='dot', wrapper='none',
                 active_dims=None, name='rbf_string',
                 embs=None, alphabet=None,
                 device='/cpu:0', batch_size=1000,
                 config=None, index=None):
        Kern.__init__(self, 1, active_dims, name)
        StringKernel.__init__(self, gap_decay, match_decay,
                              order_coefs, variance, mode, 
                              sim=sim, wrapper=wrapper, embs=embs,
                              alphabet=alphabet, device=device,
                              batch_size=batch_size, config=config,
                              index=index)
        self.gap_decay = Param('gap_decay', gap_decay, Logexp())
        self.match_decay = Param('match_decay', match_decay, Logexp())
        self.order_coefs = Param('coefs', order_coefs, Logexp())
        self.graph = None
        self.link_parameter(self.gap_decay)
        self.link_parameter(self.match_decay)
        self.link_parameter(self.order_coefs)

        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)
        self.use_invLengthscale = False
        self.ARD = False
        self.lengthscale = Param('lengthscale', 1.0, Logexp())
        self.lengthscale.constrain_fixed(1.0)

    @Cache_this(limit=3, ignore_args=())
    def _string_K(self, X, X2=None):
        #print "NOT CACHED"
        result = StringKernel.K(self, X, X2)
        gap_grads = self.gap_grads
        match_grads = self.match_grads
        coef_grads = self.coef_grads
        return result, gap_grads, match_grads, coef_grads

    @Cache_this(limit=3, ignore_args=())
    def _string_Kdiag(self, X):
        result = StringKernel.K(self, X, X, diag=True)
        gap_grads = self.gap_grads
        match_grads = self.match_grads
        coef_grads = self.coef_grads
        return result, gap_grads, match_grads, coef_grads

    @Cache_this(limit=3, ignore_args=())
    def _scaled_dist_and_grads(self, X, X2=None):
        """
        Returns the scaled distance between inputs.
        We assume lengthscale=1 since any ls changes
        can be absorbed into the sk coeficients.
        We also precalculate gradients.
        """
        #print "CALCULATING r"
        k, gap_g, match_g, coefs_g = self._string_K(X, X2)
        diag1, diag_gap_g1, diag_match_g1, diag_coefs_g1 = self._string_Kdiag(X)
        if X2 == None:
            diag2, diag_gap_g2, diag_match_g2, diag_coefs_g2 = diag1, diag_gap_g1, diag_match_g1, diag_coefs_g1
        else:
            diag2, diag_gap_g2, diag_match_g2, diag_coefs_g2 = self._string_Kdiag(X2)
        # Direct sum
        dsum = diag1[:, None] + diag2[None, :]
        r = dsum - (2 * k)
        
        dsum_dgap = diag_gap_g1[:, None] + diag_gap_g2[None, :]
        dr_dgap = dsum_dgap - (2 * gap_g)
        dsum_dmatch = diag_match_g1[:, None] + diag_match_g2[None, :]
        dr_dmatch = dsum_dmatch - (2 * match_g)
        dr_dcoefs = np.zeros_like(coefs_g)
        for i in xrange(self.order):
            dsum_dcoef = diag_coefs_g1[:, None, i] + diag_coefs_g2[None, :, i]
            dr_dcoefs[:, : , i] = dsum_dcoef - (2 * coefs_g[:, : ,i])
               
        return r, dr_dgap, dr_dmatch, dr_dcoefs

    def _scaled_dist(self, X, X2=None):
        return self._scaled_dist_and_grads(X, X2)[0]

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None):
        return GPy.kern.RBF.K(self, X, X2)

    def update_gradients_full(self, dL_dK, X, X2=None):
        #if X2 is None: 
        #    dL_dK = (dL_dK + dL_dK.T) / 2

        self.variance.gradient = np.sum(self.K(X, X2) * dL_dK) / self.variance
        r, dr_dgap, dr_dmatch, dr_dcoefs = self._scaled_dist_and_grads(X, X2)

        dterm = -self.K(X, X2) * r
        self.gap_decay.gradient = np.sum(dterm * dr_dgap * dL_dK)
        self.match_decay.gradient = np.sum(dterm * dr_dmatch * dL_dK)
        for i in xrange(self.order):
            self.order_coefs.gradient[i] = np.sum(dterm * dr_dcoefs[:, :, i] * dL_dK)            

    def _get_params(self):
        """
        Overriding this because of the way GPy handles parameters.
        """
        return [self.gap_decay[0], self.match_decay[0],
                self.order_coefs, self.variance[0]]
