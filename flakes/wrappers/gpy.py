from flakes.string import StringKernel
from paramz.transformations import Logexp
from GPy.kern import Kern
from GPy.core.parameterization import Param
import numpy as np

class GPyStringKernel(StringKernel, Kern):
    """
    Flakes string kernel wrapped in a GPy API.
    """
    def __init__(self, gap_decay=1.0, match_decay=1.0,
                 order_coefs=[1.0], mode='tf-batch', 
                 active_dims=None, name='string',
                 embs=None, alphabet=None,
                 device='/cpu:0', batch_size=1000,
                 config=None):
        Kern.__init__(self, 1, active_dims, name)
        StringKernel.__init__(self, gap_decay, match_decay,
                              order_coefs, mode, embs=embs,
                              alphabet=alphabet, device=device,
                              batch_size=batch_size, config=config)
        self.gap_decay = Param('gap_decay', gap_decay, Logexp())
        self.match_decay = Param('match_decay', match_decay, Logexp())
        self.order_coefs = Param('coefs', order_coefs, Logexp())
        self.graph = None
        self.link_parameter(self.gap_decay)
        self.link_parameter(self.match_decay)
        self.link_parameter(self.order_coefs)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: 
            dL_dK = (dL_dK + dL_dK.T) / 2
        # Workaround to enable kronecker product gradient
        # We need to update gradients to reflect a tiled gram matrix
        if dL_dK.shape[0] != self.gap_decay.shape[0]:
            n_out = dL_dK.shape[0] / self.gap_grads.shape[0]
            self.gap_decay.gradient = np.sum(np.tile(self.gap_grads, (n_out, n_out)) * dL_dK)
            self.match_decay.gradient = np.sum(np.tile(self.match_grads, (n_out, n_out)) * dL_dK)
            for i in xrange(self.order):
                self.order_coefs.gradient[i] = np.sum(np.tile(self.coef_grads[:, :, i], (n_out, n_out)) * dL_dK)
        else:
            self.gap_decay.gradient = np.sum(self.gap_grads * dL_dK)
            self.match_decay.gradient = np.sum(self.match_grads * dL_dK)
            for i in xrange(self.order):
                self.order_coefs.gradient[i] = np.sum(self.coef_grads[:, :, i] * dL_dK)

    def Kdiag(self, X):
        result = self.K(X, X, diag=True)
        return result[:len(X)]

    def _get_params(self):
        """
        Overriding this because of the way GPy handles parameters.
        """
        return [self.gap_decay[0], self.match_decay[0], self.order_coefs]
        
