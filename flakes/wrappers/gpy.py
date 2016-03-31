from flakes.string import StringKernel
from paramz.transformations import Logexp
from GPy.kern import Kern
from GPy.core.parameterization import Param
import numpy as np

class GPyStringKernel(StringKernel, Kern):
    """
    Flakes string kernel wrapped in a GPy API.
    """
    def __init__(self, decay=1.0, order_coefs=[1.0], mode='tf', 
                 active_dims=None, name='string'):
        Kern.__init__(self, 1, active_dims, name)
        StringKernel.__init__(self, decay, order_coefs, mode)
        self.decay = Param('decay', decay, Logexp())
        self.order_coefs = Param('coefs', order_coefs, Logexp())
        self.order = len(order_coefs)
        # Select implementation
        if mode == 'slow':
            self.k = self._k_slow
        elif mode == 'numpy':
            self.k = self._k_numpy
        elif mode == 'tf':
            self.k = self._k_tf
        self.graph = None
        self.link_parameter(self.decay)
        self.link_parameter(self.order_coefs)
        self.decay.constrain_fixed(decay)
        self.order_coefs.constrain_fixed(order_coefs)

    def K(self, X, X2=None):
        """
        Calculate the Gram matrix over two lists of strings.
        """
        if X2 is not None:
            self.maxlen = max([len(x[0]) for x in np.concatenate((X, X2))])
        else:
            self.maxlen = max([len(x[0]) for x in X])
        if self.graph is None:
            print self.graph
            print "BUILDING GRAPH"
            self._build_graph()
            print "GRAPH BUILT"

        if X2 is None:
            X2 = X
            symm = True
        else:
            symm = False
        result = np.zeros(shape=(len(X), len(X2)))
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                if symm and (j < i):
                    result[i, j] = result[j, i]
                else:
                    result[i, j] = self.k(x1[0], x2[0])
        return result

    def Kdiag(self, X):
        result = np.zeros(shape=(len(X),))
        for i, x1 in enumerate(X):
            result[i] = self.k(x1[0], x1[0])
        return result

    def update_gradients_full(self, dL_dK, X, X2=None):
        pass
        
