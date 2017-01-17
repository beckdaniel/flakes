import numpy as np
cimport numpy as np
from sk_util import build_input_matrix
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

class CythonNaiveStringKernel(object):
    """
    A naive, slow string kernel implementation.
    This is a direct transfer from the pseudo-code
    in Cancedda et. al (2003). It is kept for
    documentary and testing purposes, do not use
    this for general applications.
    This version also do not implement gradients.
    """
    def __init__(self, embs):
        self.embs = embs
        self.embs_dim = embs.shape[1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _k(self, s1, s2, params):
        cdef int n = len(s1)
        cdef int m = len(s2)

        cdef int i,j,k = 0
        cdef float gap = params[0]
        cdef float match = params[1]
        cdef list coefs = params[2]
        cdef int order = len(coefs)

        #if not isinstance(s1, np.ndarray):
        #    s1 = build_input_matrix(s1, self.embs, dim=self.embs_dim)
        #if not isinstance(s2, np.ndarray):
        #    s2 = build_input_matrix(s2, self.embs, dim=self.embs_dim)

        # Transform inputs into embedding matrices
        cdef np.ndarray[DTYPE_t, ndim=2] embs1 = self.embs[s1]
        cdef np.ndarray[DTYPE_t, ndim=2] embs2 = self.embs[s2]

        cdef np.ndarray[DTYPE_t, ndim=3] Kp = np.zeros(shape=(order + 1, n, m), dtype=DTYPE)
        cdef float Kpp
        for j in xrange(n):
            for k in xrange(m):
                Kp[0,j,k] = 1.0
        for i in xrange(order):
            for j in xrange(n - 1):
                Kpp = 0.0
                for k in xrange(m - 1):
                    Kpp = (gap * Kpp + 
                           match * match * (embs1[j].T.dot(embs2[k])) * Kp[i,j,k])
                    Kp[i + 1,j + 1,k + 1] = gap * Kp[i + 1,j,k + 1] + Kpp
        cdef float result = 0.0
        cdef float result_i
        for i in xrange(order):
            result_i = 0.0
            for j in xrange(n):
                for k in xrange(m):
                    result_i += (match * match * 
                                 (embs1[j].T.dot(embs2[k])) * Kp[i][j][k])
            result += coefs[i] * result_i
        return result

    def _build_input_matrix(self, s, l):
        """
        Transform an input (string or list) into a
        numpy matrix. Notice that we use an implicit
        zero padding here when l > len(s).
        """
        dim = len(self.alphabet)
        t = np.zeros(shape=(l, dim))
        for i, ch in enumerate(s):
            t[i, self.alphabet[ch]] = 1.0
        return t.T

    def K(self, X, X2, gram, params, diag=False):
        """
        """
        k_result = np.zeros(shape=(len(X), len(X2)))
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                if gram and (j < i):
                    k_result[i, j] = k_result[j, i]
                else:
                    k_result[i, j] = self._k(x1[0], x2[0], params)
        return k_result, 0, 0, 0
