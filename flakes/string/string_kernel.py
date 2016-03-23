import tensorflow as tf
import numpy as np
import flakes.util.similarities as sims

class StringKernel(object):
    """
    A general class for String Kernels.
    """

    def __init__(self, decay=1.0, order_coefs=[1.0], mode='tf',
                 sim='hard'):
        self.decay = decay
        self.order_coefs = order_coefs
        self.order = len(order_coefs)

        if mode == 'slow':
            self.k = self._k_slow
        elif mode == 'numpy':
            self.k = self._k_numpy
        elif mode == 'tf':
            self.k = self._k_tf

        if sim == 'hard':
            self.sim = sims.hard_match

    def sim(ch1, ch2):
        return ch1 == ch2

    def _k_slow(self, s1, s2):
        """
        This is a slow version using explicit loops. Useful for testing
        but shouldn't be used in practice.
        """
        n = len(s1)
        m = len(s2)
        decay = self.decay
        order = self.order
        sim = self.sim
        coefs = self.order_coefs

        Kp = np.zeros(shape=(order + 1, n, m))
        for j in xrange(n):
            for k in xrange(m):
                Kp[0][j][k] = 1.0
        for i in xrange(order):
            for j in xrange(n - 1):
                Kpp = 0.0
                for k in xrange(m - 1):
                    Kpp = (decay * Kpp + 
                           decay * decay * sim(s1[j], s2[k]) * Kp[i][j][k])
                    Kp[i + 1][j + 1][k + 1] = decay * Kp[i + 1][j][k + 1] + Kpp
                    result = 0.0
        for i in xrange(order):
            result_i = 0.0
            for j in xrange(n):
                for k in xrange(m):
                    result_i += decay * decay * sim(s1[j], s2[k]) * Kp[i][j][k]
            result += coefs[i] * result_i

        return result

    def _k_numpy(self, s1, s2):
        """
        Calculates k over two strings. Inputs can be strings or lists.
        This is a vectorized version using numpy.
        """
        n = len(s1)
        m = len(s2)
        decay = self.decay
        order = self.order
        sim = self.sim
        coefs = self.order_coefs

        Kp = np.zeros(shape=(order + 1, n, m))
        Kp[0,:,:] = 1.0
        result = 0.0

        # Store sim(j, k) values
        S = np.zeros(shape=(n,m))
        for j in xrange(n):
            for k in xrange(m):
                S[j,k] = sim(s1[j], s2[k])
        
        # Triangular matrix over decay powers
        max_len = max(n, m)
        D = np.zeros((max_len, max_len))
        d1, d2 = np.indices(D.shape)
        for k in xrange(max_len):
            D[d2-k == d1] = decay ** k 

        # Initilazing auxiliary variables
        Kpp = np.zeros(shape=(order, n, m))
        decay_sq = decay * decay

        for i in xrange(order):
            Kpp[i, :-1, 1:] = decay_sq* (S[:-1,:-1] * Kp[i,:-1,:-1]).dot(D[1:m, 1:m])
            Kp[i + 1, 1:] = Kpp[i, :-1].T.dot(D[1:n, 1:n]).T
        
        # Final calculation
        Ki = np.sum(np.sum(S * Kp[:-1], axis=1), axis=1) * decay_sq
        #Ki = np.sum(S * np.sum(Kp, axis=0), axis=0) * decay_sq

        return Ki.dot(coefs)
