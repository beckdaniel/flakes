import tensorflow as tf
import numpy as np
import flakes.util.similarities as sims

class StringKernel(object):
    """
    A general class for String Kernels.
    """

    def __init__(self, decay=1.0, order_coefs=[1.0], slow=False,
                 sim='hard'):
        self.decay = decay
        self.order_coefs = order_coefs
        self.order = len(order_coefs)
        if slow:
            self.k = self._k_slow
        else:
            self.k = self._k
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
                    Kpp = decay * Kpp + decay * decay * sim(s1[j], s2[k]) * Kp[i][j][k]
                    Kp[i + 1][j + 1][k + 1] = decay * Kp[i + 1][j][k + 1] + Kpp
                    result = 0.0
        for i in xrange(order):
            result_i = 0.0
            for j in xrange(n):
                for k in xrange(m):
                    result_i += decay * decay * sim(s1[j], s2[k]) * Kp[i][j][k]
            result += coefs[i] * result_i

        return result

    def _k(self, s1, s2):
        pass
        
