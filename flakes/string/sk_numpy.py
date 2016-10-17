import numpy as np
from sk_util import build_input_matrix


class NumpyStringKernel(object):
    """
    A vectorized string kernel implementation.
    It is faster than the naive version but
    slower than TensorFlow versions. Also
    kept for documentary and testing purposes.
    """
    def __init__(self, embs, sim='arccosine'):
        self.embs = embs
        self.embs_dim = embs.shape[1]
        if sim == 'arccosine':
            self.sim = self._arccosine
            self.norms = np.sqrt(tnp.sum(pow(embs, 2), 1, keepdims=True))
        elif sim == 'dot':
            self.sim = self._dot

    def _k(self, s1, s2, params):
        """
        The actual string kernel calculation. It also
        calculates gradients with respect to the
        multiple hyperparameters.
        """
        n = len(s1)
        m = len(s2)
        gap = params[0]
        match = params[1]
        coefs = params[2]
        order = len(coefs)

        # Transform inputs into embedding matrices
        embs1 = self.embs[s1]
        embs2 = self.embs[s2]
        
        # Triangular matrix over decay powers
        maxlen = max(n, m)
        power = np.ones((maxlen, maxlen))
        tril = np.zeros((maxlen, maxlen))
        i1, i2 = np.indices(power.shape)
        for k in xrange(maxlen - 1):
            power[i2-k-1 == i1] = k
            tril[i2-k-1 == i1] = 1.0
        gaps = np.ones((maxlen, maxlen)) * gap
        D = (gaps * tril) ** power
        dD_dgap = ((gaps * tril) ** (power - 1.0)) * tril * power

        # Store sim(j, k) values
        S = self.sim(embs1, embs2)

        # Initializing auxiliary variables
        Kp = np.ones(shape=(order + 1, n, m))
        match_sq = match * match

        for i in xrange(order):
            Kpp = match_sq * (S * Kp[i]).dot(D[0:m, 0:m])
            Kp[i + 1] = Kpp.T.dot(D[0:n, 0:n]).T

        # Final calculation
        Ki = np.sum(np.sum(S * Kp[:-1], axis=1), axis=1) * match_sq
        return Ki.dot(coefs)

    def _dot(self, embs1, embs2):
        """
        Simple dot product between two vectors of embeddings.
        This returns a matrix of positive real numbers.
        """
        return embs1.dot(embs2.T)

    def _arccosine(self, embs1, embs2):
        """
        Uses an arccosine kernel of degree 0 to calculate
        the similarity matrix between two vectors of embeddings. 
        This is just cosine similarity projected into the [0,1] interval.
        """
        normembs1 = self.norms[embs1]
        normembs2 = self.norms[embs2]
        norms = np.dot(normembs1, normembs2.T)
        dot = embs1.dot(embs2.T)
        # We clip values due to numerical errors
        # which put some values outside the arccosine range.
        cosine = np.clip(dot / norms, -1, 1)
        angle = np.arccos(cosine)
        return 1 - (angle / np.pi)

    def K(self, X, X2, gram, params, diag=False):
        """
        Calculates and returns the Gram matrix between two lists
        of strings. These should be encoded as lists of integers.
        """
        k_result = np.zeros(shape=(len(X), len(X2)))
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                if gram and (j < i):
                    k_result[i, j] = k_result[j, i]
                else:
                    k_result[i, j] = self._k(x1[0], x2[0], params)
        return k_result, 0, 0, 0
