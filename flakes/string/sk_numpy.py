import numpy as np
from sk_util import build_input_matrix


class NumpyStringKernel(object):
    """
    A vectorized string kernel implementation.
    It is faster than the naive version but
    slower than TensorFlow versions. Also
    kept for documentary and testing purposes.
    """
    def __init__(self, embs):
        self.embs = embs
        self.embs_dim = embs[embs.keys()[0]].shape[0]

    def _k(self, s1, s2, params):
        n = len(s1)
        m = len(s2)
        gap = params[0]
        match = params[1]
        coefs = params[2]
        order = len(coefs)

        if not isinstance(s1, np.ndarray):
            s1 = build_input_matrix(s1, self.embs, dim=self.embs_dim)
        if not isinstance(s2, np.ndarray):
            s2 = build_input_matrix(s2, self.embs, dim=self.embs_dim)

        # Store sim(j, k) values
        S = s1.dot(s2.T)
        
        # Triangular matrix over decay powers
        max_len = max(n, m)
        D = np.zeros((max_len, max_len))
        d1, d2 = np.indices(D.shape)
        for k in xrange(max_len):
            D[d2-k == d1] = gap ** k

        # Initializing auxiliary variables
        Kp = np.zeros(shape=(order + 1, n, m))
        Kp[0, :, :] = 1.0
        Kpp = np.zeros(shape=(order, n, m))
        match_sq = match * match

        for i in xrange(order):
            Kpp[i, :-1, 1:] = (match_sq *
                               (S[:-1, :-1] * Kp[i, :-1, :-1]).dot(D[1:m, 1:m]))
            Kp[i + 1, 1:] = Kpp[i, :-1].T.dot(D[1:n, 1:n]).T
        
        # Final calculation
        Ki = np.sum(np.sum(S * Kp[:-1], axis=1), axis=1) * match_sq
        return Ki.dot(coefs)

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
