import numpy as np
from sk_util import build_input_matrix


class NumpyStringKernel(object):
    """
    A vectorized string kernel implementation.
    It is faster than the naive version but
    slower than TensorFlow versions. Also
    kept for documentary and testing purposes.
    """
    def __init__(self, embs, sim='dot'):
        self.embs = embs
        self.embs_dim = embs.shape[1]
        if sim == 'arccosine':
            self.sim = self._arccosine
            self.norms = np.sqrt(np.sum(pow(embs, 2), 1, keepdims=True))
        elif sim == 'pos_dot':
            self.sim = self._pos_dot
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
        lengthscale = params[2]
        coefs = params[3]
        order = len(coefs)

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
        S, dS_dls = self.sim(s1, s2, lengthscale)
        #print S

        # Initializing auxiliary variables
        Kp = np.ones(shape=(order, n, m))
        dKp_dgap = np.zeros(shape=(order, n, m))
        dKp_dmatch = np.zeros(shape=(order, n, m))
        dKp_dls = np.zeros(shape=(order, n, m))
        match_sq = match * match

        for i in xrange(order - 1):
            aux1 = S * Kp[i]
            aux2 = aux1.dot(D[0:m, 0:m])
            Kpp = match_sq * aux2
            Kp[i + 1] = Kpp.T.dot(D[0:n, 0:n]).T

            daux1_dgap = S * dKp_dgap[i]
            daux2_dgap = daux1_dgap.dot(D[0:m, 0:m]) + aux1.dot(dD_dgap[0:m, 0:m])
            dKpp_dgap = match_sq * daux2_dgap
            dKp_dgap[i + 1] = dKpp_dgap.T.dot(D[0:n, 0:n]).T + Kpp.T.dot(dD_dgap[0:n, 0:n]).T

            daux1_dmatch = S * dKp_dmatch[i]
            daux2_dmatch = daux1_dmatch.dot(D[0:m, 0:m])
            dKpp_dmatch = (match_sq * daux2_dmatch) + (2 * match * aux2)
            dKp_dmatch[i + 1] = dKpp_dmatch.T.dot(D[0:n, 0:n]).T

            daux1_dls = (dS_dls * Kp[i]) + (S * dKp_dls[i])
            daux2_dls = daux1_dls.dot(D[0:m, 0:m])
            dKpp_dls = match_sq * daux2_dls
            dKp_dls[i + 1] = dKpp_dls.T.dot(D[0:n, 0:n]).T
            
        # Final calculation
        aux1 = S * Kp
        aux2 = np.sum(aux1, axis=1)
        aux3 = np.sum(aux2, axis=1)
        Ki = match_sq * aux3
        k = Ki.dot(coefs)

        daux1_dgap = S * dKp_dgap
        daux2_dgap = np.sum(daux1_dgap, axis=1)
        daux3_dgap = np.sum(daux2_dgap, axis=1)
        dKi_dgap = match_sq * daux3_dgap
        dk_dgap = dKi_dgap.dot(coefs)
        
        daux1_dmatch = S * dKp_dmatch
        daux2_dmatch = np.sum(daux1_dmatch, axis=1)
        daux3_dmatch = np.sum(daux2_dmatch, axis=1)
        dKi_dmatch = match_sq * daux3_dmatch + (2 * match * aux3)
        dk_dmatch = dKi_dmatch.dot(coefs)

        daux1_dls = (dS_dls * Kp) + (S * dKp_dls)
        daux2_dls = np.sum(daux1_dls, axis=1)
        daux3_dls = np.sum(daux2_dls, axis=1)
        dKi_dls = match_sq * daux3_dls
        dk_dls = dKi_dls.dot(coefs)

        dk_dcoefs = Ki

        return k, dk_dgap, dk_dmatch, dk_dls, dk_dcoefs

    def _dot(self, s1, s2, ls):
        """
        Simple dot product between two vectors of embeddings.
        This returns a matrix of positive real numbers.
        Also return the gradients (zero in this case).
        """
        embs1 = self.embs[s1]
        embs2 = self.embs[s2]
        return embs1.dot(embs2.T), 0.0

    def _pos_dot(self, s1, s2, ls):
        """
        Dot product with an additional SE kernel on position.
        Position is obtained by gathering the indexes of each string.
        Also return the gradients.
        """
        dot, _ = self._dot(s1, s2, ls)
        pos1 = np.arange(s1.shape[0], dtype=float)
        pos2 = np.arange(s2.shape[0], dtype=float)

        r2 = (pos1[:, None] + pos2[None, :]) ** 2
        pos_match = np.exp(-r2 / ls)
        dpos_dls_term = r2 / (ls ** 2)
        dpos_dls = dot + (pos_match * dpos_dls_term)
        return dot * pos_match, dpos_dls

    def _arccosine(self, s1, s2):
        """
        Uses an arccosine kernel of degree 0 to calculate
        the similarity matrix between two vectors of embeddings. 
        This is just cosine similarity projected into the [0,1] interval.
        """
        embs1 = self.embs[s1]
        embs2 = self.embs[s2]
        normembs1 = self.norms[s1]
        normembs2 = self.norms[s2]
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
        order = len(params[3])
        if diag:
            # Assume only X is given
            k_result = np.zeros(shape=(len(X)))
            gap_grads = np.zeros(shape=(len(X)))
            match_grads = np.zeros(shape=(len(X)))
            ls_grads = np.zeros(shape=(len(X)))
            coef_grads = np.zeros(shape=(len(X), order))
            for i, x1 in enumerate(X):
                result = self._k(x1[0], x1[0], params)
                k_result[i] = result[0]
                gap_grads[i] = result[1]
                match_grads[i] = result[2]
                ls_grads[i] = result[3]
                coef_grads[i] = np.array(result[4:])

        else:
            k_result = np.zeros(shape=(len(X), len(X2)))
            gap_grads = np.zeros(shape=(len(X), len(X2)))
            match_grads = np.zeros(shape=(len(X), len(X2)))
            ls_grads = np.zeros(shape=(len(X), len(X2)))
            coef_grads = np.zeros(shape=(len(X), len(X2), order))
            for i, x1 in enumerate(X):
                for j, x2 in enumerate(X2):
                    if gram and (j < i):
                        k_result[i, j] = k_result[j, i]
                        gap_grads[i, j] = gap_grads[j, i]
                        match_grads[i, j] = match_grads[j, i]
                        ls_grads[i, j] = ls_grads[j, i]
                        coef_grads[i, j] = coef_grads[j, i]
                    else:
                        result = self._k(x1[0], x2[0], params)
                        k_result[i, j] = result[0]
                        gap_grads[i, j] = result[1]
                        match_grads[i, j] = result[2]
                        ls_grads[i, j] = result[3]
                        coef_grads[i, j] = np.array(result[4:])
        return k_result, gap_grads, match_grads, ls_grads, coef_grads
