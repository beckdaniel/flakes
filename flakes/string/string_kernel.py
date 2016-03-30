import tensorflow as tf
import numpy as np
import flakes.util.similarities as sims
from tensorflow.python.ops import control_flow_ops as cfops
from tensorflow.python.ops import tensor_array_ops as taops
from memory_profiler import profile


class StringKernel(object):
    """
    A general class for String Kernels.
    """

    def __init__(self, decay=1.0, order_coefs=[1.0], mode='tf',
                 sim='hard'):
        self.decay = decay
        self.order_coefs = order_coefs
        self.order = len(order_coefs)
        # Select implementation
        if mode == 'slow':
            self.k = self._k_slow
        elif mode == 'numpy':
            self.k = self._k_numpy
        elif mode == 'tf':
            self.k = self._k_tf
        # Select similarity metric
        if sim == 'hard':
            self.sim = sims.hard_match
        self.graph = None

    #def _tf_init(self, maxlen):
    #    self.maxlen = maxlen
    #    self._build_graph()
    #    self.session = tf.Session(graph=self.graph)

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

        # Store sim(j, k) values
        S = np.zeros(shape=(n, m))
        for j in xrange(n):
            for k in xrange(m):
                S[j,k] = sim(s1[j], s2[k])
        
        # Triangular matrix over decay powers
        max_len = max(n, m)
        D = np.zeros((max_len, max_len))
        d1, d2 = np.indices(D.shape)
        for k in xrange(max_len):
            D[d2-k == d1] = decay ** k

        # Initializing auxiliary variables
        Kp = np.zeros(shape=(order + 1, n, m))
        Kp[0,:,:] = 1.0
        Kpp = np.zeros(shape=(order, n, m))
        decay_sq = decay * decay

        for i in xrange(order):
            Kpp[i, :-1, 1:] = decay_sq * (S[:-1,:-1] * Kp[i,:-1,:-1]).dot(D[1:m, 1:m])
            Kp[i + 1, 1:] = Kpp[i, :-1].T.dot(D[1:n, 1:n]).T
        
        # Final calculation
        Ki = np.sum(np.sum(S * Kp[:-1], axis=1), axis=1) * decay_sq
        return Ki.dot(coefs)

    def _k_tf(self, s1, s2):
        """
        Calculates k over two strings. Inputs can be strings or lists.
        This is a Tensorflow version which builds a graph and run a session
        on its own.
        """
        if not self.graph:
            self.maxlen = max(len(s1), len(s2))
            self._build_graph()

        # Now we built the input matrices and run the session
        # over the built graph.
        #import ipdb; ipdb.set_trace()
        if not isinstance(s1, np.ndarray):
            s1 = self._build_symbol_tensor(s1)
        if not isinstance(s2, np.ndarray):
            s2 = self._build_symbol_tensor(s2)
        with tf.Session(graph=self.graph) as sess:
            output = sess.run(self.result, feed_dict={self.mat1: s1, self.mat2: s2})
        #output = self.session.run(self.result, feed_dict={self.mat1: s1, self.mat2: s2})

        return output

    def _build_graph(self):
        print "BUILD GRAPH"
        n = self.maxlen
        m = self.maxlen
        decay = self.decay
        order = self.order

        # We create a Graph for the calculation
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Strings will be represented as matrices of
            # embeddings and the similarity is just
            # the dot product. Hard match is replicated
            # by using one-hot embeddings.
            self.mat1 = tf.placeholder("float", [None, n])
            self.mat2 = tf.placeholder("float", [None, m])
            S = tf.matmul(tf.transpose(self.mat1), self.mat2)

            # Initilazing auxiliary variables.
            # The zero vectors are used for padding.
            decay_sq = decay * decay
            n_zeros = tf.constant(np.zeros(shape=(1, n-1)), dtype=tf.float32)
            m_zeros = tf.constant(np.zeros(shape=(1, m)), dtype=tf.float32)

            # Triangular matrices over decay powers.
            max_len = np.max([n, m])
            npd = np.zeros((max_len, max_len))
            i1, i2 = np.indices(npd.shape)
            for k in xrange(max_len):
                npd[i2-k == i1] = decay ** k
            D1 = tf.constant(npd[1:n, 1:n], dtype=tf.float32)
            D2 = tf.constant(npd[1:m, 1:m], dtype=tf.float32)

            # Initialize Kp
            # We know the shape of Kp before building the graph
            # so we can use a "static" list of Ops here.
            # A more elegant solution would involve some scan-like
            # Op but this seems to have some memory leaks in
            # TF 0.7.1
            Kp = []
            Kp.append(tf.ones(shape=(n, m)))
            for i in xrange(1, order):
                aux1 = tf.mul(S[:n-1, :m-1], Kp[i-1][:n-1, :m-1])
                aux2 = tf.transpose(tf.matmul(aux1, D2) * decay_sq)
                aux3 = tf.concat(0, [n_zeros, aux2])
                aux4 = tf.transpose(tf.matmul(aux3, D1))
                Kp.append(tf.concat(0, [m_zeros, aux4]))


            # Final calculation. "result" contains the final kernel value.
            # Because our Kps are in a list we can't use vectorization
            # over "i" anymore... Oh well...
            results = []
            for i in xrange(order):
                coef = self.order_coefs[i]
                aux5 = S * decay_sq
                aux6 = tf.mul(aux5, Kp[i])
                aux7 = tf.reduce_sum(aux6)
                results.append(coef * aux7)
            self.result = tf.add_n(results)

    def _build_symbol_tensor(self, s):
        """
        Transform an input (string or list) into a
        numpy matrix.
        """
        dim = len(self.alphabet)
        t = np.zeros(shape=(self.maxlen, dim))
        for i, ch in enumerate(s):
            t[i, self.alphabet[ch]] = 1.0
        return t.T
       
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
