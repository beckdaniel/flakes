import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops as cfops
from tensorflow.python.ops import tensor_array_ops as taops
import sys


class StringKernel(object):
    """
    A general class for String Kernels. Default mode is
    TensorFlow mode but numpy and non-vectorized implementations
    are also available. The last two should be used only for
    testing and debugging, TF is generally faster, even in
    CPU-only environments.
    
    The parameterization is based on:
    
    Cancedda et. al (2003) "Word-Sequence Kernels" JMLR

    with two different decay parameters: one for gaps
    and another one for symbol matchings. There is 
    also a list of order coeficients which weights
    different n-grams orders. The overall order
    is implicitly obtained by the size of this list.
    This is *not* the symbol-dependent version.
    
    :param gap_decay: decay for symbols gaps, defaults to 1.0
    :param match_decay: decay for symbols matches, defaults to 1.0
    :param order_coefs: list of coefficients for different ngram
    orders, defaults to [1.0]
    :param mode: inner kernel implementation, defaults to TF
    :param sim: similarity measure used, default to hard_match
    :param device: where to run the inner kernel calculation,
    in TF nomenclature (only used if in TF mode).
    """

    def __init__(self, gap_decay=1.0, match_decay=1.0,
                 order_coefs=[1.0], mode='tf',
                 sim='hard', device='/cpu:0'):
        self.gap_decay = gap_decay
        self.match_decay = match_decay
        self.order_coefs = order_coefs
        self.mode = mode

        # This is only used in TF mode
        self.graph = None
        self.maxlen = 0
        self.device = device
        if 'gpu' in device:
            self.gpu_config = tf.ConfigProto(
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
                device_count = {'gpu': 1}
            )

    @property
    def order(self):
        """
        Kernel ngram order, defined implicitly.
        """
        return len(self.order_coefs)

    def K(self, X, X2=None):
        """
        Calculate the Gram matrix over two lists of strings. The
        underlying method used for kernel calculation depends
        on self.mode (slow, numpy or TF). 
        """
        # Symmetry check to ensure that we only calculate
        # the lower diagonal.
        if X2 is None:
            X2 = X
            symm = True
        else:
            symm = False

        # This can also be calculated for single elements but
        # we need to explicitly convert to lists before any
        # processing
        if not (isinstance(X, list) or isinstance(X, np.ndarray)):
            X = [[X]]
        if not (isinstance(X2, list) or isinstance(X2, np.ndarray)):
            X2 = [[X2]]

        # If we are in TF mode we need to do some preparations...
        if self.mode == 'tf':
            # Instead of using one graph per string pair, we
            # build a single one using the maximum length in
            # the data. Smaller strings are padded with zeroes
            # when converted to the matrix input.
            maxlen = max([len(x[0]) for x in np.concatenate((X, X2))])

            # We have to build a new graph if 1) there is no graph or
            # 2) current graph maxlen is not large enough for these inputs
            if self.graph is None:
                sys.stderr.write("No graph found. Building one.\n")
                self._build_graph(maxlen)
                self.maxlen = maxlen
            elif maxlen > self.maxlen:
                sys.stderr.write("Current graph does not have space for" +
                                 "these string sizes. Building a new one.\n")
                self._build_graph(maxlen)
                self.maxlen = maxlen

            # Finally, we also start a TF session
            if 'gpu' in self.device:
                self.sess = tf.Session(graph=self.graph, config=self.gpu_config)
            else:
                self.sess = tf.Session(graph=self.graph)
            
        # All set up. Proceed with Gram matrix calculation.
        result = np.zeros(shape=(len(X), len(X2)))
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                if symm and (j < i):
                    result[i, j] = result[j, i]
                else:
                    if self.mode == 'tf':
                        result[i, j] = self._k_tf(x1[0], x2[0])
                    elif self.mode == 'numpy':
                        result[i, j] = self._k_numpy(x1[0], x2[0])
                    elif self.mode == 'slow':
                        result[i, j] = self._k_slow(x1[0], x2[0])
            print i
        # If we are in TF mode we close the session
        if self.mode == 'tf':
            self.sess.close()

        return result

    def _build_graph(self, n):
        """
        Builds the graph for TF calculation. This should
        be usually called only once but can be called again
        if we update the maximum string length in our
        dataset.
        """
        gap_decay = self.gap_decay
        match_decay = self.match_decay
        order = self.order

        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device(self.device):

            # Strings will be represented as matrices of
            # embeddings and the similarity is just
            # the dot product. Hard match is replicated
            # by using one-hot embeddings.
            self.mat1 = tf.placeholder("float", [None, n])
            self.mat2 = tf.placeholder("float", [None, n])
            S = tf.matmul(tf.transpose(self.mat1), self.mat2)

            # Initilazing auxiliary variables.
            # The zero vectors are used for padding inside While.
            match_decay_sq = match_decay * match_decay
            n_zeros = tf.constant(np.zeros(shape=(1, n-1)), dtype=tf.float32)
            m_zeros = tf.constant(np.zeros(shape=(1, n)), dtype=tf.float32)

            # Triangular matrices over decay powers.
            npd = np.zeros((n, n))
            i1, i2 = np.indices(npd.shape)
            for k in xrange(n):
                npd[i2-k == i1] = gap_decay ** k
            D = tf.constant(npd[1:n, 1:n], dtype=tf.float32)

            # Initialize Kp
            # We know the shape of Kp before building the graph
            # so we can use a "static" list of Ops here.
            # A more elegant solution would involve some scan-like
            # Op but this seems to have some memory leaks in
            # TF 0.7.1
            Kp = []
            Kp.append(tf.ones(shape=(n, n)))
            for i in xrange(1, order):
                aux1 = tf.mul(S[:n-1, :n-1], Kp[i-1][:n-1, :n-1])
                aux2 = tf.transpose(tf.matmul(aux1, D) * match_decay_sq)
                aux3 = tf.concat(0, [n_zeros, aux2])
                aux4 = tf.transpose(tf.matmul(aux3, D))
                Kp.append(tf.concat(0, [m_zeros, aux4]))

            # Final calculation. "result" contains the final kernel value.
            # Because our Kps are in a list we can't use vectorization
            # over "i" anymore... Oh well...
            results = []
            for i in xrange(order):
                coef = self.order_coefs[i]
                aux5 = S * match_decay_sq
                aux6 = tf.mul(aux5, Kp[i])
                aux7 = tf.reduce_sum(aux6)
                results.append(coef * aux7)
            self.result = tf.add_n(results)

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

    def _k_tf(self, s1, s2):
        """
        Calculates k over two strings. Inputs can be strings or lists.
        This is a Tensorflow version 
        """
        if not isinstance(s1, np.ndarray):
            s1 = self._build_input_matrix(s1, self.maxlen)
        else:
            s1 = s1.T
        if not isinstance(s2, np.ndarray):
            s2 = self._build_input_matrix(s2, self.maxlen)
        else:
            s2 = s2.T
        output = self.sess.run(self.result, feed_dict={self.mat1: s1, 
                                                       self.mat2: s2})
        return output

    def _k_numpy(self, s1, s2):
        """
        Calculates k over two strings. Inputs can be strings or lists.
        This is a vectorized version using numpy.
        """
        n = len(s1)
        m = len(s2)
        gap_decay = self.gap_decay
        match_decay = self.match_decay
        order = self.order
        coefs = self.order_coefs

        if not isinstance(s1, np.ndarray):
            s1 = self._build_input_matrix(s1, len(s1))
        if not isinstance(s2, np.ndarray):
            s2 = self._build_input_matrix(s2, len(s2))

        # Store sim(j, k) values
        S = s1.T.dot(s2)
        
        # Triangular matrix over decay powers
        max_len = max(n, m)
        D = np.zeros((max_len, max_len))
        d1, d2 = np.indices(D.shape)
        for k in xrange(max_len):
            D[d2-k == d1] = gap_decay ** k

        # Initializing auxiliary variables
        Kp = np.zeros(shape=(order + 1, n, m))
        Kp[0,:,:] = 1.0
        Kpp = np.zeros(shape=(order, n, m))
        match_decay_sq = match_decay * match_decay

        for i in xrange(order):
            Kpp[i, :-1, 1:] = (match_decay_sq * 
                               (S[:-1,:-1] * Kp[i,:-1,:-1]).dot(D[1:m, 1:m]))
            Kp[i + 1, 1:] = Kpp[i, :-1].T.dot(D[1:n, 1:n]).T
        
        # Final calculation
        Ki = np.sum(np.sum(S * Kp[:-1], axis=1), axis=1) * match_decay_sq
        return Ki.dot(coefs)

    def _k_slow(self, s1, s2):
        """
        This is a slow version using explicit loops. Useful for testing
        but shouldn't be used in practice.
        """
        n = len(s1)
        m = len(s2)
        gap_decay = self.gap_decay
        match_decay = self.match_decay
        order = self.order
        coefs = self.order_coefs

        if not isinstance(s1, np.ndarray):
            s1 = self._build_input_matrix(s1, len(s1)).T
        if not isinstance(s2, np.ndarray):
            s2 = self._build_input_matrix(s2, len(s2)).T

        Kp = np.zeros(shape=(order + 1, n, m))
        for j in xrange(n):
            for k in xrange(m):
                Kp[0][j][k] = 1.0
        for i in xrange(order):
            for j in xrange(n - 1):
                Kpp = 0.0
                for k in xrange(m - 1):
                    Kpp = (gap_decay * Kpp + 
                           match_decay * match_decay * (s1[j].T.dot(s2[k])) * Kp[i][j][k])
                    Kp[i + 1][j + 1][k + 1] = gap_decay * Kp[i + 1][j][k + 1] + Kpp
        result = 0.0
        for i in xrange(order):
            result_i = 0.0
            for j in xrange(n):
                for k in xrange(m):
                    result_i += match_decay * match_decay * (s1[j].T.dot(s2[k])) * Kp[i][j][k]
            result += coefs[i] * result_i
        return result

       
