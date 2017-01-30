import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops as cfops
from tensorflow.python.ops import tensor_array_ops as taops
from sk_util import build_input_matrix
import sys


class TFStringKernel(object):
    """
    A TensorFlow string kernel implementation.
    """
    def __init__(self, embs, sim='dot', device='/cpu:0', config=None):    
        self.embs = embs
        self.embs_dim = embs.shape[1]
        if sim == 'arccosine':
            self.sim = self._arccosine
            self.norms = np.sqrt(np.sum(pow(embs, 2), 1, keepdims=True))
        elif sim == 'dot':
            self.sim = self._dot
        self.graph = None
        self.maxlen = 0
        self.device = device
        self.tf_config = config

    def _k(self, s1, s2, params, sess):
        """
        Calculates k over two strings. Inputs can be strings or lists.
        This method just calls a session and run the TF graph.
        The inputs are transformed into embedding matrices with
        additional zero padding if their length is smaller
        than the length used in the graph.
        """
        embs = self.embs
        dim = self.embs_dim
        s1 = self._pad(s1, self.maxlen)
        s2 = self._pad(s2, self.maxlen)
        feed_dict = {self._s1: s1, self._s2: s2,
                     self._gap: params[0], 
                     self._match: params[1],
                     self._coefs: np.array(params[3])[None, :]}
        output = sess.run(self.result, feed_dict=feed_dict)
        return output

    def _pad(self, s, length):
        new_s = np.zeros(length)
        new_s[:len(s)] = s
        return new_s

    def _build_graph(self, n, order):
        """
        Builds the graph for TF calculation. It builds all
        tensors based on a maximum string length "n".
        This should be usually called only once but can be called again
        if we update the maximum string length in our dataset.
        """
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device(self.device):
            # We preload word embeddings
            tf_embs = tf.constant(self.embs, dtype=tf.float64, name='embs')

            # Strings will be represented as matrices of
            # embeddings and the similarity is just
            # the dot product. Hard match is replicated
            # by using one-hot embeddings.
            self._s1 = tf.placeholder("int32", [n])
            self._s2 = tf.placeholder("int32", [n])
            S = self.sim(self._s1, self._s2, tf_embs)

            # Kernel hyperparameters are also placeholders.
            # The K function is responsible for tying the
            # hyper values from class to this calculation
            # and to update the hyper gradients.
            self._gap = tf.placeholder("float64", [])
            self._match = tf.placeholder("float64", [])
            self._coefs = tf.placeholder("float64", [1, order])
            match_sq = self._match ** 2

            # Triangular matrices over decay powers.
            power = np.ones((n, n))
            tril = np.zeros((n, n))
            i1, i2 = np.indices(power.shape)
            for k in xrange(n-1):
                power[i2-k-1 == i1] = k
                tril[i2-k-1 == i1] = 1.0
            tf_tril = tf.constant(tril, dtype=tf.float64)
            tf_power = tf.constant(power, dtype=tf.float64)
            gaps = tf.fill([n, n], self._gap)
            D = tf.pow(tf.mul(gaps, tril), power)

            # Main loop, where Kp values are calculated.
            Kp = []
            Kp.append(tf.ones(shape=(n, n), dtype="float64"))
            for i in xrange(order - 1):
                aux1 = S * Kp[i]
                aux2 = tf.matmul(aux1, D)
                Kpp = match_sq * aux2
                Kp.append(tf.transpose(tf.matmul(tf.transpose(Kpp), D)))
            final_Kp = tf.pack(Kp)

            # Final calculation. We gather all Kps and
            # multiply then by their coeficients.
            mul1 = S * final_Kp[:order, :, :]
            sum1 = tf.reduce_sum(mul1, 1)
            Ki = tf.reduce_sum(sum1, 1, keep_dims=True) * match_sq
            result = tf.matmul(self._coefs, Ki)
            gap_grads = tf.gradients(result, self._gap)
            match_grads = tf.gradients(result, self._match)
            #ls_grads = tf.gradients(result, self._ls)
            ls_grads = tf.zeros_like(gap_grads)
            coef_grads = tf.gradients(result, self._coefs)
            #all_stuff = [result] + gap_grads + match_grads + ls_grads
            #all_stuff = [result] + gap_grads + match_grads + ls_grads + coef_grads
            all_stuff = [result, gap_grads, match_grads, ls_grads, coef_grads]
            self.result = all_stuff

    def _dot(self, s1, s2, tf_embs):
        """
        Simple dot product between two vectors of embeddings.
        This returns a matrix of positive real numbers.
        """
        mat1 = tf.gather(tf_embs, s1)
        mat2 = tf.gather(tf_embs, s2)
        return tf.matmul(mat1, tf.transpose(mat2))

    def _arccosine(self, s1, s2, tf_embs):
        """
        Uses an arccosine kernel of degree 0 to calculate
        the similarity matrix between two vectors of embeddings. 
        This is just cosine similarity projected into the [0,1] interval.
        """
        tf_pi = tf.constant(np.pi, dtype=tf.float64)
        mat1 = tf.gather(tf_embs, s1)
        mat2 = tf.gather(tf_embs, s2)
        tf_norms = tf.constant(self.norms, dtype=tf.float64, name='norms')
        norms1 = tf.gather(tf_norms, s1)
        norms2 = tf.gather(tf_norms, s2)
        dot = tf.matmul(mat1, tf.transpose(mat2))
        norms = tf.matmul(norms1, tf.transpose(norms2))
        # We clip values due to numerical errors
        # which put some values outside the arccosine range.
        cosine = tf.clip_by_value(dot / norms, -1, 1)
        angle = tf.acos(cosine)
        # The 0 vector has norm 0, which generates a NaN.
        # We catch these NaNs and replace them with pi,
        # which ends up returning 0 similarity.
        angle = tf.select(tf.is_nan(angle), tf.ones_like(angle) * tf_pi, angle)
        return 1 - (angle / tf_pi)

    def K(self, X, X2, gram, params, diag=False):
        """
        """
        # We need a better way to name this...
        # params[2] should be always order_coefs
        order = len(params[3])

        # We have to build a new graph if 1) there is no graph or
        # 2) current graph maxlen is not large enough for these inputs
        if diag or gram:
            maxlen = max([len(x[0]) for x in X])
        else:
            maxlen = max([len(x[0]) for x in list(X) + list(X2)])

        if self.graph is None:
            #sys.stderr.write("No graph found. Building one.\n")
            self._build_graph(maxlen, order)
            self.maxlen = maxlen
        elif maxlen > self.maxlen:
            sys.stderr.write("Current graph does not have space for" +
                             "these string sizes. Building a new one.\n")
            self._build_graph(maxlen, order)
            self.maxlen = maxlen

        # We also start a TF session
        sess = tf.Session(graph=self.graph, config=self.tf_config)

        if diag:
            # Assume only X is given
            k_result = np.zeros(shape=(len(X)))
            gap_grads = np.zeros(shape=(len(X)))
            match_grads = np.zeros(shape=(len(X)))
            ls_grads = np.zeros(shape=(len(X)))
            coef_grads = np.zeros(shape=(len(X), order))
            for i, x1 in enumerate(X):
                result = self._k(x1[0], x1[0], params, sess)
                k_result[i] = result[0]
                gap_grads[i] = result[1][0]
                match_grads[i] = result[2][0]
                ls_grads[i] = result[3][0]
                coef_grads[i] = np.array(result[4:])
        else:
            # Initialize return values
            k_result = np.zeros(shape=(len(X), len(X2)))
            gap_grads = np.zeros(shape=(len(X), len(X2)))
            match_grads = np.zeros(shape=(len(X), len(X2)))
            ls_grads = np.zeros(shape=(len(X), len(X2)))
            coef_grads = np.zeros(shape=(len(X), len(X2), order))

            # All set up. Proceed with Gram matrix calculation.
            for i, x1 in enumerate(X):
                for j, x2 in enumerate(X2):
                    if gram and (j < i):
                        k_result[i, j] = k_result[j, i]
                        gap_grads[i, j] = gap_grads[j, i]
                        match_grads[i, j] = match_grads[j, i]
                        ls_grads[i, j] = ls_grads[j, i]
                        coef_grads[i, j] = coef_grads[j, i]
                    else:
                        result = self._k(x1[0], x2[0], params, sess)
                        k_result[i, j] = result[0]
                        gap_grads[i, j] = result[1][0]
                        match_grads[i, j] = result[2][0]
                        ls_grads[i, j] = result[3][0]
                        coef_grads[i, j] = np.array(result[4:])
        sess.close()
        return k_result, gap_grads, match_grads, ls_grads, coef_grads
