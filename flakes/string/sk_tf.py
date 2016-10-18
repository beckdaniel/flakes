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
            self.norms = np.sqrt(tnp.sum(pow(embs, 2), 1, keepdims=True))
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
                     self._coefs: np.array(params[2])[None, :]}
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

            # Initialize Kp, one for each n-gram order (including 0)
            initial_Kp = tf.ones(shape=(order+1, n, n), dtype=tf.float64)
            Kp = taops.TensorArray(dtype=initial_Kp.dtype, size=order+1,
                                   tensor_array_name="Kp")
            Kp = Kp.unpack(initial_Kp)

            # Auxiliary Kp for using in While.
            acc_Kp = taops.TensorArray(dtype=initial_Kp.dtype, size=order+1,
                                       tensor_array_name="ret_Kp")

            # Main loop, where Kp values are calculated.
            i = tf.constant(0)
            a = Kp.read(0)
            acc_Kp = acc_Kp.write(0, a)
            def _update_Kp(acc_Kp, a, S, i):
                aux1 = tf.mul(S, a)
                aux2 = tf.transpose(tf.matmul(aux1, D) * match_sq)
                a = tf.transpose(tf.matmul(aux2, D))
                i += 1
                acc_Kp = acc_Kp.write(i, a)
                return [acc_Kp, a, S, i]
            cond = lambda _1, _2, _3, i: i < order
            loop_vars = [acc_Kp, a, S, i]
            final_Kp, _, _, _ = tf.while_loop(cond=cond, body=_update_Kp, 
                                              loop_vars=loop_vars)
            final_Kp = final_Kp.pack()

            # Final calculation. We gather all Kps and
            # multiply then by their coeficients.
            mul1 = S * final_Kp[:order, :, :]
            sum1 = tf.reduce_sum(mul1, 1)
            Ki = tf.reduce_sum(sum1, 1, keep_dims=True) * match_sq
            result = tf.matmul(self._coefs, Ki)
            gap_grads = tf.gradients(result, self._gap)
            match_grads = tf.gradients(result, self._match)
            coef_grads = tf.gradients(result, self._coefs)
            all_stuff = [result] + gap_grads + match_grads + coef_grads
            self.result = all_stuff

    def _dot(self, s1, s2, tf_embs):
        """
        Simple dot product between two vectors of embeddings.
        This returns a matrix of positive real numbers.
        """
        mat1 = tf.gather(tf_embs, s1)
        mat2 = tf.gather(tf_embs, s2)
        #return tf.matmul(tf.transpose(mat1), mat2)
        return tf.matmul(mat1, tf.transpose(mat2))

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
        """
        # We need a better way to name this...
        # params[2] should be always order_coefs
        order = len(params[2])

        # We have to build a new graph if 1) there is no graph or
        # 2) current graph maxlen is not large enough for these inputs
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

        # Initialize return values
        k_result = np.zeros(shape=(len(X), len(X2)))
        gap_grads = np.zeros(shape=(len(X), len(X2)))
        match_grads = np.zeros(shape=(len(X), len(X2)))
        coef_grads = np.zeros(shape=(len(X), len(X2), order))

        # All set up. Proceed with Gram matrix calculation.
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                if gram and (j < i):
                    k_result[i, j] = k_result[j, i]
                    gap_grads[i, j] = gap_grads[j, i]
                    match_grads[i, j] = match_grads[j, i]
                    coef_grads[i, j] = coef_grads[j, i]
                else:
                    result = self._k(x1[0], x2[0], params, sess)
                    k_result[i, j] = result[0]
                    gap_grads[i, j] = result[1]
                    match_grads[i, j] = result[2]
                    coef_grads[i, j] = np.array(result[3:])
        sess.close()
        return k_result, gap_grads, match_grads, coef_grads
