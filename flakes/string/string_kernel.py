import tensorflow as tf
import numpy as np
import flakes.util.similarities as sims
from tensorflow.python.ops import control_flow_ops as cfops
from tensorflow.python.ops import tensor_array_ops as taops


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
        S = np.zeros(shape=(n, m))
        for j in xrange(n):
            for k in xrange(m):
                S[j,k] = sim(s1[j], s2[k])
        #print S
        
        # Triangular matrix over decay powers
        max_len = max(n, m)
        D = np.zeros((max_len, max_len))
        d1, d2 = np.indices(D.shape)
        for k in xrange(max_len):
            D[d2-k == d1] = decay ** k
        #print D

        # Initializing auxiliary variables
        Kpp = np.zeros(shape=(order, n, m))
        decay_sq = decay * decay

        for i in xrange(order):
            Kpp[i, :-1, 1:] = decay_sq * (S[:-1,:-1] * Kp[i,:-1,:-1]).dot(D[1:m, 1:m])
            Kp[i + 1, 1:] = Kpp[i, :-1].T.dot(D[1:n, 1:n]).T
        
        #print Kp
        # Final calculation
        Ki = np.sum(np.sum(S * Kp[:-1], axis=1), axis=1) * decay_sq
        #Ki = np.sum(S * np.sum(Kp, axis=0), axis=0) * decay_sq
        #print Ki

        return Ki.dot(coefs)

    def _k_tf(self, s1, s2):
        """
        Calculates k over two strings. Inputs can be strings or lists.
        This is a vectorized version using numpy.
        """
        n = len(s1)
        m = len(s2)
        decay = self.decay
        order = self.order
        sim = self.sim

        # We create a Graph for the calculation
        graph = tf.Graph()
        with graph.as_default():



            # Store sim(j, k) values
            mat1 = tf.placeholder("float", [None, n])
            mat2 = tf.placeholder("float", [None, m])
            S = tf.matmul(tf.transpose(mat1), mat2)
            #S = tf.Print(S, [S], message="S value:", summarize=20)
        
            # Triangular matrix over decay powers
            max_len = np.max([n, m])
            npd = np.zeros((max_len, max_len))
            i1, i2 = np.indices(npd.shape)
            for k in xrange(max_len):
                npd[i2-k == i1] = decay ** k
            D1 = tf.constant(npd[:n, :n], dtype=tf.float32)
            D2 = tf.constant(npd[:m, :m], dtype=tf.float32)

            # Initilazing auxiliary variables
            decay_sq = decay * decay
            n_zeros = tf.constant(np.zeros(shape=(1, n-1)), dtype=tf.float32)
            m_zeros = tf.constant(np.zeros(shape=(1, m)), dtype=tf.float32)

            # Initialize Kp
            ones = tf.ones(shape=(1, n, m))
            zeros = tf.zeros(shape=(order, n, m))
            initial_Kp = tf.concat(0, [ones, zeros])
            Kp = taops.TensorArray(dtype=initial_Kp.dtype, size=order+1,
                                   tensor_array_name="Kp")
            Kp = Kp.unpack(initial_Kp)
            acc_Kp = taops.TensorArray(dtype=initial_Kp.dtype, size=order+1,
                                   tensor_array_name="ret_Kp")

            #######################

            i = tf.constant(0)
            a = Kp.read(0)
            acc_Kp = acc_Kp.write(0, a)
            def _update_Kp(acc_Kp, a, S, i):
                aux1 = tf.mul(S, a[:n-1, :m-1])
                aux2 = tf.transpose(tf.matmul(aux1, D2[1:, 1:]) * decay_sq)
                aux3 = tf.concat(0, [n_zeros, aux2])
                aux4 = tf.transpose(tf.matmul(aux3, D1[1:, 1:]))
                a = tf.concat(0, [m_zeros, aux4])
                i += 1
                acc_Kp = acc_Kp.write(i, a)
                return [acc_Kp, a, S, i]#, t_order]

            cond = lambda _1, _2, _3, i: i < order
            loop_vars = [acc_Kp, a, S[:n-1, :m-1], i]
            final_Kp, _, _, _ = cfops.While(cond=cond, body=_update_Kp, loop_vars=loop_vars)
            final_Kp = final_Kp.pack()

            #for i in xrange(order):
            #    aux1 = tf.mul(S[:n-1, :m-1], Kp[i, :n-1, :m-1])
            #    #aux1 = tf.Print(aux1, [aux1], message="aux1 value:", summarize=20)
            #    aux2 = tf.transpose(tf.matmul(aux1, D2[1:, 1:]) * decay_sq)
            #    aux3 = tf.concat(0, [n_zeros, aux2])
            #    aux4 = tf.transpose(tf.matmul(aux3, D1[1:, 1:]))
            #    aux5 = tf.concat(0, [m_zeros, aux4])
            #    tf.scatter_update(Kp, i + 1, aux5)
            #    #tf.Print(Kp, [Kp], message="Kp value:", summarize=20)
        
            ######################

            # Final calculation
            mul1 = S * final_Kp[:order, :, :]
            sum1 = tf.reduce_sum(mul1, 1)
            Ki = tf.reduce_sum(sum1, 1, keep_dims=True) * decay_sq
            #Ki = tf.Print(Ki, [Ki], message="Ki value:", summarize=20)

            # Obtain the final result
            coefs = tf.convert_to_tensor([self.order_coefs])
            result = tf.matmul(coefs, Ki)

            init_op = tf.initialize_all_variables()

        # Now we have to build the input tensors
        t1 = self._build_symbol_tensor(s1)
        t2 = self._build_symbol_tensor(s2)
        #print t1.T.dot(t2)

        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            output = sess.run(result, feed_dict={mat1: t1, mat2: t2})

        return output


    def _build_symbol_tensor(self, s):
        """
        Transform an input (string or list) into a
        numpy matrix.
        """
        dim = len(self.alphabet)
        t = np.zeros(shape=(len(s), dim))
        for i, ch in enumerate(s):
            t[i, self.alphabet[ch]] = 1.0
        #print s
        #print t.T
        return t.T
       
