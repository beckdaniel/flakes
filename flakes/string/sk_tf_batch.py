import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops as cfops
from tensorflow.python.ops import tensor_array_ops as taops

BATCH_SIZE = 500

class TFBatchStringKernel(object):
    """
    A TensorFlow string kernel implementation.
    """
    def __init__(self, gap_decay=1.0, match_decay=1.0,
                 order_coefs=[1.0], device='/cpu:0'):    
        self.gap_decay = gap_decay
        self.match_decay = match_decay
        self.order_coefs = order_coefs
        self.graph = None
        self.maxlen = 0
        self.device = device
        if 'gpu' in device:
            self.gpu_config = tf.ConfigProto(
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0),
                device_count = {'gpu': 1}
            )

    def _build_graph_batch(self, n):
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
            self.mat_list1 = tf.placeholder("float", [None, n, None])
            self.mat_list2 = tf.placeholder("float", [None, None, n])
            S = tf.batch_matmul(self.mat_list1, self.mat_list2)
            batch_size = tf.shape(self.mat_list1)[0]

            # Initilazing auxiliary variables.
            match_decay_sq = match_decay * match_decay

            # Triangular matrices over decay powers.
            npd = np.zeros((n, n))
            i1, i2 = np.indices(npd.shape)
            for k in xrange(n-1):
                npd[i2-k-1 == i1] = gap_decay ** k
            D = tf.constant(npd, dtype=tf.float32)

            # Initialize Kp, one for each n-gram order (including 0)
            Kp_shape = tf.pack([order+1, batch_size, n, n])
            initial_Kp = tf.fill(Kp_shape, 1.0)
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
            #print [batch_size * n, n]
            def _update_Kp(acc_Kp, a, S, i):
                aux1 = tf.mul(S, a)
                #aux2 = tf.batch_matmul(aux1, D) * match_decay_sq
                #aux3 = tf.transpose(aux2, perm=[0, 2, 1])
                #aux4 = tf.batch_matmul(aux3, D)
                #a = tf.transpose(aux4, perm=[0, 2, 1])
                #shape1 = tf.mul(batch_size, n)
                aux2 = tf.reshape(aux1, tf.pack([batch_size * n, n]))
                aux3 = tf.matmul(aux2, D) * match_decay_sq
                aux4 = tf.reshape(aux3, tf.pack([batch_size, n, n]))
                aux5 = tf.transpose(aux4, perm=[0, 2, 1])
                aux6 = tf.reshape(aux5, tf.pack([batch_size * n, n]))
                aux7 = tf.matmul(aux6, D)
                aux8 = tf.reshape(aux7, tf.pack([batch_size, n, n]))
                a = tf.transpose(aux8, perm=[0, 2, 1])
                i += 1
                acc_Kp = acc_Kp.write(i, a)
                return [acc_Kp, a, S, i]
            cond = lambda _1, _2, _3, i: i < order
            loop_vars = [acc_Kp, a, S, i]
            final_Kp, _, _, _ = cfops.While(cond=cond, body=_update_Kp, 
                                            loop_vars=loop_vars)
            final_Kp = final_Kp.pack()

            # Final calculation. We gather all Kps and
            # multiply then by their coeficients.
            mul1 = S * final_Kp[:order, :, :, :]
            sum1 = tf.reduce_sum(mul1, 2)
            Ki = tf.reduce_sum(sum1, 2, keep_dims=True) * match_decay_sq
            Ki = tf.squeeze(Ki, squeeze_dims=[2])
            coefs = tf.convert_to_tensor([self.order_coefs])
            self.result = tf.matmul(coefs, Ki)

    def _k_tf_batch(self, s1, s_list2):
        """
        Calculates k over a string and a list of strings.
        This is a Tensorflow version.
        """
        mat_list1 = []
        mat_list2 = []
        for _s in s_list2:
            mat_list1.append(self._build_input_matrix(s1[0], self.maxlen).T)
            mat_list2.append(self._build_input_matrix(_s[0], self.maxlen))
        output = self.sess.run(self.result, feed_dict={self.mat_list1: mat_list1, 
                                                       self.mat_list2: mat_list2})
        return output


    def K(self, X, X2, gram, params):
        """
        UNFINISHED METHOD
        """
        for i, x1 in enumerate(X):
            print i
            if gram:
                row = X2[:i+1]
            else:
                row = X2
            row_result = np.zeros((1,0))
            for j in xrange((i / BATCH_SIZE) + 1):
                partial_result = self._k_tf_row(x1, row[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
                row_result = np.concatenate((row_result, partial_result), axis=1)
            if gram:
                result[i, :i+1] = row_result
                if i > 0:
                    result[:i+1, i] = row_result
            else:
                result[i] = row_result   
