import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops as cfops
from tensorflow.python.ops import tensor_array_ops as taops
from tensorflow.python.ops import functional_ops as fops
from sk_util import build_input_matrix
import sys


class TFGramStringKernel(object):
    """
    A TensorFlow string kernel implementation.
    """
    def __init__(self, embs, device='/cpu:0'):    
        self.embs = embs
        self.embs_dim = embs[embs.keys()[0]].shape[0]
        self.graph = None
        self.maxlen = 0
        self.device = device
        self.gram_mode = False
        if 'gpu' in device:
            self.tf_config = tf.ConfigProto(
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0),
                device_count = {'gpu': 1}
            )
        elif 'cpu' in device:
            self.tf_config = tf.ConfigProto(
                use_per_session_threads = 4,
                intra_op_parallelism_threads = 4,
                inter_op_parallelism_threads = 4,
            )

    def _build_graph(self, n, order, X, X2=None):
        """
        Builds the graph for TF calculation. This should
        be usually called only once but can be called again
        if we update the maximum string length in our
        dataset.
        """
        if X2 == None:
            X2 = X
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device(self.device):
            # Datasets are loaded as constants. Useful for GPUs.
            # Only feasible for small datasets though.
            tf_X = tf.constant(X, dtype=tf.float32)
            tf_X2 = tf.constant(X2, dtype=tf.float32)

            # Kernel hyperparameters are also placeholders.
            # The K function is responsible for tying the
            # hyper values from class to this calculation
            # and to update the hyper gradients.
            self._gap = tf.placeholder("float", [])
            self._match = tf.placeholder("float", [])
            self._coefs = tf.placeholder("float", [1, order])
            self._index1 = tf.placeholder("int32", [])
            self._index2 = tf.placeholder("int32", [])
            match_sq = tf.squeeze(self._match) ** 2

            # Select the inputs from the pre-loaded
            # dataset
            mat1 = tf.gather(tf_X, self._index1)
            mat2 = tf.gather(tf_X2, self._index2)
            #S = tf.matmul(tf.transpose(mat1), mat2)
            S = tf.matmul(mat1, tf.transpose(mat2))

            # Triangular matrices over decay powers.
            power = np.ones((n, n))
            tril = np.zeros((n, n))
            i1, i2 = np.indices(power.shape)
            for k in xrange(n-1):
                power[i2-k-1 == i1] = k
                tril[i2-k-1 == i1] = 1.0
            tf_tril = tf.constant(tril, dtype=tf.float32)
            tf_power = tf.constant(power, dtype=tf.float32)
            gaps = tf.fill([n, n], tf.squeeze(self._gap))
            D = tf.pow(tf.mul(gaps, tril), power)

            # Initialize Kp, one for each n-gram order (including 0)
            initial_Kp = tf.ones(shape=(order+1, n, n))
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
                a = tf.Print(a, [a, S], message='inside while', summarize=200)
                aux1 = tf.mul(S, a)
                aux2 = tf.transpose(tf.matmul(aux1, D) * match_sq)
                a = tf.transpose(tf.matmul(aux2, D))
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
            mul1 = S * final_Kp[:order, :, :]
            sum1 = tf.reduce_sum(mul1, 1)
            Ki = tf.reduce_sum(sum1, 1, keep_dims=True) * match_sq
            k_result = tf.matmul(self._coefs, Ki)
            gap_grad = tf.gradients(k_result, self._gap)
            match_grad = tf.gradients(k_result, self._match)
            coefs_grad = tf.gradients(k_result, self._coefs)
            #all_stuff = [k_result, gap_grad, match_grad, coefs_grad]
            self.result = [k_result, k_result, k_result, k_result]

    def K(self, X, X2, gram, params):
        """
        """
        # We need a better way to name this...
        # params[2] should be always order_coefs
        order = len(params[2])

        # If we are calculating the gram matrix we 
        # enter gram mode. In gram mode we skip
        # graph rebuilding.
        if gram:
            if not self.gram_mode:
                maxlen = max([len(x[0]) for x in X])
                X = self._code_and_pad(X, maxlen)
                self.maxlen = maxlen
                self.gram_mode = True
                #self.indices = np.array([[i, j] for i in range(len(X)) 
                #                    for j in range(len(X)) if i <= j])
                self._build_graph(maxlen, order, X)
        else: # We rebuild the graph, usually for predictions
            self.gram_mode = False
            maxlen = max([len(x[0]) for x in np.concatenate((X, X2))])
            X = self._code_and_pad(X, maxlen)
            X2 = self._code_and_pad(X2, maxlen)
            self.maxlen = maxlen
            #self.indices = np.array([[i, j] for i in range(len(X)) 
            #                    for j in range(len(X2))])
            self._build_graph(maxlen, order, X, X2)

        # Initialize return values
        k_results = np.zeros(shape=(len(X), len(X2)))
        gap_grads = np.zeros(shape=(len(X), len(X2)))
        match_grads = np.zeros(shape=(len(X), len(X2)))
        coef_grads = np.zeros(shape=(len(X), len(X2), order))

        # We start a TF session and run it
        sess = tf.Session(graph=self.graph, config=self.tf_config)
        for i in xrange(len(X)):
            for j in xrange(len(X2)):
                if gram and (j < i):
                    k_results[i, j] = k_results[j, i]
                    gap_grads[i, j] = gap_grads[j, i]
                    match_grads[i, j] = match_grads[j, i]
                    coef_grads[i, j] = coef_grads[j, i]
                else:
                    feed_dict = {self._gap: params[0], 
                                 self._match: params[1],
                                 self._coefs: np.array(params[2])[None, :],
                                 self._index1: i,
                                 self._index2: i}
                    k_result, gap_grad, match_grad, coef_grad = sess.run(self.result,
                                                                         feed_dict=feed_dict)
                    k_results[i, j] = k_result
                    gap_grads[i, j] = gap_grad
                    match_grads[i, j] = match_grad
                    coef_grads[i, j] = coef_grad
            
        sess.close()
        
        # Reshape the return values since they are vectors:
        #if gram:
        #    lenX2 = None
        #else:
        #    lenX2 = len(X2)
        #k_result = self._triangulate(k_result, self.indices, len(X), lenX2)
        #gap_grads = self._triangulate(gap_grads, self.indices, len(X), lenX2)
        #match_grads = self._triangulate(match_grads, self.indices, len(X), lenX2)
        #coef_grads = self._triangulate(coef_grads, self.indices, len(X), lenX2)
    
        return k_results, gap_grads, match_grads, coef_grads

    def _code_and_pad(self, X, maxlen):
        """
        Transform string-based inputs in embeddings and pad them with zeros.
        """
        new_X = []
        for x in X:
            new_x = np.zeros((maxlen, self.embs_dim))
            for i, word in enumerate(x[0]):
                new_x[i] = self.embs[word]
            new_X.append(new_x)
        return np.array(new_X)

    def _triangulate(self, vector, indices, lenX, lenX2=None):
        """
        Transform the return vectors from the graph into their
        original matrix form.
        """
        if lenX2 == None:
            lenX2 = lenX
            gram = True
        else:
            gram = False
        if vector[0].shape == ():
            result = np.zeros((lenX, lenX2))
        else:
            result = np.zeros((lenX, lenX2, vector[0].shape[0]))
        for i, elem in enumerate(indices):
            result[elem[0], elem[1]] = vector[i]
            if elem[0] != elem[1] and gram:
                result[elem[1], elem[0]] = vector[i]
        return result
