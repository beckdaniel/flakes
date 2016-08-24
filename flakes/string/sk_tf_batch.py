import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops as cfops
from tensorflow.python.ops import tensor_array_ops as taops
from tensorflow.python.ops import functional_ops as fops
from sk_util import build_input_matrix
import sys
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
import json
import datetime
import copy

class TFBatchStringKernel(object):
    """
    A TensorFlow string kernel implementation.
    """
    def __init__(self, embs, device='/cpu:0', 
                 batch_size=1000, config=None, trace=None):    
        self.embs = embs
        self.embs_dim = embs[embs.keys()[0]].shape[0]
        self.graph = None
        self.maxlen = 0
        self.device = device
        self.gram_mode = False
        self.trace = trace
        self.tf_config = config
        self.BATCH_SIZE = batch_size

    def _build_graph(self, n, order, X, X2=None):
        """
        Builds the graph for TF calculation. This should
        be usually called only once but can be called again
        if we update the maximum string length in our
        dataset.
        """
        if X2 == None:
            X2 = np.array([x.T for x in X])
        else:
            X2 = np.array([x.T for x in X2])
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device(self.device):
            # Datasets are loaded as constants. Useful for GPUs.
            # Only feasible for small datasets though.
            tf_X = tf.constant(X, dtype=tf.float32, name='X')
            tf_X2 = tf.constant(X2, dtype=tf.float32, name='X2')

            # Kernel hyperparameters are also placeholders.
            # The K function is responsible for tying the
            # hyper values from class to this calculation
            # and to update the hyper gradients.
            self._gap = tf.placeholder("float", [], name='gap_decay')
            self._match = tf.placeholder("float", [], name='match_decay')
            self._coefs = tf.placeholder("float", [1, order], name='coefs')
            self._indices1 = tf.placeholder("int32", [self.BATCH_SIZE], name='indices1')
            self._indices2 = tf.placeholder("int32", [self.BATCH_SIZE], name='indices2')
            
            # Triangular matrices over decay powers.
            power = np.ones((n, n))
            tril = np.zeros((n, n))
            i1, i2 = np.indices(power.shape)
            for k in xrange(n-1):
                power[i2-k-1 == i1] = k
                tril[i2-k-1 == i1] = 1.0
            tf_tril = tf.constant(tril, dtype=tf.float32, name='tril')
            tf_power = tf.constant(power, dtype=tf.float32, name='power')
            gaps = tf.fill([n, n], self._gap, name='gaps')
            D = tf.pow(tf.mul(gaps, tf_tril), tf_power, name='D_matrix')
            dD_dgap = tf.pow((tf_tril * gaps), (tf_power - 1.0)) * tf_tril * tf_power

            match_sq = tf.pow(self._match, 2, name='match_sq')

            # Gather matlists and generate similarity matrices
            matlist1 = tf.gather(tf_X, self._indices1, name='matlist1')
            matlist2 = tf.gather(tf_X2, self._indices2, name='matlist2')
            S = tf.batch_matmul(matlist1, matlist2)
            
            # Kp calculation
            Kp = []
            dKp_dgap = []
            dKp_dmatch = []
            Kp.append(tf.ones(shape=(self.BATCH_SIZE, n, n)))
            dKp_dgap.append(tf.zeros(shape=(self.BATCH_SIZE, n, n)))
            dKp_dmatch.append(tf.zeros(shape=(self.BATCH_SIZE, n, n)))

            for i in xrange(order - 1):
                aux1 = tf.mul(S, Kp[i])
                aux2 = tf.reshape(aux1, tf.pack([self.BATCH_SIZE * n, n]))
                aux3 = tf.matmul(aux2, D)
                aux4 = aux3 * match_sq
                aux5 = tf.reshape(aux4, tf.pack([self.BATCH_SIZE, n, n]))
                aux6 = tf.transpose(aux5, perm=[0, 2, 1])
                aux7 = tf.reshape(aux6, tf.pack([self.BATCH_SIZE * n, n]))
                aux8 = tf.matmul(aux7, D)
                aux9 = tf.reshape(aux8, tf.pack([self.BATCH_SIZE, n, n]))
                aux10 = tf.transpose(aux9, perm=[0, 2, 1])
                Kp.append(aux10)

                daux1_dgap = tf.mul(S, dKp_dgap[i])
                daux2_dgap = tf.reshape(daux1_dgap, tf.pack([self.BATCH_SIZE * n, n]))
                daux3_dgap = tf.matmul(daux2_dgap, D) + tf.matmul(aux2, dD_dgap)
                daux4_dgap = daux3_dgap * match_sq
                daux5_dgap = tf.reshape(daux4_dgap, tf.pack([self.BATCH_SIZE, n, n]))
                daux6_dgap = tf.transpose(daux5_dgap, perm=[0, 2, 1])
                daux7_dgap = tf.reshape(daux6_dgap, tf.pack([self.BATCH_SIZE * n, n]))
                daux8_dgap = tf.matmul(daux7_dgap, D) + tf.matmul(aux7, dD_dgap)
                daux9_dgap = tf.reshape(daux8_dgap, tf.pack([self.BATCH_SIZE, n, n]))
                daux10_dgap = tf.transpose(daux9_dgap, perm=[0, 2, 1])
                dKp_dgap.append(daux10_dgap)

                daux1_dmatch = tf.mul(S, dKp_dmatch[i])
                daux2_dmatch = tf.reshape(daux1_dmatch, tf.pack([self.BATCH_SIZE * n, n]))
                daux3_dmatch = tf.matmul(daux2_dmatch, D)
                daux4_dmatch = (daux3_dmatch * match_sq) + (2 * self._match * aux3)
                daux5_dmatch = tf.reshape(daux4_dmatch, tf.pack([self.BATCH_SIZE, n, n]))
                daux6_dmatch = tf.transpose(daux5_dmatch, perm=[0, 2, 1])
                daux7_dmatch = tf.reshape(daux6_dmatch, tf.pack([self.BATCH_SIZE * n, n]))
                daux8_dmatch = tf.matmul(daux7_dmatch, D)
                daux9_dmatch = tf.reshape(daux8_dmatch, tf.pack([self.BATCH_SIZE, n, n]))
                daux10_dmatch = tf.transpose(daux9_dmatch, perm=[0, 2, 1])
                dKp_dmatch.append(daux10_dmatch)

            final_Kp = tf.pack(Kp)
            final_dKp_dgap = tf.pack(dKp_dgap)
            final_dKp_dmatch = tf.pack(dKp_dmatch)

            # Final calculation. We gather all Kps and
            # multiply then by their coeficients.
            mul1 = tf.mul(S, final_Kp)
            sum1 = tf.reduce_sum(mul1, 2)
            sum2 = tf.reduce_sum(sum1, 2, keep_dims=True)
            Ki = tf.mul(sum2, match_sq)
            Ki = tf.squeeze(Ki, squeeze_dims=[2])
            k_result = tf.squeeze(tf.matmul(self._coefs, Ki))

            dmul1_dgap = tf.mul(S, final_dKp_dgap)
            dsum1_dgap = tf.reduce_sum(dmul1_dgap, 2)
            dsum2_dgap = tf.reduce_sum(dsum1_dgap, 2, keep_dims=True)
            dKi_dgap = tf.mul(dsum2_dgap, match_sq)
            dKi_dgap = tf.squeeze(dKi_dgap, squeeze_dims=[2])
            dk_dgap = tf.squeeze(tf.matmul(self._coefs, dKi_dgap))

            dmul1_dmatch = tf.mul(S, final_dKp_dmatch)
            dsum1_dmatch = tf.reduce_sum(dmul1_dmatch, 2)
            dsum2_dmatch = tf.reduce_sum(dsum1_dmatch, 2, keep_dims=True)
            dKi_dmatch = tf.mul(dsum2_dmatch, match_sq) + (2 * self._match * sum2)
            dKi_dmatch = tf.squeeze(dKi_dmatch, squeeze_dims=[2])
            dk_dmatch = tf.squeeze(tf.matmul(self._coefs, dKi_dmatch))

            dk_dcoefs = Ki

            self.result = (k_result, dk_dgap, dk_dmatch, dk_dcoefs)


    def K(self, X, X2, gram, params, diag=False):
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
                self._build_graph(maxlen, order, X)
            indices = [[i1, i2] for i1 in range(len(X)) for i2 in range(len(X2)) if i1 >= i2]
        else: # We rebuild the graph, usually for predictions
            self.gram_mode = False
            if diag:
                maxlen = max([len(x[0]) for x in X])
                X = self._code_and_pad(X, maxlen)
                self.maxlen = maxlen
                self._build_graph(maxlen, order, X)
                indices = [[i1, i1] for i1 in range(len(X))]
            else:
                maxlen = max([len(x[0]) for x in np.concatenate((X, X2))])
                X = self._code_and_pad(X, maxlen)
                X2 = self._code_and_pad(X2, maxlen)
                self.maxlen = maxlen
                self._build_graph(maxlen, order, X, X2)
                indices = [[i1, i2] for i1 in range(len(X)) for i2 in range(len(X2))]

        # Initialize return values
        k_results = [] 
        gap_grads = []
        match_grads = []
        coef_grads = []

        # Add optional tracing for profiling
        if self.trace is not None:
            run_options = config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # We start a TF session and run it
        sess = tf.Session(graph=self.graph, config=self.tf_config)

        indices_copy = copy.deepcopy(indices)
        while indices != []:
            items = indices[:self.BATCH_SIZE]
            if len(items) < self.BATCH_SIZE:
                # padding
                items += [[0, 0]] * (self.BATCH_SIZE - len(items))
            items1 = [elem[0] for elem in items]
            items2 = [elem[1] for elem in items]
            feed_dict = {self._gap: params[0], 
                         self._match: params[1],
                         self._coefs: np.array(params[2])[None, :],
                         self._indices1: np.array(items1),
                         self._indices2: np.array(items2)
                     }
            before = datetime.datetime.now()
            result = sess.run(self.result, feed_dict=feed_dict,
                              options=run_options, 
                              run_metadata=run_metadata)
            after = datetime.datetime.now()
            k, gapg, matchg, coefsg = result
            if self.trace is not None:
                tl = timeline.Timeline(run_metadata.step_stats, graph=self.graph)
                trace = tl.generate_chrome_trace_format()
                with open(self.trace, 'w') as f:
                    f.write(trace)
            for i in xrange(self.BATCH_SIZE):
                k_results.append(k[i])
                gap_grads.append(gapg[i])
                match_grads.append(matchg[i])
                coef_grads.append(coefsg[:, i])
            indices = indices[self.BATCH_SIZE:]
        sess.close()
        
        # Reshape the return values since they are vectors:
        if not diag:
            if gram:
                lenX2 = None
            else:
                lenX2 = len(X2)
            k_results = self._triangulate(k_results, indices_copy, len(X), lenX2)
            gap_grads = self._triangulate(gap_grads, indices_copy, len(X), lenX2)
            match_grads = self._triangulate(match_grads, indices_copy, len(X), lenX2)
            coef_grads = self._triangulate(coef_grads, indices_copy, len(X), lenX2)
        else:
            k_results = np.array(k_results)
            gap_grads = np.array(gap_grads)
            match_grads = np.array(match_grads)
            coef_grads = np.array(coef_grads)
        return k_results, gap_grads, match_grads, coef_grads

    def _code_and_pad(self, X, maxlen):
        """
        Transform string-based inputs in embeddings and pad them with zeros.
        """
        new_X = []
        for x in X:
            new_x = np.zeros((maxlen, self.embs_dim))
            for i, word in enumerate(x[0]):
                try:
                    new_x[i] = self.embs[word]
                except KeyError:
                    pass # OOV, stick with zeros
            new_X.append(new_x)
        return np.array(new_X)

    def _triangulate(self, vector, indices, lenX, lenX2=None):
        """
        Transform the return vectors from the graph into their
        original matrix form.
        """
        vector = np.squeeze(np.array(vector))
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
