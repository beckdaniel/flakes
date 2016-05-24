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

class TFGramStringKernel(object):
    """
    A TensorFlow string kernel implementation.
    """
    def __init__(self, embs, device='/cpu:0', trace=None):    
        self.embs = embs
        self.embs_dim = embs[embs.keys()[0]].shape[0]
        self.graph = None
        self.maxlen = 0
        self.device = device
        self.gram_mode = False
        self.trace = trace
        if 'gpu' in device:
            self.tf_config = tf.ConfigProto(
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
                device_count = {'gpu': 1}
            )
        elif 'cpu' in device:
            self.tf_config = tf.ConfigProto(
                use_per_session_threads = 4,
                intra_op_parallelism_threads = 4,
                inter_op_parallelism_threads = 4,
            )
        self.BATCH_SIZE = 10

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
            tf_X = tf.constant(X, dtype=tf.float32, name='X')
            tf_X2 = tf.constant(X2, dtype=tf.float32, name='X2')

            # Kernel hyperparameters are also placeholders.
            # The K function is responsible for tying the
            # hyper values from class to this calculation
            # and to update the hyper gradients.
            self._gap = tf.placeholder("float", [], name='gap_decay')
            self._match = tf.placeholder("float", [], name='match_decay')
            self._coefs = tf.placeholder("float", [1, order], name='coefs')
            #self._index1 = tf.placeholder("int32", [], name='index1')
            #self._index2 = tf.placeholder("int32", [], name='index2')
            self._indices = tf.placeholder("float32", [self.BATCH_SIZE, 2], name='indices')
            match_sq = tf.pow(self._match, 2, name='match_sq')

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
            D = tf.pow(tf.mul(gaps, tril), power, name='D_matrix')

            ks = [] 
            gapgs = []
            matchgs = []
            coefsgs = []

            # Kernel calculation + gradients
            for i in xrange(self.BATCH_SIZE):
                _index = tf.gather(self._indices, i, name='index_%d' % i)
                _index1 = tf.to_int32(tf.gather(_index, 0, name='index1_%d' % i))
                _index2 = tf.to_int32(tf.gather(_index, 1, name='index2_%d' % i))
                k, gapg, matchg, coefsg = self._build_k(_index1, _index2,
                                                        tf_X, tf_X2, D, match_sq,
                                                        self._gap, self._match,
                                                        self._coefs, n, order)
                #print k
                #print gapg
                #print matchg
                #print coefsg
                #gapg = tf.Print(gapg, gapg)
                ks.append(k)
                gapgs.append(gapg)
                matchgs.append(matchg)
                coefsgs.append(coefsg)
            
            #print ks
            #print gapgs
            #print coefsgs
            k_result = tf.pack(ks)
            gap_result = tf.pack(gapgs)
            match_result = tf.pack(matchgs)
            coefs_result = tf.pack(coefsgs)
            #all_stuff = [k_result] + gap_result + match_result + coefs_result
            #all_stuff = [k] + gapg + matchg + coefsg
            all_stuff = (k_result, gap_result, match_result, coefs_result)
            self.result = all_stuff

    def _build_k(self, index1, index2, tf_X, tf_X2, D, match_sq,
                 gap, match, coefs, n, order):
        # Select the inputs from the pre-loaded
        # dataset
        mat1 = tf.gather(tf_X, index1)#, name='mat1')
        mat2 = tf.gather(tf_X2, index2)#, name='mat2')
        S = tf.matmul(mat1, tf.transpose(mat2))#, name='S_matrix')

        # Kp calculation
        Kp = []
        Kp.append(tf.ones(shape=(n, n)))
        for i in xrange(order):
            aux1 = tf.mul(S, Kp[i])#, name="aux1_%d" % i)
            aux2 = tf.transpose(tf.matmul(aux1, D) * match_sq)#, name="aux2_%d" % i)
            aux3 = tf.transpose(tf.matmul(aux2, D))#, name="aux3_%d" % i)
            Kp.append(aux3)
        final_Kp = tf.pack(Kp)

        # Final calculation. We gather all Kps and
        # multiply then by their coeficients.
        mul1 = tf.mul(S, final_Kp[:order, :, :])#, name='mul1')
        sum1 = tf.reduce_sum(mul1, 1)#, name='sum1')
        Ki = tf.mul(tf.reduce_sum(sum1, 1, keep_dims=True), match_sq)#, name='Ki')
        k_result = tf.matmul(coefs, Ki)#, name='k_result')
        gap_grad = tf.gradients(k_result, gap)[0]#, name='gradients_gap_grad')[0]
        match_grad = tf.gradients(k_result, match)[0]#, name='gradients_match_grad')[0]
        coefs_grad = tf.gradients(k_result, coefs)[0]#, name='gradients_coefs_grad')[0]
        return k_result, gap_grad, match_grad, coefs_grad

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
                self._build_graph(maxlen, order, X)
            indices = [[i1, i2] for i1 in range(len(X)) for i2 in range(len(X2)) if i1 >= i2]
        else: # We rebuild the graph, usually for predictions
            self.gram_mode = False
            maxlen = max([len(x[0]) for x in np.concatenate((X, X2))])
            X = self._code_and_pad(X, maxlen)
            X2 = self._code_and_pad(X2, maxlen)
            self.maxlen = maxlen
            self._build_graph(maxlen, order, X, X2)
            indices = [[i1, i2] for i1 in range(len(X)) for i2 in range(len(X2))]

        # Initialize return values
        #k_results = np.zeros(shape=(len(X), len(X2)))
        #gap_grads = np.zeros(shape=(len(X), len(X2)))
        #match_grads = np.zeros(shape=(len(X), len(X2)))
        #coef_grads = np.zeros(shape=(len(X), len(X2), order))
        #k_results = np.zeros(len(indices))
        #gap_grads = np.zeros(len(indices))
        #match_grads = np.zeros(len(indices))
        #coef_grads = np.zeros((len(indices), order))
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

        #########################
        # for i in xrange(len(X)):
        #     for j in xrange(len(X2)):
        #         if gram and (j < i):
        #             k_results[i, j] = k_results[j, i]
        #             gap_grads[i, j] = gap_grads[j, i]
        #             match_grads[i, j] = match_grads[j, i]
        #             coef_grads[i, j] = coef_grads[j, i]
        #         else:
        #             feed_dict = {self._gap: params[0], 
        #                          self._match: params[1],
        #                          self._coefs: np.array(params[2])[None, :],
        #                          self._index1: i,
        #                          self._index2: j}
        #             import datetime
        #             before = datetime.datetime.now()
        #             k_result, gap_grad, match_grad, coef_grad = sess.run(self.result,
        #                                                                  feed_dict=feed_dict,
        #                                                                  options=run_options, 
        #                                                                  run_metadata=run_metadata)
        #             after = datetime.datetime.now()
        #             print 'SESSION RUN: ',
        #             print after - before
        #             if self.trace is not None:
        #                 tl = timeline.Timeline(run_metadata.step_stats, graph=self.graph)
        #                 trace = tl.generate_chrome_trace_format()
        #                 with open(self.trace, 'w') as f:
        #                     f.write(trace)
        #             k_results[i, j] = k_result
        #             gap_grads[i, j] = gap_grad
        #             match_grads[i, j] = match_grad
        #             coef_grads[i, j] = coef_grad
        ###########################

        indices_copy = copy.deepcopy(indices)
        while indices != []:
            items = indices[:self.BATCH_SIZE]
            if len(items) < self.BATCH_SIZE:
                # padding
                items += [[0, 0]] * (self.BATCH_SIZE - len(items))
            feed_dict = {self._gap: params[0], 
                         self._match: params[1],
                         self._coefs: np.array(params[2])[None, :],
                         self._indices: np.array(items)}
            before = datetime.datetime.now()
            k, gapg, matchg, coefsg = sess.run(self.result, feed_dict=feed_dict,
                                               options=run_options, 
                                               run_metadata=run_metadata)
            after = datetime.datetime.now()
            print 'SESSION RUN: ',
            print after - before
            if self.trace is not None:
                tl = timeline.Timeline(run_metadata.step_stats, graph=self.graph)
                trace = tl.generate_chrome_trace_format()
                with open(self.trace, 'w') as f:
                    f.write(trace)
            for i in xrange(self.BATCH_SIZE):
                k_results.append(k[i])
                gap_grads.append(gapg[i])
                match_grads.append(matchg[i])
                coef_grads.append(coefsg[i])
            indices = indices[self.BATCH_SIZE:]
        sess.close()
        ############################
        
        # Reshape the return values since they are vectors:
        if gram:
            lenX2 = None
        else:
            lenX2 = len(X2)
        #print k_results
        #print gap_grads
        k_results = self._triangulate(k_results, indices_copy, len(X), lenX2)
        gap_grads = self._triangulate(gap_grads, indices_copy, len(X), lenX2)
        match_grads = self._triangulate(match_grads, indices_copy, len(X), lenX2)
        coef_grads = self._triangulate(coef_grads, indices_copy, len(X), lenX2)
    
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
