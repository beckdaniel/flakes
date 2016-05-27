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
                #device_count = {'gpu': 1}
                #intra_op_parallelism_threads = 1,
                #inter_op_parallelism_threads = 1,
            )
        elif 'cpu' in device:
            #self.tf_config = None
            self.tf_config = tf.ConfigProto(
            #    use_per_session_threads = 4,
                intra_op_parallelism_threads = 1,
                inter_op_parallelism_threads = 2,
            )
        self.BATCH_SIZE = 20

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
            self._indices = tf.placeholder("float32", [self.BATCH_SIZE, 2], name='indices')


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
            #dD_dgap = tf.mul(tf.mul(tf_power, tf_tril), gaps, name='dD_dgap')

            ####
            sum_dD_dgap = tf.reduce_sum(dD_dgap)
            dD_dgap = tf.Print(dD_dgap, [sum_dD_dgap], summarize=100)
            
            true_dD_dgap = tf.gradients(D, self._gap)
            dD_dgap = tf.Print(dD_dgap, true_dD_dgap, summarize=100)
            ####

            match_sq = tf.pow(self._match, 2, name='match_sq')

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
                                                        self._coefs, n, order,
                                                        dD_dgap)
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
            #k_result = tf.pack(ks)
            #gap_result = tf.pack(gapgs)
            #match_result = tf.pack(matchgs)
            #coefs_result = tf.pack(coefsgs)
            #all_stuff = [k_result] + gap_result + match_result + coefs_result
            #all_stuff = [k] + gapg + matchg + coefsg
            #all_stuff = (k_result, gap_result, match_result, coefs_result)
            all_stuff = ks + gapgs + matchgs + coefsgs
            self.result = all_stuff

    def _build_k(self, index1, index2, tf_X, tf_X2, D, match_sq,
                 gap, match, coefs, n, order, dD_dgap):
        # Select the inputs from the pre-loaded
        # dataset
        mat1 = tf.gather(tf_X, index1)#, name='mat1')
        mat2 = tf.gather(tf_X2, index2)#, name='mat2')
        S = tf.matmul(mat1, tf.transpose(mat2))#, name='S_matrix')

        # Kp calculation
        Kp = []
        dKp_dgap = []
        dKp_dmatch = []
        Kp.append(tf.ones(shape=(n, n)))
        dKp_dgap.append(tf.zeros(shape=(n, n)))
        dKp_dmatch.append(tf.zeros(shape=(n, n)))

        for i in xrange(order - 1):
            aux1 = tf.mul(S, Kp[i])
            aux2 = tf.matmul(tf.transpose(D), aux1)
            aux3 = aux2 * match_sq
            aux4 = tf.matmul(aux3, D)
            Kp.append(aux4)
            
            daux1_dgap = tf.mul(S, dKp_dgap[i])
            daux2_dgap = (tf.matmul(tf.transpose(dD_dgap), aux1) +
                          tf.matmul(tf.transpose(D), daux1_dgap))
            daux3_dgap = daux2_dgap * match_sq
            daux4_dgap = (tf.matmul(daux3_dgap, D) +
                          tf.matmul(aux3, dD_dgap))
            dKp_dgap.append(daux4_dgap)


            daux1_dmatch = tf.mul(S, dKp_dmatch[i])
            daux2_dmatch = tf.matmul(tf.transpose(D), daux1_dmatch)
            daux3_dmatch = (daux2_dmatch * match_sq) + (2 * match * aux2)
            daux4_dmatch = tf.matmul(daux3_dmatch, D)
            dKp_dmatch.append(daux4_dmatch)

        final_Kp = tf.pack(Kp)
        final_dKp_dgap = tf.pack(dKp_dgap)
        final_dKp_dmatch = tf.pack(dKp_dmatch)

        # Final calculation. We gather all Kps and
        # multiply then by their coeficients.
        mul1 = tf.mul(S, final_Kp)
        sum1 = tf.reduce_sum(mul1, 1)
        sum2 = tf.reduce_sum(sum1, 1, keep_dims=True)
        Ki = tf.mul(sum2, match_sq)
        k_result = tf.matmul(coefs, Ki)
        
        dmul1_dgap = tf.mul(S, final_dKp_dgap)
        dsum1_dgap = tf.reduce_sum(dmul1_dgap, 1)
        dsum2_dgap = tf.reduce_sum(dsum1_dgap, 1, keep_dims=True)
        dKi_dgap = tf.mul(dsum2_dgap, match_sq)
        dk_dgap = tf.matmul(coefs, dKi_dgap)

        dmul1_dmatch = tf.mul(S, final_dKp_dmatch)
        dsum1_dmatch = tf.reduce_sum(dmul1_dmatch, 1)
        dsum2_dmatch = ((tf.reduce_sum(dsum1_dmatch, 1, keep_dims=True) * match_sq) +
                        (2 * match * sum2))
        dk_dmatch = tf.matmul(coefs, dsum2_dmatch)

        dk_dcoefs = Ki

        return k_result, dk_dgap, dk_dmatch, dk_dcoefs

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
                self.sess = tf.Session(graph=self.graph, config=self.tf_config)
            indices = [[i1, i2] for i1 in range(len(X)) for i2 in range(len(X2)) if i1 >= i2]
        else: # We rebuild the graph, usually for predictions
            try:
                self.sess.close()
            except AttributeError:
                # we do not have a session yet: not a problem in theory...
                pass
            self.gram_mode = False
            maxlen = max([len(x[0]) for x in np.concatenate((X, X2))])
            X = self._code_and_pad(X, maxlen)
            X2 = self._code_and_pad(X2, maxlen)
            self.maxlen = maxlen
            self._build_graph(maxlen, order, X, X2)
            self.sess = tf.Session(graph=self.graph, config=self.tf_config)
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
        #sess = tf.Session(graph=self.graph, config=self.tf_config)

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
        first = True
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
            #k, gapg, matchg, coefsg = sess.run(self.result, feed_dict=feed_dict,
            #                                   options=run_options, 
            #                                   run_metadata=run_metadata)
            result = self.sess.run(self.result, feed_dict=feed_dict,
                              options=run_options, 
                              run_metadata=run_metadata)
            after = datetime.datetime.now()
            k = result[:self.BATCH_SIZE]
            gapg = result[self.BATCH_SIZE:(self.BATCH_SIZE * 2)]
            matchg = result[(self.BATCH_SIZE * 2):(self.BATCH_SIZE * 3)]
            coefsg = result[(self.BATCH_SIZE * 3):]

            if first:
                first = False
                print 'FIRST SESSION RUN: ',
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

        #sess.close()
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
