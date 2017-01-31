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
    def __init__(self, embs, sim='dot', wrapper='none',
                 index=None, device='/cpu:0', 
                 batch_size=1000, normalise=False,
                 config=None, trace=None):    
        self.embs = embs
        self.embs_dim = embs.shape[1]
        self.wrapper = wrapper
        self.index = index
        if sim == 'arccosine':
            self.sim = self._arccosine
            self.norms = np.sqrt(np.sum(pow(embs, 2), 1, keepdims=True))
        elif sim == 'pos_dot':
            self.sim = self._pos_dot
        elif sim == 'dot':
            self.sim = self._dot
        self.graph = None
        self.maxlen = 0
        self.device = device
        self.gram_mode = False
        self.trace = trace
        self.tf_config = config
        self.BATCH_SIZE = batch_size
        self.normalise = normalise
        self.sess = None

    def _build_graph(self, n, order):
        """
        Builds the graph for TF calculation. This should
        be usually called only once but can be called again
        if we update the maximum string length in our
        dataset.
        """
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device(self.device):

            # Embeddings are loaded as constants
            tf_embs = tf.constant(self.embs, dtype=tf.float64, name='embs')

            # Kernel hyperparameters are also placeholders.
            # The K function is responsible for tying the
            # hyper values from class to this calculation
            # and to update the hyper gradients.
            self._gap = tf.placeholder("float64", [], name='gap_decay')
            self._match = tf.placeholder("float64", [], name='match_decay')
            self._ls = tf.placeholder("float64", [], name='lengthscale')
            self._coefs = tf.placeholder("float64", [1, order], name='coefs')
            self._slist1 = tf.placeholder("int32", [self.BATCH_SIZE, n], name='slist1')
            self._slist2 = tf.placeholder("int32", [self.BATCH_SIZE, n], name='slist2')
            
            # Decay constants are initialised here
            D, dD_dgap, match_sq = self._init_constants(n)

            # Similarity matrix calculation. This is the only
            # place where the embeddings are used.
            S, dS_dls = self.sim(self._slist1, self._slist2, tf_embs, self._ls)

            # Kp and gradient matrices initialisation
            Kp, dKp_dgap, dKp_dmatch, dKp_dls = self._init_Kp(n)

            # Main kernel calculation happens here.
            # "i" correspond to the ngram order.
            for i in xrange(order - 1):
                aux2, aux3, aux7, aux10 = self._calc_Kp_i(Kp[i], S, D, n, match_sq)
                Kp.append(aux10)
                dKp_dgap.append(self._calc_dKp_dgap_i(dKp_dgap[i], S, D, n, match_sq, 
                                                      dD_dgap, aux2, aux7))
                dKp_dmatch.append(self._calc_dKp_dmatch_i(dKp_dmatch[i], S, D, n, 
                                                          match_sq, aux3))
                dKp_dls.append(self._calc_dKp_dls_i(dKp_dls[i], S, D, n, 
                                                    match_sq, Kp[i], dS_dls))

            # Final calculation. We gather all Kps and
            # multiply then by their coeficients.
            # Also do the same with gradients.
            final_Kp = tf.pack(Kp)
            final_dKp_dgap = tf.pack(dKp_dgap)
            final_dKp_dmatch = tf.pack(dKp_dmatch)
            final_dKp_dls = tf.pack(dKp_dls)
            k_result, Ki, sum2 = self._final_calc(final_Kp, S, match_sq)
            dk_dgap, _, _ = self._final_calc(final_dKp_dgap, S, match_sq)
            dk_dmatch, _, _  = self._final_calc(final_dKp_dmatch, S, match_sq,
                                                prev_sum2=sum2)
            dk_dls, _, _  = self._final_calc(final_dKp_dls, S, match_sq,
                                             Kp=final_Kp, dS_dls=dS_dls)

            #dk_dls = tf.zeros_like(dk_dgap)
            
            # Coef gradients are simply the individual
            # ngram kernel values
            dk_dcoefs = Ki

            self.result = (k_result, dk_dgap, dk_dmatch, dk_dls, dk_dcoefs)

    def _init_constants(self, n):
        """
        Initialise gap decay matrix and match decay constants.
        Also initialise the gay decay matrix gradients.
        """
        power = np.ones((n, n))
        tril = np.zeros((n, n))
        i1, i2 = np.indices(power.shape)
        for k in xrange(n-1):
            power[i2-k-1 == i1] = k
            tril[i2-k-1 == i1] = 1.0
        tf_tril = tf.constant(tril, dtype=tf.float64, name='tril')
        tf_power = tf.constant(power, dtype=tf.float64, name='power')
        gaps = tf.fill([n, n], self._gap, name='gaps')
        D = tf.pow(tf.mul(gaps, tf_tril), tf_power, name='D_matrix')
        dD_dgap = tf.pow((tf_tril * gaps), (tf_power - 1.0)) * tf_tril * tf_power
        match_sq = tf.pow(self._match, 2, name='match_sq')
        return D, dD_dgap, match_sq

    def _dot(self, slist1, slist2, tf_embs, ls):
        """
        Simple dot product between two vectors of embeddings.
        This returns a matrix of positive real numbers.
        """
        matlist1 = tf.gather(tf_embs, slist1, name='matlist1')
        matlist2 = tf.matrix_transpose(tf.gather(tf_embs, slist2, name='matlist2'))
        dot = tf.batch_matmul(matlist1, matlist2)
        return dot, tf.zeros_like(dot)

    def _pos_dot(self, slist1, slist2, tf_embs, ls):
        """
        Dot product with an additional SE kernel on position.
        Position is obtained by gathering the indexes of each string.
        """
        dot, _ = self._dot(slist1, slist2, tf_embs, ls)

        # All vectors have the same length, just padded.
        # We compute only a single positional matrix here,
        # final results rely on broadcast through the shape
        # of 'dot'.
        pos = tf.range(tf.shape(slist1)[1])
        r2 = tf.to_double((pos[:, None] - pos[None, :]) ** 2)
        pos_match = tf.exp(tf.truediv(-r2, ls))
        result = tf.mul(dot, pos_match)
        dpos_dls_term = tf.truediv(r2, tf.pow(ls, 2.0))
        dpos_dls = tf.mul(dot, tf.mul(pos_match, dpos_dls_term))
        return result, dpos_dls

    def _arccosine(self, slist1, slist2, tf_embs, ls):
        """
        Uses an arccosine kernel of degree 0 to calculate
        the similarity matrix between two vectors of embeddings. 
        This is just cosine similarity projected into the [0,1] interval.
        """
        dot = self._dot(slist1, slist2, tf_embs)
        # This calculation corresponds to an arc-cosine with 
        # degree 0. It can be interpreted as cosine
        # similarity but projected into a [0,1] interval.
        # TODO: arc-cosine with degree 1.
        tf_pi = tf.constant(np.pi, dtype=tf.float64)
        tf_norms = tf.constant(self.norms, dtype=tf.float64, name='norms')
        normlist1 = tf.gather(tf_norms, slist1, name='normlist1')
        normlist2 = tf.matrix_transpose(tf.gather(tf_norms, slist2, name='normlist2'))
        norms = tf.batch_matmul(normlist1, normlist2)
        cosine = tf.clip_by_value(tf.truediv(dot, norms), -1, 1)
        angle = tf.acos(cosine)
        angle = tf.select(tf.is_nan(angle), tf.ones_like(angle) * tf_pi, angle)
        return 1 - (angle / tf_pi), tf.zeros_like(angle)

    def _init_Kp(self, n):
        """
        Initialise Kp and gradients. Check paper/notes for
        details on the nomenclature.
        """
        Kp = []
        dKp_dgap = []
        dKp_dmatch = []
        dKp_dls = []
        Kp.append(tf.ones(shape=(self.BATCH_SIZE, n, n), dtype=tf.float64))
        dKp_dgap.append(tf.zeros(shape=(self.BATCH_SIZE, n, n), dtype=tf.float64))
        dKp_dmatch.append(tf.zeros(shape=(self.BATCH_SIZE, n, n), dtype=tf.float64))
        dKp_dls.append(tf.zeros(shape=(self.BATCH_SIZE, n, n), dtype=tf.float64))
        return Kp, dKp_dgap, dKp_dmatch, dKp_dls
                          
    def _calc_Kp_i(self, Kp_i, S, D, n, match_sq):
        """
        Calculate an element from Kp. We use auxiliary
        variables for the sake of clarity.
        """
        aux1 = tf.mul(S, Kp_i)
        aux2 = tf.reshape(aux1, tf.pack([self.BATCH_SIZE * n, n]))
        aux3 = tf.matmul(aux2, D)
        aux4 = aux3 * match_sq
        aux5 = tf.reshape(aux4, tf.pack([self.BATCH_SIZE, n, n]))
        aux6 = tf.transpose(aux5, perm=[0, 2, 1])
        aux7 = tf.reshape(aux6, tf.pack([self.BATCH_SIZE * n, n]))
        aux8 = tf.matmul(aux7, D)
        aux9 = tf.reshape(aux8, tf.pack([self.BATCH_SIZE, n, n]))
        aux10 = tf.transpose(aux9, perm=[0, 2, 1])
        return aux2, aux3, aux7, aux10

    def _calc_dKp_dgap_i(self, dKp_dgap_i, S, D, n, match_sq,
                         dD_dgap, aux2, aux7):
        """
        Calculate the gradient of Kp with respect
        to the gap decay.
        """
        daux1_dgap = tf.mul(S, dKp_dgap_i)
        daux2_dgap = tf.reshape(daux1_dgap, tf.pack([self.BATCH_SIZE * n, n]))
        daux3_dgap = tf.matmul(daux2_dgap, D) + tf.matmul(aux2, dD_dgap)
        daux4_dgap = daux3_dgap * match_sq
        daux5_dgap = tf.reshape(daux4_dgap, tf.pack([self.BATCH_SIZE, n, n]))
        daux6_dgap = tf.transpose(daux5_dgap, perm=[0, 2, 1])
        daux7_dgap = tf.reshape(daux6_dgap, tf.pack([self.BATCH_SIZE * n, n]))
        daux8_dgap = tf.matmul(daux7_dgap, D) + tf.matmul(aux7, dD_dgap)
        daux9_dgap = tf.reshape(daux8_dgap, tf.pack([self.BATCH_SIZE, n, n]))
        daux10_dgap = tf.transpose(daux9_dgap, perm=[0, 2, 1])
        return daux10_dgap

    def _calc_dKp_dmatch_i(self, dKp_dmatch_i, S, D, n, match_sq, aux3):
        """
        Calculate the gradient of Kp with respect
        to the match decay.
        """
        daux1_dmatch = tf.mul(S, dKp_dmatch_i)
        daux2_dmatch = tf.reshape(daux1_dmatch, tf.pack([self.BATCH_SIZE * n, n]))
        daux3_dmatch = tf.matmul(daux2_dmatch, D)
        daux4_dmatch = (daux3_dmatch * match_sq) + (2 * self._match * aux3)
        daux5_dmatch = tf.reshape(daux4_dmatch, tf.pack([self.BATCH_SIZE, n, n]))
        daux6_dmatch = tf.transpose(daux5_dmatch, perm=[0, 2, 1])
        daux7_dmatch = tf.reshape(daux6_dmatch, tf.pack([self.BATCH_SIZE * n, n]))
        daux8_dmatch = tf.matmul(daux7_dmatch, D)
        daux9_dmatch = tf.reshape(daux8_dmatch, tf.pack([self.BATCH_SIZE, n, n]))
        daux10_dmatch = tf.transpose(daux9_dmatch, perm=[0, 2, 1])
        return daux10_dmatch

    def _calc_dKp_dls_i(self, dKp_dls_i, S, D, n, match_sq, Kp_i, dS_dls):
        """
        Calculate the gradient of Kp with respect
        to the lengthscale.
        """
        daux1_dls = tf.mul(dS_dls, Kp_i) + tf.mul(S, dKp_dls_i)
        daux2_dls = tf.reshape(daux1_dls, tf.pack([self.BATCH_SIZE * n, n]))
        daux3_dls = tf.matmul(daux2_dls, D)
        daux4_dls = daux3_dls * match_sq
        daux5_dls = tf.reshape(daux4_dls, tf.pack([self.BATCH_SIZE, n, n]))
        daux6_dls = tf.transpose(daux5_dls, perm=[0, 2, 1])
        daux7_dls = tf.reshape(daux6_dls, tf.pack([self.BATCH_SIZE * n, n]))
        daux8_dls = tf.matmul(daux7_dls, D)
        daux9_dls = tf.reshape(daux8_dls, tf.pack([self.BATCH_SIZE, n, n]))
        daux10_dls = tf.transpose(daux9_dls, perm=[0, 2, 1])
        return daux10_dls
        
    def _final_calc(self, final_tensor, S, match_sq, prev_sum2=None, Kp=None,
                    dS_dls=None):
        """
        Common method to wrap up the kernel calculations.
        This is also used for gradients since they are 
        mostly similar (except for match).
        """
        if Kp is not None:
            mul1 = tf.mul(dS_dls, Kp) + tf.mul(S, final_tensor)
        else:
            mul1 = tf.mul(S, final_tensor)
        sum1 = tf.reduce_sum(mul1, 2)
        sum2 = tf.reduce_sum(sum1, 2, keep_dims=True)
        mul2 = tf.mul(sum2, match_sq)
        if prev_sum2 is not None:
            mul2 = mul2 + (2 * self._match * prev_sum2)
        squeezed = tf.squeeze(mul2, squeeze_dims=[2])
        result = tf.squeeze(tf.matmul(self._coefs, squeezed))
        return result, squeezed, sum2
                  
    def K(self, X, X2, gram, params, diag=False):
        """
        Actual kernel calculation, for now this is
        doing arc-cosine wrapping stuff.
        """
        k_result, gap_grads, match_grads, ls_grads, coef_grads = self._K_unnorm(X, X2, gram, params, diag)
        if self.wrapper == 'none':
            return k_result, gap_grads, match_grads, ls_grads, coef_grads

        # Else: we assume there is a variance parameter
        # and proceed calculation.
        self.variance = params[4]
        # WARNING: This is done outside Tensorflow, we might
        # want to move this inside the GPU in the future
        if self.gram_mode:
            # In gram mode we have all information in the
            # diagonals of the result matrices
            ktt = np.outer(np.diag(k_result) + self.variance,
                           np.diag(k_result) + self.variance)
            sqrt_ktt = np.sqrt(ktt)
            #print params
            #if np.isnan(np.sum(ktt)):
            #    raise ValueError

            norm_k_result = k_result / sqrt_ktt
            gap_grads = self._normalise_grads(gap_grads, k_result, 
                                              norm_k_result, ktt, sqrt_ktt)
            match_grads = self._normalise_grads(match_grads, k_result,
                                                norm_k_result, ktt, sqrt_ktt)
            ls_grads = self._normalise_grads(ls_grads, k_result,
                                             norm_k_result, ktt, sqrt_ktt)
            order = len(params[3])
            for i in range(order):
                coef_grads[:, :, i] = self._normalise_grads(coef_grads[:, :, i], 
                                                            k_result,
                                                            norm_k_result,
                                                            ktt, sqrt_ktt)
            var_vec = np.tile(np.diag(k_result) + self.variance,
                              (k_result.shape[0], 1))
            var_num = var_vec + var_vec.T
            var_denom = 2 * ktt
            var_grads = - norm_k_result * (var_num / var_denom)
            k_result = norm_k_result

            #######################
            # ARCCOS
            if self.wrapper == 'arccos0':
                acos_result = 1 - ((np.arccos(k_result)) / np.pi)
                #acos_grad_term = - (1 / np.sqrt(1 - k_result + 1e-60))
                acos_grad_term = (1 / (np.pi * np.sqrt(1 - (k_result ** 2))))
                k_result = acos_result
                gap_grads *= acos_grad_term
                match_grads *= acos_grad_term
                ls_grads *= acos_grad_term
                for i in range(order):
                    coef_grads[:, :, i] *= acos_grad_term
                var_grads *= acos_grad_term
            #######################
        else:
            # We need to calculate the diagonals explicitly
            # but we do not need to update gradients
            # (risky...)
            #if self.wrapper == 'norm':
            #    return np.ones(X.shape[0]), gap_grads, match_grads, coef_grads, 0
            all_X = np.concatenate((X, X2))
            diag_k_result, _, _, _, _ = self._K_unnorm(all_X, None, gram=False, params=params, diag=True)
            diag_X = diag_k_result[:X.shape[0]] + self.variance
            diag_X2 = diag_k_result[X.shape[0]:] + self.variance
            ktt = np.outer(diag_X, diag_X2)
            k_result = k_result / np.sqrt(ktt)
            ####################
            # ARCCOS
            if self.wrapper == 'arccos0':
                acos_result = 1 - ((np.arccos(k_result)) / np.pi)
                k_result = acos_result
            ####################
            var_grads = None
        return k_result, gap_grads, match_grads, ls_grads, coef_grads, var_grads

    def _normalise_grads(self, grad, k_result, norm_k_result, ktt, sqrt_ktt):
        first = grad / sqrt_ktt
        sec_num = (np.outer(np.diag(grad), np.diag(k_result) + self.variance) +
                   np.outer(np.diag(k_result) + self.variance, np.diag(grad)))
        second = norm_k_result * (sec_num / (2 * ktt))
        return first - second

    #@profile
    def _K_unnorm(self, X, X2, gram, params, diag=False):
        """
        Calculate the unnormalized kernel value.
        """
        assert not (gram and diag), "gram and diag cannot be true at the same time."

        # We need a better way to name this...
        # params[2] should be always order_coefs
        order = len(params[3])

        # Check input length, unlesse we are in Gram matrix mode.
        if gram:
            if not self.gram_mode:
                maxlen = max([len(x[0]) for x in X])
                self.gram_mode = True
            else:
                maxlen = self.maxlen
            indices = [[i1, i2] for i1 in range(len(X)) for i2 in range(len(X)) if i1 >= i2]
        else:
            self.gram_mode = False
            if diag:
                maxlen = max([len(x[0]) for x in X])
                indices = [[i1, i1] for i1 in range(len(X))]
            else:
                maxlen = max([len(x[0]) for x in list(X) + list(X2)])
                indices = [[i1, i2] for i1 in range(len(X)) for i2 in range(len(X2))]

        # We have to rebuild the graph if we have new inputs
        # which are larger than the graph capacity.
        # This is usually only called at prediction time.
        if maxlen > self.maxlen:
            self.maxlen = maxlen
            self._build_graph(maxlen, order)
            if self.sess is not None:
                self.sess.close()
                self.sess = None

        # We also have to pad inputs so each instance
        # has maximum length.
        X = [self._pad(x[0], self.maxlen) for x in X]
        if gram or diag:
            X2 = X
        else:
            X2 = [self._pad(x[0], self.maxlen) for x in X2]

        # Initialize return values
        k_results = np.zeros((len(X), len(X2)))
        gap_grads = np.zeros((len(X), len(X2)))
        match_grads = np.zeros((len(X), len(X2)))
        ls_grads = np.zeros((len(X), len(X2)))
        coef_grads = np.zeros((len(X), len(X2), order))

        # Add optional tracing for profiling
        if self.trace is not None:
            run_options = config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # We start a TF session and run it
        if self.sess is None:
            self.sess = tf.Session(graph=self.graph, config=self.tf_config)

        batches = (len(indices) / self.BATCH_SIZE) + 1

        for i in xrange(batches):
            items = indices[i * self.BATCH_SIZE: (i+1) * self.BATCH_SIZE]
            slist1 = [X[i[0]] for i in items]
            slist2 = [X2[i[1]] for i in items]
            if len(items) < self.BATCH_SIZE:
                # Padding
                slist1 = np.array(slist1 + [[0] * self.maxlen] * (self.BATCH_SIZE - len(items)))
                slist2 = np.array(slist2 + [[0] * self.maxlen] * (self.BATCH_SIZE - len(items)))
            else:
                slist1 = np.array(slist1)
                slist2 = np.array(slist2)
            feed_dict = {self._gap: params[0], 
                         self._match: params[1],
                         self._ls: params[2],
                         self._coefs: np.array(params[3])[None, :],
                         self._slist1: slist1,
                         self._slist2: slist2
                     }
            before = datetime.datetime.now()
            result = self.sess.run(self.result, feed_dict=feed_dict,
                                   options=run_options, 
                                   run_metadata=run_metadata)
            after = datetime.datetime.now()
            k, gapg, matchg, lsg, coefsg = result
            if self.trace is not None:
                tl = timeline.Timeline(run_metadata.step_stats, graph=self.graph)
                trace = tl.generate_chrome_trace_format()
                with open(self.trace, 'w') as f:
                    f.write(trace)

            # Populate the return matrices:
            for j, item in enumerate(items):
                k_results[item[0], item[1]] = k[j]
                gap_grads[item[0], item[1]] = gapg[j]
                match_grads[item[0], item[1]] = matchg[j]
                ls_grads[item[0], item[1]] = lsg[j]
                coef_grads[item[0], item[1]] = coefsg[:, j]

        # Postprocess the matrices and return results.
        if gram:
            # Symmetrize results
            new_coef_grads = np.zeros((len(X), len(X), order))
            for i in xrange(order):
                new_coef_grads[:, :, i] = self._symmetrize(coef_grads[:, :, i])
            return (self._symmetrize(k_results), self._symmetrize(gap_grads),
                    self._symmetrize(match_grads), self._symmetrize(ls_grads),
                    new_coef_grads)
        elif diag:
            # Get only the diagonals (the other elements are zeroes)
            new_coef_grads = np.zeros((len(X), order))
            for i in xrange(order):
                new_coef_grads[:, i] = np.diag(coef_grads[:, :, i])
            return (np.diag(k_results), np.diag(gap_grads),
                    np.diag(match_grads), np.diag(ls_grads),
                    new_coef_grads)
        else:
            # Return full results
            return k_results, gap_grads, match_grads, ls_grads, coef_grads

    def _symmetrize(self, matrix):
        """
        Symmetrize a lower triangular matrix with
        non-zero diagonal. Called when the kernel
        is in Gram matrix mode.
        """
        return matrix + matrix.T - np.diag(np.diag(matrix))

    def _pad(self, s, length):
        """
        Ensure a string has maximum length by adding
        zeroes. Notice we assume zero as the index
        for a zero embedding so the kernel result
        is not affected by padding.
        """
        new_s = np.zeros(length)
        new_s[:len(s)] = s
        return new_s

