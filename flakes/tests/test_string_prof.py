import flakes
import unittest
import numpy as np
import GPy
import datetime
import sys


#@unittest.skip('profiling')
class StringKernelProfiling(unittest.TestCase):
    DEVICE = '/cpu:0'
    TRACE_FILE = None

    def setUp(self):
        self.s1 = "Scientists throughout the world and all the friends of the late Baron Sir Ferdinand von Mueller, who was Government Botanist of Victoria, will be pleased to learn that his executors, the Rev. W. Potter, F.R.G.S., Alexander Buttner, and Hermann Buttner, are now making an effort to erect over the grave a monument worthy of the deceased savant's fame. The monument will be of grey granite, 23 feet in height."
        self.s2 = "To say he shall leave a wall of strong Towns behind him is to say nothing at all in this case, while there is an Army of 60000 Men in the field there; to say he shall want Provisions or any Assistance whatever is to say nothing while we are Masters of the Seas and can in four Hours come from Dover to Bologn, with Supplies of all Sorts, a passage so easie that you might bake his very Bread for him in Kent if you pleas'd."
        self.s3 = 'cgtagctagcgacgcagccaatcgatcg'
        self.s4 = 'cgagatgccaatagagagagcgctgta'
        alphabet = list(set(self.s1 + self.s2))
        self.k_slow = flakes.string.StringKernel(mode='slow', alphabet=alphabet)
        self.k_np = flakes.string.StringKernel(mode='numpy', alphabet=alphabet)
        self.k_tf = flakes.string.StringKernel(alphabet=alphabet, device=self.DEVICE)
        self.k_tf_gram = flakes.string.StringKernel(mode='tf-gram', alphabet=alphabet,
                                                    device=self.DEVICE,
                                                    trace=self.TRACE_FILE)

    @unittest.skip('profiling')
    def test_prof_1(self):
        self.k_tf.order = 40
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1, 1, 1] + ([0.5] * 30)
        self.k_tf.decay = 0.8
        result1 = self.k_tf.k(self.s1, self.s2)
        print result1

        self.k_np.order = 40
        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1, 1, 1] + ([0.5] * 30)
        self.k_np.decay = 0.8
        result2 = self.k_np.k(self.s1, self.s2)
        print result2

    @unittest.skip('profiling')
    def test_prof_2(self):
        self.k_tf.order = 8
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.k(self.s1, self.s1)
        print result1

        self.k_np.order = 8
        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1]
        self.k_np.decay = 0.8
        result2 = self.k_np.k(self.s1, self.s1)
        print result2

    @unittest.skip('profiling')
    def test_prof_3(self):
        data = np.loadtxt('flakes/tests/trial2', dtype=object, delimiter='\t')[:2]
        inputs = data[:, 1:]
        self.k_tf.order = 30
        self.k_tf.order_coefs = [0.1, 0.7, 0.5, 0.3, 0.1] + ([0.1] * 25)
        self.k_tf.decay = 0.1
        alphabet = list(set(''.join(inputs.flatten())))
        self.k_tf.alphabet = {elem: i for i, elem in enumerate(alphabet)}
        #result = self.k_tf.K(inputs)
        for i in range(50):
            print i
            #import ipdb; ipdb.set_trace()
            result = self.k_tf.K(inputs[0][0], inputs[1][0])
        print inputs
        print result

    @unittest.skip('profiling')
    def test_prof_4(self):
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1] + 32 * [1.0]
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        #result1 = self.k_tf.k(self.s1, self.s1)
        #print result1

        #self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1]
        #self.k_np.decay = 0.8
        #print "START PROF 4"
        before = datetime.datetime.now()
        for i in range(25):
            print i
            result2 = self.k_tf.K(self.s1, self.s2)
        after = datetime.datetime.now()
        print result2
        print after - before

    @unittest.skip('profiling')
    def test_prof_5(self):
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1] + 32 * [1.0]
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        #result1 = self.k_tf.k(self.s1, self.s1)
        #print result1

        #self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1]
        #self.k_np.decay = 0.8
        print "START PROF NORMAL"
        X = [[self.s3]] * 20
        X2 = [[self.s4]] * 20
        before = datetime.datetime.now()
        result2 = self.k_tf.K(X, X2)
        after = datetime.datetime.now()
        print result2
        print after - before

    #@unittest.skip('profiling')
    def test_prof_gram_1(self):
        self.k_tf_gram.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1] + 32 * [1.0]
        self.k_tf_gram.gap_decay = 0.8
        self.k_tf_gram.match_decay = 0.8
        #result1 = self.k_tf.k(self.s1, self.s1)
        #print result1

        #self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1]
        #self.k_np.decay = 0.8
        print "START PROF GRAM"
        X = [[self.s1]] * 2
        X2 = [[self.s2]] * 2
        before = datetime.datetime.now()
        result2 = self.k_tf_gram.K(X, X2)
        after = datetime.datetime.now()
        print result2
        print after - before


if __name__ == "__main__":
    if len(sys.argv) > 2:
        StringKernelProfiling.TRACE_FILE = sys.argv.pop()
        StringKernelProfiling.DEVICE = sys.argv.pop()
    unittest.main()
    
