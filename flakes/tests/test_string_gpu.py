import flakes
import unittest
import numpy as np
#import GPy
import datetime


class GPUStringKernelProfiling(unittest.TestCase):

    def setUp(self):
        self.s1 = "Scientists throughout the world and all the friends of the late Baron Sir Ferdinand von Mueller, who was Government Botanist of Victoria, will be pleased to learn that his executors, the Rev. W. Potter, F.R.G.S., Alexander Buttner, and Hermann Buttner, are now making an effort to erect over the grave a monument worthy of the deceased savant's fame. The monument will be of grey granite, 23 feet in height."
        self.s2 = "To say he shall leave a wall of strong Towns behind him is to say nothing at all in this case, while there is an Army of 60000 Men in the field there; to say he shall want Provisions or any Assistance whatever is to say nothing while we are Masters of the Seas and can in four Hours come from Dover to Bologn, with Supplies of all Sorts, a passage so easie that you might bake his very Bread for him in Kent if you pleas'd."
        self.s3 = 'cgtagctagcgacgcagccaatcgatcg'
        self.s4 = 'cgagatgccaatagagagagcgctgta'
        self.k_tf = flakes.string.StringKernel(device='/gpu:6')
        self.k_tf.alphabet = {elem: i for i, elem in enumerate(list(set(self.s1 + self.s2)))}
        self.k_tf_row = flakes.string.StringKernel(device='/gpu:6', mode='tf-row')
        self.k_tf_row.alphabet = {elem: i for i, elem in enumerate(list(set(self.s1 + self.s2)))}

    @unittest.skip('profiling')
    def test_prof_4_gpu(self):
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1]# + 32 * [1.0]
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        for i in range(25):
            print i
            result2 = self.k_tf.k(self.s1, self.s2)
        print result2

    #@unittest.skip('profiling')
    def test_prof_5_gpu(self):
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]#, 1, 1, 1] + 32 * [1.0]
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        X = [[self.s3]] * 100
        #X2 = [[self.s2]] * 5
        before = datetime.datetime.now()
        result2 = self.k_tf.K(X)#, X2)
        after = datetime.datetime.now()
        print result2
        print 'CELL-BASED'
        print after - before

    @unittest.skip('profiling')
    def test_prof_5_gpu_row(self):
        self.k_tf_row.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7, 1, 1, 1]# + 32 * [1.0]
        self.k_tf_row.gap_decay = 0.8
        self.k_tf_row.match_decay = 0.8
        X = [[self.s1]] * 800
        before = datetime.datetime.now()
        result2 = self.k_tf_row.K(X)#, X2)
        after = datetime.datetime.now()
        print result2
        print 'ROW-BASED'
        print after - before

    def test_compare_row_based(self):
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.K(X)
        self.k_tf_row.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf_row.decay = 0.8
        result2 = self.k_tf_row.K(X)
        print result1
        print result2
        self.assertAlmostEqual(np.sum(result1), np.sum(result2), places=7)

if __name__ == "__main__":
    unittest.main()
