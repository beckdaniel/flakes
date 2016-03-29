import flakes
import unittest
import numpy as np


class StringKernelTests(unittest.TestCase):
    
    def setUp(self):
        self.s1 = 'cata'
        self.s2 = 'gatta'
        self.s3 = 'cgtagctagcgacgcagccaatcgatcg'
        self.s4 = 'cgagatgccaatagagagagcgctgta'
        self.k_slow = flakes.string.StringKernel(mode='slow')
        self.k_np = flakes.string.StringKernel(mode='numpy')
        self.k_tf = flakes.string.StringKernel()
        self.k_tf.alphabet = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        
    def test_sk_slow_1(self):
        self.k_slow.order = 5
        self.k_slow.order_coefs = [1.] * 5
        self.k_slow.decay = 2.0
        expected = 504.0
        result = self.k_slow.k(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_numpy_1(self):
        self.k_np.order = 5
        self.k_np.order_coefs = [1.] * 5
        self.k_np.decay = 2.0
        expected = 504.0
        result = self.k_np.k(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_tf_1(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.decay = 2.0
        expected = 504.0
        result = self.k_tf.k(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_tf_2(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.decay = 0.8
        expected = 5.943705
        result = self.k_tf.k(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_compare_1(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.k(self.s1, self.s2)

        self.k_np.order = 5
        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.decay = 0.8
        result2 = self.k_np.k(self.s1, self.s2)
        self.assertAlmostEqual(result1, result2)

    def test_compare_2(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.k(self.s3, self.s4)

        self.k_np.order = 5
        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.decay = 0.8
        result2 = self.k_np.k(self.s3, self.s4)
        self.assertAlmostEqual(result1, result2, places=2)

    def test_compare_3(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.k(self.s1, self.s4)

        self.k_np.order = 5
        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.decay = 0.8
        result2 = self.k_np.k(self.s1, self.s4)
        self.assertAlmostEqual(result1, result2, places=2)
        

@unittest.skip('profiling')
class StringKernelProfiling(unittest.TestCase):

    def setUp(self):
        self.s1 = "Scientists throughout the world and all the friends of the late Baron Sir Ferdinand von Mueller, who was Government Botanist of Victoria, will be pleased to learn that his executors, the Rev. W. Potter, F.R.G.S., Alexander Buttner, and Hermann Buttner, are now making an effort to erect over the grave a monument worthy of the deceased savant's fame. The monument will be of grey granite, 23 feet in height."
        self.s2 = "To say he shall leave a wall of strong Towns behind him is to say nothing at all in this case, while there is an Army of 60000 Men in the field there; to say he shall want Provisions or any Assistance whatever is to say nothing while we are Masters of the Seas and can in four Hours come from Dover to Bologn, with Supplies of all Sorts, a passage so easie that you might bake his very Bread for him in Kent if you pleas'd."
        self.k_slow = flakes.string.StringKernel(mode='slow')
        self.k_np = flakes.string.StringKernel(mode='numpy')
        self.k_tf = flakes.string.StringKernel()
        self.k_tf.alphabet = {elem: i for i, elem in enumerate(list(set(self.s1 + self.s2)))}

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
        data = np.loadtxt('flakes/tests/trial2', dtype=object, delimiter='\t')[:10]
        inputs = data[:, 1:]
        self.k_tf.order = 30
        self.k_tf.order_coefs = [0.1, 0.7, 0.5, 0.3, 0.1] + ([0.1] * 25)
        self.k_tf.decay = 0.1
        alphabet = list(set(''.join(inputs.flatten())))
        self.k_tf.alphabet = {elem: i for i, elem in enumerate(alphabet)}
        result = self.k_tf.K(inputs)
        print inputs
        print result


class GPyTests(unittest.TestCase):
    
    def test_gpy_1(self):
        data = np.loadtxt('flakes/tests/trial2', dtype=object, delimiter='\t')[:5]
        inputs = data[:, 1:]
        k = flakes.wrappers.gpy.GPyStringKernel()
        k.order = 30
        k.order_coefs = [0.1, 0.7, 0.5, 0.3, 0.1] + ([0.1] * 25)
        k.decay = 0.1
        alphabet = list(set(''.join(inputs.flatten())))
        k.alphabet = {elem: i for i, elem in enumerate(alphabet)}
        result = k.K(inputs)
        print inputs
        print result

if __name__ == "__main__":
    unittest.main()
