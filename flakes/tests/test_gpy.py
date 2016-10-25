import flakes
import unittest
import numpy as np
import GPy
from paramz import ObsAr


class GPyStringKernelTests(unittest.TestCase):

    def setUp(self):
        self.s1 = 'ababa'
        self.s2 = 'bbbab'
        self.s1_bow = [3., 2.]
        self.s2_bow = [1., 4.]
        self.s1_bow2 = [3., 2., 3., 3., 3., 1.]
        self.s2_bow2 = [1., 4., 0., 1., 3., 6.]

        self.s3 = 'ababab'
        self.s4 = 'bababaa'
        self.s3_bow2 = [3., 3., 3., 6., 3., 3.]
        self.s4_bow2 = [4., 3., 6., 3., 9., 3.]

        alphabet = 'ab'
        self.k_tf = flakes.wrappers.gpy.GPyStringKernel(mode='tf', alphabet=alphabet, order_coefs=[1.0])
        self.k_tf2 = flakes.wrappers.gpy.GPyStringKernel(mode='tf-batch', alphabet=alphabet, order_coefs=[1.0, 1.0])
        self.k_tf_batch = flakes.wrappers.gpy.GPyStringKernel(mode='tf-batch', alphabet=alphabet, order_coefs=[1.0], wrapper='none')
        self.k_tf_batch2 = flakes.wrappers.gpy.GPyStringKernel(mode='tf-batch', alphabet=alphabet, order_coefs=[1.0, 1.0], sim='arccosine', wrapper='arccos0')
        self.k_tf_rbf = flakes.wrappers.gpy.RBFStringKernel(mode='tf-batch', alphabet=alphabet)
        
    @unittest.skip('')
    def test_linear_vs_sk(self):
        self.k_tf_batch.order_coefs = [1.]
        self.k_tf_batch.gap_decay = 1.0
        self.k_tf_batch.match_decay = 1.0
        
        sk_result = self.k_tf_batch.K(self.s1, self.s2)
        print sk_result
        print np.dot(self.s1_bow, self.s2_bow)

    @unittest.skip('')
    def test_caching(self):
        self.k_tf_batch.order_coefs = [1.] * 5
        self.k_tf_batch.gap_decay = 1.0
        self.k_tf_batch.match_decay = 1.0
        X = np.array([[self.s1], [self.s2], [self.s3], [self.s4]])
        # To allow caching
        X = ObsAr(X)

        print "REPETITION"
        sk_result = self.k_tf_batch.K(X)
        sk_result = self.k_tf_batch.K(X)
        sk_result = self.k_tf_batch.K(X)
        sk_result = self.k_tf_batch.K(X)
        sk_result = self.k_tf_batch.K(X)
        sk_result = self.k_tf_batch.K(X)
        print "REPETITION END"
        print sk_result
        print np.dot(self.s1_bow, self.s2_bow)

    def test_rbf(self):
        self.k_tf_rbf.order_coefs = [1.] * 5
        self.k_tf_rbf.gap_decay = 0.1
        self.k_tf_rbf.match_decay = 0.1
        X = np.array([[self.s1], [self.s2], [self.s3], [self.s4]])
        # To allow caching
        X = ObsAr(X)

        print "REPETITION"
        sk_result = self.k_tf_rbf.K(X)
        sk_result = self.k_tf_rbf.K(X)
        sk_result = self.k_tf_rbf.K(X)
        sk_result = self.k_tf_rbf.K(X)
        sk_result = self.k_tf_rbf.K(X)
        sk_result = self.k_tf_rbf.K(X)
        print "REPETITION END"
        print sk_result

    def test_rbf_inside_gp_regression(self):
        #self.k_tf_rbf.order_coefs = [1.] * 5
        self.k_tf_rbf.gap_decay = 0.1
        self.k_tf_rbf.match_decay = 0.1
        X = np.array([[self.s1], [self.s2], [self.s3], [self.s4]])
        Y = np.array([[3.0], [5.0], [8.0], [14.0]])
        m = GPy.models.GPRegression(X, Y, kernel=self.k_tf_rbf)
        print m
        print m.checkgrad(verbose=True)
        m.optimize(messages=True)
        print m
        print m.checkgrad(verbose=True)
        

    @unittest.skip('')
    def test_linear_vs_sk_gpy(self):
        #self.k_tf_batch.order_coefs = np.array([1.])
        self.k_tf_batch.gap_decay = 1.0
        self.k_tf_batch.match_decay = 1.0
        X1 = np.array([[self.s1], [self.s2]])
        Y1 = np.array([[3.0], [5.0]])
        m1 = GPy.models.GPRegression(X1, Y1, kernel=self.k_tf_batch)
        #m1['.*match.*'].constrain_fixed(1.0)
        m1['.*coefs.*'].constrain_fixed([1.0])
        print m1
        print m1.checkgrad(verbose=True)

        X2 = np.array([self.s1_bow, self.s2_bow])
        Y2 = np.array([[3.0], [5.0]])
        m2 = GPy.models.GPRegression(X2, Y2, kernel=GPy.kern.Linear(2))
        print m2

        m1['.*noise.*'].constrain_fixed(1e-4)        
        m1.optimize(messages=True)
        print m1
        print m1.checkgrad(verbose=True)

        m2['.*noise.*'].constrain_fixed(1e-4)
        m2.optimize(messages=True)
        print m2
        print m1.predict(X1)
        print m2.predict(X2)

    @unittest.skip('')
    def test_linear_vs_sk_gpy_2(self):
        #self.k_tf_batch.order_coefs = [1., 1.]
        #self.k_tf_batch.gap_decay = 1.0
        #self.k_tf_batch.match_decay = 1.0
        X1 = np.array([[self.s3], [self.s4]])
        Y1 = np.array([[3.0], [5.0]])
        m1 = GPy.models.GPRegression(X1, Y1, kernel=self.k_tf_batch2)
        m1['.*gap.*'].constrain_fixed(1.0)
        #m1['.*match.*'].constrain_fixed(1.0)
        m1['.*coefs.*'].constrain_fixed([1.0])
        m1['.*noise.*'].constrain_fixed(1e-4)
        print m1
        print m1['.*coefs.*']
        print m1.checkgrad(verbose=True)

        X2 = np.array([self.s3_bow2, self.s4_bow2])
        Y2 = np.array([[3.0], [5.0]])
        m2 = GPy.models.GPRegression(X2, Y2, kernel=GPy.kern.Linear(6))
        m2['.*noise.*'].constrain_fixed(1e-4)
        print m2
        print m2.checkgrad(verbose=True)        

        m1.optimize(messages=True)
        print m1
        print m1['.*coefs.*']
        print m1.checkgrad(verbose=True)
        m2.optimize(messages=True)
        print m2
        print m2.checkgrad(verbose=True)        

    @unittest.skip('')
    def test_linear_vs_sk_autograd_gpy(self):
        #self.k_tf_batch.order_coefs = np.array([1.])
        self.k_tf.gap_decay = 1.0
        self.k_tf.match_decay = 1.0
        X1 = np.array([[self.s1], [self.s2]])
        Y1 = np.array([[3.0], [5.0]])
        m1 = GPy.models.GPRegression(X1, Y1, kernel=self.k_tf)
        #m1['.*match.*'].constrain_fixed(1.0)
        m1['.*coefs.*'].constrain_fixed([1.0])
        print m1
        print m1.checkgrad(verbose=True)

        X2 = np.array([self.s1_bow, self.s2_bow])
        Y2 = np.array([[3.0], [5.0]])
        m2 = GPy.models.GPRegression(X2, Y2, kernel=GPy.kern.Linear(2))
        print m2

        m1['.*noise.*'].constrain_fixed(1e-4)        
        m1.optimize(messages=True)
        print m1
        print m1.checkgrad(verbose=True)

        m2['.*noise.*'].constrain_fixed(1e-4)
        m2.optimize(messages=True)
        print m2

    @unittest.skip('')
    def test_linear_vs_sk_autograd_gpy_2(self):
        #self.k_tf_batch.order_coefs = [1., 1.]
        #self.k_tf_batch.gap_decay = 1.0
        #self.k_tf_batch.match_decay = 1.0
        X1 = np.array([[self.s3], [self.s4]])
        Y1 = np.array([[3.0], [5.0]])
        m1 = GPy.models.GPRegression(X1, Y1, kernel=self.k_tf2)
        m1['.*gap.*'].constrain_fixed(1.0)
        #m1['.*match.*'].constrain_fixed(1.0)
        m1['.*coefs.*'].constrain_fixed([1.0])
        m1['.*noise.*'].constrain_fixed(1e-4)
        print m1
        print m1['.*coefs.*']
        print m1.checkgrad(verbose=True)

        X2 = np.array([self.s3_bow2, self.s4_bow2])
        Y2 = np.array([[3.0], [5.0]])
        m2 = GPy.models.GPRegression(X2, Y2, kernel=GPy.kern.Linear(6))
        m2['.*noise.*'].constrain_fixed(1e-4)
        print m2
        print m2.checkgrad(verbose=True)        

        m1.optimize(messages=True)
        print m1
        print m1['.*coefs.*']
        print m1.checkgrad(verbose=True)
        m2.optimize(messages=True)
        print m2
        print m2.checkgrad(verbose=True)        

    @unittest.skip('')
    def test_manualgrad_vs_sk_autograd_gpy_2(self):
        #self.k_tf_batch.order_coefs = [1., 1.]
        #self.k_tf_batch.gap_decay = 1.0
        #self.k_tf_batch.match_decay = 1.0
        X1 = np.array([[self.s3], [self.s4]])
        Y1 = np.array([[3.0], [5.0]])
        m1 = GPy.models.GPRegression(X1, Y1, kernel=self.k_tf2)
        #m1['.*gap.*'].constrain_fixed(1.0)
        #m1['.*match.*'].constrain_fixed(1.0)
        #m1['.*coefs.*'].constrain_fixed([1.0])
        #m1['.*noise.*'].constrain_fixed(1e-4)
        print m1
        print m1['.*coefs.*']
        print m1.checkgrad(verbose=True)

        X2 = np.array([[self.s3], [self.s4]])
        Y2 = np.array([[3.0], [5.0]])
        m2 = GPy.models.GPRegression(X2, Y2, kernel=self.k_tf_batch2)
        #m2['.*noise.*'].constrain_fixed(1e-4)
        print m2
        print m2.checkgrad(verbose=True)        

        m1.optimize(messages=True)
        print m1
        print m1['.*coefs.*']
        print m1.checkgrad(verbose=True)
        m2.optimize(messages=True)
        print m2
        print m2['.*coefs.*']
        print m2.checkgrad(verbose=True)        

    @unittest.skip('')
    def test_linear_vs_sk_autograd_gpy_2(self):
        #self.k_tf_batch.order_coefs = np.array([1.])
        self.k_tf.gap_decay = 1.0
        self.k_tf.match_decay = 1.0
        X1 = np.array([[self.s1], [self.s2]])
        Y1 = np.array([[3.0], [5.0]])
        m1 = GPy.models.GPRegression(X1, Y1, kernel=self.k_tf)
        m1['.*gap.*'].constrain_fixed(1.0)
        m1['.*match.*'].constrain_fixed(1.0)
        #m1['.*coefs.*'].constrain_fixed([1.0])
        #print m1
        #print m1.checkgrad(verbose=True)
        #print m1['.*gap.*'].gradient
        #print m1['.*match.*'].gradient
        #print m1['.*coefs.*'].gradient

        X2 = np.array([self.s1_bow, self.s2_bow])
        Y2 = np.array([[3.0], [5.0]])
        m2 = GPy.models.GPRegression(X2, Y2, kernel=GPy.kern.Linear(2))
        #print m2
        #print m2['.*variances.*'].gradient

        m1['.*noise.*'].constrain_fixed(1e-4)        
        #m1.optimize(messages=True, max_iters=200)
        #print m1


        m2['.*noise.*'].constrain_fixed(1e-4)
        #m2.optimize(messages=True, max_iters=200)
        #print m2

        m1['.*coefs.*'] = 20
        m2['.*variances.*'] = 20

        print m1
        print m2

        
        m1['.*coefs.*'] = 2.5
        m2['.*variances.*'] = 2.5
        #print m1['.*coefs.*'].gradient
        #print m2['.*variances.*'].gradient
        #print m1.checkgrad(verbose=True)
        #print m2.checkgrad(verbose=True)
        #print m1
        #print m1.kern._get_params()


    @unittest.skip('')
    def test_linear_vs_sk_autograd_gpy_3(self):
        #self.k_tf_batch.order_coefs = np.array([1.])
        self.k_tf.gap_decay = 1.0
        self.k_tf.match_decay = 1.0
        X1 = np.array([[self.s3], [self.s4]])
        Y1 = np.array([[3.0], [5.0]])
        m1 = GPy.models.GPRegression(X1, Y1, kernel=self.k_tf_batch2)
        #m1['.*gap.*'].constrain_fixed(1.0)
        #m1['.*match.*'].constrain_fixed(1.0)
        #m1['.*coefs.*'].constrain_fixed([1.0])
        #m1['.*string.variance.*'].constrain_fixed(0.0)
        print m1
        #print m1.checkgrad(verbose=True)
        print m1['.*gap.*'].gradient
        print m1['.*match.*'].gradient
        print m1['.*coefs.*'].gradient

        X2 = np.array([self.s3_bow2, self.s4_bow2])
        Y2 = np.array([[3.0], [5.0]])
        m2 = GPy.models.GPRegression(X2, Y2, kernel=GPy.kern.Linear(6))
        print m2
        #print m2['.*variances.*'].gradient

        #m1['.*noise.*'].constrain_fixed(1e-4)        
        #m1.optimize(messages=True, max_iters=200)
        print m1
        print m1['.*coefs.*']


        #m2['.*noise.*'].constrain_fixed(1e-4)
        #m2.optimize(messages=True, max_iters=200)
        print m2

        
        #m1['.*decay.*'] = 0.5
        #m1['.*coefs.*'] = 2
        #m2['.*variances.*'] = 0.1
        #print m1['.*coefs.*'].gradient
        #print np.sum(m1['.*coefs.*'].gradient)
        #print m2['.*variances.*'].gradient
        print m1.checkgrad(verbose=True)
        print m2.checkgrad(verbose=True)
        print m1.predict(X1)
        print m2.predict(X2)

if __name__ == "__main__":
    unittest.main()
