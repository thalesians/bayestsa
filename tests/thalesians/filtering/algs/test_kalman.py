import datetime
import unittest

class KalmanFilterTest(unittest.TestCase):
    
    def test_kalman_filter_with_prior_predict(self):
        t0 = datetime.datetime(2014, 2, 12, 16, 18, 25, 204000)
        print(t0)
        
        self.assertEqual(1., 1.)
        
    def test_kalman_filter_without_prior_predict(self):
        pass
    
    def test_kalman_filter_with_low_variance_observation(self):
        pass
    
    def test_kalman_filter_multidim(self):
        pass
        
if __name__ == '__main__':
    unittest.main()
    