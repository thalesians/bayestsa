import unittest

import numpy.testing as npt

import thalesians.maths.stats as stats
from thalesians.utilities.collections import SubdiagonalArray

class StatsTest(unittest.TestCase):
    def test_cor2cov(self):
        cor = SubdiagonalArray.create((-.25, -.5, .3))
        var = (4., 3., 5.)
        cov = stats.cor2cov(cor, var)
        npt.assert_almost_equal(cov, (
                ( 4.        , -0.8660254, -2.23606798),
                (-0.8660254 ,  3.       ,  1.161895  ),
                (-2.23606798,  1.161895 ,  5.        )))
        
    def test_cov2cor(self):
        cov = (( 4.        , -0.8660254, -2.23606798),
               (-0.8660254 ,  3.       ,  1.161895  ),
               (-2.23606798,  1.161895 ,  5.        ))
        cor = stats.cov2cor(cov)
        npt.assert_almost_equal(cor, (
                ( 1.0 , -0.25, -0.5),
                (-0.25,  1.0 ,  0.3),
                (-0.5 ,  0.3 ,  1.0)))
            
if __name__ == '__main__':
    unittest.main()
    