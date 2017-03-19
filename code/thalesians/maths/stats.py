import numpy as np

import thalesians.maths.numpyutils as npu
from thalesians.utilities.collections import DiagonalArray, SubdiagonalArray
from thalesians.utilities.conditions import precondition, exactlyonenotnone

@precondition(lambda cor, var=None, sd=None, copy=True: exactlyonenotnone(var, sd),
        'Exactly one of var, sd must be specified (not None)')
def cor2cov(cor, var=None, sd=None, copy=True):
    sd = np.sqrt(var) if var is not None else sd
    if isinstance(cor, (DiagonalArray, SubdiagonalArray)):
        cor = cor.tonumpyarray()
    cor = npu.tondim2(cor, copy=copy)
    dim = len(var)
    assert dim == np.shape(cor)[0] and dim == np.shape(cor)[1]
    np.fill_diagonal(cor, 1.)
    cor = (sd.T * (sd * cor).T).T
    npu.lowertosymmetric(cor, copy=False)
    return cor

def cov2cor(covs):
    var = np.diag(covs)
    sd = np.sqrt(var)
    return ((covs / sd).T / sd.T).T

def choleskysqrt2d(sd1, sd2, cor):
    return np.array(((sd1, 0.), (sd2 * cor, sd2 * np.sqrt(1. - cor * cor))))

class OnlineMeanAndVarCalculator(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.__n = 0
        self.__mean = 0.0
        self.__meansq = 0.0
        self.__M2 = 0.0
        
    def __get_count(self):
        return self.__n
    
    count = property(fget=__get_count)
    
    def __get_mean(self):
        return self.__mean
    
    mean = property(fget=__get_mean)
    
    def __get_meansq(self):
        return self.__meansq
    
    meansq = property(fget=__get_meansq)
    
    def __get_rms(self):
        return np.sqrt(self.meansq)
    
    rms = property(fget=__get_rms)
    
    def __get_varN(self):
        return self.__M2 / self.__n
    
    varN = property(fget=__get_varN)
    
    def __get_var(self):
        return self.__M2 / (self.__n - 1)
    
    var = property(fget=__get_var)
    
    def __get_sd(self):
        return np.sqrt(self.var)
    
    sd = property(fget=__get_sd)
    
    def __get_sdN(self):
        return np.sqrt(self.varN)
    
    sdN = property(fget=__get_sdN)
    
    def add(self, x):
        self.__n += 1
        delta = x - self.__mean
        self.__mean += delta / self.__n
        deltasq = x * x - self.__meansq
        self.__meansq += deltasq / self.__n
        if self.__n > 1:
            self.__M2 += delta * (x - self.__mean)

    def addall(self, xs):
        for x in xs: self.add(x)
        