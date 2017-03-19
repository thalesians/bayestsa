# TODO Don't use the word measurement
# TODO Don't say "matrix" too often
# TODO Shorten observation to obs
# TODO Shorten innovation to innov
# TODO innovcov, not innovvar
# TODO Cater for the multidimensional observation case
# TODO Complete the docs
# TODO Mention all parameters in the docs
# TODO __str__
# DONE There are four dimensions, not just self.n, self.q
# TODO Don't store priors, innovations, etc., return them if necessary
# TODO Stop referring to Haykin, refer to the handbook chapter

import warnings

import numpy as np

import thalesians.maths.numpyutils as npu
from thalesians.maths.constants import MINUS_HALF_LN_2PI

class KalmanFilter(object):
    r"""
The Kalman filter.

:param state: The initial procdim-dimensional estimate of the state of the system
:param statecov: The procdim-by-procdim-dimensional a posteriori error covariance matrix (a measure of the estimated accuracy of the state estimate)
:param procnoisecov: The procdim-by-procdim-dimensional covariance matrix of the state noise process due to disturbances and modelling errors
:param obsnoisecov: The obsdim-by-obsdim-dimensional covariance matrix of the measurement noise process
:param procmap: The procdim-by-procdim-dimensional transition matrix taking the state from time k to time k+1
:param obsmap: The obsdim-by-procdim-dimensional measurement matrix. Applied to a state, it produces the corresponding observable
    """
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Constructor
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, state=None, statecov=None, procnoisecov=None, obsnoisecov=None, procmap=None, obsmap=None, procoffset=None, obsoffset=None, procnoisemap=None, obsnoisemap=None):
        self.procdim = None
        self.obsdim = None

        # The procdim-dimensional state of the system.
        self.state = state
        self.statecov = statecov
        self.__priorstate = None
        self.__priorstatecov = None
        self.procnoisecov = procnoisecov
        self.obsnoisecov = obsnoisecov
        self.procmap = procmap
        self.obsmap = obsmap
        self.procoffset = procoffset
        self.obsoffset = obsoffset
        self.procnoisemap = procnoisemap
        self.obsnoisemap = obsnoisemap

        # We shall be storing the latest innovation and its covariance.
        self.predictedobs = None
        self.lastobs = None
        self.innov = None
        self.innovcov = None
        self.gain = None

        self.loglikelihood = 0.0

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Predict and observe
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# The core of the implementation.
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

    def predict(self, **kwargs):
        assert not self.procnoisecov is None, 'The process noise covariance is not set'

        # By default, our transition matrix is the procdim-by-procdim-dimensional identity
        # matrix: the state stays the same as time passes.
        if self.procmap is None:
            if not self.procdim is None:
                warnings.warn('The transition matrix procmap is not set. Defaulting to procdim-by-procdim-dimensional identity')
                self.procmap = np.eye(self.procdim)

        assert not self.procmap is None, 'The transition matrix procmap is not set'
        
        if self.procnoisemap is None:
            self.procnoisemap = np.eye(self.procdim)
            
        if self.procnoisemap is None:
            if not self.procdim is None:
                warnings.warn('The matrix procnoisemap is not set. Defaulting to procdim-by-procdim-dimensional identity')
                self.procnoisemap = np.eye(self.procdim)

        assert not self.procnoisemap is None, 'The matrix procnoisemap is not set'

        # Here we shall refer to the steps given in [Haykin-2001]_.

        # State estimate propagation (step 1):
        self.state = np.dot(self.procmap, self.state)
        if self.procoffset is not None:
            self.state += self.procoffset

        # Error covariance propagation (step 2):
        self.statecov = np.dot(np.dot(self.procmap, self.statecov), self.procmap.T) + np.dot(np.dot(self.procnoisemap, self.procnoisecov), self.procnoisemap.T)

        self.__priorstate = self.state
        self.__priorstatecov = self.statecov

        return self.state

    def observe(self, obs, **kwargs):
        if 'obsnoisecov' in kwargs:
            obsnoisecov = kwargs['obsnoisecov']
        else:
            obsnoisecov = self.obsnoisecov

        assert not obsnoisecov is None, 'The covariance matrix obsnoisecov is not set'

        # By default, our measurement matrix is the procdim-by-procdim-dimensional identity
        # matrix: we are observing the state directly. This only makes sense if
        # procdim == obsdim.
        if self.obsmap is None:
            if (not self.procdim is None) and (self.procdim == self.obsdim):
                warnings.warn('The measurement matrix obsmap is not set. Defaulting to procdim-by-procdim-dimensional identity')
                self.obsmap = np.eye(self.procdim)

        assert not self.obsmap is None, 'The measurement matrix obsmap is not set'
        
        if self.obsnoisemap is None:
            if self.obsdim is not None:
                warnings.warn('The matrix obsnoisemap is not set. Defaulting to obsdim-by-obsdim-dimensional identity')
                self.obsnoisemap = np.eye(self.obsdim)

        assert not self.obsnoisemap is None, 'The matrix obsnoisemap is not set'

        obs = npu.tondim2(obs, ndim1tocolumn=True, copy=False)

        # Here we shall refer to the steps given in [Haykin-2001]_.

        # Kalman gain matrix (step 3):
        self.innovcov = np.dot(np.dot(self.obsmap, self.statecov), self.obsmap.T) + np.dot(np.dot(self.obsnoisemap, obsnoisecov), self.obsnoisemap.T)

        self.gain = np.dot(np.dot(self.statecov, self.obsmap.T), np.linalg.pinv(self.innovcov))
        
        self.predictedobs = np.dot(self.obsmap, self.state)
        if self.obsoffset is not None:
            self.predictedobs += self.obsoffset

        # State estimate update (step 4):
        self.innov = obs - self.predictedobs
        self.state = self.state + np.dot(self.gain, self.innov)
        self.statecov = np.dot(np.identity(self.procdim) - np.dot(self.gain, self.obsmap), self.statecov)

        # TODO Cater for the multidimensional observation case
        self.loglikelihood += MINUS_HALF_LN_2PI - .5 * (np.log(self.innovcov) + self.innov * self.innov / self.innovcov)
        
        self._lastobs = obs

        return self.state

    def predictAndObserve(self, obs, **kwargs):
        self.predict(**kwargs)
        return self.observe(obs, **kwargs)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# This code is mostly wrappers and glue. The reason it is here is to ensure that
# we are dealing with numpy arrays of the right rank and shape. We are coercing
# inputs to (two-dimensional) matrices. This helps us avoid subtle
# bugs.
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __get_state(self):
        return self.__state

    def __set_state(self, value):
        if value is not None:
            self.__state = npu.tondim2(value, ndim1tocolumn=True, copy=True)
            shape = np.shape(self.__state)
            assert (self.procdim is None) or (self.procdim == shape[0]), 'The state must be procdim-dimensional'
            self.procdim = shape[0]
        else:
            self.__state = None

    state = property(fget=__get_state, fset=__set_state, doc='The procdim-dimensional estimate of the state of the system')

    def __get_statecov(self):
        return self.__statecov

    def __set_statecov(self, value):
        if value is not None:
            self.__statecov = npu.tondim2(value, copy=True)
            shape = np.shape(self.__statecov)
            assert shape[0] == shape[1], 'The state covariance must be square'
            assert (self.procdim is None) or (self.procdim == shape[0]), 'The state covariance must be procdim-by-procdim-dimensional'
            self.procdim = shape[0]
        else:
            self.__statecov = None

    statecov = property(fget=__get_statecov, fset=__set_statecov, doc='The procdim-by-procdim-dimensional a posteriori error covariance matrix (a measure of the estimated accuracy of the state estimate)')

    def __get_priorstate(self):
        return self.__priorstate

    priorstate = property(fget=__get_priorstate, doc='The procdim-dimensional predicted state of the system')

    def __get_priorstatecov(self):
        return self.__priorstatecov

    priorstatecov = property(fget=__get_priorstatecov, doc='The procdim-by-procdim-dimensional a priori error covariance matrix (a measure of the estimated accuracy of the state estimate)')

    def __get_procnoisecov(self):
        return self.__procnoisecov

    def __set_procnoisecov(self, value):
        if value is not None:
            self.__procnoisecov = npu.tondim2(value, copy=True)
            shape = np.shape(self.__procnoisecov)
            assert shape[0] == shape[1], 'The covariance matrix procnoisecov must be square'
            assert (self.procdim is None) or (self.procdim == shape[0]), 'The process noise covariance must be procdim-by-procdim-dimensional'
            self.procdim = shape[0]
        else:
            self.__procnoisecov = None

    procnoisecov = property(fget=__get_procnoisecov, fset=__set_procnoisecov, doc='The procdim-by-procdim-dimensional covariance matrix of the state noise process due to disturbances and modelling errors')

    def __get_obsnoisecov(self):
        return self.__obsnoisecov

    def __set_obsnoisecov(self, value):
        if value is not None:
            self.__obsnoisecov = npu.tondim2(value, copy=True)
            shape = np.shape(self.__obsnoisecov)
            assert shape[0] == shape[1], 'The covariance matrix obsnoisecov must be square'
            assert (self.obsdim is None) or (self.obsdim == shape[0]), 'The covariance matrix obsnoisecov must be obsdim-by-obsdim-dimensional'
            self.obsdim = shape[0]
        else:
            self.__obsnoisecov = None

    obsnoisecov = property(fget=__get_obsnoisecov, fset=__set_obsnoisecov, doc='The obsdim-by-obsdim-dimensional covariance matrix of the measurement noise process')

    def __get_procmap(self):
        return self.__procmap

    def __set_procmap(self, value):
        if value is not None:
            self.__procmap = npu.tondim2(value, copy=True)
            shape = np.shape(self.__procmap)
            assert shape[0] == shape[1], 'The transition matrix procmap must be square'
            assert (self.procdim is None) or (self.procdim == shape[0]), 'The transition matrix procmap must be procdim-by-procdim-dimensional'
            self.procdim = shape[0]
        else:
            self.__procmap = None

    procmap = property(fget=__get_procmap, fset=__set_procmap, doc='The procdim-by-procdim-dimensional transition matrix taking the state from time k to time k+1')

    def __get_obsmap(self):
        return self.__obsmap

    def __set_obsmap(self, value):
        if value is not None:
            self.__obsmap = npu.tondim2(value, copy=True)
            shape = np.shape(self.__obsmap)
            assert (self.obsdim is None) or (self.obsdim == shape[0]), 'The measurement matrix obsmap must have obsdim (%d) rows; it has %d rows' % (self.obsdim, shape[0])
            assert (self.procdim is None) or (self.procdim == shape[1]), 'The measurement matrix obsmap must have procdim (%d) columns; it has %d columns' % (self.procdim, shape[1])
            self.obsdim = shape[0]
            self.procdim = shape[1]
        else:
            self.__obsmap = None

    obsmap = property(fget=__get_obsmap, fset=__set_obsmap, doc='The obsdim-by-procdim-dimensional measurement matrix. Applied to a state, it produces the corresponding observable')
    
    def __get_procoffset(self):
        return self.__procoffset
    
    def __set_procoffset(self, value):
        if value is not None:
            self.__procoffset = npu.tondim2(value, ndim1tocolumn=True, copy=True)
        else:
            self.__procoffset = None
    
    procoffset = property(fget=__get_procoffset, fset=__set_procoffset)

    def __get_obsoffset(self):
        return self.__obsoffset
    
    def __set_obsoffset(self, value):
        if value is not None:
            self.__obsoffset = npu.tondim2(value, ndim1tocolumn=True, copy=True)
        else:
            self.__obsoffset = None
    
    obsoffset = property(fget=__get_obsoffset, fset=__set_obsoffset)
    
    @property
    def mean(self): return self.state
    
    @property
    def var(self): return self.statecov

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Special methods
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __str__(self):
        return 'KalmanFilter(procdim=%s, obsdim=%s, state=%s, statecov=%s, procnoisecov=%s, obsnoisecov=%s, procmap=%s, obsmap=%s, innov=%s, innovcov=%s)' % (
            self.procdim, self.obsdim, self.state, self.statecov, self.procnoisecov, self.obsnoisecov, self.procmap, self.obsmap, self.innov, self.innovcov)
