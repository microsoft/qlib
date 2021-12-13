# cython: profile=False
# cython: boundscheck=False, wraparound=False, cdivision=True
cimport cython
cimport numpy as np
import numpy as np

from libc.math cimport sqrt, isnan, NAN
from libcpp.deque cimport deque


cdef class Rolling:
    """1-D array rolling"""
    cdef int window
    cdef deque[double] barv
    cdef int na_count
    def __init__(self, int window):
        self.window = window
        self.na_count = window
        cdef int i
        for i in range(window):
            self.barv.push_back(NAN)

    cdef double update(self, double val):
        pass


cdef class Mean(Rolling):
    """1-D array rolling mean"""
    cdef double vsum
    def __init__(self, int window):
        super(Mean, self).__init__(window)
        self.vsum = 0
        
    cdef double update(self, double val):
        self.barv.push_back(val)
        if not isnan(self.barv.front()):
            self.vsum -= self.barv.front()
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
            # return NAN
        else:
            self.vsum += val
        return self.vsum / (self.window - self.na_count)


cdef class Slope(Rolling):
    """1-D array rolling slope"""
    cdef double i_sum # can be used as i2_sum
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double xy_sum
    def __init__(self, int window):
        super(Slope, self).__init__(window)
        self.i_sum  = 0
        self.x_sum  = 0
        self.x2_sum = 0
        self.y_sum  = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2*self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        cdef double _val
        _val = self.barv.front()
        if not isnan(_val):
            self.i_sum -= 1
            self.y_sum -= _val
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
            # return NAN
        else:
            self.i_sum  += 1
            self.x_sum  += self.window
            self.x2_sum += self.window * self.window
            self.y_sum  += val
            self.xy_sum += self.window * val
        cdef int N = self.window - self.na_count
        return (N*self.xy_sum - self.x_sum*self.y_sum) / \
            (N*self.x2_sum - self.x_sum*self.x_sum)

    
cdef class Resi(Rolling):
    """1-D array rolling residuals"""
    cdef double i_sum # can be used as i2_sum
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double xy_sum
    def __init__(self, int window):
        super(Resi, self).__init__(window)
        self.i_sum  = 0
        self.x_sum  = 0
        self.x2_sum = 0
        self.y_sum  = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2*self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        cdef double _val
        _val = self.barv.front()
        if not isnan(_val):
            self.i_sum -= 1
            self.y_sum -= _val
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
            # return NAN
        else:
            self.i_sum  += 1
            self.x_sum  += self.window
            self.x2_sum += self.window * self.window
            self.y_sum  += val
            self.xy_sum += self.window * val
        cdef int N = self.window - self.na_count
        slope = (N*self.xy_sum - self.x_sum*self.y_sum) / \
                (N*self.x2_sum - self.x_sum*self.x_sum)
        x_mean = self.x_sum / N
        y_mean = self.y_sum / N
        interp = y_mean - slope*x_mean
        return val - (slope*self.window + interp)

    
cdef class Rsquare(Rolling):
    """1-D array rolling rsquare"""
    cdef double i_sum
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double y2_sum
    cdef double xy_sum
    def __init__(self, int window):
        super(Rsquare, self).__init__(window)
        self.i_sum  = 0
        self.x_sum  = 0
        self.x2_sum = 0
        self.y_sum  = 0
        self.y2_sum = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2*self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        cdef double _val
        _val = self.barv.front()
        if not isnan(_val):
            self.i_sum  -= 1
            self.y_sum  -= _val
            self.y2_sum -= _val * _val
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
            # return NAN
        else:
            self.i_sum  += 1
            self.x_sum  += self.window
            self.x2_sum += self.window * self.window
            self.y_sum  += val
            self.y2_sum += val * val
            self.xy_sum += self.window * val
        cdef int N = self.window - self.na_count
        cdef double rvalue
        rvalue = (N*self.xy_sum - self.x_sum*self.y_sum) / \
            sqrt((N*self.x2_sum - self.x_sum*self.x_sum) * (N*self.y2_sum - self.y_sum*self.y_sum))
        return rvalue * rvalue

    
cdef np.ndarray[double, ndim=1] rolling(Rolling r, np.ndarray a):
    cdef int  i
    cdef int  N = len(a)
    cdef np.ndarray[double, ndim=1] ret = np.empty(N)
    for i in range(N):
        ret[i] = r.update(a[i])
    return ret

def rolling_mean(np.ndarray a, int window):
    cdef Mean r = Mean(window)
    return rolling(r, a)

def rolling_slope(np.ndarray a, int window):
    cdef Slope r = Slope(window)
    return rolling(r, a)

def rolling_rsquare(np.ndarray a, int window):
    cdef Rsquare r = Rsquare(window)
    return rolling(r, a)

def rolling_resi(np.ndarray a, int window):
    cdef Resi r = Resi(window)
    return rolling(r, a)
