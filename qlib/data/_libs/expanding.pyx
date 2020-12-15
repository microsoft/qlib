# cython: profile=False
# cython: boundscheck=False, wraparound=False, cdivision=True
cimport cython
cimport numpy as np
import numpy as np

from libc.math cimport sqrt, isnan, NAN
from libcpp.vector cimport vector


cdef class Expanding:
    """1-D array expanding"""
    cdef vector[double] barv
    cdef int na_count
    def __init__(self):
        self.na_count = 0

    cdef double update(self, double val):
        pass


cdef class Mean(Expanding):
    """1-D array expanding mean"""
    cdef double vsum
    def __init__(self):
        super(Mean, self).__init__()
        self.vsum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        if isnan(val):
            self.na_count += 1
        else:
            self.vsum += val
        return self.vsum / (self.barv.size() - self.na_count)


cdef class Slope(Expanding):
    """1-D array expanding slope"""
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double xy_sum
    def __init__(self):
        super(Slope, self).__init__()
        self.x_sum  = 0
        self.x2_sum = 0
        self.y_sum  = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        cdef size_t size = self.barv.size()
        if isnan(val):
            self.na_count += 1
        else:
            self.x_sum  += size
            self.x2_sum += size * size
            self.y_sum  += val
            self.xy_sum += size * val
        cdef int N = size - self.na_count
        return (N*self.xy_sum - self.x_sum*self.y_sum) / \
            (N*self.x2_sum - self.x_sum*self.x_sum)


cdef class Resi(Expanding):
    """1-D array expanding residuals"""
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double xy_sum
    def __init__(self):
        super(Resi, self).__init__()
        self.x_sum  = 0
        self.x2_sum = 0
        self.y_sum  = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        cdef size_t size = self.barv.size()
        if isnan(val):
            self.na_count += 1
        else:
            self.x_sum  += size
            self.x2_sum += size * size
            self.y_sum  += val
            self.xy_sum += size * val
        cdef int N = size - self.na_count
        slope = (N*self.xy_sum - self.x_sum*self.y_sum) / \
                (N*self.x2_sum - self.x_sum*self.x_sum)
        x_mean = self.x_sum / N
        y_mean = self.y_sum / N
        interp = y_mean - slope*x_mean
        return val - (slope*size + interp)


cdef class Rsquare(Expanding):
    """1-D array expanding rsquare"""
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double y2_sum
    cdef double xy_sum
    def __init__(self):
        super(Rsquare, self).__init__()
        self.x_sum  = 0
        self.x2_sum = 0
        self.y_sum  = 0
        self.y2_sum = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        cdef size_t size = self.barv.size()
        if isnan(val):
            self.na_count += 1
        else:
            self.x_sum  += size
            self.x2_sum += size * size
            self.y_sum  += val
            self.y2_sum += val * val
            self.xy_sum += size * val
        cdef int N = size - self.na_count
        cdef double rvalue = (N*self.xy_sum - self.x_sum*self.y_sum) / \
            sqrt((N*self.x2_sum - self.x_sum*self.x_sum) * (N*self.y2_sum - self.y_sum*self.y_sum))
        return rvalue * rvalue


cdef np.ndarray[double, ndim=1] expanding(Expanding r, np.ndarray a):
    cdef int  i
    cdef int  N = len(a)
    cdef np.ndarray[double, ndim=1] ret = np.empty(N)
    for i in range(N):
        ret[i] = r.update(a[i])
    return ret

def expanding_mean(np.ndarray a):
    cdef Mean r = Mean()
    return expanding(r, a)

def expanding_slope(np.ndarray a):
    cdef Slope r = Slope()
    return expanding(r, a)

def expanding_rsquare(np.ndarray a):
    cdef Rsquare r = Rsquare()
    return expanding(r, a)

def expanding_resi(np.ndarray a):
    cdef Resi r = Resi()
    return expanding(r, a)
