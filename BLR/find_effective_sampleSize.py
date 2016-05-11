__author__ = 'nt357'
import numpy as np
from rpy2 import robjects


# Class should be imported as from devang_code import RCodaTools
# Can use functions in this class by calling RCodaTools.ess_coda_vec(samples)
# which will compute coordinate-wise ESS using the Rcodatools
# also some stuff to do geweke validation


class RCodaTools(object):
    @staticmethod
    def ess_coda(data):
        """
        Computes the effective samples size of a 1d-array using R-coda via
        an external R call. The python package rpy2 and the R-library
        "library(coda)" have to be installed. Inspired by Charles Blundell's
        neat little python script :)
        """
        robjects.r('library(coda)')
        r_ess = robjects.r['effectiveSize']
        data = robjects.r.matrix(robjects.FloatVector(data), nrow=len(data))
        return r_ess(data)[0]

    @staticmethod
    def ess_coda_vec(samples):
        sample_array = np.array(samples)
        l, h = sample_array.shape

        ess_vec = np.zeros(h)
        for i in xrange(h):
            ess_vec[i] = RCodaTools.ess_coda(sample_array[:, i])

        return ess_vec

    @staticmethod
    def geweke(data):
        robjects.r('library(coda)')
        r_geweke = robjects.r['geweke.diag']
        data = robjects.r.matrix(robjects.FloatVector(data), nrow=len(data))

        return r_geweke(data)[0]
