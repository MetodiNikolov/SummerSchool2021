import math

from numpy.random import default_rng
import numpy as np

class Model(object):

    def __init__(self, data, burn_in, sample, nu):
        print("making a model")
        self._data = np.asarray(data)
        self._data_size = len(data)
        self._burn_in = burn_in
        self._sample_size = sample
        self._nu = nu

        self._rng = default_rng()

        self._extended_vars = np.zeros(self._data_size)
        self._tau2 = 1
        self._mu = sum(data) / self._data_size
        self._alpha2 = 1

        self._update_extended_vars()
        self._results_mu = np.zeros(self._sample_size)
        self._results_sigma2 = np.zeros(self._sample_size)

        self._tmp_with_data_size = np.zeros(self._data_size)
        self._tmp_with_data_size2 = np.zeros(self._data_size)

        
    def _sampleScaledInvChiSquare(self, ni, scale):
        x = self._rng.chisquare(ni)
        return ni * scale / x

    def _update_mu(self):
        np.reciprocal(self._extended_vars, out=self._tmp_with_data_size)
        self._tmp_with_data_size2 = self._data * self._tmp_with_data_size

        variance = self._tmp_with_data_size.sum()
        expected_value = self._tmp_with_data_size2.sum()

        # for i in range(self._data_size):
        #     variance += 1.0 / self._extended_vars[i]
        #     expected_value += self._data[i] / self._extended_vars[i]

        variance /= self._alpha2
        expected_value /= self._alpha2
        variance = 1.0 / variance
        expected_value = expected_value * variance
        self._mu = self._rng.normal(expected_value, math.sqrt(variance))
    
    def _update_tau2(self):
        np.reciprocal(self._extended_vars, out=self._tmp_with_data_size)
        x = self._tmp_with_data_size.sum()
        # x = 0.0
        # for i in range(self._data_size):
        #     x += 1.0 / self._extended_vars[i]
        
        self._tau2 = self._rng.gamma(self._data_size * self._nu / 2.0, 2.0 / (self._nu * x))


    def _update_alpha2(self):
        x = 0.0
        self._tmp_with_data_size = self._data - self._mu
        self._tmp_with_data_size = (self._tmp_with_data_size * self._tmp_with_data_size) / self._extended_vars
        x = self._tmp_with_data_size.sum()
        # for i in range(self._data_size):
        #     x += (self._data[i]  - self._mu) * (self._data[i]  - self._mu) / self._extended_vars[i]
        x /= self._data_size
        self._alpha2 = self._sampleScaledInvChiSquare(self._data_size, x)

    def _update_extended_vars(self):
        for i in range(self._data_size):
            x = (self._data[i] - self._mu) * (self._data[i] - self._mu) / self._alpha2
            self._extended_vars[i] = self._sampleScaledInvChiSquare(self._nu + 1, (self._nu * self._tau2 + x) / (self._nu + 1))

    def run(self):
        print("Starting Burn-in")
        for _ in range(self._burn_in):
            self._update_extended_vars()
            self._update_alpha2()
            self._update_mu()
            self._update_tau2

        print("Starting Data run")
        for i in range(self._sample_size):
            self._update_extended_vars()
            self._update_alpha2()
            self._update_mu()
            self._update_tau2
            self._results_mu[i] = self._mu
            self._results_sigma2[i] = self._alpha2 * self._tau2
    
    def get_mu(self):
        return self._results_mu

    def get_sigma2(self):
        return self._results_sigma2


if __name__ == '__main__':

    sss = Model([1, 2, 3, 4, 5], 1000, 1000, 4)
    sss.run()