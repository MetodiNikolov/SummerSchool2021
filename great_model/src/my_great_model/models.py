import math

from numpy.random import default_rng

class Model(object):

    def __init__(self, data, burn_in, sample, nu):
        self._data = data
        self._data_size = len(data)
        self._burn_in = burn_in
        self._sample_size = sample
        self._nu = nu

        self._rng = default_rng()

        self._extended_vars = [0.0] * self._data_size 
        self._tau2 = 1
        self._mu = sum(data) / self._data_size
        self._alpha2 = 1

        self._update_extended_vars()
        self._results_mu = [0.0] * self._sample_size
        self._results_sigma2 = [0.0] * self._sample_size

        
    def _sampleScaledInvChiSquare(self, ni, scale):
        x = self._rng.chisquare(ni)
        return ni * scale / x

    def _update_mu(self):
        variance = 0.0
        expected_value = 0.0
        for i in range(self._data_size):
            variance += 1.0 / self._extended_vars[i]
            expected_value += self._data[i] / self._extended_vars[i]

        variance /= self._alpha2
        expected_value /= self._alpha2
        variance = 1.0 / variance
        expected_value = expected_value * variance
        self._mu = self._rng.normal(expected_value, math.sqrt(variance))
    
    def _update_tau2(self):
        x = 0.0
        for i in range(self._data_size):
            x += 1.0 / self._extended_vars[i]
        
        self._tau2 = self._rng.gamma(self._data_size * self._nu / 2, 1.0 / (self._nu * x))


    def _update_alpha2(self):
        x = 0.0
        for i in range(self._data_size):
            x += (self._data[i]  - self._mu) * (self._data[i]  - self._mu) / self._extended_vars[i]
        x /= self._data_size
        self._alpha2 = self._sampleScaledInvChiSquare(self._data_size, x)

    def _update_extended_vars(self):
        for i in range(self._data_size):
            x = ((self._data[i] - self._mu) * (self._data[i] - self._mu) / self._alpha2)
            self._extended_vars[i] = self._sampleScaledInvChiSquare(self._nu + 1, self._nu * self._tau2 + x / self._nu)

    def run(self):
        for _ in range(self._burn_in):
            self._update_extended_vars()
            self._update_alpha2()
            self._update_mu()
            self._update_tau2

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



sss = Model([1, 2, 3, 4, 5], 1000, 1000, 4)
sss.run()