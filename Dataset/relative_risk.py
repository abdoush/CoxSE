# Adapted from the pycox library:
# https://github.com/havakv/pycox
#
# This file contains portions of code from the original project, which is
# licensed under the BSD 2-Clause License. The code has been modified by
# Abdallah Alabdallah in 2025 to fit the needs of our project.
#
# ----------------------------------------------------------------------
# BSD 2-Clause License
#
# Copyright (c) 2018, Haavard Kvamme
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd

from .base import dict2df, _SimBase


class _SimStudyRelativeRisk(_SimBase):
    '''Abstract class for simulation relative risk survival data,
    with constant baseline, and constant censoring distribution

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
        c0: Constant censoring distribution
    '''

    def __init__(self, h0, right_c=30., c0=30., surv_grid=None, x_seed=12345, t_seed=12345):
        self.h0 = h0
        self.right_c = right_c
        self.c0 = c0
        self.surv_grid = surv_grid
        self.rng1 = np.random.default_rng(x_seed)
        self.rng2 = np.random.default_rng(t_seed)

    def simulate(self, n, m=3, surv_df=False):
        covs = self.sample_covs(n, m).astype('float32')
        v = self.rng2.exponential(size=n)
        t = self.inv_cum_hazard(v, covs)
        c = self.c0 * self.rng2.exponential(size=n)
        tt = t.copy()
        tt[c < t] = c[c < t]
        tt[tt > self.right_c] = self.right_c
        d = tt == t
        surv_df = self.surv_df(covs, self.surv_grid) if surv_df else None
        # censor_surv_df = NotImplemented if censor_df else None
        return dict(covs=covs, durations=tt, events=d, surv_df=surv_df, durations_true=t,
                    events_true=np.ones_like(t), censor_durations=c,
                    censor_events=np.ones_like(c))

    # @staticmethod
    def sample_covs(self, n, m=3):
        raise NotImplementedError

    def inv_cum_hazard(self, v, covs):
        '''The inverse of the cumulative hazard.'''
        raise NotImplementedError

    def cum_hazard(self, t, covs):
        '''The the cumulative hazard function.'''
        raise NotImplementedError

    def survival_func(self, t, covs):
        '''Returns the survival function.'''
        return np.exp(-self.cum_hazard(t, covs))

    def survival_grid_single(self, covs, t=None):
        covs = covs.reshape(1, -1)
        if t is None:
            t = np.arange(0, 31, 0.5)
        return pd.Series(self.survival_func(t, covs), index=t)

    def surv_df(self, covs, t=None):
        if t is None:
            t = np.linspace(0, 30, 100)
        s = [self.survival_grid_single(xx, t) for xx in covs]
        return pd.concat(s, axis=1)

    @staticmethod
    def dict2df(data, add_true=True):
        """Make a pd.DataFrame from the dict obtained when simulating.

        Arguments:
            data {dict} -- Dict from simulation.

        Keyword Arguments:
            add_true {bool} -- If we should include the true duration and censoring times
                (default: {True})

        Returns:
            pd.DataFrame -- A DataFrame
        """
        return dict2df(data, add_true)


class SimStudyLinearPH(_SimStudyRelativeRisk):
    '''Survival simulations study for linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is linear.

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''

    def __init__(self, h0=0.1, right_c=30., c0=30., surv_grid=None, x_seed=12345, t_seed=12345):
        super().__init__(h0, right_c, c0, surv_grid, x_seed=x_seed, t_seed=t_seed)

    # @staticmethod
    def sample_covs(self, n, m=3):
        return self.rng1.uniform(-1, 1, size=(n, m))

    # @staticmethod
    def g(self, covs):
        x = covs
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        print(f'0.44 * x0 ** 2 + 0.66 * x1 ** 2 + 0.88 * x2')
        return 0.44 * x0 + 0.66 * x1 + 0.88 * x2

    def inv_cum_hazard(self, v, covs):
        '''The inverse of the cumulative hazard.'''
        return v / (self.h0 * np.exp(self.g(covs)))

    def cum_hazard(self, t, covs):
        '''The the cumulative hazard function.'''
        return self.h0 * t * np.exp(self.g(covs))


class SimStudyNonLinearPH(SimStudyLinearPH):
    '''Survival simulations study for non-linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is non-linear.

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''

    def __init__(self, intaract_w=0, h0=0.02, right_c=30., c0=30., surv_grid=None, x_seed=12345, t_seed=12345):
        super().__init__(h0, right_c, c0, surv_grid, x_seed, t_seed)
        self.intaract_w = intaract_w

    # @staticmethod
    def g(self, covs):
        x = covs
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        # beta = 2/3
        linear = SimStudyLinearPH.g(self, x)
        nonlinear = self.intaract_w * (x0 ** 2 + x2 ** 2 + x0 * x1 + x1 * x2 + x1 * x2)
        return linear + nonlinear


class SimStudyNonLinearNonPH(SimStudyNonLinearPH):
    '''Survival simulations study for non-linear non-prop. hazard model.
        h(t | x) = h0 * exp[g(t, x)],
        with constant h_0, and g(t, x) = a(x) + b(x)*t.

        Cumulative hazard:
        H(t | x) = h0 / b(x) * exp[a(x)] * (exp[b(x) * t] - 1)
        Inverse:
        H^{-1}(v, x) = 1/b(x) log{1 +  v * b(x) / h0 exp[-a(x)]}

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''

    def __init__(self, h0=0.02, right_c=30., c0=30., surv_grid=None, x_seed=12345, t_seed=12345):
        super().__init__(h0, right_c, c0, surv_grid, x_seed, t_seed)

    @staticmethod
    def a(x):
        _, _, x2 = x[:, 0], x[:, 1], x[:, 2]
        return np.sign(x2) + SimStudyNonLinearPH.g(x)

    @staticmethod
    def b(x):
        x0, x1, _ = x[:, 0], x[:, 1], x[:, 2]
        return np.abs(0.2 * (x0 + x1) + 0.5 * x0 * x1)

    # @staticmethod
    def g(self, t, covs):
        x = covs
        return SimStudyNonLinearNonPH.a(x) + SimStudyNonLinearNonPH.b(x) * t

    def inv_cum_hazard(self, v, covs):
        x = covs
        return 1 / self.b(x) * np.log(1 + v * self.b(x) / self.h0 * np.exp(-self.a(x)))

    def cum_hazard(self, t, covs):
        x = covs
        return self.h0 / self.b(x) * np.exp(self.a(x)) * (np.exp(self.b(x) * t) - 1)
