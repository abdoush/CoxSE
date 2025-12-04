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

def dict2df(data, add_true=True, add_censor_covs=False):
    """Make a pd.DataFrame from the dict obtained when simulating.

    Arguments:
        data {dict} -- Dict from simulation.

    Keyword Arguments:
        add_true {bool} -- If we should include the true duration and censoring times
            (default: {True})
        add_censor_covs {bool} -- If we should include the censor covariates as covariates.
            (default: {False})

    Returns:
        pd.DataFrame -- A DataFrame
    """
    covs = data['covs']
    if add_censor_covs:
        covs = np.concatenate([covs, data['censor_covs']], axis=1)
    df = (pd.DataFrame(covs, columns=[f"x{i}" for i in range(covs.shape[1])])
          .assign(duration=data['durations'].astype('float32'),
                  event=data['events'].astype('float32')))
    if add_true:
        df = df.assign(duration_true=data['durations_true'].astype('float32'),
                       event_true=data['events_true'].astype('float32'),
                       censoring_true=data['censor_durations'].astype('float32'))
    return df


class _SimBase:
    def simulate(self, n, p=3, surv_df=False):
        """Simulate dataset of size `n`.
        
        Arguments:
            n {int} -- Number of simulations
        
        Keyword Arguments:
            surv_df {bool} -- If a dataframe containing the survival function should be returned.
                (default: {False})
        
        Returns:
            [dict] -- A dictionary with the results.
        """
        raise NotImplementedError

    def surv_df(self, *args):
        """Returns a data frame containing the survival function.
        """
        raise NotImplementedError
