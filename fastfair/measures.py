# Copyright 2023 Gradient Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementations of the fairness measures."""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, bernoulli

#
# Ratio measures
#

# General binary class ratio calculation
def ratio(p, q):
    assert p.shape[1] == 2
    assert q.shape[1] == 2

    r_i = p[:, 1] / p[:, 0] * q[:, 0] / q[:, 1]
    r = np.mean(r_i)

    return r

#
# Conditional Mutual Information Measures
#

# Conditional Mutual Information, probabilistic
def cmi_prob(p, q, normalise=True):
    N = p.shape[0]
    mi_n = p * (np.log(p) - np.log(q))
    mi = mi_n.sum() / N
    mi = mi / cent_prob(q) if normalise else mi
    return mi


# Conditional Mutual Information, empirical
def cmi_emp(p, q, a, normalise=True):
    N = len(a)
    ind = np.arange(N)
    p_emp, q_emp = p[ind, a], q[ind, a]
    if normalise:
        mi = 1. - np.mean(np.log(p_emp)) / np.mean(np.log(q_emp))
    else:
        mi = np.mean(np.log(p_emp) - np.log(q_emp))
    return mi


# Empirical or Probabilistic approaches depending on args
def cmi(p, q, a=None, normalise=True):
    if a is None:
        mi = cmi_prob(p, q, normalise)
    else:
        mi = cmi_emp(p, q, a, normalise)
    return mi


# Conditional entropy normaliser, probabilistic
def cent_prob(p):
    N = p.shape[0]
    h_i = - p * np.log(p)
    h = h_i.sum() / N
    return h