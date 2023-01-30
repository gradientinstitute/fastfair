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
"""Dataset creation and loading utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import KFold


class Dataset:
    def __init__(self, random_state, settings=None):
        # make sure you initialise, self.X, self.y, self.a and
        # self.random_state
        raise NotImplementedError

    def splits(self, n_splits):
        cv = KFold(n_splits=n_splits, shuffle=True,
                   random_state=self.random_state)
        splits = list(cv.split(self.a))
        return splits

    def get(self):
        return self.X, self.y, self.a


@dataclass
class CommunitiesSettings:
    pass


communities_settings = CommunitiesSettings()


class Communities(Dataset):

    def __init__(self, random_state, settings=None):
        self.random_state = random_state
        s = communities_settings if settings is None else settings
        
        # Load data
        datafname = Path("../data/communities.data")
        data = pd.read_csv(datafname, index_col=None, header=None)

        # Get column names
        colsfname = Path("../data/communities.names")
        columns = []
        with colsfname.open() as f:
            while line := f.readline():
                if line.startswith("@attribute"):
                    parts = line.split()
                    columns.append(parts[1])

        data.columns = columns
        data = data.iloc[:, 5:] # The first 5 cols are non-predictive
        
        # Remove columns with missing values
        data = data.select_dtypes(exclude=['object'])

        # Apply Simon's processing
        # Identify a protected class
        racepctblack = data['racepctblack']
        racepctother = data[['racePctWhite', 'racePctAsian', 'racePctHisp']].sum(axis=1)
        # This version seems closer to the number of protect instances in the convex fairness paper.
        protected_ind = racepctblack > 0.5

        # This one seems more logical though.
        # TODO implement as option and test
        alternative_protected_ind = racepctblack > racepctother

        a = protected_ind.values.astype(int)

        data['protected'] = a
        y = data['ViolentCrimesPerPop'].values
        X = data.drop(['ViolentCrimesPerPop', 'protected'], axis=1).values
        self.a = a
        self.y = y
        self.X = X


@dataclass
class ToySettings:
    N: int
    P: float
    P_CORRUPT_1: float
    P_CORRUPT_0: float
    SIGMA: float
    MU: float
    SLOPE_1: float
    D: int


toy_settings = ToySettings(
    N=1000,
    P=0.10,
    P_CORRUPT_1=0.4,
    P_CORRUPT_0=0.2,
    SIGMA=3,
    MU=3,
    SLOPE_1=2.0,
    D=10
)


class Toy(Dataset):

    def __init__(self, random_state, settings=None):
        self.random_state = random_state
        s = toy_settings if settings is None else settings

        rand = np.random.RandomState(random_state)
        x = rand.uniform(-10, 10, size=(s.N,))
        a = rand.binomial(n=1, p=s.P, size=(s.N,))
        N1 = sum(a)
        N0 = s.N - N1

        y = np.zeros_like(x)
        y[a == 1] = x[a == 1] * s.SLOPE_1 + rand.normal(scale=s.SIGMA,
                                                        size=(N1,)) + s.MU
        y[a == 0] = x[a == 0] + rand.normal(scale=s.SIGMA, size=(N0,)) - s.MU

        corrupt0 = rand.binomial(n=1, p=s.P_CORRUPT_0, size=(N0,))
        corrupt1 = rand.binomial(n=1, p=1 - s.P_CORRUPT_1, size=(N1,))
        a_corr = np.zeros_like(a)
        a_corr[a == 0] = corrupt0
        a_corr[a == 1] = corrupt1

        w_proj = rand.normal(size=(2, s.D))
        X = np.dot(np.vstack((x, a_corr)).T, w_proj)
        self.a = a
        self.y = y
        self.X = X