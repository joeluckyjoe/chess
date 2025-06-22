#
# Copyright (c) 2016, Johannes L.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of bayesian-changepoint-detection nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
from scipy import stats


class Bocd:
    def __init__(self, model, hazard_lambda=200):
        self.model = model
        self.hazard_lambda = hazard_lambda
        self.reset()

    def reset(self):
        self.t = 0
        self.R = np.array([1])
        self.model.reset()

    def update(self, x):
        # 1. Evaluate predictive probabilities.
        pred_probs = self.model.pdf(x)

        # 2. Calculate growth probabilities.
        growth_probs = pred_probs * self.R * (1 - 1 / self.hazard_lambda)

        # 3. Calculate changepoint probabilities.
        cp_prob = np.sum(pred_probs * self.R * 1 / self.hazard_lambda)

        # 4. Calculate evidence
        new_R = np.append(cp_prob, growth_probs)

        # 5. Determine run-length distribution.
        self.R = new_R / np.sum(new_R)

        # 6. Update sufficient statistics.
        self.model.update(x)
        self.t += 1

        return self.R


class NormalUnknownMean:
    def __init__(self, mu=0, kappa=1, alpha=1, beta=1):
        self.mu0 = mu
        self.kappa0 = kappa
        self.alpha0 = alpha
        self.beta0 = beta
        self.reset()

    def reset(self):
        self.mu = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])

    def pdf(self, x):
        return stats.t.pdf(x,
                           2 * self.alpha,
                           self.mu,
                           np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa)))

    def update(self, x):
        muT = (self.kappa * self.mu + x) / (self.kappa + 1)
        kappaT = self.kappa + 1
        alphaT = self.alpha + 0.5
        betaT = self.beta + (self.kappa * (x - self.mu)**2) / (2 * (self.kappa + 1))

        self.mu = np.append(self.mu0, muT)
        self.kappa = np.append(self.kappa0, kappaT)
        self.alpha = np.append(self.alpha0, alphaT)
        self.beta = np.append(self.beta0, betaT)