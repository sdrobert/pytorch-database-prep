# Copyright 2019 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import numpy as np
import pytest
import ngram_lm

from pydrobert.torch.util import parse_arpa_lm


KATZ_DIR = os.path.join(os.path.dirname(__file__), 'katz')

@pytest.fixture
def katz_ngram_counts():
    # CMU derives lower-order n-grams from higher-order n-grams by summing
    # up prefixes, e.g.:
    # C(a, b) = sum_c C(a, b, c)
    # This is not sufficient to recover the actual counts since we lose
    # sentence-final (n-1)-grams, but whatever.
    ngram_counts = [dict(), dict(), dict()]
    with open(os.path.join(KATZ_DIR, 'republic.wngram')) as f:
        for line in f:
            w1, w2, w3, n = line.strip().split()
            n = int(n)
            ngram_counts[0][w1] = ngram_counts[0].get(w1, 0) + n
            ngram_counts[1][(w1, w2)] = ngram_counts[1].get((w1, w2), 0) + n
            ngram_counts[2][(w1, w2, w3)] = (
                ngram_counts[1].get((w1, w2, w3), 0) + n
            )
    return ngram_counts


def test_katz_discounts(katz_ngram_counts):
    exp_dcs = []
    discount_pattern = re.compile(r'^(\d+)-gram discounting ratios :(.*)$')
    with open(os.path.join(KATZ_DIR, 'republic.arpa')) as f:
        for line in f:
            m = discount_pattern.match(line.strip())
            if m is not None:
                n, dcs = m.groups()
                n = int(n)
                if n == 1:
                    exp_dcs.append([])
                    continue
                exp_dcs.append([float(x) for x in dcs.strip().split()])
                if n == 3:
                    break
    for order in range(1, 3):  # n=2 and n=3
        exp_dc = np.array(exp_dcs[order])
        act_dc = ngram_lm._get_katz_discounted_counts(
            katz_ngram_counts[order], len(exp_dc))
        act_dc = act_dc[1:]  # exclude r=0
        act_dc -= np.log10(np.arange(1, len(act_dc) + 1))
        act_dc = 10 ** act_dc
        assert np.allclose(act_dc[len(exp_dc):], 1.)
        # discount ratios have been rounded to precision 2 in arpa file
        assert np.allclose(act_dc[:len(exp_dc)], exp_dc, atol=1e-2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_katz_backoff(katz_ngram_counts):
    exp_prob_list = parse_arpa_lm(os.path.join(KATZ_DIR, 'republic.arpa'))
    act_prob_list = ngram_lm.ngram_counts_to_prob_list_katz_backoff(
        katz_ngram_counts, _cmu_hacks=True)
    for order in range(3):
        exp_probs, act_probs = exp_prob_list[order], act_prob_list[order]
        assert set(exp_probs) == set(act_probs)
        for ngram, exp_prob in exp_probs.items():
            act_prob = act_probs[ngram]
            assert np.allclose(exp_prob, act_prob, atol=1e-2), ngram