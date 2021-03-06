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

from collections import Counter

import numpy as np
import pytest
import ngram_lm

from pydrobert.torch.util import parse_arpa_lm


KK_DIR = os.path.join(os.path.dirname(__file__), "kneser_ney")
KATZ_DIR = os.path.join(os.path.dirname(__file__), "katz")
RE_DIR = os.path.join(os.path.dirname(__file__), "re_pruning")


@pytest.fixture
def katz_ngram_counts():
    # CMU derives lower-order n-grams from higher-order n-grams by summing
    # up prefixes, e.g.:
    # C(a, b) = sum_c C(a, b, c)
    # This is not sufficient to recover the actual counts since we lose
    # sentence-final (n-1)-grams, but whatever.
    ngram_counts = [dict(), dict(), dict()]
    with open(os.path.join(KATZ_DIR, "republic.wngram")) as f:
        for line in f:
            w1, w2, w3, n = line.strip().split()
            n = int(n)
            ngram_counts[0][w1] = ngram_counts[0].get(w1, 0) + n
            ngram_counts[1][(w1, w2)] = ngram_counts[1].get((w1, w2), 0) + n
            ngram_counts[2][(w1, w2, w3)] = ngram_counts[1].get((w1, w2, w3), 0) + n
    return ngram_counts


def test_katz_discounts(katz_ngram_counts):
    exp_dcs = []
    discount_pattern = re.compile(r"^(\d+)-gram discounting ratios :(.*)$")
    with open(os.path.join(KATZ_DIR, "republic.arpa")) as f:
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
            katz_ngram_counts[order], len(exp_dc)
        )
        act_dc = act_dc[1:]  # exclude r=0
        act_dc -= np.log10(np.arange(1, len(act_dc) + 1))
        act_dc = 10 ** act_dc
        assert np.allclose(act_dc[len(exp_dc) :], 1.0)
        # discount ratios have been rounded to precision 2 in arpa file
        assert np.allclose(act_dc[: len(exp_dc)], exp_dc, atol=1e-2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_katz_backoff(katz_ngram_counts):
    exp_prob_list = parse_arpa_lm(os.path.join(KATZ_DIR, "republic.arpa"))
    act_prob_list = ngram_lm.ngram_counts_to_prob_list_katz_backoff(
        katz_ngram_counts, _cmu_hacks=True
    )
    for order in range(3):
        exp_probs, act_probs = exp_prob_list[order], act_prob_list[order]
        assert set(exp_probs) == set(act_probs)
        for ngram, exp_prob in exp_probs.items():
            act_prob = act_probs[ngram]
            assert np.allclose(exp_prob, act_prob, atol=1e-4), ngram


@pytest.fixture
def kneser_ney_ngram_counts():
    ngram_counts = [Counter({"<unk>": 0, "<s>": 0, "</s>": 0}), Counter(), Counter()]
    # Note: KenLM doesn't count <s> or </s> in unigrams
    with open(os.path.join(KK_DIR, "republic.txt")) as f:
        for line in f:
            s = tuple(["<s>"] + line.strip().split() + ["</s>"])
            for order, counts in enumerate(ngram_counts):
                counts.update(
                    s[i : i + order + 1] if order else s[i]
                    for i in range(
                        max(1 - order, 0), len(s) - order - max(1 - order, 0)
                    )
                )
    return ngram_counts


def test_kneser_ney_unpruned(kneser_ney_ngram_counts):
    exp_prob_list = parse_arpa_lm(os.path.join(KK_DIR, "republic.arpa"))
    exp_vocab = set(exp_prob_list[0])
    count_vocab = set(kneser_ney_ngram_counts[0])
    assert exp_vocab == count_vocab, exp_vocab - count_vocab
    act_prob_list = ngram_lm.ngram_counts_to_prob_list_kneser_ney(
        kneser_ney_ngram_counts
    )
    # While it doesn't really matter - you shouldn't ever need to predict <s> -
    # it's strange that KenLM sets the unigram probability of <s> to 1
    act_prob_list[0]["<s>"] = (0.0, act_prob_list[0]["<s>"][1])
    for order in range(3):
        exp_probs, act_probs = exp_prob_list[order], act_prob_list[order]
        assert set(exp_probs) == set(act_probs)
        for ngram, exp_prob in exp_probs.items():
            act_prob = act_probs[ngram]
            assert np.allclose(exp_prob, act_prob, atol=1e-4), ngram


def test_kneser_ney_pruning(kneser_ney_ngram_counts):
    exp_prob_list = parse_arpa_lm(os.path.join(KK_DIR, "republic.pruned.arpa"))
    # prune out bigrams and trigrams with count 1:
    pruned = set()
    for counts in kneser_ney_ngram_counts[1:]:
        pruned.update(k for (k, v) in counts.items() if v == 1)
    act_prob_list = ngram_lm.ngram_counts_to_prob_list_kneser_ney(
        kneser_ney_ngram_counts, to_prune=pruned
    )
    act_prob_list[0]["<s>"] = (0.0, act_prob_list[0]["<s>"][1])
    for order in range(3):
        exp_probs, act_probs = exp_prob_list[order], act_prob_list[order]
        for ngram, exp_prob in exp_probs.items():
            act_prob = act_probs[ngram]
            assert np.allclose(exp_prob, act_prob, atol=1e-4), ngram


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_relative_entropy_pruning():
    with open(os.path.join(RE_DIR, "republic.arpa")) as f:
        unpruned_prob_list = parse_arpa_lm(f)
    lm = ngram_lm.BackoffNGramLM(unpruned_prob_list)
    lm.relative_entropy_pruning(1e-5, _srilm_hacks=True)
    act_prob_list = lm.to_prob_list()
    del lm, unpruned_prob_list
    exp_prob_list = parse_arpa_lm(os.path.join(RE_DIR, "republic.pruned.arpa"))
    for order in range(3):
        exp_probs, act_probs = exp_prob_list[order], act_prob_list[order]
        assert set(exp_probs) == set(act_probs)
        for ngram, exp_prob in exp_probs.items():
            act_prob = act_probs[ngram]
            assert np.allclose(exp_prob, act_prob, atol=1e-4), ngram


LIPSUM = """\
Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium
doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore
veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim
ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia
consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque
porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur,
adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et
dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis
nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex
ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea
voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem
eum fugiat quo voluptas nulla pariatur?!!!.?!
"""


def test_text_to_sents():
    sents = ngram_lm.text_to_sents(LIPSUM)
    assert len(sents) == 5
    assert sents[2][3:6] == ("EST", "QUI", "DOLOREM")
    assert sents[4][-4:] == ("QUO", "VOLUPTAS", "NULLA", "PARIATUR")


def test_sents_to_ngram_counts():
    sent = "what am I chopped liver".split()
    sents = [sent]
    for s in range(1, len(sent)):
        sents.append(sent[s:] + sent[:s])
    ngram_counts = ngram_lm.sents_to_ngram_counts(sents, len(sent) + 2)
    assert not ngram_counts[0]["<S>"]
    assert ngram_counts[0]["</S>"] == len(sents)
    for word in sent:
        assert ngram_counts[0][word] == len(sents)
        assert ngram_counts[1][("<S>", word)] == 1
        assert ngram_counts[1][(word, "</S>")] == 1
    assert len(ngram_counts[-1]) == len(sents)
    assert all(ngram_counts[-1][("<S>",) + tuple(s) + ("</S>",)] == 1 for s in sents)
    ngram_counts_b = ngram_lm.sents_to_ngram_counts(
        sents, len(sent) + 2, count_unigram_sos=True
    )
    assert ngram_counts_b[0]["<S>"] == len(sents)
    ngram_counts_b[0]["<S>"] = 0
    assert ngram_counts == ngram_counts_b


def test_prune_by_threshold():
    sents = ngram_lm.text_to_sents(LIPSUM)
    ngram_counts = ngram_lm.sents_to_ngram_counts(sents, 3)
    ngram_counts[0]["<UNK>"] = 0
    prob_list = ngram_lm.ngram_counts_to_prob_list_mle(ngram_counts)
    will_keep = set(prob_list[0])
    will_discard = set()
    for dict_ in prob_list[:0:-1]:
        for ngram, lprob in dict_.items():
            if isinstance(lprob, tuple):
                lprob = lprob[0]
            if lprob <= -0.1 and not any(
                high[: len(ngram)] == ngram for high in will_keep
            ):
                will_discard.add(ngram)
            else:
                will_keep.add(ngram)
    lm = ngram_lm.BackoffNGramLM(prob_list)
    lm.prune_by_threshold(-0.1)
    pruned_prob_list = lm.to_prob_list()
    for pruned_dict, dict_ in zip(pruned_prob_list, prob_list):
        kept = set(pruned_dict)
        assert not (kept - will_keep)
        assert not (kept & will_discard)
        for ngram in kept:
            v1, v2 = pruned_dict[ngram], dict_[ngram]
            if isinstance(v1, tuple):
                v1, v2 = v1[0], v2[0]
            assert np.isclose(v1, v2)
    lm.prune_by_threshold(0.0)
    pruned_prob_list = lm.to_prob_list()
    assert len(pruned_prob_list) == 1
    assert len(pruned_prob_list[0]) == len(ngram_counts[0])


def test_prune_by_name():
    eps_lprob = -30
    sents = ngram_lm.text_to_sents(LIPSUM)
    ngram_counts = ngram_lm.sents_to_ngram_counts(sents, 3)
    ngram_counts[0]["<UNK>"] = 0
    prob_list = ngram_lm.ngram_counts_to_prob_list_mle(ngram_counts)
    to_prune = set(ngram_counts[0])
    to_prune.remove("QUI")
    to_prune |= set(k[:-1] for k in ngram_counts[-1])
    lm = ngram_lm.BackoffNGramLM(prob_list)
    lm.prune_by_name(to_prune, eps_lprob=eps_lprob)
    pruned_prob_list = lm.to_prob_list()
    assert len(pruned_prob_list) == len(prob_list)
    for pruned_dict_, dict_ in zip(pruned_prob_list, prob_list):
        assert set(pruned_dict_) == set(dict_)
    assert all(
        pruned_prob_list[1][k][0] == eps_lprob
        for k in set(pruned_prob_list[1]) & to_prune
    )
    assert not any(
        pruned_prob_list[1][k][0] == eps_lprob
        for k in set(pruned_prob_list[1]) - to_prune
    )
    assert all(
        k == "QUI" or v[0] == eps_lprob for (k, v) in pruned_prob_list[0].items()
    )
    assert np.isclose(pruned_prob_list[0]["QUI"][0], 0.0)
    to_prune = (set(ngram_counts[1]) | set(ngram_counts[2])) - to_prune
    lm.prune_by_name(to_prune, eps_lprob=eps_lprob)
    pruned_prob_list = lm.to_prob_list()
    assert len(pruned_prob_list) == 1
    assert all(k == "QUI" or v == eps_lprob for (k, v) in pruned_prob_list[0].items())
    assert np.isclose(pruned_prob_list[0]["QUI"], 0.0)
