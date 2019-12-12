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

import warnings
import sys
import locale

from collections import OrderedDict, Counter
from itertools import product

import numpy as np

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'BackoffNGramLM',
    'write_arpa',
    'ngram_counts_to_prob_list_mle',
    'ngram_counts_to_prob_list_add_k',
    'ngram_counts_to_prob_list_simple_good_turing',
]

locale.setlocale(locale.LC_ALL, 'C')
warnings.simplefilter("error", RuntimeWarning)


class BackoffNGramLM(object):
    '''A backoff NGram language model, stored as a trie

    This class is intended for two things: one, to prune backoff language
    models, and two, to calculate the perplexity of a language model on a
    corpus. It is very inefficient.

    Relative entropy pruning based on [stolcke2000]_

    Parameters
    ----------
    prob_list : sequence
        See :mod:`pydrobert.torch.util.parse_arpa_lm`
    sos : str, optional
        The start-of-sequence symbol. When calculating the probability of a
        sequence, :math:`P(sos) = 1` when `sos` starts the sequence. Defaults
        to ``'<S>'`` if that symbol is in the vocabulary, otherwise
        ``'<s>'``
    eos : str, optional
        The end-of-sequence symbol. This symbol is expected to terminate each
        sequence when calculating sequence or corpus perplexity. Defaults to
        ``</S>`` if that symbol is in the vocabulary, otherwise ``</s>``
    unk : str, optional
        The out-of-vocabulary symbol. If a unigram probability does not exist
        for a token, the token is replaced with this symbol. Defaults to
        ``'<UNK>'`` if that symbol is in the vocabulary, otherwise
        ``'<unk>'``

    References
    ----------
    .. [stolcke2000] A. Stolcke "Entropy-based Pruning of Backoff Language
       Models," ArXiv ePrint, 2000
    '''

    def __init__(self, prob_list, sos=None, eos=None, unk=None):
        self.trie = self.TrieNode(0.0, 0.0)
        self.vocab = set()
        if not len(prob_list) or not len(prob_list[0]):
            raise ValueError('prob_list must contain (all) unigrams')
        for order, dict_ in enumerate(prob_list):
            is_first = not order
            is_last = order == len(prob_list) - 1
            for context, value in dict_.items():
                if is_first:
                    self.vocab.add(context)
                    context = (context,)
                if is_last:
                    lprob, bo = value, 0.0
                else:
                    lprob, bo = value
                self.trie.add_child(context, lprob, bo)
        if sos is None:
            if '<S>' in self.vocab:
                sos = '<S>'
            else:
                sos = '<s>'
        if sos not in self.vocab:
            raise ValueError(
                'start-of-sequence symbol "{}" does not have unigram '
                'entry.'.format(sos))
        self.trie.sos = sos
        if eos is None:
            if '</S>' in self.vocab:
                eos = '</S>'
            else:
                eos = '</s>'
        if eos not in self.vocab:
            raise ValueError(
                'end-of-sequence symbol "{}" does not have unigram '
                'entry.'.format(eos))
        self.eos = eos
        if unk is None:
            if '<UNK>' in self.vocab:
                unk = '<UNK>'
            else:
                unk = '<unk>'
        if unk in self.vocab:
            self.unk = unk
        else:
            warnings.warn(
                'out-of-vocabulary symbol "{}" does not have unigram count. '
                'Out-of-vocabulary tokens will raise an error'.formart(unk))
            self.unk = None
        assert self.trie.depth == len(prob_list)

    class TrieNode(object):

        def __init__(self, lprob, bo):
            self.lprob = lprob
            self.bo = bo
            self.children = OrderedDict()
            self.depth = 0
            self.sos = None

        def add_child(self, context, lprob, bo):
            assert len(context)
            next_, rest = context[0], context[1:]
            child = self.children.setdefault(
                next_, type(self)(None, 0.0))
            if rest:
                child.add_child(rest, lprob, bo)
            else:
                child.lprob = lprob
                child.bo = bo
            self.depth = max(self.depth, child.depth + 1)

        def conditional(self, context):
            assert context and self.depth
            context = context[-self.depth:]
            cond = 0.0
            while True:
                assert len(context)
                cur_node = self
                idx = 0
                while idx < len(context):
                    token = context[idx]
                    next_node = cur_node.children.get(token, None)
                    if next_node is None:
                        if idx == len(context) - 1:
                            cond += cur_node.bo
                        break
                    else:
                        cur_node = next_node
                    idx += 1
                if idx == len(context):
                    return cond + cur_node.lprob
                assert len(context) > 1  # all unigrams should exist
                context = context[1:]
            # should never get here

        def log_prob(self, context):
            joint = 0.0
            for prefix in range(
                    2 if context[0] == self.sos else 1, len(context) + 1):
                joint += self.conditional(context[:prefix])
            return joint

        def _gather_nodes_by_depth(self, order):
            nodes = [(tuple(), self)]
            nodes_by_depth = []
            for _ in range(order):
                last, nodes = nodes, []
                nodes_by_depth.append(nodes)
                for ctx, parent in last:
                    nodes.extend(
                        (ctx + (k,), v) for (k, v) in parent.children.items())
            return nodes_by_depth

        def relative_entropy_pruning(self, threshold, eps=1e-8):
            nodes_by_depth = self._gather_nodes_by_depth(self.depth - 1)
            base_10 = np.log(10)
            while nodes_by_depth:
                nodes = nodes_by_depth.pop()  # highest order first
                for h, node in nodes:
                    if not len(node.children):
                        node.bo = 0.0  # no need for a backoff
                        continue
                    num = 0.0
                    denom = 0.0
                    logP_w_given_hprimes = []  # log P(w | h')
                    P_h = 10. ** self.log_prob(h)  # log P(h)
                    for w, child in node.children.items():
                        assert child.lprob is not None
                        num -= 10. ** child.lprob
                        logP_w_given_hprime = self.conditional(h[1:] + (w,))
                        logP_w_given_hprimes.append(logP_w_given_hprime)
                        denom -= 10. ** logP_w_given_hprime
                    if num + 1 < eps or denom + 1 < eps:
                        warnings.warn(
                            'Malformed backoff weight for context {}. Leaving '
                            'as is'.format(h))
                        continue
                    # alpha = (1 + num) / (1 + denom)
                    log_alpha = (np.log1p(num) - np.log1p(denom)) / base_10
                    if abs(log_alpha - node.bo) > 1e-2:
                        warnings.warn(
                            'Calculated backoff ({}) differs from stored '
                            'backoff ({}) for context {}'
                            ''.format(log_alpha, node.bo, h))
                    num_change = 0.
                    denom_change = 0.
                    for idx, w in enumerate(tuple(node.children)):
                        child = node.children[w]
                        logP_w_given_h = child.lprob
                        P_w_given_h = 10 ** child.lprob
                        logP_w_given_hprime = logP_w_given_hprimes[idx]
                        P_w_given_hprime = 10 ** logP_w_given_hprime
                        new_num = num + P_w_given_h
                        new_denom = denom + P_w_given_hprime
                        log_alphaprime = np.log1p(new_num)
                        log_alphaprime -= np.log1p(new_denom)
                        log_alphaprime /= base_10
                        KL = -P_h * (
                            P_w_given_h * (
                                logP_w_given_hprime + log_alphaprime
                                - logP_w_given_h
                            ) +
                            (log_alphaprime - log_alpha) * (1. + num)
                        )
                        delta_perplexity = 10. ** KL - 1
                        if delta_perplexity < threshold:
                            node.children.pop(w)
                            # we update the backoff weights only after we've
                            # pruned all the children, as per paper instruction
                            num_change += P_w_given_h
                            denom_change += P_w_given_hprime
                    if len(node.children):
                        node.bo = np.log1p(num + num_change)
                        node.bo -= np.log1p(denom + denom_change)
                        node.bo /= base_10
                    else:
                        node.bo = 0.0
            # recalculate depth in case it's changed
            self.depth = -1
            cur_nodes = (self,)
            while cur_nodes:
                self.depth += 1
                next_nodes = []
                for parent in cur_nodes:
                    next_nodes.extend(parent.children.values())
                cur_nodes = next_nodes
            assert self.depth >= 1

        def to_ngram_list(self):
            nodes_by_depth = self._gather_nodes_by_depth(self.depth)
            prob_list = []
            for order, nodes in enumerate(nodes_by_depth):
                is_first = not order
                is_last = order == self.depth - 1
                dict_ = dict()
                for context, node in nodes:
                    if is_first:
                        context = context[0]
                    if is_last:
                        assert not node.bo
                        value = node.lprob
                    else:
                        value = (node.lprob, node.bo)
                    dict_[context] = value
                prob_list.append(dict_)
            return prob_list

    def conditional(self, context):
        r'''Return the log probability of the last word in the context

        `context` is a non-empty sequence of tokens ``[w_1, w_2, ...,
        w_N]``. This method determines

        .. math::

            \log Pr(w_N | w_{N-1}, w_{N-2}, ... w_{N-C})

        Where ``C`` is this model's maximum n-gram size. If an exact entry
        cannot be found, the model backs off to a shorter context.

        Parameters
        ----------
        context : sequence

        Returns
        -------
        cond : float or :obj:`None`
        '''
        if self.unk is None:
            context = tuple(context)
        else:
            context = tuple(
                t if t in self.vocab else self.unk for t in context)
        if not len(context):
            raise ValueError('context must have at least one token')
        return self.trie.conditional(context)

    def log_prob(self, context):
        r'''Return the log probability of the whole context

        `context` is a non-empty sequence of tokens ``[w_1, w_2, ..., w_N]``.
        This method determines

        .. math::

            \log Pr(w_1, w_2, ..., w_{N})

        Which it decomposes according to the markov assumption
        (see :func:`conditional`)

        Parameters
        ----------
        context : sequence

        Returns
        -------
        joint : float
        '''
        if self.unk is None:
            context = tuple(context)
        else:
            context = tuple(
                t if t in self.vocab else self.unk for t in context)
        if not len(context):
            raise ValueError('context must have at least one token')
        return self.trie.log_prob(context)

    def to_ngram_list(self):
        return self.trie.to_ngram_list()

    def relative_entropy_pruning(self, threshold):
        return self.trie.relative_entropy_pruning(threshold)

    def sequence_perplexity(self, sequence, include_delimiters=True):
        r'''Return the perplexity of the sequence using this language model

        Given a `sequence` of tokens ``[w_1, w_2, ..., w_N]``, the perplexity
        of the sequence is

        .. math::

            Pr(sequence)^{-1/N} = Pr(w_1, w_2, ..., w_N)^{-1/N}

        Parameters
        ----------
        sequence : sequence
        include_delimiters : bool, optional
            If :obj:`True`, the sequence will be prepended with the
            start-of-sequence symbol and appended with an end-of-sequence
            symbol, assuming they do not already exist as prefix and suffix of
            `sequence`

        Notes
        -----
        If the first token in `sequence` is the start-of-sequence token (or
        it is added using `include_delimiters`), it will not be included in
        the count ``N`` because ``Pr(sos) = 1`` always. An end-of-sequence
        token is always included in ``N``.
        '''
        sequence = list(sequence)
        if include_delimiters:
            if not len(sequence) or sequence[0] != self.sos:
                sequence.insert(0, self.sos)
            if sequence[-1] != self.eos:
                sequence.append(self.eos)
        if not len(sequence):
            raise ValueError(
                'sequence cannot be empty when include_delimiters is False')
        N = len(sequence)
        if sequence[0] == self.sos:
            N -= 1
        return 10. ** (-self.log_prob(sequence) / N)

    def corpus_perplexity(self, corpus, include_delimiters=True):
        r'''Calculate the perplexity of an entire corpus using this model

        A `corpus` is a sequence of sequences ``[s_1, s_2, ..., s_S]``. Each
        sequence ``s_i`` is a sequence of tokens ``[w_1, w_2, ..., w_N_i]``.
        Assuming sentences are independent,

        .. math::

            Pr(corpus) = Pr(s_1, s_2, ..., s_S) = Pr(s_1)Pr(s_2)...Pr(s_S)

        We calculate the corpus perplexity as the corpus probablity normalized
        by the total number of tokens in the corpus. Letting
        :math:`M = \sum_i^S N_i`, the corpus perplexity is

        .. math::

            Pr(corpus)^{-1/M}

        Parameters
        ----------
        corpus : sequence
        include_delimiters : bool, optional
            Whether to add start- and end-of-sequence delimiters to each
            sequence (if necessary). See :func:`sequence_complexity` for more
            info
        '''
        joint = 0.0
        M = 0
        for sequence in corpus:
            sequence = list(sequence)
            if include_delimiters:
                if not len(sequence) or sequence[0] != self.sos:
                    sequence.insert(0, self.sos)
                if sequence[-1] != self.eos:
                    sequence.append(self.eos)
            if not len(sequence):
                warnings.warn(
                    'skipping empty sequence (include_delimiters is False)')
                continue
            N = len(sequence)
            if sequence[0] == self.sos:
                N -= 1
            M += N
            joint += self.log_prob(sequence)
        return 10. ** (-joint / M)


def write_arpa(prob_list, out=sys.stdout):
    '''Convert an lists of n-gram probabilities to arpa format

    The inverse operation of :func:`pydrobert.torch.util.parse_arpa_lm`

    Parameters
    ----------
    prob_list : list of dict
    out : file or str, optional
        Path or file object to output to
    '''
    if isinstance(out, str):
        with open(out, 'w') as f:
            return write_arpa(prob_list, f)
    entries_by_order = []
    for idx, dict_ in enumerate(prob_list):
        entries = sorted(
            (k, v) if idx else ((k,), v)
            for (k, v) in dict_.items()
        )
        entries_by_order.append(entries)
    out.write('\\data\\\n')
    for idx in range(len(entries_by_order)):
        out.write('ngram {}={}\n'.format(idx + 1, len(entries_by_order[idx])))
    out.write('\n')
    for idx, entries in enumerate(entries_by_order):
        out.write('{}-grams:\n'.format(idx + 1))
        if idx == len(entries_by_order) - 1:
            for entry in entries:
                out.write('{} {}\n'.format(' '.join(entry[0]), entry[1]))
        else:
            for entry in entries:
                out.write('{} {} {}\n'.format(
                    entry[1][0], ' '.join(entry[0]), entry[1][1]))
        out.write('\n')
    out.write('\\end\\\n')


def ngram_counts_to_prob_list_mle(ngram_counts, eps_lprob=-99.999):
    '''Determine probabilities based on MLE of observed n-gram counts

    For a given n-gram :math:`w`, the maximum likelihood
    estimate of the last token given the first is:

    .. math::

        Pr(w) = C(w) / N

    Where :math:`C(w)` is the count of the n-gram and :math:`N` is the sum of
    all counts of the same order. Many counts will be zero, especially for
    large n-grams or rare words, making this a not terribly generalizable
    solution.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to
        unigram counts in a corpus, ``ngram_counts[1]`` to bi-grams, etc.
        Keys are tuples of tokens (n-grams) of the appropriate length, with
        the exception of unigrams, whose keys are the tokens themselves.
        Values are the counts of those n-grams in the corpus
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability"

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Examples
    --------
    >>> from collections import Counter
    >>> text = 'a man a plan a canal panama'
    >>> ngram_counts = [
    >>>     Counter(
    >>>         tuple(text[offs:offs + order]) if order > 1
    >>>         else text[offs:offs + order]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 4)
    >>> ]
    >>> ngram_counts[0]['<unk>'] = 0  # add oov to vocabulary
    >>> ngram_counts[0]['a']
    10
    >>> sum(ngram_counts[0].values())
    27
    >>> ngram_counts[1][('a', ' ')]
    3
    >>> sum(ngram_counts[1].values())
    26
    >>> prob_list = ngram_counts_to_prob_list_mle(ngram_counts)
    >>> prob_list[0]['a']   # (log10(10 / 27), eps_lprob)
    (-0.43136376415898736, -99.99)
    >>> '<unk>' in prob_list[0]  # no probability mass gets removed
    False
    >>> prob_list[1][('a', ' ')]  # (log10((3 / 26) / (10 / 27)), eps_lprob)
    (-0.5064883290921682, -99.99)

    Notes
    -----
    To be compatible with back-off models, MLE estimates assign a negligible
    backoff probability (`eps_lprob`) to n-grams where necessary. This means
    the probability mass might not exactly sum to one.
    '''
    return ngram_counts_to_prob_list_add_k(
        ngram_counts, eps_lprob=-99.99, k=0.)


def ngram_counts_to_prob_list_add_k(ngram_counts, eps_lprob=-99.999, k=.5):
    r'''MLE probabilities with constant discount factor added to counts

    Similar to :func:`ngram_counts_to_prob_list_mle`, but with a constant
    added to each count to smooth out probabilities:

    .. math::

        Pr(w) = (C(w) + k)/(N + k|V|)

    Where :math:`V` is the vocabulary set. The initial vocabulary set is
    determined from the unique unigrams :math:`V = U`. The bigram vocabulary
    set is the Cartesian product :math:`V = U \times U`, trigrams
    :math:`V = U \times U \times U`, and so on.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to
        unigram counts in a corpus, ``ngram_counts[1]`` to bi-grams, etc.
        Keys are tuples of tokens (n-grams) of the appropriate length, with
        the exception of unigrams, whose keys are the tokens themselves.
        Values are the counts of those n-grams in the corpus
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability"

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Examples
    --------
    >>> from collections import Counter
    >>> text = 'a man a plan a canal panama'
    >>> ngram_counts = [
    >>>     Counter(
    >>>         tuple(text[offs:offs + order]) if order > 1
    >>>         else text[offs:offs + order]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 4)
    >>> ]
    >>> ngram_counts[0]['<unk>'] = 0  # add oov to vocabulary
    >>> ngram_counts[0]['a']
    10
    >>> sum(ngram_counts[0].values())
    27
    >>> sum(ngram_counts[1].values())
    26
    >>> prob_list = ngram_counts_to_prob_list_add_k(ngram_counts, k=1)
    >>> prob_list[0]['a']   # (log10((10 + 1) / (27 + 8)), eps_lprob)
    (-0.5026753591920505, -99.999)
    >>> # Pr('<unk>') = 1 / (27 + 8) = 1 / 35
    >>> # Pr('<unk>', 'a') = 1 / (26 + 8 * 8) = 1 / 90
    >>> # Pr('a' | '<unk>') = Pr('<unk>', 'a') / Pr('<unk>') = 35 / 90
    >>> prob_list[1][('<unk>', 'a')]  # (log10(35 / 90), eps_lprob)
    (-0.4101744650890493, -99.999)
    '''
    max_order = len(ngram_counts) - 1
    if not len(ngram_counts):
        raise ValueError('At least unigram counts must exist')
    vocab = list(ngram_counts[0])
    prob_list = [{tuple(): (0, None)}]  # Pr(empty) = 1., will remove at end
    for order, counts in enumerate(ngram_counts):
        new_counts = dict()
        N = 0
        for ngram in product(vocab, repeat=order + 1):
            count = counts.get(ngram if order else ngram[0], 0) + k
            if not count:
                continue
            new_counts[ngram] = count
            N += count
        if N < 1:
            if not order:
                raise ValueError('Total unigram count is not positive')
            warnings.warn(
                'total {}-gram count is not positive. Skipping this and all '
                'higher order n-grams in return'.format(order + 1))
            break
        log_N = np.log10(N)
        probs = dict(
            (ngram, np.log10(count) - log_N - prob_list[-1][ngram[:-1]][0])
            for (ngram, count) in new_counts.items()
        )
        if order != max_order:
            probs = dict(
                (ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    del prob_list[0]
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def _get_simple_good_turing_log_scores(counts, eps_lprob):
    # this follows GT smoothing w/o tears section 6 pretty closely. You might
    # not know what's happening otherwise
    N_r = Counter(counts.values())
    max_r = max(N_r.keys())
    N_r = np.array(tuple(N_r.get(i, 0) for i in range(max_r + 2)))
    N_r[0] = 0
    r = np.arange(max_r + 2)
    N = (N_r * r).sum()
    log_N = np.log10(N)

    # find S(r) = a r^b
    nonzeros = np.where(N_r != 0)[0]
    Z_rp1 = 2. * N_r[1:-1]
    j = r[1:-1]
    diff = nonzeros - j[..., None]
    i = j - np.where(-diff < 1, max_r, -diff).min(1)
    i[0] = 0
    k = j + np.where(diff < 1, max_r, diff).min(1)
    k[-1] = 2 * j[-1] - i[-1]
    Z_rp1 /= k - i
    y = np.log10(Z_rp1[nonzeros - 1])  # Z_rp1 does not include r=0
    x = np.log10(r[nonzeros])
    # regress on y = bx + a
    mu_x, mu_y = x.mean(), y.mean()
    num = ((x - mu_x) * (y - mu_y)).sum()
    denom = ((x - mu_x) ** 2).sum()
    b = num / denom if denom else 0.0
    a = mu_y - b * mu_x
    log_Srp1 = a + b * np.log10(r[1:])

    # determine direct estimates of r* (x) as well as regressed estimates of
    # r* (y). Use x until absolute difference between x and y is statistically
    # significant (> 2 std dev of gauss defined by x)
    log_r_star = np.empty(max_r + 1, dtype=float)
    log_Nr = log_r_star[0] = np.log10(N_r[1]) if N_r[1] else eps_lprob + log_N
    switched = False
    C, ln_10 = np.log10(1.69), np.log(10)
    for r_ in range(1, max_r + 1):
        switched |= not N_r[r_]
        log_rp1 = np.log10(r_ + 1)
        log_y = log_rp1 + log_Srp1[r_] - log_Srp1[r_ - 1]
        if not switched:
            if N_r[r_ + 1]:
                log_Nrp1 = np.log10(N_r[r_ + 1])
            else:
                log_Nrp1 = eps_lprob + log_N + log_Nr
            log_x = log_rp1 + log_Nrp1 - log_Nr
            if log_y > log_x:
                log_abs_diff = log_y + np.log1p(-np.exp(log_x - log_y))
            elif log_x < log_y:
                log_abs_diff = log_x + np.log1p(-np.exp(log_y - log_x))
            else:
                log_abs_diff = -float('inf')
            log_z = C + log_rp1 - log_Nr + .5 * log_Nrp1
            log_z += .5 * np.log1p(N_r[r_ + 1] / N_r[r_]) / ln_10
            if log_abs_diff <= log_z:
                switched = True
            else:
                log_r_star[r_] = log_x
            log_Nr = log_Nrp1
        if switched:
            log_r_star[r_] = log_y

    # G&S tell us to renormalize the prob mass among the nonzero r terms. i.e.
    # p[0] = r_star[0] / N
    # p[i] = (1 - p[0]) r_star[i] / N'
    # where N' = \sum_i>0 N_r[i] r_star[i]
    max_log_r_star = np.max(log_r_star[1:][nonzeros[:-1] - 1])
    log_Np = np.log10(
        (N_r[1:-1] * 10 ** (log_r_star[1:] - max_log_r_star)).sum())
    log_Np += max_log_r_star
    log_p_0 = log_r_star[0] - log_N
    log_p_r = log_r_star - log_Np + np.log10(1 - 10 ** log_p_0)
    log_p_r[0] = log_p_0
    return log_p_r


def ngram_counts_to_prob_list_simple_good_turing(
        ngram_counts, eps_lprob=-99.999):
    r'''Determine probabilities based on n-gram counts using simple good-turing

    Simple Good-Turing smoothing discounts counts of n-grams according to the
    following scheme:

    .. math::

        r_* = (r + 1) N_{r + 1} / N_r

    Where :math:`r` is the original count of the n-gram in question,
    :math:`r_*` the discounted, and :math:`N_r` is the count of the number of
    times any n-gram had a count `r`.

    When :math:`N_r` becomes sparse, it is replaced with a log-linear
    regression of :math:`N_r` values, :math:`S(r) = a + b \log r`.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to
        unigram counts in a corpus, ``ngram_counts[1]`` to bi-grams, etc.
        Keys are tuples of tokens (n-grams) of the appropriate length, with
        the exception of unigrams, whose keys are the tokens themselves.
        Values are the counts of those n-grams in the corpus
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability"

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Warnings
    --------
    This function manually defines all n-grams of the target order given a
    vocabulary. This means that higher-order n-grams will be very large.
    Further, it makes no distinction between start tokens, end tokens, and
    regular tokens. This means inappropriate n-grams, such as
    ``('<s>', '<s>', 'a')`` or ``('</s>', 'a')``, will have to be manually
    removed.

    Examples
    --------
    >>> from collections import Counter
    >>> text = 'a man a plan a canal panama'
    >>> ngram_counts = [
    >>>     Counter(
    >>>         tuple(text[offs:offs + order]) if order > 1
    >>>         else text[offs:offs + order]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 4)
    >>> ]
    >>> ngram_counts[0]['<unk>'] = 0  # add oov to vocabulary
    >>> ngram_counts[0]['a']
    (10, 1)
    >>> sum(ngram_counts[0].values())
    27
    >>> Counter(ngram_counts[0].values())
    Counter({2: 3, 10: 1, 6: 1, 4: 1, 1: 1, 0: 1})
    >>> # N_1 = 1, N_2 = 3, N_3 = 1
    >>> prob_list = ngram_counts_to_prob_list_simple_good_turing(ngram_counts)
    >>> # Pr('<unk>') = Pr(r=0) = N_1 / N_0 / N *= 1 / 27
    >>> prob_list[0]['<unk>']   # (log10(1 / 27), eps_lprob)
    (-1.4313637641589874, -99.999)

    References
    ----------
    .. [gale1995] W. A. Gale and G. Sampson, "Good‐Turing frequency estimation
       without tears," Journal of Quantitative Linguistics, vol. 2, no. 3, pp.
       217-237, Jan. 1995.
    '''
    if len(ngram_counts) < 1:
        raise ValueError('At least unigram counts must exist')
    max_order = len(ngram_counts) - 1
    vocab = set(ngram_counts[0])
    prob_list = [{tuple(): (0, None)}]
    for order, counts in enumerate(ngram_counts):
        probs = dict()
        log_scores = _get_simple_good_turing_log_scores(counts, eps_lprob)
        N_0_vocab = set()
        for ngram in product(vocab, repeat=order + 1):
            r = counts.get(ngram if order else ngram[0], 0)
            probs[ngram] = log_scores[r] - prob_list[-1][ngram[:-1]][0]
            if not r:
                N_0_vocab.add(ngram)
        if N_0_vocab:
            log_N_0 = np.log10(len(N_0_vocab))
            for ngram in N_0_vocab:
                # distribute r=0 probability mass over unseen n-grams
                probs[ngram] -= log_N_0
        if order != max_order:
            probs = dict(
                (ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    del prob_list[0]
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list

