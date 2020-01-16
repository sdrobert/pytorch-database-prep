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
from collections.abc import Iterable
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
    'ngram_counts_to_prob_list_katz_backoff',
    'ngram_counts_to_prob_list_absolute_discounting',
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
    r'''Determine probabilities based on MLE of observed n-gram counts

    For a given n-gram :math:`p, w`, where :math:`p` is a prefix, :math:`w` is
    the next word, the maximum likelihood estimate of the last token given the
    prefix is:

    .. math::

        Pr(w | p) = C(p, w) / (\sum_w' C(p, w'))

    Where :math:`C(x)` Is the count of the sequence :math:`x`. Many counts will
    be zero, especially for large n-grams or rare words, making this a not
    terribly generalizable solution.

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
    >>> sum(v for (k, v) in ngram_counts[1].items() if k[0] == 'a')
    9
    >>> prob_list = ngram_counts_to_prob_list_mle(ngram_counts)
    >>> prob_list[0]['a']   # (log10(10 / 27), eps_lprob)
    (-0.43136376415898736, -99.99)
    >>> '<unk>' in prob_list[0]  # no probability mass gets removed
    False
    >>> prob_list[1][('a', ' ')]  # (log10(3 / 9), eps_lprob)
    (-0.47712125471966244, -99.99)

    Notes
    -----
    To be compatible with back-off models, MLE estimates assign a negligible
    backoff probability (`eps_lprob`) to n-grams where necessary. This means
    the probability mass might not exactly sum to one.
    '''
    return ngram_counts_to_prob_list_add_k(
        ngram_counts, eps_lprob=-99.99, k=0.)


def _get_cond_mle(order, counts, vocab, k):
    n_counts = dict()  # C(p, w) + k
    d_counts = dict()  # \sum_w' C(p, w') + k|V|
    for ngram in product(vocab, repeat=order + 1):
        c = counts.get(ngram if order else ngram[0], 0) + k
        if not c:
            continue
        n_counts[ngram] = c
        d_counts[ngram[:-1]] = d_counts.get(ngram[:-1], 0) + c
    return dict(
        (ng, np.log10(num) - np.log10(d_counts[ng[:-1]]))
        for ng, num in n_counts.items()
    )


def ngram_counts_to_prob_list_add_k(ngram_counts, eps_lprob=-99.999, k=.5):
    r'''MLE probabilities with constant discount factor added to counts

    Similar to :func:`ngram_counts_to_prob_list_mle`, but with a constant
    added to each count to smooth out probabilities:

    .. math::

        Pr(w|p) = (C(p,w) + k)/(\sum_w' C(p, w') + k|V|)

    Where :math:`p` is a prefix, :math:`w` is the next word, and
    :math:`V` is the vocabulary set. The initial vocabulary set is
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
    >>> ('a', '<unk>') not in ngram_counts[1]
    True
    >>> sum(v for (k, v) in ngram_counts[1].items() if k[0] == 'a')
    9
    >>> prob_list = ngram_counts_to_prob_list_add_k(ngram_counts, k=1)
    >>> prob_list[0]['a']   # (log10((10 + 1) / (27 + 8)), eps_lprob)
    (-0.5026753591920505, -99.999)
    >>> # Pr('a' | '<unk>') = (C('<unk>', 'a') + k) / (C('<unk>', .) + k|V|)
    >>> #                   = 1 / 8
    >>> prob_list[1][('<unk>', 'a')]  # (log10(1 / 8), eps_lprob)
    (-0.9030899869919435, -99.999)
    >>> # Pr('<unk>' | 'a') = (C('a', '<unk>') + k) / (C('a', .) + k|V|)
    >>> #                   = 1 / (9 + 8)
    >>> prob_list[1][('a', '<unk>')]  # (log10(1 / 17), eps_lprob)
    (-1.2304489213782739, -99.999)
    '''
    max_order = len(ngram_counts) - 1
    if not len(ngram_counts):
        raise ValueError('At least unigram counts must exist')
    vocab = set(ngram_counts[0])
    prob_list = []
    for order, counts in enumerate(ngram_counts):
        probs = _get_cond_mle(order, counts, vocab, k)
        if order != max_order:
            probs = dict(
                (ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def _log10sumexp(*args):
    if len(args) > 1:
        return _log10sumexp(np.array(args, dtype=float))
    x = args[0]
    if np.any(np.isnan(x)):
        return np.nan
    if np.any(np.isposinf(x)):
        return np.inf
    x = x[np.isfinite(x)]
    if not len(x):
        return 0.
    max_ = np.max(x)
    return np.log10((10 ** (x - max_)).sum()) + max_


def _simple_good_turing_counts(counts, eps_lprob):
    # this follows GT smoothing w/o tears section 6 pretty closely. You might
    # not know what's happening otherwise
    N_r = Counter(counts.values())
    max_r = max(N_r.keys())
    N_r = np.array(tuple(N_r.get(i, 0) for i in range(max_r + 2)))
    N_r[0] = 0
    r = np.arange(max_r + 2)
    N = (N_r * r).sum()
    log_N = np.log10(N)
    nonzeros = np.where(N_r != 0)[0]

    # find S(r) = a r^b
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
    # we convert back to counts so that our conditional MLEs are accurate
    max_log_r_star = np.max(log_r_star[1:][nonzeros[:-1] - 1])
    log_Np = np.log10(
        (N_r[1:-1] * 10 ** (log_r_star[1:] - max_log_r_star)).sum())
    log_Np += max_log_r_star
    log_p_0 = log_r_star[0] - log_N
    log_r_star[1:] += -log_Np + np.log10(1 - 10 ** log_p_0) + log_N

    return log_r_star


def ngram_counts_to_prob_list_simple_good_turing(
        ngram_counts, eps_lprob=-99.999):
    r'''Determine probabilities based on n-gram counts using simple good-turing

    Simple Good-Turing smoothing discounts counts of n-grams according to the
    following scheme:

    .. math::

        r^* = (r + 1) N_{r + 1} / N_r

    Where :math:`r` is the original count of the n-gram in question,
    :math:`r^*` the discounted, and :math:`N_r` is the count of the number of
    times any n-gram had a count `r`.

    When :math:`N_r` becomes sparse, it is replaced with a log-linear
    regression of :math:`N_r` values, :math:`S(r) = a + b \log r`.
    :math:`r^*` for :math:`r > 0` are renormalized so that
    :math:`\sum_r N_r r^* = \sum_r N_r r`.

    We assume a closed vocabulary and that, for any order n-gram, :math:`N_0`
    is the size of the set of n-grams with frequency zero. This method differs
    from traditional Good-Turing, which assumes one unseen "event" (i.e.
    n-gram) per level. See below notes for more details.

    If, for a given order of n-gram, none of the terms have frequency
    zero, this function will warn and use MLEs.

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

    Notes
    -----
    The traditional definition of Good-Turing is somewhat vague about how to
    assign probability mass among unseen events. By setting
    :math:`r^* = N_1 / N` for :math:`r = 0`, it's implicitly stating that
    :math:`N_0 = 1`, that is, there's only one possible unseen event. This is
    consistent with introducing a special token, e.g. ``"<unk>"``, that does
    not occur in the corpus. It also collapses unseen n-grams into one event.

    We cannot bootstrap the backoff penalty to be the probability of the
    unseen term because the backoff will be combined with a lower-order
    estimate, and Good-Turing uses a fixed unseen probability.

    As our solution, we assume the vocabulary is closed. Any term that appears
    zero times is added to :math:`N_0`. If all terms appear, then
    :math:`N_0 = 0` and we revert to the MLE. While you can simulate the
    traditional Good-Turing at the unigram-level by introducing ``"<unk>"``
    with count 0, this will not hold for higher-order n-grams.

    Warnings
    --------
    This function manually defines all n-grams of the target order given a
    vocabulary. This means that higher-order n-grams will be very large.

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
    >>> sum(ngram_counts[0].values())
    27
    >>> Counter(ngram_counts[0].values())
    Counter({2: 3, 10: 1, 6: 1, 4: 1, 1: 1, 0: 1})
    >>> # N_1 = 1, N_2 = 3, N_3 = 1
    >>> prob_list = ngram_counts_to_prob_list_simple_good_turing(ngram_counts)
    >>> # Pr('<unk>') = Pr(r=0) = N_1 / N_0 / N = 1 / 27
    >>> prob_list[0]['<unk>']   # (log10(1 / 27), eps_lprob)
    (-1.4313637641589874, -99.999)
    >>> # Pr('a'|'<unk>') = Cstar('<unk>', 'a') / (Cstar('unk', .))
    >>> #                 = rstar[0] / (|V| * rstar[0]) = 1 / 8
    >>> prob_list[1][('<unk>', 'a')]  # (log10(1 / 8), eps_lprob)
    (-0.9030899869919435, -99.999)

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
    prob_list = []
    for order, counts in enumerate(ngram_counts):
        N_0_vocab = set()
        log_r_stars = _simple_good_turing_counts(counts, eps_lprob)
        n_counts = dict()
        d_counts = dict()
        for ngram in product(vocab, repeat=order + 1):
            r = counts.get(ngram if order else ngram[0], 0)
            if r:
                c = 10. ** log_r_stars[r]
                n_counts[ngram] = c
                d_counts[ngram[:-1]] = d_counts.get(ngram[:-1], 0) + c
            else:
                N_0_vocab.add(ngram)
        N_0 = len(N_0_vocab)
        if N_0:
            c = (10 ** log_r_stars[0]) / N_0
            for ngram in N_0_vocab:
                n_counts[ngram] = c
                d_counts[ngram[:-1]] = d_counts.get(ngram[:-1], 0) + c
            probs = dict(
                (ng, np.log10(n_counts[ng]) - np.log10(d_counts[ng[:-1]]))
                for ng in n_counts
            )
        else:
            warnings.warn(
                'No {}-grams were missing. Using MLE instead'
                ''.format(order + 1))
            probs = _get_cond_mle(order, counts, vocab, 0)
        if order != max_order:
            probs = dict(
                (ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def _get_katz_discounted_counts(counts, k):
    N_r = Counter(counts.values())
    max_r = max(N_r.keys())
    N_r = np.array(tuple(N_r.get(i, 0) for i in range(max_r + 2)))
    N_r[0] = 1
    r = np.arange(max_r + 2)
    N = (N_r * r).sum()
    log_N = np.log10(N)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_Nr = np.log10(N_r)
        log_rp1 = np.log10(r + 1)
        log_r_star = log_rp1[:-1] + log_Nr[1:] - log_Nr[:-1]
    if k + 1 < len(N_r):
        log_d_rp1 = np.zeros(max_r, dtype=float)
        log_num_minu = log_r_star[1:k + 1] - log_rp1[:k]
        log_subtra = np.log10(k + 1) + log_Nr[k + 1] - log_Nr[1]
        if log_subtra >= 0:
            raise ValueError('Your corpus is too small for this')
        # np.log10((10 ** (x - max_)).sum()) + max_
        log_num = log_num_minu + np.log1p(
            -10 ** (log_subtra - log_num_minu)) / np.log(10)
        log_denom = np.log1p(-(10 ** log_subtra)) / np.log(10)
        log_d_rp1[:k] = log_num - log_denom
    else:
        log_d_rp1 = log_r_star[1:] - log_rp1[:-1]
    log_r_star = np.empty(max_r + 1, dtype=float)
    log_r_star[0] = log_Nr[1]
    log_r_star[1:] = log_d_rp1 + log_rp1[:-2]
    assert np.isclose(_log10sumexp(log_r_star + log_Nr[:-1]), log_N)
    return log_r_star


def ngram_counts_to_prob_list_katz_backoff(
        ngram_counts, k=7, eps_lprob=-99.999, _cmu_hacks=False):
    r'''Determine probabilities based on Katz's backoff algorithm

    Kat'z backoff algorithm determines the conditional probability of the last
    token in n-gram :math:`w = (w_1, w_2, ..., w_n)` as

    .. math::

        Pr_{BO}(w_n|w_{n-1}, w_{n-2} ..., w_1) = \begin{cases}
            d_w Pr_{MLE}(w_n|w_{n-1}, w_{n-1}, ..., w_1) & \text{if }C(w) > 0
            \alpha(w_1, ..., w_{n-1}) Pr_{BO}(w_n|w_{n-1}, ..., w_2)&
                                                                    \text{else}
        \end{cases}

    Where :math:`Pr_{MLE}` is the maximum likelihood estimate (based on
    frequencies), :math:`d_w` is some discount factor (based on Good-Turing
    for low-frequency n-grams), and :math:`\alpha` is an allowance of the
    leftover probability mass from discounting.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to
        unigram counts in a corpus, ``ngram_counts[1]`` to bi-grams, etc.
        Keys are tuples of tokens (n-grams) of the appropriate length, with
        the exception of unigrams, whose keys are the tokens themselves.
        Values are the counts of those n-grams in the corpus
    k : int, optional
        `k` is a threshold such that, if :math:`C(w) > k`, no discounting will
        be applied to the term. That is, the probability mass assigned for
        backoff will be entirely from n-grams s.t. :math:`C(w) \leq k`
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability"

    Warnings
    --------
    If the counts of the extensions of a prefix are all above k, no discounting
    will be applied to those counts, meaning no probability mass can be
    assigned to unseen events.

    For example, in the Brown corpus, "Hong" is always followed by "Kong". The
    bigram "Hong Kong" occurs something like 10 times, so it's not discounted.
    Thus :math:`P_{BO}(Kong|Hong) = 1` :math:`P_{BO}(not Kong|Hong) = 0`.

    A :obj:`UserWarning` will be issued whenever ths happens. If this bothers
    you, you could try increasing `k` or, better yet, abandon Katz Backoff
    altogether.

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Examples
    --------
    >>> from nltk.corpus import brown
    >>> from collections import Counter
    >>> text = tuple(brown.words())[:20000]
    >>> ngram_counts = [
    >>>     Counter(
    >>>         text[offs:offs + order] if order > 1
    >>>         else text[offs]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 4)
    >>> ]
    >>> del text
    >>> prob_list = ngram_counts_to_prob_list_katz_backoff(ngram_counts)

    References
    ----------
    .. [katz1987] S. Katz, "Estimation of probabilities from sparse data for
       the language model component of a speech recognizer," IEEE Transactions
       on Acoustics, Speech, and Signal Processing, vol. 35, no. 3, pp.
       400-401, Mar. 1987.
    '''
    if len(ngram_counts) < 1:
        raise ValueError('At least unigram counts must exist')
    if k < 1:
        raise ValueError('k too low')
    prob_list = []
    max_order = len(ngram_counts) - 1
    probs = _get_cond_mle(0, ngram_counts[0], set(ngram_counts[0]), 0)
    if 0 != max_order:
        probs = dict(
            (ngram, (prob, eps_lprob))
            for (ngram, prob) in probs.items())
    prob_list.append(probs)
    log_r_stars = [
        _get_katz_discounted_counts(counts, k)
        for counts in ngram_counts[1:]
    ]
    if _cmu_hacks:
        # A note on CMU compatibility. First, the standard non-ML estimate of
        # P(w|p) = C(p, w) / C(p) instead of P(w|p) = C(p, w) / sum_w' C(p, w')
        # Second, this below loop. We add one to the count of a prefix whenever
        # that prefix has only one child and that child's count is greater than
        # k (in increment_context.cc). This ensures there's a non-zero backoff
        # to assign to unseen contexts starting with that prefix (N.B. this
        # hack should be extended to the case where all children have count
        # greater than k, but I don't want to reinforce this behaviour). Note
        # that it is applied AFTER the MLE for unigrams, and AFTER deriving
        # discounted counts.
        for order in range(len(ngram_counts) - 1, 0, -1):
            prefix2children = dict()
            for ngram, count in ngram_counts[order].items():
                prefix2children.setdefault(ngram[:-1], []).append(ngram)
            for prefix, children in prefix2children.items():
                if (
                        len(children) == 1 and
                        ngram_counts[order][children[0]] > k):
                    for oo in range(order):
                        pp = prefix[:oo + 1]
                        if not oo:
                            pp = pp[0]
                        ngram_counts[oo][pp] += 1
    for order in range(1, len(ngram_counts)):
        counts = ngram_counts[order]
        probs = dict()
        # P_katz(w|pr) = C*(pr, w) / \sum_x C*(pr, x) if C(pr, w) > 0
        #                alpha(pr) Pr_katz(w|pr[1:]) else
        # alpha(pr) = (1 - sum_{c(pr, w) > 0} Pr*(w|pr)
        #             / (1 - sum_{c(pr, w) > 0} Pr*(w|pr[1:]))
        # note: \sum_w C*(pr, w) = \sum_w C(pr, w), which is why we can
        # normalize by the true counts
        lg_num_subtras = dict()  # logsumexp(log c*(pr,w)) for c(pr,w) > 0
        lg_den_subtras = dict()  # logsumexp(log Pr(w|pr[1:]) for c(pr, w) > 0
        lg_pref_counts = dict()  # logsumexp(log c(pr)) for c(pr,w) > 0
        for ngram, r in counts.items():
            if not r:
                continue
            log_r_star = log_r_stars[order - 1][r]
            probs[ngram] = log_r_star
            lg_num_subtras[ngram[:-1]] = _log10sumexp(
                lg_num_subtras.get(ngram[:-1], -np.inf), log_r_star)
            lg_den_subtras[ngram[:-1]] = _log10sumexp(
                lg_den_subtras.get(ngram[:-1], -np.inf),
                prob_list[-1][ngram[1:]][0]
            )
            lg_pref_counts[ngram[:-1]] = _log10sumexp(
                lg_pref_counts.get(ngram[:-1], -np.inf), np.log10(r))
        for ngram in probs:
            prefix = ngram[:-1]
            if _cmu_hacks:
                if order == 1:
                    prefix = prefix[0]
                lg_norm = np.log10(ngram_counts[order - 1][prefix])
            else:
                lg_norm = lg_pref_counts[prefix]
            probs[ngram] -= lg_norm
        for prefix, lg_num_subtra in lg_num_subtras.items():
            lg_den_subtra = lg_den_subtras[prefix]
            if _cmu_hacks:
                if order == 1:
                    lg_norm = np.log10(ngram_counts[order - 1][prefix[0]])
                else:
                    lg_norm = np.log10(ngram_counts[order - 1][prefix])
            else:
                lg_norm = lg_pref_counts[prefix]
            num_subtra = 10. ** (lg_num_subtra - lg_norm)
            den_subtra = 10. ** lg_den_subtra
            if np.isclose(num_subtra, 1.) or np.isclose(den_subtra, 1.):
                warnings.warn(
                    'Cannot back off to prefix {}. Will assign negligible '
                    'probability. If this is an issue, try increasing k'
                    ''.format(prefix))
                continue
            log_alpha = np.log1p(-num_subtra) - np.log1p(-den_subtra)
            log_alpha /= np.log(10)
            log_prob, bad_backoff = prob_list[-1][prefix]
            assert bad_backoff == eps_lprob
            prob_list[-1][prefix] = (log_prob, log_alpha)
        if order != max_order:
            probs = dict(
                (ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def _N_xplus_suff(s, counts, x=1, gte=True):
    # Calculates N_{1+}(\cdot s) when x=1.
    # Counts the number of unique prefixes with which s occurs which have
    # count >= x. Make it only c == x when gte is false
    if not isinstance(s, tuple):
        s = (s,)
    if gte:
        return sum(
            1 for ng, v in counts.items() if ng[-len(s):] == s and v >= x)
    else:
        return sum(
            1 for ng, v in counts.items() if ng[-len(s):] == s and v == x)


def _N_xplus_pref(p, counts, x=1, gte=True):
    # Calculates N_{1+}(p \cdot) when x=1.
    # Counts the number of unique suffixes with which p occurs which have
    # count >= x
    if not isinstance(p, tuple):
        p = (p,)
    if gte:
        return sum(
            1 for ng, v in counts.items() if ng[:len(p)] == p and v >= x)
    else:
        return sum(
            1 for ng, v in counts.items() if ng[:len(p)] == p and v == x)


def _optimal_deltas(counts, y):
    N_r = Counter(counts.values())
    if not all(N_r[r] for r in range(1, y + 2)):
        raise ValueError(
            'Your dataset is too small to use the default discount '
            '(or maybe you removed the hapax before estimating probs?)')
    Y = N_r[1] / (N_r[1] + 2 * N_r[2])
    return [r - Y * N_r[r + 1] / N_r[r] for r in range(1, y + 1)]


def _absolute_discounting(ngram_counts, deltas, eps_lprob):
    V = len(set(ngram_counts[0]))
    prob_list = [{tuple(): (-np.log10(V), eps_lprob)}]
    max_order = len(ngram_counts) - 1
    for order, counts, delta in zip(
            range(len(ngram_counts)), ngram_counts, deltas):
        delta = np.array(delta, dtype=float)
        n_counts = dict()
        d_counts = dict()
        pr2bin = dict()
        for ngram, count in counts.items():
            if not count:
                continue
            if not order:
                ngram = (ngram,)
            bin_ = min(count - 1, len(delta) - 1)
            d = delta[bin_]
            assert count - d >= 0.
            prefix = ngram[:-1]
            n_counts[ngram] = n_counts.get(ngram, 0) + count - d
            d_counts[prefix] = d_counts.get(prefix, 0) + count
            prefix_bins = pr2bin.setdefault(prefix, np.zeros(len(delta)))
            prefix_bins[bin_] += 1
        for prefix, prefix_bins in pr2bin.items():
            with np.errstate(divide='ignore'):
                prefix_bins = np.log10(prefix_bins)
                prefix_bins += np.log10(delta)
            gamma = _log10sumexp(prefix_bins)
            gamma -= np.log10(d_counts[prefix])
            lprob, bo = prob_list[-1][prefix]
            assert np.isclose(bo, eps_lprob)
            prob_list[-1][prefix] = (lprob, gamma)
        probs = dict()
        for ngram, n_count in n_counts.items():
            prefix = ngram[:-1]
            lprob = np.log10(n_count) - np.log10(d_counts[prefix])
            lower_order = prob_list[-1][prefix][1]  # gamma(prefix)
            lower_order += prob_list[-1][ngram[1:]][0]  # Pr(w|prefix[1:])
            lprob = _log10sumexp(lprob, lower_order)
            if order != max_order:
                lprob = (lprob, eps_lprob)
            probs[ngram] = lprob
        prob_list.append(probs)
    del prob_list[0]  # zero-th order
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def ngram_counts_to_prob_list_absolute_discounting(
        ngram_counts, delta=None, eps_lprob=-99.99):
    r'''Determine probabilities from n-gram counts using absolute discounting

    Absolute discounting (based on the formulation in [chen1999]_) interpolates
    between higher-order and lower-order n-grams as

    .. math::

        Pr_{abs}(w_n|w_{n-1}, \ldots w_1) =
            \frac{\max\left\{C(w_1, \ldots, w_n) - \delta, 0\right\}}
                {\sum_w' C(w_1, \ldots, w_{n-1}, w')}
            - \gamma(w_1, \ldots, w_{n-1})
               Pr_{abs}(w_n|w_{n-1}, \ldots, w_2)

    Where :math:`\gamma` are chosen so :math:`Pr_{abs}(\cdot)` sum to one and
    :math:`\delta \in [0, 1]`. For the base case, we pretend there's such a
    thing as a zeroth-order n-gram, and
    :math:`Pr_{abs}(\emptyset) = 1 / \left\|V\right\|`.

    Letting

    .. math::

        N_c = \left|\left\{
                (w'_1, \ldots, w'_n): C(w'_1, \ldots, w'_n) = c
            \right\}\right|

    :math:`\delta` is often chosen to be :math:`\delta = N_1 / (N_1 + 2N_2)`
    for a given order n-gram. We can use different :math:`\delta` for different
    orders of the recursion.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to
        unigram counts in a corpus, ``ngram_counts[1]`` to bi-grams, etc.
        Keys are tuples of tokens (n-grams) of the appropriate length, with
        the exception of unigrams, whose keys are the tokens themselves.
        Values are the counts of those n-grams in the corpus
    delta : float or tuple or :obj:`None`, optional
        The absolute discount to apply to non-zero values. `delta` can take
        one of three forms: a :class:`float` to be used identically for all
        orders of the recursion; :obj:`None` specifies that the above formula
        for calculating `delta` should be used separately for each order of
        the recursion; or a tuple of length `ngram_counts`, where
        each element is either a :class:`float` or :obj:`None`, specifying
        either a fixed value or the default value for every order of the
        recursion (except the zeroth-order)
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability"

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
    >>> ngram_counts[0]['a']
    10
    >>> sum(ngram_counts[0].values())
    27
    >>> len(set(ngram_counts[0]))
    7
    >>> sum(1 for k in ngram_counts[1] if k[0] == 'a')
    4
    >>> sum(v for k, v in ngram_counts[1].items() if k[0] == 'a')
    9
    >>> prob_list = ngram_counts_to_prob_list_absolute_discounting(
    >>>     ngram_counts, delta=0.5)
    >>> # gamma_0() = 0.5 * 7 / 27
    >>> # Pr(a) = (10 - 0.5) / 27 + 0.5 (7 / 27) (1 / 7) = 10 / 27
    >>> # BO(a) = gamma_1(a) = 0.5 * 4 / 9 = 2 / 9
    >>> prob_list[0]['a']  # (log10 Pr(a), log10 gamma_1(a))
    (-0.4313637641589874, -0.6532125137753437)
    >>> ngram_counts[1][('a', 'n')]
    4
    >>> ngram_counts[0]['n']
    4
    >>> sum(1 for k in ngram_counts[2] if k[:2] == ('a', 'n'))
    2
    >>> sum(v for k, v in ngram_counts[2].items() if k[:2] == ('a', 'n'))
    4
    >>> # Pr(n) = (4 - 0.5) / 27 + 0.5 (7 / 27) (1 / 7) = 4 / 27
    >>> # Pr(n|a) = (4 - 0.5) / 9 + gamma_1(a) Pr(n)
    >>> #         = (4 - 0.5) / 9 + 0.5 (4 / 9) (4 / 27)
    >>> #         = (108 - 13.5 + 8) / 243
    >>> #         = 102.5 / 243
    >>> # BO(a, n) = gamma_2(a, n) = 0.5 (2 / 4) = 1 / 4
    >>> prob_list[1][('a', 'n')]  # (log10 Pr(n|a), log10 gamma_2(a, n))
    (-0.37488240820653906, -0.6020599913279624)

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    References
    ----------
    .. [chen1999] S. F. Chen and J. Goodman, "An empirical study of smoothing
       techniques for language modeling," Computer Speech & Language, vol. 13,
       no. 4, pp. 359-394, Oct. 1999, doi: 10.1006/csla.1999.0128.
    '''
    if len(ngram_counts) < 1:
        raise ValueError('At least unigram counts must exist')
    if not isinstance(delta, Iterable):
        delta = (delta,) * len(ngram_counts)
    if len(delta) != len(ngram_counts):
        raise ValueError(
            'Expected {} deltas, got {}'.format(len(ngram_counts), len(delta)))
    delta = tuple(
        _optimal_deltas(counts, 1)
        if d is None else [d]
        for (d, counts) in zip(delta, ngram_counts)
    )
    if not all(all(0. <= dd <= 1. for dd in d) for d in delta):
        raise ValueError('deltas {} must be in [0, 1]'.format(delta))
    return _absolute_discounting(ngram_counts, delta, eps_lprob)