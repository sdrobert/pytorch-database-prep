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

from collections import OrderedDict

import numpy as np

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'BackoffNGramLM',
    'write_arpa'
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
    ngram_list : sequence
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

    def __init__(self, ngram_list, sos=None, eos=None, unk=None):
        self.trie = self.TrieNode(0.0, 0.0)
        self.vocab = set()
        if not len(ngram_list) or not len(ngram_list[0]):
            raise ValueError('ngram_list must contain (all) unigrams')
        for order, dict_ in enumerate(ngram_list):
            is_first = not order
            is_last = order == len(ngram_list) - 1
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
        assert self.trie.depth == len(ngram_list)

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
            ngram_list = []
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
                ngram_list.append(dict_)
            return ngram_list

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


def write_arpa(ngram_list, out=sys.stdout):
    '''Convert an lists of n-gram probabilities to arpa format

    The inverse operation of :func:`pydrobert.torch.util.parse_arpa_lm`

    Parameters
    ----------
    ngram_list : list of dict
    out : file or str, optional
        Path or file object to output to
    '''
    if isinstance(out, str):
        with open(out, 'w') as f:
            return write_arpa(ngram_list, f)
    entries_by_order = []
    for idx, dict_ in enumerate(ngram_list):
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
