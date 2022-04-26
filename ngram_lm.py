# Copyright 2021 Sean Robertson
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

import warnings
import sys
import locale
import re

from collections import OrderedDict, Counter
from collections.abc import Iterable
from itertools import product

import numpy as np

__all__ = [
    "BackoffNGramLM",
    "write_arpa",
    "ngram_counts_to_prob_list_mle",
    "ngram_counts_to_prob_list_add_k",
    "ngram_counts_to_prob_list_simple_good_turing",
    "ngram_counts_to_prob_list_katz_backoff",
    "ngram_counts_to_prob_list_absolute_discounting",
    "ngram_counts_to_prob_list_kneser_ney",
    "text_to_sents",
    "sents_to_ngram_counts",
]

locale.setlocale(locale.LC_ALL, "C")
warnings.simplefilter("error", RuntimeWarning)


class BackoffNGramLM(object):
    """A backoff NGram language model, stored as a trie

    This class is intended for two things: one, to prune backoff language models, and
    two, to calculate the perplexity of a language model on a corpus. It is very
    inefficient.

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
    """

    def __init__(self, prob_list, sos=None, eos=None, unk=None):
        self.trie = self.TrieNode(0.0, 0.0)
        self.vocab = set()
        if not len(prob_list) or not len(prob_list[0]):
            raise ValueError("prob_list must contain (all) unigrams")
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
            if "<S>" in self.vocab:
                sos = "<S>"
            else:
                sos = "<s>"
        if sos not in self.vocab:
            raise ValueError(
                'start-of-sequence symbol "{}" does not have unigram '
                "entry.".format(sos)
            )
        self.sos = self.trie.sos = sos
        if eos is None:
            if "</S>" in self.vocab:
                eos = "</S>"
            else:
                eos = "</s>"
        if eos not in self.vocab:
            raise ValueError(
                'end-of-sequence symbol "{}" does not have unigram '
                "entry.".format(eos)
            )
        self.eos = self.trie.eos = eos
        if unk is None:
            if "<UNK>" in self.vocab:
                unk = "<UNK>"
            else:
                unk = "<unk>"
        if unk in self.vocab:
            self.unk = unk
        else:
            warnings.warn(
                'out-of-vocabulary symbol "{}" does not have unigram count. '
                "Out-of-vocabulary tokens will raise an error".format(unk)
            )
            self.unk = None
        assert self.trie.depth == len(prob_list)

    class TrieNode(object):
        def __init__(self, lprob, bo):
            self.lprob = lprob
            self.bo = bo
            self.children = OrderedDict()
            self.depth = 0
            self.sos = None
            self.eos = None

        def add_child(self, context, lprob, bo):
            assert len(context)
            next_, rest = context[0], context[1:]
            child = self.children.setdefault(next_, type(self)(None, 0.0))
            if rest:
                child.add_child(rest, lprob, bo)
            else:
                child.lprob = lprob
                child.bo = bo
            self.depth = max(self.depth, child.depth + 1)

        def conditional(self, context):
            assert context and self.depth
            context = context[-self.depth :]
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

        def log_prob(self, context, _srilm_hacks=False):
            joint = 0.0
            for prefix in range(2 if context[0] == self.sos else 1, len(context) + 1):
                joint += self.conditional(context[:prefix])
            if _srilm_hacks and context[0] == self.sos:
                # this is a really silly thing that SRI does - it estimates
                # the initial SOS probability with an EOS probability. Why?
                # The unigram probability of an SOS is 0. However, we assume
                # the sentence-initial SOS exists prior to the generation task,
                # and isn't a "real" part of the vocabulary
                joint += self.conditional((self.eos,))
            return joint

        def _gather_nodes_by_depth(self, order):
            nodes = [(tuple(), self)]
            nodes_by_depth = []
            for _ in range(order):
                last, nodes = nodes, []
                nodes_by_depth.append(nodes)
                for ctx, parent in last:
                    nodes.extend((ctx + (k,), v) for (k, v) in parent.children.items())
            return nodes_by_depth

        def _gather_nodes_at_depth(self, order):
            nodes = [(tuple(), self)]
            for _ in range(order):
                last, nodes = nodes, []
                for ctx, parent in last:
                    nodes.extend((ctx + (k,), v) for (k, v) in parent.children.items())
            return nodes

        def _renormalize_backoffs_for_order(self, order):
            nodes = self._gather_nodes_at_depth(order)
            base_10 = np.log(10)
            for h, node in nodes:
                if not len(node.children):
                    node.bo = 0.0
                    continue
                num = 0.0
                denom = 0.0
                for w, child in node.children.items():
                    assert child.lprob is not None
                    num -= 10.0 ** child.lprob
                    denom -= 10.0 ** self.conditional(h[1:] + (w,))
                # these values may be ridiculously close to 1, but still valid.
                if num < -1.0:
                    raise ValueError(
                        "Too much probability mass {} on children of n-gram {}"
                        "".format(-num, h)
                    )
                elif denom <= -1.0:
                    # We'll never back off. By convention, this is 0. (Pr(1.))
                    new_bo = 0.0
                elif num == -1.0:
                    if node.bo > -10:
                        warnings.warn(
                            "Found a non-negligible backoff {} for n-gram {} "
                            "when no backoff mass should exist".format(node.bo, h)
                        )
                    continue
                else:
                    new_bo = (np.log1p(num) - np.log1p(denom)) / base_10
                node.bo = new_bo

        def recalculate_depth(self):
            max_depth = 0
            stack = [(max_depth, self)]
            while stack:
                depth, node = stack.pop()
                max_depth = max(max_depth, depth)
                stack.extend((depth + 1, c) for c in node.children.values())
            self.depth = max_depth

        def renormalize_backoffs(self):
            for order in range(1, self.depth):  # final order has no backoffs
                self._renormalize_backoffs_for_order(order)

        def relative_entropy_pruning(self, threshold, eps=1e-8, _srilm_hacks=False):
            nodes_by_depth = self._gather_nodes_by_depth(self.depth - 1)
            base_10 = np.log(10)
            while nodes_by_depth:
                nodes = nodes_by_depth.pop()  # highest order first
                for h, node in nodes:
                    num = 0.0
                    denom = 0.0
                    logP_w_given_hprimes = []  # log P(w | h')
                    P_h = 10 ** self.log_prob(h, _srilm_hacks=_srilm_hacks)
                    for w, child in node.children.items():
                        assert child.lprob is not None
                        num -= 10.0 ** child.lprob
                        logP_w_given_hprime = self.conditional(h[1:] + (w,))
                        logP_w_given_hprimes.append(logP_w_given_hprime)
                        denom -= 10.0 ** logP_w_given_hprime
                    if num + 1 < eps or denom + 1 < eps:
                        warnings.warn(
                            "Malformed backoff weight for context {}. Leaving "
                            "as is".format(h)
                        )
                        continue
                    # alpha = (1 + num) / (1 + denom)
                    log_alpha = (np.log1p(num) - np.log1p(denom)) / base_10
                    if abs(log_alpha - node.bo) > 1e-2:
                        warnings.warn(
                            "Calculated backoff ({}) differs from stored "
                            "backoff ({}) for context {}"
                            "".format(log_alpha, node.bo, h)
                        )
                    if _srilm_hacks:
                        # technically these should match when well-formed, but
                        # re-calculating alpha allows us to re-normalize an ill-formed
                        # language model
                        log_alpha = node.bo
                    for idx, w in enumerate(tuple(node.children)):
                        child = node.children[w]
                        if child.bo:
                            continue  # don't prune children with backoffs
                        logP_w_given_h = child.lprob
                        P_w_given_h = 10 ** logP_w_given_h
                        logP_w_given_hprime = logP_w_given_hprimes[idx]
                        P_w_given_hprime = 10 ** logP_w_given_hprime
                        new_num = num + P_w_given_h
                        new_denom = denom + P_w_given_hprime
                        log_alphaprime = np.log1p(new_num)
                        log_alphaprime -= np.log1p(new_denom)
                        log_alphaprime /= base_10
                        log_delta_prob = logP_w_given_hprime + log_alphaprime
                        log_delta_prob -= logP_w_given_h
                        KL = -P_h * (
                            P_w_given_h * log_delta_prob
                            + (log_alphaprime - log_alpha) * (1.0 + num)
                        )
                        delta_perplexity = 10.0 ** KL - 1
                        if delta_perplexity < threshold:
                            node.children.pop(w)
                    # we don't have to set backoff properly (we'll renormalize at end).
                    # We just have to signal whether we can be pruned to our parents (do
                    # *we* have children?)
                    node.bo = float("nan") if len(node.children) else None
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
            self.renormalize_backoffs()

        def to_prob_list(self):
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

        def prune_by_threshold(self, lprob):
            for order in range(self.depth - 1, 0, -1):
                for _, parent in self._gather_nodes_at_depth(order):
                    for w in set(parent.children):
                        child = parent.children[w]
                        if not child.children and child.lprob <= lprob:
                            del parent.children[w]
            self.renormalize_backoffs()
            self.recalculate_depth()

        def prune_by_name(self, to_prune, eps_lprob):
            to_prune = set(to_prune)
            # we'll prune by threshold in a second pass, so no need to worry about
            # parent-child stuff
            extra_mass = -float("inf")
            remainder = set()
            stack = [((w,), c) for w, c in self.children.items()]
            while stack:
                ctx, node = stack.pop()
                stack.extend((ctx + (w,), c) for w, c in node.children.items())
                if len(ctx) == 1:
                    ctx = ctx[0]
                    if ctx in to_prune:
                        extra_mass = _log10sumexp(extra_mass, node.lprob)
                        node.lprob = eps_lprob
                    elif node.lprob > eps_lprob:
                        remainder.add(ctx)
                elif ctx in to_prune:
                    node.lprob = eps_lprob
            # we never *actually* remove unigrams - we set their probablities to roughly
            # zero and redistribute the collected mass across the remainder
            if not remainder:
                raise ValueError("No unigrams are left unpruned!")
            extra_mass -= np.log10(len(remainder))
            for w in remainder:
                child = self.children[w]
                child.lprob = _log10sumexp(child.lprob, extra_mass)
            self.prune_by_threshold(eps_lprob)

    def conditional(self, context):
        r"""Return the log probability of the last word in the context

        `context` is a non-empty sequence of tokens ``[w_1, w_2, ..., w_N]``. This
        method determines

        .. math::

            \log Pr(w_N | w_{N-1}, w_{N-2}, ... w_{N-C})

        Where ``C`` is this model's maximum n-gram size. If an exact entry cannot be
        found, the model backs off to a shorter context.

        Parameters
        ----------
        context : sequence

        Returns
        -------
        cond : float or :obj:`None`
        """
        if self.unk is None:
            context = tuple(context)
        else:
            context = tuple(t if t in self.vocab else self.unk for t in context)
        if not len(context):
            raise ValueError("context must have at least one token")
        return self.trie.conditional(context)

    def log_prob(self, context):
        r"""Return the log probability of the whole context

        `context` is a non-empty sequence of tokens ``[w_1, w_2, ..., w_N]``. This
        method determines

        .. math::

            \log Pr(w_1, w_2, ..., w_{N})

        Which it decomposes according to the markov assumption (see :func:`conditional`)

        Parameters
        ----------
        context : sequence

        Returns
        -------
        joint : float
        """
        if self.unk is None:
            context = tuple(context)
        else:
            context = tuple(t if t in self.vocab else self.unk for t in context)
        if not len(context):
            raise ValueError("context must have at least one token")
        return self.trie.log_prob(context)

    def to_prob_list(self):
        return self.trie.to_prob_list()

    def renormalize_backoffs(self):
        r"""Ensure backoffs induce a valid probability distribution

        Backoff models follow the same recursive formula for determining the probability
        of the next token:

        .. math::

            Pr(w_n|w_1, \ldots w_{n-1}) = \begin{cases}
                Entry(w_1, \ldots, w_n) &
                                    \text{if }Entry(\ldots)\text{ exists}\\
                Backoff(w_1, \ldots, w_{n-1})P(w_n|w_{n-1}, \ldots, w_2) &
                                    \text{otherwise}
            \end{cases}

        Calling this method renormalizes :math:`Backoff(\ldots)` such that,
        where possible, :math:`\sum_w Pr(w|\ldots) = 1`
        """
        return self.trie.renormalize_backoffs()

    def relative_entropy_pruning(self, threshold, _srilm_hacks=False):
        r"""Prune n-grams with negligible impact on model perplexity

        This method iterates through n-grams, highest order first, looking to absorb
        their explicit probabilities into a backoff. The language model defines a
        distribution over sequences, :math:`s \sim p(\cdot|\theta)`. Assuming this is
        the true distribution of sequences, we can define an approximation of
        :math:`p(\cdot)`, :math:`q(\cdot)`, as one that replaces one explicit n-gram
        probability with a backoff. [stolcke2000]_ defines the relative change in model
        perplexity as:

        .. math::

            \Delta PP = e^{D_{KL}(p\|q)} - 1

        Where :math:`D_{KL}` is the KL-divergence between the two distributions. This
        method will prune an n-gram whenever the change in model perplexity is
        negligible (below `threshold`). More details can be found in [stolcke2000]_.

        Parameters
        ----------
        threshold : float

        References
        ----------
        .. [stolcke2000] A. Stolcke "Entropy-based pruning of Backoff Language Models,"
           ArXiv ePrint, 2000
        """
        return self.trie.relative_entropy_pruning(threshold, _srilm_hacks=_srilm_hacks)

    def sequence_perplexity(self, sequence, include_delimiters=True):
        r"""Return the perplexity of the sequence using this language model

        Given a `sequence` of tokens ``[w_1, w_2, ..., w_N]``, the perplexity of the
        sequence is

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
        """
        sequence = list(sequence)
        if include_delimiters:
            if not len(sequence) or sequence[0] != self.sos:
                sequence.insert(0, self.sos)
            if sequence[-1] != self.eos:
                sequence.append(self.eos)
        if not len(sequence):
            raise ValueError(
                "sequence cannot be empty when include_delimiters is False"
            )
        N = len(sequence)
        if sequence[0] == self.sos:
            N -= 1
        return 10.0 ** (-self.log_prob(sequence) / N)

    def corpus_perplexity(self, corpus, include_delimiters=True):
        r"""Calculate the perplexity of an entire corpus using this model

        A `corpus` is a sequence of sequences ``[s_1, s_2, ..., s_S]``. Each
        sequence ``s_i`` is a sequence of tokens ``[w_1, w_2, ..., w_N_i]``.
        Assuming sentences are independent,

        .. math::

            Pr(corpus) = Pr(s_1, s_2, ..., s_S) = Pr(s_1)Pr(s_2)...Pr(s_S)

        We calculate the corpus perplexity as the inverse corpus probablity
        normalized by the total number of tokens in the corpus. Letting
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
        """
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
                warnings.warn("skipping empty sequence (include_delimiters is False)")
                continue
            N = len(sequence)
            if sequence[0] == self.sos:
                N -= 1
            M += N
            joint += self.log_prob(sequence)
        return 10.0 ** (-joint / M)

    def prune_by_threshold(self, lprob):
        """Prune n-grams with a log-probability <= a threshold

        This method prunes n-grams with a conditional log-probability less than or equal
        to some fixed threshold. The reclaimed probability mass is sent to the
        (n-1)-gram's backoff.

        This method never prunes unigrams. Further, it cannot prune n-grams which are a
        prefix of some higher-order n-gram that has a conditional probability above that
        threshold, since the higher-order n-gram may have need of the lower-order's
        backoff.

        Parameters
        ----------
        lprob : float
            The base-10 log probability of conditionals, below or at which the n-gram
            will be pruned.
        """
        self.trie.prune_by_threshold(lprob)

    def prune_by_name(self, to_prune, eps_lprob=-99.999):
        """Prune n-grams by name

        This method prunes n-grams of arbitrary order by name. For n-grams of order > 1,
        the reclaimed probability mass is allotted to the appropriate backoff. For
        unigrams, the reclaimed probability mass is distributed uniformly across the
        remaining unigrams.

        This method prunes nodes by setting their probabilities a small log-probability
        (`eps_lprob`), then calling :func:`prune_by_threshold` with that small
        log-probability. This ensures we do not remove the backoff of higher-order
        n-grams (instead setting the probability of "pruned" nodes very low), and gets
        rid of lower-order nodes that were previously "pruned" but had to exist for
        their backoff when their backoff is now no longer needed.

        Unigrams are never fully pruned - their log probabilities are set to
        `eps_lprob`.

        Parameters
        ----------
        to_prune : set
            A set of all n-grams of all orders to prune.
        eps_lprob : float, optional
            A base 10 log probability considered negligible
        """
        self.trie.prune_by_name(to_prune, eps_lprob)


def write_arpa(prob_list, out=sys.stdout):
    """Convert an lists of n-gram probabilities to arpa format

    The inverse operation of :func:`pydrobert.torch.util.parse_arpa_lm`

    Parameters
    ----------
    prob_list : list of dict
    out : file or str, optional
        Path or file object to output to
    """
    if isinstance(out, str):
        with open(out, "w") as f:
            return write_arpa(prob_list, f)
    entries_by_order = []
    for idx, dict_ in enumerate(prob_list):
        entries = sorted((k, v) if idx else ((k,), v) for (k, v) in dict_.items())
        entries_by_order.append(entries)
    out.write("\\data\\\n")
    for idx in range(len(entries_by_order)):
        out.write("ngram {}={}\n".format(idx + 1, len(entries_by_order[idx])))
    out.write("\n")
    for idx, entries in enumerate(entries_by_order):
        out.write("\\{}-grams:\n".format(idx + 1))
        if idx == len(entries_by_order) - 1:
            for entry in entries:
                out.write("{:f} {}\n".format(entry[1], " ".join(entry[0])))
        else:
            for entry in entries:
                out.write(
                    "{:f} {} {:f}\n".format(
                        entry[1][0], " ".join(entry[0]), entry[1][1]
                    )
                )
        out.write("\n")
    out.write("\\end\\\n")


def ngram_counts_to_prob_list_mle(ngram_counts, eps_lprob=-99.999):
    r"""Determine probabilities based on MLE of observed n-gram counts

    For a given n-gram :math:`p, w`, where :math:`p` is a prefix, :math:`w` is the next
    word, the maximum likelihood estimate of the last token given the prefix is:

    .. math::

        Pr(w | p) = C(p, w) / (\sum_w' C(p, w'))

    Where :math:`C(x)` Is the count of the sequence :math:`x`. Many counts will be zero,
    especially for large n-grams or rare words, making this a not terribly generalizable
    solution.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to unigram counts
        in a corpus, ``ngram_counts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
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
    To be compatible with back-off models, MLE estimates assign a negligible backoff
    probability (`eps_lprob`) to n-grams where necessary. This means the probability
    mass might not exactly sum to one.
    """
    return ngram_counts_to_prob_list_add_k(ngram_counts, eps_lprob=eps_lprob, k=0.0)


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


def ngram_counts_to_prob_list_add_k(ngram_counts, eps_lprob=-99.999, k=0.5):
    r"""MLE probabilities with constant discount factor added to counts

    Similar to :func:`ngram_counts_to_prob_list_mle`, but with a constant added to each
    count to smooth out probabilities:

    .. math::

        Pr(w|p) = (C(p,w) + k)/(\sum_w' C(p, w') + k|V|)

    Where :math:`p` is a prefix, :math:`w` is the next word, and :math:`V` is the
    vocabulary set. The initial vocabulary set is determined from the unique unigrams
    :math:`V = U`. The bigram vocabulary set is the Cartesian product :math:`V = U
    \times U`, trigrams :math:`V = U \times U \times U`, and so on.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to unigram counts
        in a corpus, ``ngram_counts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
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
    """
    max_order = len(ngram_counts) - 1
    if not len(ngram_counts):
        raise ValueError("At least unigram counts must exist")
    vocab = set(ngram_counts[0])
    prob_list = []
    for order, counts in enumerate(ngram_counts):
        probs = _get_cond_mle(order, counts, vocab, k)
        if not order:
            for v in vocab:
                probs.setdefault((v,), eps_lprob)
        if order != max_order:
            probs = dict((ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def _log10sumexp(*args):
    if len(args) > 1:
        return _log10sumexp(args)
    args = np.array(args, dtype=float, copy=False)
    x = args[0]
    if np.any(np.isnan(x)):
        return np.nan
    if np.any(np.isposinf(x)):
        return np.inf
    x = x[np.isfinite(x)]
    if not len(x):
        return 0.0
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
    Z_rp1 = 2.0 * N_r[1:-1]
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
                log_abs_diff = -float("inf")
            log_z = C + log_rp1 - log_Nr + 0.5 * log_Nrp1
            log_z += 0.5 * np.log1p(N_r[r_ + 1] / N_r[r_]) / ln_10
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
    log_Np = np.log10((N_r[1:-1] * 10 ** (log_r_star[1:] - max_log_r_star)).sum())
    log_Np += max_log_r_star
    log_p_0 = log_r_star[0] - log_N
    log_r_star[1:] += -log_Np + np.log10(1 - 10 ** log_p_0) + log_N

    return log_r_star


def ngram_counts_to_prob_list_simple_good_turing(ngram_counts, eps_lprob=-99.999):
    r"""Determine probabilities based on n-gram counts using simple good-turing

    Simple Good-Turing smoothing discounts counts of n-grams according to the following
    scheme:

    .. math::

        r^* = (r + 1) N_{r + 1} / N_r

    Where :math:`r` is the original count of the n-gram in question, :math:`r^*` the
    discounted, and :math:`N_r` is the count of the number of times any n-gram had a
    count `r`.

    When :math:`N_r` becomes sparse, it is replaced with a log-linear regression of
    :math:`N_r` values, :math:`S(r) = a + b \log r`. :math:`r^*` for :math:`r > 0` are
    renormalized so that :math:`\sum_r N_r r^* = \sum_r N_r r`.

    We assume a closed vocabulary and that, for any order n-gram, :math:`N_0` is the
    size of the set of n-grams with frequency zero. This method differs from traditional
    Good-Turing, which assumes one unseen "event" (i.e. n-gram) per level. See below
    notes for more details.

    If, for a given order of n-gram, none of the terms have frequency zero, this
    function will warn and use MLEs.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to unigram counts
        in a corpus, ``ngram_counts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability."

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Notes
    -----
    The traditional definition of Good-Turing is somewhat vague about how to assign
    probability mass among unseen events. By setting :math:`r^* = N_1 / N` for :math:`r
    = 0`, it's implicitly stating that :math:`N_0 = 1`, that is, there's only one
    possible unseen event. This is consistent with introducing a special token, e.g.
    ``"<unk>"``, that does not occur in the corpus. It also collapses unseen n-grams
    into one event.

    We cannot bootstrap the backoff penalty to be the probability of the unseen term
    because the backoff will be combined with a lower-order estimate, and Good-Turing
    uses a fixed unseen probability.

    As our solution, we assume the vocabulary is closed. Any term that appears zero
    times is added to :math:`N_0`. If all terms appear, then :math:`N_0 = 0` and we
    revert to the MLE. While you can simulate the traditional Good-Turing at the
    unigram-level by introducing ``"<unk>"`` with count 0, this will not hold for
    higher-order n-grams.

    Warnings
    --------
    This function manually defines all n-grams of the target order given a vocabulary.
    This means that higher-order n-grams will be very large.

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
    .. [gale1995] W. A. Gale and G. Sampson, "Good‐Turing frequency estimation without
       tears," Journal of Quantitative Linguistics, vol. 2, no. 3, pp. 217-237, Jan.
       1995.
    """
    if len(ngram_counts) < 1:
        raise ValueError("At least unigram counts must exist")
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
                c = 10.0 ** log_r_stars[r]
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
                "No {}-grams were missing. Using MLE instead" "".format(order + 1)
            )
            probs = _get_cond_mle(order, counts, vocab, 0)
        if order != max_order:
            probs = dict((ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
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
    with np.errstate(divide="ignore", invalid="ignore"):
        log_Nr = np.log10(N_r)
        log_rp1 = np.log10(r + 1)
        log_r_star = log_rp1[:-1] + log_Nr[1:] - log_Nr[:-1]
    if k + 1 < len(N_r):
        log_d_rp1 = np.zeros(max_r, dtype=float)
        log_num_minu = log_r_star[1 : k + 1] - log_rp1[:k]
        log_subtra = np.log10(k + 1) + log_Nr[k + 1] - log_Nr[1]
        if log_subtra >= 0:
            raise ValueError("Your corpus is too small for this")
        # np.log10((10 ** (x - max_)).sum()) + max_
        log_num = log_num_minu + np.log1p(
            -(10 ** (log_subtra - log_num_minu))
        ) / np.log(10)
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
    ngram_counts, k=7, eps_lprob=-99.999, _cmu_hacks=False
):
    r"""Determine probabilities based on Katz's backoff algorithm

    Kat'z backoff algorithm determines the conditional probability of the last token in
    n-gram :math:`w = (w_1, w_2, ..., w_n)` as

    .. math::

        Pr_{BO}(w_n|w_{n-1}, w_{n-2} ..., w_1) = \begin{cases}
            d_w Pr_{MLE}(w_n|w_{n-1}, w_{n-1}, ..., w_1) & \text{if }C(w) > 0
            \alpha(w_1, ..., w_{n-1}) Pr_{BO}(w_n|w_{n-1}, ..., w_2)&
                                                                    \text{else}
        \end{cases}

    Where :math:`Pr_{MLE}` is the maximum likelihood estimate (based on frequencies),
    :math:`d_w` is some discount factor (based on Good-Turing for low-frequency
    n-grams), and :math:`\alpha` is an allowance of the leftover probability mass from
    discounting.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to unigram counts
        in a corpus, ``ngram_counts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    k : int, optional
        `k` is a threshold such that, if :math:`C(w) > k`, no discounting will be
        applied to the term. That is, the probability mass assigned for backoff will be
        entirely from n-grams s.t. :math:`C(w) \leq k`.
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability."

    Warnings
    --------
    If the counts of the extensions of a prefix are all above `k`, no discounting will
    be applied to those counts, meaning no probability mass can be assigned to unseen
    events.

    For example, in the Brown corpus, "Hong" is always followed by "Kong". The bigram
    "Hong Kong" occurs something like 10 times, so it's not discounted. Thus
    :math:`P_{BO}(Kong|Hong) = 1` :math:`P_{BO}(not Kong|Hong) = 0`.

    A :class:`UserWarning` will be issued whenever ths happens. If this bothers you, you
    could try increasing `k` or, better yet, abandon Katz Backoff altogether.

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
    .. [katz1987] S. Katz, "Estimation of probabilities from sparse data for the
       language model component of a speech recognizer," IEEE Transactions on Acoustics,
       Speech, and Signal Processing, vol. 35, no. 3, pp. 400-401, Mar. 1987.
    """
    if len(ngram_counts) < 1:
        raise ValueError("At least unigram counts must exist")
    if k < 1:
        raise ValueError("k too low")
    prob_list = []
    max_order = len(ngram_counts) - 1
    probs = _get_cond_mle(0, ngram_counts[0], set(ngram_counts[0]), 0)
    if 0 != max_order:
        probs = dict((ngram, (prob, 0.0)) for (ngram, prob) in probs.items())
    prob_list.append(probs)
    log_r_stars = [
        _get_katz_discounted_counts(counts, k) for counts in ngram_counts[1:]
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
                if len(children) == 1 and ngram_counts[order][children[0]] > k:
                    for oo in range(order):
                        pp = prefix[: oo + 1]
                        if not oo:
                            pp = pp[0]
                        ngram_counts[oo][pp] += 1
    for order in range(1, len(ngram_counts)):
        counts = ngram_counts[order]
        probs = dict()
        # P_katz(w|pr) = C*(pr, w) / \sum_x C*(pr, x) if C(pr, w) > 0
        #                alpha(pr) Pr_katz(w|pr[1:]) else
        # alpha(pr) = (1 - sum_{c(pr, w) > 0} Pr_katz(w|pr)
        #             / (1 - sum_{c(pr, w) > 0} Pr_katz(w|pr[1:]))
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
                lg_num_subtras.get(ngram[:-1], -np.inf), log_r_star
            )
            lg_den_subtras[ngram[:-1]] = _log10sumexp(
                lg_den_subtras.get(ngram[:-1], -np.inf), prob_list[-1][ngram[1:]][0]
            )
            lg_pref_counts[ngram[:-1]] = _log10sumexp(
                lg_pref_counts.get(ngram[:-1], -np.inf), np.log10(r)
            )
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
            num_subtra = 10.0 ** (lg_num_subtra - lg_norm)
            den_subtra = 10.0 ** lg_den_subtra
            if np.isclose(den_subtra, 1.0):  # 1 - den_subtra = 0
                # If the denominator is zero, it means nothing we're backing
                # off to has a nonzero probability. It doesn't really matter
                # what we put here, but let's not warn about it (we've already
                # warned about the prefix)
                log_alpha = 0.0
            elif np.isclose(num_subtra, 1.0):
                warnings.warn(
                    "Cannot back off to prefix {}. Will assign negligible "
                    "probability. If this is an issue, try increasing k"
                    "".format(prefix)
                )
                # If the numerator is zero and the denominator is nonzero,
                # this means we did not discount any probability mass for
                # unseen terms. The only way to make a proper distribution is
                # to set alpha to zero
                log_alpha = eps_lprob
            else:
                log_alpha = np.log1p(-num_subtra) - np.log1p(-den_subtra)
                log_alpha /= np.log(10)
            log_prob, bad_backoff = prob_list[-1][prefix]
            prob_list[-1][prefix] = (log_prob, log_alpha)
        if order != max_order:
            probs = dict((ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def _optimal_deltas(counts, y):
    N_r = Counter(counts.values())
    if not all(N_r[r] for r in range(1, y + 2)):
        raise ValueError(
            "Your dataset is too small to use the default discount "
            "(or maybe you removed the hapax before estimating probs?)"
        )
    Y = N_r[1] / (N_r[1] + 2 * N_r[2])
    deltas = [r - (r + 1) * Y * N_r[r + 1] / N_r[r] for r in range(1, y + 1)]
    if any(d <= 0.0 for d in deltas):
        raise ValueError(
            "Your dataset is too small to use the default discount "
            "(or maybe you removed the hapax before estimating probs?)"
        )
    return deltas


def _absolute_discounting(ngram_counts, deltas, to_prune):
    V = len(set(ngram_counts[0]) - to_prune)
    prob_list = [{tuple(): (-np.log10(V), 0.0)}]
    max_order = len(ngram_counts) - 1
    for order, counts, delta in zip(range(len(ngram_counts)), ngram_counts, deltas):
        delta = np.array(delta, dtype=float)
        n_counts = dict()
        d_counts = dict()
        pr2bin = dict()
        for ngram, count in counts.items():
            in_prune = ngram in to_prune
            if not order:
                ngram = (ngram,)
                if not in_prune:
                    n_counts.setdefault(ngram, 0)
            if not count:
                continue
            bin_ = min(count - 1, len(delta) - 1)
            d = delta[bin_]
            assert count - d >= 0.0
            prefix = ngram[:-1]
            d_counts[prefix] = d_counts.get(prefix, 0) + count
            prefix_bins = pr2bin.setdefault(prefix, np.zeros(len(delta) + 1))
            if in_prune:
                prefix_bins[-1] += count
            else:
                prefix_bins[bin_] += 1
                n_counts[ngram] = count - d
        for prefix, prefix_bins in pr2bin.items():
            if (order == 1 and prefix[0] in to_prune) or prefix in to_prune:
                continue
            with np.errstate(divide="ignore"):
                prefix_bins = np.log10(prefix_bins)
                prefix_bins[:-1] += np.log10(delta)
            gamma = _log10sumexp(prefix_bins)
            gamma -= np.log10(d_counts[prefix])
            lprob, bo = prob_list[-1][prefix]
            prob_list[-1][prefix] = (lprob, gamma)
        probs = dict()
        for ngram, n_count in n_counts.items():
            prefix = ngram[:-1]
            if n_count:
                lprob = np.log10(n_count) - np.log10(d_counts[prefix])
            else:
                lprob = -float("inf")
            lower_order = prob_list[-1][prefix][1]  # gamma(prefix)
            lower_order += prob_list[-1][ngram[1:]][0]  # Pr(w|prefix[1:])
            lprob = _log10sumexp(lprob, lower_order)
            if order != max_order:
                # the only time the backoff will not be recalculated is if
                # no words ever follow the prefix. In this case, we actually
                # want to back off to a lower-order estimate
                # Pr(w|prefix) = P(w|prefix[1:]). We can achieve this by
                # setting gamma(prefix) = 1 and treating the higher-order
                # contribution to Pr(w|prefix) as zero
                lprob = (lprob, 0.0)
            probs[ngram] = lprob
        prob_list.append(probs)
    del prob_list[0]  # zero-th order
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def ngram_counts_to_prob_list_absolute_discounting(
    ngram_counts, delta=None, to_prune=set()
):
    r"""Determine probabilities from n-gram counts using absolute discounting

    Absolute discounting (based on the formulation in [chen1999]_) interpolates between
    higher-order and lower-order n-grams as

    .. math::

        Pr_{abs}(w_n|w_{n-1}, \ldots w_1) =
            \frac{\max\left\{C(w_1, \ldots, w_n) - \delta, 0\right\}}
                {\sum_w' C(w_1, \ldots, w_{n-1}, w')}
            - \gamma(w_1, \ldots, w_{n-1})
               Pr_{abs}(w_n|w_{n-1}, \ldots, w_2)

    Where :math:`\gamma` are chosen so :math:`Pr_{abs}(\cdot)` sum to one. For the base
    case, we pretend there's such a thing as a zeroth-order n-gram, and
    :math:`Pr_{abs}(\emptyset) = 1 / \left\|V\right\|`.

    Letting

    .. math::

        N_c = \left|\left\{
                (w'_1, \ldots, w'_n): C(w'_1, \ldots, w'_n) = c
            \right\}\right|

    :math:`\delta` is often chosen to be :math:`\delta = N_1 / (N_1 + 2N_2)` for a given
    order n-gram. We can use different :math:`\delta` for different orders of the
    recursion.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to unigram counts
        in a corpus, ``ngram_counts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    delta : float or tuple or :obj:`None`, optional
        The absolute discount to apply to non-zero values. `delta` can take one of three
        forms: a :class:`float` to be used identically for all orders of the recursion;
        :obj:`None` specifies that the above formula for calculating `delta` should be
        used separately for each order of the recursion; or a tuple of length
        ``len(ngram_counts)``, where each element is either a :class:`float` or
        :obj:`None`, specifying either a fixed value or the default value for every
        order of the recursion (except the zeroth-order), unigrams first.
    to_prune : set, optional
        A set of n-grams that will not be explicitly set in the return value. This
        differs from simply removing those n-grams from `ngram_counts` in some key ways.
        First, pruned counts can still be used to calculate default `delta` values.
        Second, as per [chen1999]_, pruned counts are still summed in the denominator,
        :math:`\sum_w' C(w_1, \ldots, w_{n-1}, w')`, which then make their way into the
        numerator of :math:`gamma(w_1, \ldots, w_{n-1})`.

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
    >>> ngram_counts[0]['a']
    10
    >>> sum(ngram_counts[0].values())
    27
    >>> len(ngram_counts[0])
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

    References
    ----------
    .. [chen1999] S. F. Chen and J. Goodman, "An empirical study of smoothing
       techniques for language modeling," Computer Speech & Language, vol. 13,
       no. 4, pp. 359-394, Oct. 1999, doi: 10.1006/csla.1999.0128.
    """
    if len(ngram_counts) < 1:
        raise ValueError("At least unigram counts must exist")
    if not isinstance(delta, Iterable):
        delta = (delta,) * len(ngram_counts)
    if len(delta) != len(ngram_counts):
        raise ValueError(
            "Expected {} deltas, got {}".format(len(ngram_counts), len(delta))
        )
    delta = tuple(
        _optimal_deltas(counts, 1) if d is None else [d]
        for (d, counts) in zip(delta, ngram_counts)
    )
    return _absolute_discounting(ngram_counts, delta, to_prune)


def ngram_counts_to_prob_list_kneser_ney(
    ngram_counts, delta=None, sos=None, to_prune=set()
):
    r"""Determine probabilities from counts using Kneser-Ney(-like) estimates

    Chen and Goodman's implemented Kneser-Ney smoothing [chen1999]_ is the same as
    absolute discounting, but with lower-order n-gram counts ((n-1)-grams, (n-2)-grams,
    etc.) replaced with adjusted counts:

    .. math::

        C'(w_1, \ldots, w_k) = \begin{cases}
            C(w_1, \ldots, w_k) & k = n \lor w_1 = sos \\
            \left|\left\{v : C(v, w_1, \ldots, w_k) > 0\right\}\right| & else\\
        \end{cases}

    The adjusted count is the number of unique prefixes the n-gram can occur with. We do
    not modify n-grams starting with the start-of-sequence `sos` token (as per
    [heafield2013]_) as they cannot have a preceding context.

    By default, modified Kneser-Ney is performed, which uses different absolute
    discounts for different adjusted counts:

    .. math::

        Pr_{KN}(w_1, \ldots, w_n) =
            \frac{C'(w_1, \ldots, w_n) - \delta(C'(w_1, \ldots, w_n))}
                 {\sum_{w'} C'(w_1, \ldots, w_{n-1}, w')}
            + \gamma(w_1, \ldots, w_{n-1}) Pr_{KN}(w_n|w_1, \ldots, w_{n-1})

    :math:`\gamma` are chosen so that :math:`Pr_{KN}(\cdot)` sum to one. As a base case,
    :math:`Pr_{KN}(\emptyset) = 1 / \left\|V\right\|`.

    Letting :math:`N_c` be defined as in
    :func:`ngram_counts_to_prob_list_absolute_discounting`, and :math:`y = N_1 / (N_1 +
    2 N_2)`, the default value for :math:`\delta(\cdot)` is

    .. math::

        \delta(k) = k - (k + 1) y (N_{k + 1} / N_k)

    Where we set :math:`\delta(0) = 0` and :math:`\delta(>3) = \delta(3)`.

    Parameters
    ----------
    ngram_counts : sequence
        A list of dictionaries. ``ngram_counts[0]`` should correspond to unigram counts
        in a corpus, ``ngram_counts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    delta : float or tuple or :obj:`None`, optional
        The absolute discount to apply to non-zero values. `delta` may be a
        :class:`float`, at which point a fixed discount will be applied to all orders of
        the recursion. If :obj:`None`, the default values defined above will be
        employed. `delta` can be a :class:`tuple` of the same length as
        ``len(ngram_counts)``, which can be used to specify discounts at each level of
        the recursion (excluding the zero-th order), unigrams first. If an element is a
        :class:`float`, that fixed discount will be applied to all nonzero counts at
        that order. If :obj:`None`, `delta` will be calculated in the default manner for
        that order. Finally, an element can be a :class:`tuple` itself of positive
        length. In this case, the elements of that tuple will correspond to the values
        of :math:`\delta(k)` where the i-th indexed element is :math:`\delta(i+1)`.
        Counts above the last :math:`\delta(k)` will use the same discount as the last
        :math:`\delta(k)`. Elements within the tuple can be either :class:`float`
        (use this value) or :obj:`None` (use defafult)
    sos : str or :obj:`None`, optional
        The start-of-sequence symbol. Defaults to ``'<S>'`` if that symbol is in the
        vocabulary, otherwise ``'<s>'``
    to_prune : set, optional
        A set of n-grams that will not be explicitly set in the return value. This
        differs from simply removing those n-grams from `ngram_counts` in some key ways.
        First, nonzero counts of pruned n-grams are used when calculating adjusted
        counts of the remaining terms. Second, pruned counts can still be used to
        calculate default `delta` values. Third, as per [chen1999]_, pruned counts are
        still summed in the denominator, :math:`\sum_w' C(w_1, \ldots, w_{n-1}, w')`,
        which then make their way into the numerator of
        :math:`gamma(w_1, \ldots, w_{n-1})`.

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
    >>>     for order in range(1, 5)
    >>> ]
    >>> ngram_counts[0]
    Counter({'a': 10, ' ': 6, 'n': 4, 'm': 2, 'p': 2, 'l': 2, 'c': 1})
    >>> adjusted_unigrams = dict(
    >>>     (k, sum(1 for kk in ngram_counts[1] if kk[1] == k))
    >>>     for k in ngram_counts[0]
    >>> )
    >>> adjusted_unigrams
    {'a': 6, ' ': 3, 'm': 2, 'n': 1, 'p': 1, 'l': 2, 'c': 1}
    >>> adjusted_bigrams = dict(
    >>>     (k, sum(1 for kk in ngram_counts[2] if kk[1:] == k))
    >>>     for k in ngram_counts[1]
    >>> )
    >>> adjusted_trigrams = dict(
    >>>     (k, sum(1 for kk in ngram_counts[3] if kk[1:] == k))
    >>>     for k in ngram_counts[2]
    >>> )
    >>> len(adjusted_unigrams)
    7
    >>> sum(adjusted_unigrams.values())
    16
    >>> sum(1 for k in adjusted_bigrams if k[0] == 'a')
    4
    >>> sum(v for k, v in adjusted_bigrams.items() if k[0] == 'a')
    7
    >>> prob_list = ngram_counts_to_prob_list_kneser_ney(
    >>>     ngram_counts, delta=.5)
    >>> # gamma_0() = 0.5 * 7 / 16
    >>> # Pr(a) = (6 - 0.5) / 16 + 0.5 * (7 / 16) * (1 / 7)
    >>> # Pr(a) = 3 / 8
    >>> # BO(a) = gamma_1(a) = 0.5 * 4 / 7 = 2 / 7
    >>> prob_list[0]['a']  # (log10 Pr(a), log10 BO(a))
    (-0.42596873227228116, -0.5440680443502757)
    >>> adjusted_bigrams[('a', 'n')]
    4
    >>> adjusted_unigrams['n']
    1
    >>> sum(1 for k in adjusted_trigrams if k[:2] == ('a', 'n'))
    2
    >>> sum(v for k, v in adjusted_trigrams.items() if k[:2] == ('a', 'n'))
    4
    >>> # Pr(n) = (1 - 0.5) / 16 + 0.5 (7 / 16) (1 / 7) = 1 / 16
    >>> # Pr(n|a) = (4 - 0.5) / 7 + gamma_1(a) Pr(n)
    >>> #         = (4 - 0.5) / 7 + (2 / 7) (1 / 16)
    >>> #         = (64 - 8 + 2) / 112
    >>> #         = 29 / 56
    >>> # BO(a, n) = gamma_2(a, n) = 0.5 (2 / 4) = 1 / 4
    (-0.2857900291072443, -0.6020599913279624)

    Notes
    -----
    As discussed in [chen1999]_, Kneser-Ney is usually formulated so that only unigram
    counts are adjusted. However, they themselves experiment with modified counts for
    all lower orders.

    References
    ----------
    .. [chen1999] S. F. Chen and J. Goodman, "An empirical study of smoothing techniques
       for language modeling," Computer Speech & Language, vol. 13, no. 4, pp. 359-394,
       Oct. 1999, doi: 10.1006/csla.1999.0128.
    .. [heafield2013] K. Heafield, I. Pouzyrevsky, J. H. Clark, and P. Koehn, "Scalable
       modified Kneser-Ney language model estimation,” in Proceedings of the 51st Annual
       Meeting of the Association for Computational Linguistics, Sofia, Bulgaria, 2013,
       vol. 2, pp. 690-696.
    """
    if len(ngram_counts) < 1:
        raise ValueError("At least unigram counts must exist")
    if not isinstance(delta, Iterable):
        delta = (delta,) * len(ngram_counts)
    if len(delta) != len(ngram_counts):
        raise ValueError(
            "Expected {} deltas, got {}".format(len(ngram_counts), len(delta))
        )
    if sos is None:
        sos = "<S>" if "<S>" in ngram_counts[0] else "<s>"
    new_ngram_counts = [ngram_counts[-1]]
    for order in range(len(ngram_counts) - 2, -1, -1):
        if order:
            new_counts = dict()
        else:  # preserve vocabulary
            new_counts = dict.fromkeys(ngram_counts[order], 0)
        for ngram, count in ngram_counts[order + 1].items():
            if not count:
                continue
            suffix = ngram[1:] if order else ngram[1]
            new_counts[suffix] = new_counts.get(suffix, 0) + 1
        new_counts.update(
            (k, v)
            for (k, v) in ngram_counts[order].items()
            if ((order and k[0] == sos) or (not order and k == sos))
        )
        new_ngram_counts.insert(0, new_counts)
    ngram_counts = new_ngram_counts
    delta = list(delta)
    for i in range(len(delta)):
        ds, counts = delta[i], ngram_counts[i]
        if ds is None:
            ds = (None, None, None)
        if not isinstance(ds, Iterable):
            ds = (ds,)
        ds = tuple(ds)
        try:
            last_deft = len(ds) - ds[::-1].index(None) - 1
            optimals = _optimal_deltas(counts, last_deft + 1)
            assert len(optimals) == len(ds)
            ds = tuple(y if x is None else x for (x, y) in zip(ds, optimals))
        except ValueError:
            if None in ds:
                raise
        delta[i] = ds
    return _absolute_discounting(ngram_counts, delta, to_prune)


def text_to_sents(
    text,
    sent_end_expr=r"[.?!]+",
    word_delim_expr=r"\W+",
    to_case="upper",
    trim_empty_sents=False,
):
    """Convert a block of text to a list of sentences, each a list of words

    Parameters
    ----------
    text : str
        The text to parse
    set_end_expr : str or re.Pattern, optional
        A regular expression indicating an end of a sentence. By default, this is one or
        more of the characters ".?!"
    word_delim_expr : str or re.Pattern, optional
        A regular expression used for splitting words. By default, it is one or more of
        any non-alphanumeric character (including ' and -). Any empty words are removed
        from the sentence
    to_case : {'lower', 'upper', :obj:`None`}, optional
        Convert all words to a specific case: ``'lower'`` is lower case, ``'upper'`` is
        upper case, anything else performs no conversion
    trim_empty_sents : bool, optional
        If :obj:`True`, any sentences with no words in them will be removed from the
        return value. The exception is an empty final string, which is always removed.

    Returns
    -------
    sents : list of tuples
        A list of sentences from `text`. Each sentence/element is actually a tuple of
        the words in the sentences
    """
    if not isinstance(sent_end_expr, re.Pattern):
        sent_end_expr = re.compile(sent_end_expr)
    if not isinstance(word_delim_expr, re.Pattern):
        word_delim_expr = re.compile(word_delim_expr)
    sents = sent_end_expr.split(text)
    i = 0
    while i < len(sents):
        sent = word_delim_expr.split(sents[i])
        sent = tuple(w for w in sent if w)
        if to_case == "lower":
            sent = tuple(w.lower() for w in sent)
        elif to_case == "upper":
            sent = tuple(w.upper() for w in sent)
        if trim_empty_sents and not sent:
            del sents[i]
        else:
            sents[i] = sent
            i += 1
    if sents and not sents[-1]:
        del sents[-1]
    return sents


def sents_to_ngram_counts(
    sents, max_order, sos="<S>", eos="</S>", count_unigram_sos=False
):
    """Count n-grams in sentence lists up to a maximum order

    Parameters
    ----------
    sents : list of tuples
        A list of sentences, where each sentence is a tuple of its words.
    max_order : int
        The maximum order (inclusive) of n-gram to count.
    sos : str, optional
        A token representing the start-of-sequence.
    eos : str, optional
        A token representing the end-of-sequence.
    count_unigram_sos : bool, optional
        If :obj:`False`, the unigram count of the start-of-sequence token will always be
        zero (though higher-order n-grams beginning with the SOS can have counts).

    Returns
    -------
    ngram_counts : list of dicts
        A list of length `max_order` where ``ngram_counts[0]`` is a dictionary of
        unigram counts, ``ngram_counts[1]`` of bigram counts, etc.

    Notes
    -----
    The way n-grams count start-of-sequence and end-of-sequence tokens differ from
    package to package. For example, some tools left-pad sentences with n - 1
    start-of-sequence tokens (e.g. making ``Pr(x|<s><s>)`` a valid conditional). Others
    only count sos and eos tokens for n > 1.

    This function adds one (and only one) sos and eos to the beginning and end of each
    sentence before counting n-grams with only one exception. By default,
    `count_unigram_sos` is set to :obj:`False`, meaning the start-of-sequence token will
    not be counted as a unigram. This makes sense in the word prediction context since a
    language model should never predict the next word to be the start-of-sequence token.
    Rather, it always exists prior to the first word being predicted. This exception can
    be disabled by setting `count_unigram_sos` to :obj:`True`.
    """
    if max_order < 1:
        raise ValueError("max_order ({}) must be >= 1".format(max_order))
    ngram_counts = [Counter() for _ in range(max_order)]
    ngram_counts[0].setdefault(sos, 0)
    for sent in sents:
        if {sos, eos} & set(sent):
            raise ValueError(
                "start-of-sequence ({}) or end-of-sequence ({}) found in "
                'sentence "{}"'.format(sos, eos, " ".join(sent))
            )
        sent = (sos,) + tuple(sent) + (eos,)
        for order, counter in zip(range(1, max_order + 1), ngram_counts):
            if order == 1:
                counter.update(sent if count_unigram_sos else sent[1:])
            else:
                counter.update(
                    sent[s : s + order] for s in range(len(sent) - order + 1)
                )
    return ngram_counts
