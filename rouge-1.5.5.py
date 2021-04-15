#! /usr/bin/env python

# Calculation code from https://github.com/Diego999/py-rouge. The code is supposed to
# be licensed by Apache 2.0, but does not have the requisite header in the source.
# The files located in resources/py-rouge are also in this repo.
#
# Best effort:
#
#   Copyright 2020 Diego Antognini
#
# Command-line wrapper code subject to Apache 2.0 as well by me:
#
#   Copyright 2020 Sean Robertson
#
# The code was also automatically formatted by Black and includes some of my own
# formatting to stop the linter from complaining.
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

import nltk
import os
import re
import itertools
import collections
import argparse
import xml.etree.ElementTree as et

from io import open


RESOURCES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", "py-rouge"
)


class Rouge:
    DEFAULT_METRICS = {"rouge-n"}
    DEFAULT_N = 1
    STATS = ["f", "p", "r"]
    AVAILABLE_METRICS = {"rouge-n", "rouge-l", "rouge-w"}
    AVAILABLE_LENGTH_LIMIT_TYPES = {"words", "bytes"}
    REMOVE_CHAR_PATTERN = re.compile("[^A-Za-z0-9]")

    # Hack to not tokenize "cannot" to "can not" and consider them different as in the
    # official ROUGE script
    KEEP_CANNOT_IN_ONE_WORD = re.compile("cannot")
    KEEP_CANNOT_IN_ONE_WORD_REVERSED = re.compile("_cannot_")

    WORDNET_KEY_VALUE = {}
    WORDNET_DB_FILEPATH = os.path.join(RESOURCES_PATH, "wordnet_key_value.txt")
    WORDNET_DB_FILEPATH_SPECIAL_CASE = os.path.join(
        RESOURCES_PATH, "wordnet_key_value_special_cases.txt"
    )
    WORDNET_DB_DELIMITER = "|"
    STEMMER = None

    def __init__(
        self,
        metrics=None,
        max_n=None,
        limit_length=True,
        length_limit=665,
        length_limit_type="bytes",
        apply_avg=True,
        apply_best=False,
        stemming=True,
        alpha=0.5,
        weight_factor=1.0,
        ensure_compatibility=True,
    ):
        """Handle the ROUGE score computation as in the official perl script.

        Note 1: Small differences might happen if the resampling of the perl script is
        not high enough (as the average depends on this).

        Note 2: Stemming of the official Porter Stemmer of the ROUGE perl script is
        slightly different and the Porter one implemented in NLTK. However, special
        cases of DUC 2004 have been traited. The solution would be to rewrite the whole
        perl stemming in python from the original script

        Parameters
        ----------
        metrics: {'rouge-n', 'rouge-l', 'rouge-w'}, optional
            What ROUGE score to compute.
        max_n: int, optional
            N-grams for ROUGE-N if specify.
        limit_length: bool, optional
            If the summaries must be truncated.
        length_limit: int, optional
            Number of the truncation where the unit is express int length_limit_Type. In
            bytes.
        length_limit_type: {'bytes', 'words'}, optional
            Unit of length_limit.
        apply_avg: bool, optional
            If we should average the score of multiple samples. If `apply_avg` and
            `apply_best` are :obj:`False`, then each ROUGE scores are independent.
        apply_best: bool, optional
            Take the best instead of the average.
        stemming: bool, optional
            Apply stemming to summaries.
        alpha: float, optional
            Alpha use to compute f1 score: ``P*R/((1-a)*P + a*R)``
        weight_factor: float, optional
            Weight factor to be used for ROUGE-W. Official rouge score defines it at
            1.2
        ensure_compatibility: bool, optional
            Use same stemmer and special "hacks" to product same results as in the
            official perl script (besides the number of sampling if not high enough)

        Raises
        ------
        ValueError
            If metric is not among AVAILABLE_METRICS, `length_limit_type` is not amon
            AVAILABLE_LENGTH_LIMIT_TYPES, or `weight_factor` < 0
        """
        self.metrics = metrics[:] if metrics is not None else Rouge.DEFAULT_METRICS
        for m in self.metrics:
            if m not in Rouge.AVAILABLE_METRICS:
                raise ValueError("Unknown metric '{}'".format(m))

        self.max_n = max_n if "rouge-n" in self.metrics else None
        # Add all rouge-n metrics
        if self.max_n is not None:
            index_rouge_n = self.metrics.index("rouge-n")
            del self.metrics[index_rouge_n]
            self.metrics += ["rouge-{}".format(n) for n in range(1, self.max_n + 1)]
        self.metrics = set(self.metrics)

        self.limit_length = limit_length
        if self.limit_length:
            if length_limit_type not in Rouge.AVAILABLE_LENGTH_LIMIT_TYPES:
                raise ValueError(
                    "Unknown length_limit_type '{}'".format(length_limit_type)
                )

        self.length_limit = length_limit
        if self.length_limit == 0:
            self.limit_length = False
        self.length_limit_type = length_limit_type
        self.stemming = stemming

        self.apply_avg = apply_avg
        self.apply_best = apply_best
        self.alpha = alpha
        self.weight_factor = weight_factor
        if self.weight_factor <= 0:
            raise ValueError("ROUGE-W weight factor must greater than 0.")
        self.ensure_compatibility = ensure_compatibility

        # Load static objects
        if len(Rouge.WORDNET_KEY_VALUE) == 0:
            Rouge.load_wordnet_db(ensure_compatibility)
        if Rouge.STEMMER is None:
            Rouge.load_stemmer(ensure_compatibility)

    @staticmethod
    def load_stemmer(ensure_compatibility):
        """Load the stemmer that is going to be used if stemming is enabled

        Parameters
        ----------
        ensure_compatibility: bool
            Use same stemmer and special "hacks" to product same results as in the
            official perl script (besides the number of sampling if not high enough)
        """
        Rouge.STEMMER = (
            nltk.stem.porter.PorterStemmer("ORIGINAL_ALGORITHM")
            if ensure_compatibility
            else nltk.stem.porter.PorterStemmer()
        )

    @staticmethod
    def load_wordnet_db(ensure_compatibility):
        """Load WordNet database

        In order to to apply specific rules instead of stemming + load file for special
        cases to ensure kind of compatibility (at list with DUC 2004) with the original
        stemmer used in the Perl script

        Parameters
        ----------
        ensure_compatibility: bool
            Use same stemmer and special "hacks" to product same results as in the
            official perl script (besides the number of sampling if not high enough)

        Raises
        ------
        FileNotFoundError
            If one of both databases is not found
        """
        files_to_load = [Rouge.WORDNET_DB_FILEPATH]
        if ensure_compatibility:
            files_to_load.append(Rouge.WORDNET_DB_FILEPATH_SPECIAL_CASE)

        for filepath in files_to_load:
            if not os.path.exists(filepath):
                raise FileNotFoundError("The file '{}' does not exist".format(filepath))

            with open(filepath, "r", encoding="utf-8") as fp:
                for line in fp:
                    k, v = line.strip().split(Rouge.WORDNET_DB_DELIMITER)
                    assert k not in Rouge.WORDNET_KEY_VALUE
                    Rouge.WORDNET_KEY_VALUE[k] = v

    @staticmethod
    def tokenize_text(text, language="english"):
        """Tokenize text in the specific language

        Parameters
        ----------
        text: str
            The string text to tokenize
        language: str, optional
            Language of the text

        Returns
        -------
        tokens : list of str
            List of tokens of text
        """
        return nltk.word_tokenize(text, language)

    @staticmethod
    def split_into_sentences(text, ensure_compatibility, language="english"):
        """Split text into sentences, using specified language.

        Uses PunktSentenceTokenizer

        Parameters
        ----------
        text: str
            The string text to tokenize
        ensure_compatibility: bool
            Split sentences by `'\n'` instead of NLTK sentence tokenizer model
        language: str, optional
            Language of the text

        Returns
        -------
        tokens : list of str
            List of tokens of text
        """
        if ensure_compatibility:
            return text.split("\n")
        else:
            return nltk.sent_tokenize(text, language)

    @staticmethod
    def stem_tokens(tokens):
        """Apply WordNetDB rules or stem each token of tokens in place

        Parameters
        ----------
        tokens: lis of str
            List of tokens to apply WordNetDB rules or to stem

        Returns
        -------
        tokens : list of str
          List of final stems (same list)
        """
        # Stemming & Wordnet apply only if token has at least 3 chars
        for i, token in enumerate(tokens):
            if len(token) > 0:
                if len(token) > 3:
                    if token in Rouge.WORDNET_KEY_VALUE:
                        token = Rouge.WORDNET_KEY_VALUE[token]
                    else:
                        token = Rouge.STEMMER.stem(token)
                    tokens[i] = token

        return tokens

    @staticmethod
    def _get_ngrams(n, text):
        """Calcualtes n-gram counts

        Parameters
        ----------
        n: int
            Which order n-grams to calculate.
        text: sequence of str
            An array of tokens

        Returns
        -------
        ngram_set : dict
            A set of n-grams with their number of occurences
        """
        # Modified from
        # https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        ngram_set = collections.defaultdict(int)
        max_index_ngram_start = len(text) - n
        for i in range(max_index_ngram_start + 1):
            ngram_set[tuple(text[i : i + n])] += 1
        return ngram_set

    @staticmethod
    def _split_into_words(sentences):
        """Splits multiple sentences into words and flattens the result

        Parameters
        ----------
        sentences: list of str

        Returns
        -------
        words : list of str
            A list of words (split by white space)
        """
        # Modified from
        # https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        return list(itertools.chain(*[_.split() for _ in sentences]))

    @staticmethod
    def _get_word_ngrams_and_length(n, sentences):
        """Calculates word n-grams for multiple sentences.

        Parameters
        ----------
        n: int
            Wich order n-grams to calculate
        sentences: list of str

        Returns
        -------
        ngram_set, tokens, num : dict, list of str, int
            A set of n-grams, their frequency and #n-grams in sentences
        """
        # Modified from
        # https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        assert len(sentences) > 0
        assert n > 0

        tokens = Rouge._split_into_words(sentences)
        return Rouge._get_ngrams(n, tokens), tokens, len(tokens) - (n - 1)

    @staticmethod
    def _get_unigrams(sentences):
        """Calcualtes uni-grams.

        Parameters
        ----------
        sentences: list of str

        Returns
        -------
        unigram_set, num : dict, int
            A set of n-grams and their frequencies
        """
        assert len(sentences) > 0

        tokens = Rouge._split_into_words(sentences)
        unigram_set = collections.defaultdict(int)
        for token in tokens:
            unigram_set[token] += 1
        return unigram_set, len(tokens)

    @staticmethod
    def _compute_p_r_f_score(
        evaluated_count,
        reference_count,
        overlapping_count,
        alpha=0.5,
        weight_factor=1.0,
    ):
        """Compute precision, recall and f1_score

        Using `alpha`: ``P*R / ((1-alpha)*P + alpha*R)``

        Parameters
        ----------
        evaluated_count: int
            #n-grams in the hypothesis
        reference_count: int
            #n-grams in the reference
        overlapping_count: int
            #n-grams in common between hypothesis and reference
        alpha: float, optional
            Value to use for the F1 score
        weight_factor: float
            Weight factor if we have use ROUGE-W

        Returns
        -------
        scores : dict
            A dict with 'p', 'r' and 'f' as keys fore precision, recall, f1 score
        """
        precision = (
            0.0 if evaluated_count == 0 else overlapping_count / float(evaluated_count)
        )
        if weight_factor != 1.0:
            precision = precision ** (1.0 / weight_factor)
        recall = (
            0.0 if reference_count == 0 else overlapping_count / float(reference_count)
        )
        if weight_factor != 1.0:
            recall = recall ** (1.0 / weight_factor)
        f1_score = Rouge._compute_f_score(precision, recall, alpha)
        return {"f": f1_score, "p": precision, "r": recall}

    @staticmethod
    def _compute_f_score(precision, recall, alpha=0.5):
        """Compute f1_score

        Using alpha: ``P*R / ((1-alpha)*P + alpha*R))``

        Parameters
        ----------
        precision: float
        recall: float
        alpha: float, optional

        Returns
        -------
        f1_score : float
        """
        return (
            0.0
            if (recall == 0.0 or precision == 0.0)
            else precision * recall / ((1 - alpha) * precision + alpha * recall)
        )

    @staticmethod
    def _compute_ngrams(evaluated_sentences, reference_sentences, n):
        """Computes n-grams overlap of two text collections of sentences.

        `Source
        <http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf>`_

        Parameters
        ----------
        evaluated_sentences: sequence of str
            The sentences that have been picked by the summarizer.
        reference_sentences: sequence of str
            The sentences from the referene set.
        n: int
            Order of n-gram.

        Returns
        -------
        evaluated_count, reference_count, overlapping_count : int, int, int
            Number of n-grams for evaluated_sentences, reference_sentences and
            intersection of both. Intersection of both counts multiple of occurences in
            n-grams match several times.

        Raises
        ------
        ValueError
            If a param has len <= 0
        """
        # Modified from
        # https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams, _, evaluated_count = Rouge._get_word_ngrams_and_length(
            n, evaluated_sentences
        )
        reference_ngrams, _, reference_count = Rouge._get_word_ngrams_and_length(
            n, reference_sentences
        )

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = set(evaluated_ngrams.keys()).intersection(
            set(reference_ngrams.keys())
        )
        overlapping_count = 0
        for ngram in overlapping_ngrams:
            overlapping_count += min(evaluated_ngrams[ngram], reference_ngrams[ngram])

        return evaluated_count, reference_count, overlapping_count

    @staticmethod
    def _compute_ngrams_lcs(
        evaluated_sentences, reference_sentences, weight_factor=1.0
    ):
        """Computes ROUGE-L (summary level) of two text collections of sentences.

        `Source
        <http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf>`_

        Parameters
        ----------
        evaluated_sentences: sequence of str
            The sentences that have been picked by the summarizer.
        reference_sentence: str
            One of the sentences in the reference summaries.
        weight_factor: float, optional
            Weight factor to be used for WLCS

        Returns
        -------
        evaluated_count, reference_count, overlapping_count : int, int, int
            Number of LCS n-grams for evaluated_sentences, reference_sentences and
            intersection of both. Intersection of both counts multiple of occurences in
            n-grams match several times.

        Raises
        ------
        ValueError
            If a param has len <= 0
        """

        def _lcs(x, y):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(int)
            dirs = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        vals[i, j] = vals[i - 1, j - 1] + 1
                        dirs[i, j] = "|"
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = "^"
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = "<"

            return vals, dirs

        def _wlcs(x, y, weight_factor):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(float)
            dirs = collections.defaultdict(int)
            lengths = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        length_tmp = lengths[i - 1, j - 1]
                        vals[i, j] = (
                            vals[i - 1, j - 1]
                            + (length_tmp + 1) ** weight_factor
                            - length_tmp ** weight_factor
                        )
                        dirs[i, j] = "|"
                        lengths[i, j] = length_tmp + 1
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = "^"
                        lengths[i, j] = 0
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = "<"
                        lengths[i, j] = 0

            return vals, dirs

        def _mark_lcs(mask, dirs, m, n):
            while m != 0 and n != 0:
                if dirs[m, n] == "|":
                    m -= 1
                    n -= 1
                    mask[m] = 1
                elif dirs[m, n] == "^":
                    m -= 1
                elif dirs[m, n] == "<":
                    n -= 1
                else:
                    raise UnboundLocalError("Illegal move")

            return mask

        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_unigrams_dict, evaluated_count = Rouge._get_unigrams(
            evaluated_sentences
        )
        reference_unigrams_dict, reference_count = Rouge._get_unigrams(
            reference_sentences
        )

        # Has to use weight factor for WLCS
        use_WLCS = weight_factor != 1.0
        if use_WLCS:
            evaluated_count = evaluated_count ** weight_factor
            reference_count = 0

        overlapping_count = 0.0
        for reference_sentence in reference_sentences:
            reference_sentence_tokens = reference_sentence.split()
            if use_WLCS:
                reference_count += len(reference_sentence_tokens) ** weight_factor
            hit_mask = [0 for _ in range(len(reference_sentence_tokens))]

            for evaluated_sentence in evaluated_sentences:
                evaluated_sentence_tokens = evaluated_sentence.split()

                if use_WLCS:
                    _, lcs_dirs = _wlcs(
                        reference_sentence_tokens,
                        evaluated_sentence_tokens,
                        weight_factor,
                    )
                else:
                    _, lcs_dirs = _lcs(
                        reference_sentence_tokens, evaluated_sentence_tokens
                    )
                _mark_lcs(
                    hit_mask,
                    lcs_dirs,
                    len(reference_sentence_tokens),
                    len(evaluated_sentence_tokens),
                )

            overlapping_count_length = 0
            for ref_token_id, val in enumerate(hit_mask):
                if val == 1:
                    token = reference_sentence_tokens[ref_token_id]
                    if (
                        evaluated_unigrams_dict[token] > 0
                        and reference_unigrams_dict[token] > 0
                    ):
                        evaluated_unigrams_dict[token] -= 1
                        reference_unigrams_dict[ref_token_id] -= 1

                        if use_WLCS:
                            overlapping_count_length += 1
                            if (
                                ref_token_id + 1 < len(hit_mask)
                                and hit_mask[ref_token_id + 1] == 0
                            ) or ref_token_id + 1 == len(hit_mask):
                                overlapping_count += (
                                    overlapping_count_length ** weight_factor
                                )
                                overlapping_count_length = 0
                        else:
                            overlapping_count += 1

        if use_WLCS:
            reference_count = reference_count ** weight_factor

        return evaluated_count, reference_count, overlapping_count

    def get_scores(self, hypothesis, references):
        """Compute precision, recall and f1 score between hypothesis and references

        Parameters
        ----------
        hypothesis: str or seqeuence of str
            Hypothesis summary.
        references: str or sequence of str
            Reference summary/ies, either one or many.

        Returns
        ------
        scores : dict
            Return precision, recall and f1 score between hypothesis and references

        Raises
        ------
        ValueError
            If a type of hypothesis is different than the one of reference or if a len
            of hypothesis is different than the one of reference.
        """
        if isinstance(hypothesis, str):
            hypothesis, references = [hypothesis], [references]

        if type(hypothesis) != type(references):
            raise ValueError("'hyps' and 'refs' are not of the same type")

        if len(hypothesis) != len(references):
            raise ValueError("'hyps' and 'refs' do not have the same length")
        scores = {}
        has_rouge_n_metric = (
            len([metric for metric in self.metrics if metric.split("-")[-1].isdigit()])
            > 0
        )
        if has_rouge_n_metric:
            scores.update(self._get_scores_rouge_n(hypothesis, references))

        has_rouge_l_metric = (
            len(
                [
                    metric
                    for metric in self.metrics
                    if metric.split("-")[-1].lower() == "l"
                ]
            )
            > 0
        )
        if has_rouge_l_metric:
            scores.update(self._get_scores_rouge_l_or_w(hypothesis, references, False))

        has_rouge_w_metric = (
            len(
                [
                    metric
                    for metric in self.metrics
                    if metric.split("-")[-1].lower() == "w"
                ]
            )
            > 0
        )
        if has_rouge_w_metric:
            scores.update(self._get_scores_rouge_l_or_w(hypothesis, references, True))

        return scores

    def _get_scores_rouge_n(self, all_hypothesis, all_references):
        """Computes precision, recall and f1 score between all hypothesis and references

        Parameters
        ----------
        all_hypothesis: list of str
            Hypothesis summaries.
        all_references: list of str
            Reference summary/ies, either string or list of strings (if multiple)

        Returns
        -------
        scores : dict
            Return precision, recall and f1 score between all hypothesis and references
        """
        metrics = [metric for metric in self.metrics if metric.split("-")[-1].isdigit()]

        if self.apply_avg or self.apply_best:
            scores = {metric: {stat: 0.0 for stat in Rouge.STATS} for metric in metrics}
        else:
            scores = {
                metric: [
                    {stat: [] for stat in Rouge.STATS}
                    for _ in range(len(all_hypothesis))
                ]
                for metric in metrics
            }

        for sample_id, (hypothesis, references) in enumerate(
            zip(all_hypothesis, all_references)
        ):
            assert isinstance(hypothesis, str)
            has_multiple_references = False
            if isinstance(references, list):
                has_multiple_references = len(references) > 1
                if not has_multiple_references:
                    references = references[0]

            # Prepare hypothesis and reference(s)
            hypothesis = self._preprocess_summary_as_a_whole(hypothesis)
            references = (
                [
                    self._preprocess_summary_as_a_whole(reference)
                    for reference in references
                ]
                if has_multiple_references
                else [self._preprocess_summary_as_a_whole(references)]
            )

            # Compute scores
            for metric in metrics:
                suffix = metric.split("-")[-1]
                n = int(suffix)

                # Aggregate
                if self.apply_avg:
                    # average model
                    total_hypothesis_ngrams_count = 0
                    total_reference_ngrams_count = 0
                    total_ngrams_overlapping_count = 0

                    for reference in references:
                        (
                            hypothesis_count,
                            reference_count,
                            overlapping_ngrams,
                        ) = Rouge._compute_ngrams(hypothesis, reference, n)
                        total_hypothesis_ngrams_count += hypothesis_count
                        total_reference_ngrams_count += reference_count
                        total_ngrams_overlapping_count += overlapping_ngrams

                    score = Rouge._compute_p_r_f_score(
                        total_hypothesis_ngrams_count,
                        total_reference_ngrams_count,
                        total_ngrams_overlapping_count,
                        self.alpha,
                    )

                    for stat in Rouge.STATS:
                        scores[metric][stat] += score[stat]
                else:
                    # Best model
                    if self.apply_best:
                        best_current_score = None
                        for reference in references:
                            (
                                hypothesis_count,
                                reference_count,
                                overlapping_ngrams,
                            ) = Rouge._compute_ngrams(hypothesis, reference, n)
                            score = Rouge._compute_p_r_f_score(
                                hypothesis_count,
                                reference_count,
                                overlapping_ngrams,
                                self.alpha,
                            )
                            if (
                                best_current_score is None
                                or score["r"] > best_current_score["r"]
                            ):
                                best_current_score = score

                        for stat in Rouge.STATS:
                            scores[metric][stat] += best_current_score[stat]
                    # Keep all
                    else:
                        for reference in references:
                            (
                                hypothesis_count,
                                reference_count,
                                overlapping_ngrams,
                            ) = Rouge._compute_ngrams(hypothesis, reference, n)
                            score = Rouge._compute_p_r_f_score(
                                hypothesis_count,
                                reference_count,
                                overlapping_ngrams,
                                self.alpha,
                            )
                            for stat in Rouge.STATS:
                                scores[metric][sample_id][stat].append(score[stat])

        # Compute final score with the average or the the max
        if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
            for metric in metrics:
                for stat in Rouge.STATS:
                    scores[metric][stat] /= len(all_hypothesis)

        return scores

    def _get_scores_rouge_l_or_w(self, all_hypothesis, all_references, use_w=False):
        """Computes precision, recall and f1 score between all hypothesis and references

        Parameters
        ----------
        all_hypothesis: list of str
            Hypothesis summaries.
        all_references: list of str
            Reference summary/ies, either string or list of strings (if multiple)
        use_w : bool, optional
            If true, compute ROUGE_W scores; otherwise ROUGE-L

        Returns
        -------
        scores : dict
            Return precision, recall and f1 score between all hypothesis and references
        """
        metric = "rouge-w" if use_w else "rouge-l"
        if self.apply_avg or self.apply_best:
            scores = {metric: {stat: 0.0 for stat in Rouge.STATS}}
        else:
            scores = {
                metric: [
                    {stat: [] for stat in Rouge.STATS}
                    for _ in range(len(all_hypothesis))
                ]
            }

        for sample_id, (hypothesis_sentences, references_sentences) in enumerate(
            zip(all_hypothesis, all_references)
        ):
            assert isinstance(hypothesis_sentences, str)
            has_multiple_references = False
            if isinstance(references_sentences, list):
                has_multiple_references = len(references_sentences) > 1
                if not has_multiple_references:
                    references_sentences = references_sentences[0]

            # Prepare hypothesis and reference(s)
            hypothesis_sentences = self._preprocess_summary_per_sentence(
                hypothesis_sentences
            )
            references_sentences = (
                [
                    self._preprocess_summary_per_sentence(reference)
                    for reference in references_sentences
                ]
                if has_multiple_references
                else [self._preprocess_summary_per_sentence(references_sentences)]
            )

            # Compute scores
            # Aggregate
            if self.apply_avg:
                # average model
                total_hypothesis_ngrams_count = 0
                total_reference_ngrams_count = 0
                total_ngrams_overlapping_count = 0

                for reference_sentences in references_sentences:
                    (
                        hypothesis_count,
                        reference_count,
                        overlapping_ngrams,
                    ) = Rouge._compute_ngrams_lcs(
                        hypothesis_sentences,
                        reference_sentences,
                        self.weight_factor if use_w else 1.0,
                    )
                    total_hypothesis_ngrams_count += hypothesis_count
                    total_reference_ngrams_count += reference_count
                    total_ngrams_overlapping_count += overlapping_ngrams

                score = Rouge._compute_p_r_f_score(
                    total_hypothesis_ngrams_count,
                    total_reference_ngrams_count,
                    total_ngrams_overlapping_count,
                    self.alpha,
                    self.weight_factor if use_w else 1.0,
                )
                for stat in Rouge.STATS:
                    scores[metric][stat] += score[stat]
            else:
                # Best model
                if self.apply_best:
                    best_current_score = None
                    best_current_score_wlcs = None
                    for reference_sentences in references_sentences:
                        (
                            hypothesis_count,
                            reference_count,
                            overlapping_ngrams,
                        ) = Rouge._compute_ngrams_lcs(
                            hypothesis_sentences,
                            reference_sentences,
                            self.weight_factor if use_w else 1.0,
                        )
                        score = Rouge._compute_p_r_f_score(
                            total_hypothesis_ngrams_count,
                            total_reference_ngrams_count,
                            total_ngrams_overlapping_count,
                            self.alpha,
                            self.weight_factor if use_w else 1.0,
                        )

                        if use_w:
                            reference_count_for_score = reference_count ** (
                                1.0 / self.weight_factor
                            )
                            overlapping_ngrams_for_score = overlapping_ngrams
                            score_wlcs = (
                                overlapping_ngrams_for_score / reference_count_for_score
                            ) ** (1.0 / self.weight_factor)

                            if (
                                best_current_score_wlcs is None
                                or score_wlcs > best_current_score_wlcs
                            ):
                                best_current_score = score
                                best_current_score_wlcs = score_wlcs
                        else:
                            if (
                                best_current_score is None
                                or score["r"] > best_current_score["r"]
                            ):
                                best_current_score = score

                    for stat in Rouge.STATS:
                        scores[metric][stat] += best_current_score[stat]
                # Keep all
                else:
                    for reference_sentences in references_sentences:
                        (
                            hypothesis_count,
                            reference_count,
                            overlapping_ngrams,
                        ) = Rouge._compute_ngrams_lcs(
                            hypothesis_sentences,
                            reference_sentences,
                            self.weight_factor if use_w else 1.0,
                        )
                        score = Rouge._compute_p_r_f_score(
                            hypothesis_count,
                            reference_count,
                            overlapping_ngrams,
                            self.alpha,
                            self.weight_factor,
                        )

                        for stat in Rouge.STATS:
                            scores[metric][sample_id][stat].append(score[stat])

        # Compute final score with the average or the the max
        if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
            for stat in Rouge.STATS:
                scores[metric][stat] /= len(all_hypothesis)

        return scores

    def _preprocess_summary_as_a_whole(self, summary):
        """Preprocessing of a summary as a whole

        - truncate text if enabled
        - tokenization
        - stemming if enabled
        - lowering

        Parameters
        ----------
        summary: str

        Returns
        -------
        preprocessed_summary : list of str
            The preprocessed summary.
        """
        sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)

        # Truncate
        if self.limit_length:
            # By words
            if self.length_limit_type == "words":
                summary = " ".join(sentences)
                all_tokens = summary.split()  # Counting as in the perls script
                summary = " ".join(all_tokens[: self.length_limit])

            # By bytes
            elif self.length_limit_type == "bytes":
                summary = ""
                current_len = 0
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)

                    if current_len + sentence_len < self.length_limit:
                        if current_len != 0:
                            summary += " "
                        summary += sentence
                        current_len += sentence_len
                    else:
                        if current_len > 0:
                            summary += " "
                        summary += sentence[: self.length_limit - current_len]
                        break
        else:
            summary = " ".join(sentences)

        summary = Rouge.REMOVE_CHAR_PATTERN.sub(" ", summary.lower()).strip()

        # Preprocess. Hack: because official ROUGE script bring "cannot" as "cannot" and
        # "can not" as "can not", we have to hack nltk tokenizer to not transform
        # "cannot/can not" to "can not"
        if self.ensure_compatibility:
            tokens = self.tokenize_text(
                Rouge.KEEP_CANNOT_IN_ONE_WORD.sub("_cannot_", summary)
            )
        else:
            tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(" ", summary))

        if self.stemming:
            self.stem_tokens(tokens)  # stemming in-place

        if self.ensure_compatibility:
            preprocessed_summary = [
                Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub("cannot", " ".join(tokens))
            ]
        else:
            preprocessed_summary = [" ".join(tokens)]

        return preprocessed_summary

    def _preprocess_summary_per_sentence(self, summary):
        """Preprocessing of a summary by sentences

        - truncate text if enabled
        - tokenization
        - stemming if enabled
        - lowering

        Parameters
        ----------
        summary: str

        Returns
        -------
        final_sentences : list of str
            The preprocessed summary
        """
        sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)

        # Truncate
        if self.limit_length:
            final_sentences = []
            current_len = 0
            # By words
            if self.length_limit_type == "words":
                for sentence in sentences:
                    tokens = sentence.strip().split()
                    tokens_len = len(tokens)
                    if current_len + tokens_len < self.length_limit:
                        sentence = " ".join(tokens)
                        final_sentences.append(sentence)
                        current_len += tokens_len
                    else:
                        sentence = " ".join(tokens[: self.length_limit - current_len])
                        final_sentences.append(sentence)
                        break
            # By bytes
            elif self.length_limit_type == "bytes":
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)
                    if current_len + sentence_len < self.length_limit:
                        final_sentences.append(sentence)
                        current_len += sentence_len
                    else:
                        sentence = sentence[: self.length_limit - current_len]
                        final_sentences.append(sentence)
                        break
            sentences = final_sentences

        final_sentences = []
        for sentence in sentences:
            sentence = Rouge.REMOVE_CHAR_PATTERN.sub(" ", sentence.lower()).strip()

            # Preprocess. Hack: because official ROUGE script bring "cannot" as "cannot"
            # and "can not" as "can not", we have to hack nltk tokenizer to not
            # transform "cannot/can not" to "can not"
            if self.ensure_compatibility:
                tokens = self.tokenize_text(
                    Rouge.KEEP_CANNOT_IN_ONE_WORD.sub("_cannot_", sentence)
                )
            else:
                tokens = self.tokenize_text(
                    Rouge.REMOVE_CHAR_PATTERN.sub(" ", sentence)
                )

            if self.stemming:
                self.stem_tokens(tokens)  # stemming in-place

            if self.ensure_compatibility:
                sentence = Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub(
                    "cannot", " ".join(tokens)
                )
            else:
                sentence = " ".join(tokens)

            final_sentences.append(sentence)

        return final_sentences


def parse_args(args):
    def get_ranged(start=None, stop=None, type_=int):
        def f(x):
            x = type_(x)
            if start is not None and x < start:
                raise argparse.ArgumentTypeError("{} must be gte {}".format(x, start))
            if stop is not None and x > stop:
                raise argparse.ArgumentTypeError("{} must be lte {}".format(x, stop))
            return x

        return f

    parser = argparse.ArgumentParser(
        description=main.__doc__,
        epilog="""
A ROUGE eval config file is of the following format. Hopefully your recipe (e.g.
ggws.py) gives you a way to auto-generate this.

  <ROUGE_EVAL version="1.5.5">
    <EVAL ID="<doc-id-1>">
      <MODEL-ROOT>/path/to/model/files</MODEL-ROOT>
      <PEER-ROOT>/path/to/peer/files</PEER-ROOT>
      <INPUT-FORMAT TYPE="<input-type>" />
      <PEERS>
        <P ID="<system-id-1>">relative/path/to/peer-1/summary/for/doc-1</P>
        <P ID="<system-id-2>">relative/path/to/peer-2/summary/for/doc-1</P>
        ...
      </PEERS>
      <MODELS>
        <M ID="<model-id-1>">relative/path/to/model-1/summary/for/doc-1</P>
        <M ID="<model-id-2>">relative/path/to/model-2/summary/for/doc-1</P>
        ...
      </MODELS>
    </EVAL>
    <EVAL ID="<doc-id-2>">
      ...
    </EVAL>
  </ROUGE_EVAL>

where
- <doc-id-X> is a unique ID for one document you are summarizing
- <system-id-X> is a unique ID specifying a single peer/hypothesis/ML system that
  summarizes various documents
- <model-id-X> is a unique ID specifying a single model/reference/gold-standard
  summarizer that summarizes various documents
- <input-type> is one of "SEE", "SPL", or "ISI", though this script is currently limited
  to "SPL"

The "SPL" format is Sentence Per Line. That is, both the peer and model summaries are
formatted such that sentences are delimited by newlines. Ex for
relative/path/to/model-1/summary/for/doc-1:

  Here is the first sentence of the gold standard summary.
  Here is the second.

Be very careful not to confuse "peers" and "models." Peers are the output of your
algorithms. Models are gold-standard summaries.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "rouge_eval_config_file",
        type=argparse.FileType("r"),
        help="Specify the evaluation setup. See below for more details.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "system_id",
        help="Only process results for peer/system matching this ID",
        nargs="?",
        default=None,
    )
    group.add_argument(
        "-a",
        action="store_true",
        default=False,
        help="Process results for all peers/systems",
    )

    parser.add_argument(
        "-n",
        type=get_ranged(0),
        default=0,
        help="Calculate n-gram ROUGE up to and including n",
    )
    parser.add_argument(
        "-f",
        choices=("A", "B"),
        default="A",
        help="Select scoring formula: 'A' => model average; 'B' => best model",
    )
    parser.add_argument(
        "-m",
        action="store_true",
        default=False,
        help="Stem both model and system summaries using Porter stemmer before "
        "computing various statistics.",
    )
    parser.add_argument(
        "-p",
        type=get_ranged(0.0, 1.0, float),
        default=0.5,
        help="Relative importance of recall and precision ROUGE scores. Alpha -> 1 "
        "favors precision, Alpha -> 0 favors recall.",
    )
    parser.add_argument(
        "-x", action="store_true", default=False, help="Do not calculate ROUGE-L.",
    )
    parser.add_argument(
        "-w",
        type=get_ranged(0.0, type_=float),
        default=None,
        help="Compute ROUGE-W that gives consecutive matches of length L in an LCS a "
        "weight of 'L^weight' instead of just 'L' as in LCS. Typically this is set to "
        "1.2 or other number greater than 1.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-b",
        type=get_ranged(1),
        default=0,
        help="Only use the first n bytes in the system/peer summary for evaluation",
    )
    group.add_argument(
        "-l",
        type=get_ranged(1),
        default=0,
        help="Only use the first n words in the system/peer summary for evaluation",
    )

    return parser.parse_args(args)


def build_scorer(opts):
    metrics = {"rouge-l"}
    if opts.x:
        metrics.remove("rouge-l")
    if opts.n:
        metrics.add("rouge-n")
    if opts.w is not None:
        metrics.add("rouge-w")
        w = opts.w
    else:
        w = 1.0
    metrics = sorted(metrics)
    limit_length = True
    length_limit_type = "bytes"
    length_limit = opts.b
    if opts.l:
        length_limit_type = "words"
        length_limit = opts.l
    elif not opts.b:
        limit_length = False
    if opts.f == "A":
        apply_avg, apply_best = True, False
    else:
        apply_avg, apply_best = False, True
    return Rouge(
        metrics=metrics,
        max_n=opts.n,
        limit_length=limit_length,
        length_limit=length_limit,
        length_limit_type=length_limit_type,
        apply_avg=apply_avg,
        apply_best=apply_best,
        stemming=opts.m,
        alpha=opts.p,
        weight_factor=w,
    )


def parse_rouge_eval_config(xml):

    eval_id = peer_id = model_id = None

    def err(msg):
        if model_id is not None:
            msg = "model " + model_id + ": " + msg
        elif peer_id is not None:
            msg = "peer " + peer_id + ": " + msg
        if eval_id is not None:
            msg = "event " + eval_id + ": " + msg
        msg = "file " + xml.name + ": " + msg
        raise IOError(msg)

    evals = dict()
    tree = et.parse(xml)
    root = tree.getroot()
    if root.tag != "ROUGE_EVAL":
        err("root tag must be ROUGE_EVAL, got " + root.tag)
    if "version" in root.attrib and root.attrib["version"] != "1.5.5":
        err("version set to {}; must be 1.5.5".format(root.attrib["version"]))

    for eval_ in root.findall("EVAL"):
        peer_id = model_id = eval_id = None
        if "ID" not in eval_.attrib:
            err("EVAL element does not have ID attribute")
        eval_id = eval_.attrib["ID"]
        if eval_id in evals:
            err("Duplicate EVAL ids")
        input_format = eval_.find("INPUT-FORMAT")
        if (
            input_format is None
            or "TYPE" not in input_format.attrib
            or input_format.attrib["TYPE"] != "SPL"
        ):
            err(
                "INPUT-FORMAT element must exist in EVAL and have TYPE attribute set "
                'to "SPL"'
            )
        peer_root = eval_.findtext("PEER-ROOT")
        model_root = eval_.findtext("MODEL-ROOT")

        peer2summary = dict()
        peers = eval_.find("PEERS")
        if peers is None:
            err("PEERS element must exist in EVAL")
        for peer in peers.findall("P"):
            peer_id = None
            if "ID" not in peer.attrib:
                err("P element does not have ID attribute")
            peer_id = peer.attrib["ID"]
            if peer_id in peer2summary:
                err("Duplicate P ids in same EVAL")
            if peer.text is None:
                err("P elem must have text")
            if peer_root:
                path = os.path.join(peer_root, peer.text)
            else:
                path = peer.text
            if not os.path.isfile(path):
                err("{} is not a file or does not exist".format(path))
            with open(path) as file_:
                summary = file_.read()
            peer2summary[peer_id] = summary
        peer_id = None
        if not len(peer2summary):
            err("PEERS element listed no peers")

        model2summary = dict()
        models = eval_.find("MODELS")
        if models is None:
            err("MODELS element must exist in EVAL")
        for model in models.findall("M"):
            model_id = None
            if "ID" not in model.attrib:
                err("M element does not have ID attribute")
            model_id = model.attrib["ID"]
            if model_id in model2summary:
                err("Duplicate M ids in same EVAL")
            if model.text is None:
                err("M elem must have text")
            if model_root:
                path = os.path.join(model_root, model.text)
            else:
                path = model.text
            if not os.path.isfile(path):
                err("{} is not a file or does not exist".format(path))
            with open(path) as file_:
                summary = file_.read()
            model2summary[model_id] = summary
        model_id = None
        if not len(model2summary):
            err("MODELS element listed no models")

        evals[eval_id] = (peer2summary, model2summary)

    return evals


def main(args=None):
    """Simple CLI wrapper for py-rouge"""

    opts = parse_args(args)

    scorer = build_scorer(opts)

    evals = parse_rouge_eval_config(opts.rouge_eval_config_file)

    if opts.a:
        peers = set()
        for peer2summary, _ in evals.values():
            peers |= set(peer2summary)
        peers = sorted(peers)
    else:
        peers = [opts.system_id]

    for peer in sorted(peers):
        hypotheses = []
        references = []
        for peer2summary, model2summary in evals.values():
            if peer not in peer2summary:
                continue
            hypotheses.append(peer2summary[peer])
            references.append(list(model2summary.values()))
        metric2scores = scorer.get_scores(hypotheses, references)
        for metric in sorted(metric2scores):
            scores = metric2scores[metric]
            print("---------------------------------------------")
            metric = metric.upper().replace("_", "-")
            for name, score in (
                ("Average_R", scores["r"]),
                ("Average_P", scores["p"]),
                ("Average_F", scores["f"]),
            ):
                print("{} {} {}: {:.05f}".format(peer, metric, name, score))


if __name__ == "__main__":
    main()
