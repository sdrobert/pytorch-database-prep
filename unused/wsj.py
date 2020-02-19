# Copyright 2020 Sean Robertson
#
# at the time of writing, kaldi/egs/wsj/s5/local/wsj_extend_dict.sh, as well
# as the files from kaldi/egs/wsj/s5/local/dict, don't have copyright info.
# I think they're probably written by Daniel Povey.
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

# some relevant files to the setup:
# https://catalog.ldc.upenn.edu/docs/LDC93S6A/readme.txt
# https://catalog.ldc.upenn.edu/docs/LDC94S13A/wsj1.txt

'''Some unused routines based on Kaldi's recipe for wsj'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import io
import gzip

from collections import Counter, OrderedDict
from shutil import copytree, rmtree
from itertools import chain

import ngram_lm

from common import glob, mkdir, sort, cat, pipe_to
from unlzw import unlzw

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"


try:
    import urllib.request as request
except ImportError:
    import urllib2 as request


def wsj_prepare_dict(data_root, dict_suffix=''):
    dir_ = os.path.join(data_root, 'local', 'dict' + dict_suffix)
    cmudict = os.path.join(dir_, 'cmudict')
    mkdir(cmudict)

    # we use the github URL mentioned here
    # http://www.speech.cs.cmu.edu/cgi-bin/cmudict
    # to avoid using subversion.
    url = (
        'https://raw.githubusercontent.com/Alexir/CMUdict/'
        '7a37de79f7e650fd6334333b1b5d2bcf0dee8ad3/'
    )
    for x in {'cmudict.0.7a', 'cmudict-0.7b.symbols'}:
        # 0.7b.symbols the same as 0.7a.symbols
        path = os.path.join(cmudict, x)
        if not os.path.exists(path):
            request.urlretrieve(url + x, path)

    silence_phones_txt = os.path.join(dir_, 'silence_phones.txt')
    optional_silence_txt = os.path.join(dir_, 'optional_silence.txt')
    pipe_to(['SIL', 'SPN', 'NSN'], silence_phones_txt)
    pipe_to(['SIL'], optional_silence_txt)

    nonsilence_phones_txt = os.path.join(dir_, 'nonsilence_phones.txt')
    phone_pattern = re.compile(r'^(\D+)\d*$')
    phones_of = dict()
    for phone in cat(os.path.join(cmudict, 'cmudict-0.7b.symbols')):
        match = phone_pattern.match(phone)
        if not match:
            raise ValueError("Bad phone {}".format(phone))
        base = match.groups(1)  # no stress
        phones_of.setdefault(base, []).append(phone)
    pipe_to(
        (' '.join(x) for x in phones_of.values()),
        nonsilence_phones_txt
    )
    del phones_of

    # skip extra_questions.txt

    # there were a few updates to 0.7.a that make the resulting lexicon
    # slightly different from Kaldi's
    lexicon_raw_nosil_txt = os.path.join(dir_, 'lexicon1_raw_nosil.txt')
    entry_pattern = re.compile(r'^(\S+)\(\d+\) (.*)$')
    lexicon_raw_nosil_lines = []
    for line in cat(os.path.join(cmudict, 'cmudict.0.7a')):
        if line.startswith(';;;'):
            continue
        match = entry_pattern.match(line)
        if match is None:
            lexicon_raw_nosil_lines.append(line)
        else:
            lexicon_raw_nosil_lines.append(
                ' '.join([match.group(1), match.group(2)]))
    pipe_to(lexicon_raw_nosil_lines, lexicon_raw_nosil_txt)
    del lexicon_raw_nosil_lines

    lexicon_txt = os.path.join(dir_, 'lexicon.txt')
    pipe_to(
        sort(set(cat(
            ['!SIL  SIL', '<SPOKEN_NOISE>  SPN', '<UNK>  SPN', '<NOISE>  NSN'],
            lexicon_raw_nosil_txt
        ))),
        lexicon_txt
    )


def wsj_prepare_char_dict(data_root, src_dict_suffix, dst_dict_suffix):
    phone_dir = os.path.join(data_root, 'local', 'dict' + src_dict_suffix)
    dir_ = os.path.join(data_root, 'local', 'dict' + dst_dict_suffix)
    mkdir(dir_)

    lexicon1_raw_nosil_txt = os.path.join(phone_dir, 'lexicon1_raw_nosil.txt')
    phn_lexicon2_raw_nosil_txt = os.path.join(
        phone_dir, 'lexicon2_raw_nosil.txt')
    unique = OrderedDict()
    for entry in cat(lexicon1_raw_nosil_txt):
        unique.setdefault(entry.split(' ')[0], entry)
    pipe_to(unique.values(), phn_lexicon2_raw_nosil_txt)

    char_lexicon2_raw_nosil_txt = os.path.join(dir_, 'lexicon2_raw_nosil.txt')
    bad_chars = set("!~@#$%^&*()+=/\",;:?_{}-")
    pipe_to(
        (
            ' '.join([x] + [y for y in x if y not in bad_chars])
            for x in unique.keys()
        ),
        char_lexicon2_raw_nosil_txt
    )
    del unique

    pipe_to(['SIL', 'SPN', 'NSN'], os.path.join(dir_, 'silence_phones.txt'))
    pipe_to(['SIL'], os.path.join(dir_, 'optional_silence.txt'))

    pipe_to(
        sort(set(cat(
            ['!SIL  SIL', '<SPOKEN_NOISE>  SPN', '<NOISE>  NSN'],
            char_lexicon2_raw_nosil_txt,
        ))),
        os.path.join(dir_, 'lexicon.txt')
    )

    pipe_to(
        sort(set(cat(*(
            x.split(' ')[1:] for x in cat(char_lexicon2_raw_nosil_txt))))),
        os.path.join(dir_, 'nonsilence_phones.txt'),
    )


def dict_get_acronym_prons(oovlist, dict_):
    # this function first extracts single-letter acronyms from the CMU dict.
    # It then looks for words in the oovlist file that look like acronyms, and
    # builds up some pronunciations based on them
    #
    # consult Kaldi's wsj/s5/local/dict/get_acronym_prons.pl for more details

    def get_letter_prons(letters, letter_prons):
        acronym = list(letters)
        prons = [""]
        while acronym:
            letter = acronym.pop(0)
            n = 1
            while acronym and acronym[0] == letter:
                acronym.pop(0)
                n += 1
            letter_pron = letter_prons[letter]
            prons_of_block = []
            if n == 2:
                for lpron in letter_pron:
                    prons_of_block.append("D AH1 B AH0 L " + lpron)
                    prons_of_block.append(lpron + " " + lpron)
            elif n == 3:
                for lpron in letter_pron:
                    prons_of_block.append("T R IH1 P AH0 L " + lpron)
                    prons_of_block.append(" ".join([lpron] * 3))
            else:
                for lpron in letter_pron:
                    prons_of_block.append(" ".join([lpron] * n))
            new_prons = []
            for pron in prons:
                for pron_of_block in prons_of_block:
                    if pron:
                        new_prons.append(pron + " " + pron_of_block)
                    else:
                        new_prons.append(pron_of_block)
            prons = new_prons
        assert prons[0] != ""
        for pron in prons:
            yield pron

    if isinstance(dict_, str):
        dict_ = cat(dict_)

    letter_pattern = re.compile(r'^[A-Z]\.$')
    letter_prons = dict()
    for line in dict_:
        word, pron = line.strip().split(' ', maxsplit=1)
        if letter_pattern.match(word):
            letter = word[0]
            letter_prons.setdefault(letter, []).append(pron.strip())

    if isinstance(oovlist, str):
        oovlist = cat(oovlist)

    acro_wo_points_pattern = re.compile(r'^[A-Z]{1,5}$')
    acro_w_points_pattern = re.compile(r'^([A-Z]\.){1,4}[A-Z]\.?$')
    for word in oovlist:
        word = word.strip()
        if acro_wo_points_pattern.match(word):
            for pron in get_letter_prons(word, letter_prons):
                yield word + "  " + pron
        elif acro_w_points_pattern.match(word):
            for pron in get_letter_prons(word.replace('.', ''), letter_prons):
                yield word + "  " + pron


def dict_get_rules(
        in_, disallow_empty_suffix=True, min_prefix_len=3,
        ignore_prefix_stress=True, min_suffix_count=20):
    # this function looks at a dictionary and finds pairs of suffixes whose
    # prefixes overlap in both spelling and pronunciation. For example, because
    # entries like (fictitious)
    #
    # NUMBERS N U M B E R S
    # NUMBERING N U M B E R I N G
    #
    # Share a common prefix in both spelling "NUMBER" and pronunciation
    # "N U M B E R", we'll generate the pairs ("S", ["S"]) and ("ING", ["I",
    # "N", "G"]). Those pairs will be kept if those suffixes reoccur exactly
    # (in both pronunciation and spelling) above some threshold. Down the line,
    # we can use these rules to create new pronunciations, e.g. from
    #
    # FIELDS F I E L D S
    #
    # we can produce FIELDING by removing the S suffing and adding the ING one
    #
    # consult kaldi's wsj/s5/local/dict/get_rules.pl for a formal description
    prons = dict()
    for entry in in_:
        word, pron = entry.split(' ', 1)
        prons.setdefault(word, []).append(pron)

    suffix_count = Counter(chain.from_iterable(
        (w[i:] for i in range(min_prefix_len, len(w) + 1))
        for w in prons
    ))
    suffix_count = dict(
        (w, count) for (w, count) in suffix_count.items()
        if count >= min_suffix_count
    )

    suffixes_of = dict()
    for word in prons:
        for i in range(min_prefix_len, len(word) + 1):
            prefix, suffix = word[:i], word[i:]
            if suffix in suffix_count:
                suffixes_of.setdefault(prefix, []).append(suffix)

    suffix_set_count = Counter()
    for suffixes in suffixes_of.values():
        suffixes = tuple(sorted(suffixes))
        suffix_set_count[suffixes] += 1

    suffix_pair_count = Counter()
    for suffixes, count in suffix_set_count.items():
        for suffix_a in suffixes:
            for suffix_b in suffixes:
                if suffix_a != suffix_b:
                    suffix_pair_count[(suffix_a, suffix_b)] += count
    suffix_pair_count = dict(
        (k, count) for (k, count) in suffix_pair_count.items()
        if count >= min_suffix_count
    )

    quadruple_count = Counter()
    digits = set(str(x) for x in range(10))
    for prefix, suffixes in suffixes_of.items():
        for suffix_a in suffixes:
            for suffix_b in suffixes:
                if (suffix_a, suffix_b) not in suffix_pair_count:
                    continue
                for pron_a in prons[prefix + suffix_a]:
                    pron_a = pron_a.split(' ')
                    for pron_b in prons[prefix + suffix_b]:
                        pron_b = pron_b.split(' ')
                        for pos in range(min(len(pron_a), len(pron_b)) + 1):
                            psuffix_a = ' '.join(pron_a[pos:])
                            psuffix_b = ' '.join(pron_b[pos:])
                            quad = (suffix_a, suffix_b, psuffix_a, psuffix_b)
                            quadruple_count[quad] += 1
                            pron_ai = '' if pos == len(pron_a) else pron_a[pos]
                            pron_bi = '' if pos == len(pron_b) else pron_b[pos]
                            if ignore_prefix_stress:
                                if set(pron_ai[-1:]) & digits:
                                    pron_ai = pron_ai[:-1]
                                if set(pron_bi[-1:]) & digits:
                                    pron_bi = pron_bi[:-1]
                            if pron_ai != pron_bi:
                                break

    del suffix_count, suffixes_of, suffix_pair_count
    for quad, count in quadruple_count.items():
        if count >= min_suffix_count:
            yield ','.join(quad)


def dict_get_rule_hierarchy(rules_path):
    # this function looks at the pairs of suffixes from dict_get_rules.
    # Whenever those spelling/pronunciation suffixes share a non-empty prefix,
    # it implies that that rule could be generalized to one that excludes that
    # shared prefix. For (a real) example: ("TICS", ["S"])  and ("TIC", [])
    # are a paired rule. Recall that this says you can generate a new word by
    # removing "TICS" from the word and adding "TIC", and removing ["S"] from
    # its pronunciation and adding [] (or vice versa). Because the suffixes
    # share a prefix "TIC", a more general rule might exclude the "TIC" part:
    # ("S", ["S"]), ("", []). If this more general rule is also included in the
    # set of rules, then the output stores the specific -> general relationship
    #
    # See wsj/s5/local/dict/get_rule_hierarchy.pl for a more formal description
    rules = set(cat(rules_path))
    for rule in rules:
        A = rule.split(',')
        suffix_a, suffix_b = tuple(A[0]), tuple(A[1])
        psuffix_a, psuffix_b = A[2].split(), A[3].split()
        common_suffix_len = 0
        while common_suffix_len < min(len(suffix_a), len(suffix_b)):
            if suffix_a[common_suffix_len] != suffix_b[common_suffix_len]:
                break
            common_suffix_len += 1
        common_psuffix_len = 0
        while common_psuffix_len < min(len(psuffix_a), len(psuffix_b)):
            if psuffix_a[common_psuffix_len] != psuffix_b[common_psuffix_len]:
                break
            common_psuffix_len += 1
        for m in range(common_suffix_len + 1):
            sa, sb = ''.join(suffix_a[m:]), ''.join(suffix_b[m:])
            for n in range(common_psuffix_len + 1):
                if not n and not m:
                    continue
                psa, psb = ' '.join(psuffix_a[n:]), ' '.join(psuffix_b[n:])
                more_general_rule = ','.join((sa, sb, psa, psb))
                if more_general_rule in rules:
                    yield ';'.join((rule, more_general_rule))


def dict_get_candidate_prons(rules, dict_, words, min_prefix_len=3):
    # the purpose of this script is to apply the rules from dict_get_rules to
    # the word list, given the dictionary. It does no pruning based on
    # hierarchy. We're basically removing one suffix and attaching another.
    #
    # see wsj/s5/local/dict/get_candidate_prons.pl for a more formal descript.
    isrule = set()
    suffix2rule = dict()
    rule_and_stress_to_rule_score = dict()
    for rule in cat(rules):
        rule = rule.split(';', 3)
        rule_score = rule.pop() if len(rule) == 3 else -1
        destress = rule.pop() if len(rule) == 2 else None
        assert len(rule) == 1
        rule = rule[0]
        R = rule.split(',')
        if len(R) != 4:
            raise ValueError('Bad rule {}'.format(rule))
        suffix = R[0]
        if rule not in isrule:
            isrule.add(rule)
            suffix2rule.setdefault(suffix, []).append(R)
        if destress is None:
            rule_and_stress_to_rule_score[rule + ';yes'] = rule_score
            rule_and_stress_to_rule_score[rule + ';no'] = rule_score
        else:
            rule_and_stress_to_rule_score[rule + ';' + destress] = rule_score

    word2prons = dict()
    for entry in cat(dict_):
        word, pron = entry.split(maxsplit=1)
        word2prons.setdefault(word, []).append(pron)
    prefixcount = Counter(chain.from_iterable(
        (w[:p] for p in range(len(w) + 1)) for w in word2prons))

    if isinstance(words, str):
        words = cat(words)

    for word in words:
        word = word.split(maxsplit=1)[0]
        ownword = 1 if word in word2prons else 0
        for prefix_len in range(min_prefix_len, len(word) + 1):
            prefix, suffix = word[:prefix_len], word[prefix_len:]
            if prefixcount.get(prefix, 0) - ownword == 0:
                continue
            rules_array_ref = suffix2rule.get(suffix, None)
            if rules_array_ref is None:
                continue
            for R in rules_array_ref:
                _, base_suffix, psuffix, base_psuffix = R
                base_word = prefix + base_suffix
                base_prons_ref = word2prons.get(base_word, None)
                if base_prons_ref is None:
                    continue
                if base_psuffix:
                    base_psuffix = " " + base_psuffix
                # FIXME(sdrobert): I think this might split up phones. This'll
                # be bad when some phones share prefixes with others.
                for base_pron in base_prons_ref:
                    base_pron_prefix_len = len(base_pron) - len(base_psuffix)
                    if (
                            base_pron_prefix_len < 0 or
                            base_pron[base_pron_prefix_len:] != base_psuffix):
                        continue
                    pron_prefix = base_pron[:base_pron_prefix_len]
                    rule = ','.join(R)
                    for destress in range(2):
                        if destress:
                            pron_prefix = pron_prefix.replace('2', '1')
                            destress_mark = 'yes'
                        else:
                            destress_mark = 'no'
                        pron = pron_prefix
                        if psuffix:
                            pron += ' ' + psuffix
                        rule_score = rule_and_stress_to_rule_score.get(
                            rule + ';' + destress_mark, None)
                        if rule_score is None:
                            continue
                        output = [
                            word, pron, base_word, base_pron, rule,
                            destress_mark
                        ]
                        if rule_score != -1:
                            output.append(str(rule_score))
                        yield ';'.join(output)


def dict_limit_candidate_prons(rule_hierarchy, candidate_prons):
    # This takes the candidate pronunciations from dict_get_candidate_prons
    # and the rule hierarchy from dict_get_rule_hierarchy and limits the
    # candidate pronunciations to those that use the most specific rules
    #
    # see wsj/s5/local/dict/limit_candidate_prons.pl etc etc...
    hierarchy = set(cat(rule_hierarchy))

    def process_word(cur_lines):
        pair2rule_list = dict()
        for line in cur_lines:
            baseword, basepron = line.split(';', 4)[2:4]
            key = baseword + ';' + basepron
            pair2rule_list.setdefault(key, []).append(line)
        for lines in pair2rule_list.values():
            stress, rules = [], []
            for line in lines:
                rulename, destress = line.split(';')[4:6]
                stress.append(destress)
                rules.append(rulename)
            for m in range(len(lines)):
                ok = True
                for n in range(len(lines)):
                    if m == n or stress[m] != stress[n]:
                        continue
                    if (rules[n] + ';' + rules[m]) in hierarchy:
                        ok = False
                        break
                if ok:
                    yield lines[m]

    if isinstance(candidate_prons, str):
        candidate_prons = cat(candidate_prons)

    cur_word = None
    cur_lines = []
    for line in candidate_prons:
        word = line.split(';', 1)[0]
        if cur_word is not None and cur_word != word:
            for x in process_word(cur_lines):
                yield x
            cur_lines.clear()
        cur_lines.append(line)
        cur_word = word
    for x in process_word(cur_lines):
        yield x


def dict_score_prons(dict_, candidate_prons):
    # adds "score" information to candidate pronunciations. If a candidate
    # pronunciation matches a dictionary entry, it's "right." If it matches
    # except for stress, it's "partial." Otherwise, it's "wrong."
    #
    # wsj/s5/local/dict/score_prons.pl
    word_and_pron = set()
    word_and_pron_nostress = set()
    num_pattern = re.compile(r'\d')
    for entry in cat(dict_):
        word, pron = entry.split(maxsplit=1)
        pron_nostress = num_pattern.sub('', pron)
        word_and_pron.add(word + ';' + pron)
        word_and_pron_nostress.add(word + ';' + pron_nostress)

    if isinstance(candidate_prons, str):
        candidate_prons = cat(candidate_prons)

    for line in candidate_prons:
        word, pron = line.split(';', 2)[:2]
        pron_nostress = num_pattern.sub('', pron)
        if (word + ';' + pron) in word_and_pron:
            score = ';right'
        elif (word + ';' + pron_nostress) in word_and_pron_nostress:
            score = ';partial'
        else:
            score = ';wrong'
        yield line + score


def dict_count_rules(scored_prons):
    # count the number of times a rule, stress pair was scored right, partial
    # or wrong in dict_score_prons
    #
    # wsj/s5/local/dict/count_rules.pl
    counts = dict()
    if isinstance(scored_prons, str):
        scored_prons = cat(scored_prons)

    for scored_pron in scored_prons:
        rulename, destress, score = scored_pron.split(';')[4:]
        ref = counts.setdefault(rulename + ';' + destress, [0, 0, 0])
        if score == 'right':
            ref[0] += 1
        elif score == 'partial':
            ref[1] += 1
        elif score == 'wrong':
            ref[2] += 1
        else:
            raise ValueError('Bad score')

    for key, value in counts.items():
        yield ';'.join([key] + [str(x) for x in value])


def dict_score_rules(
        counted_rules, partial_score=0.8, ballast=1, destress_penalty=1e-5):
    # weigh the counted rules to derive a score for each rule
    #
    # wsj/s5/local/dict/score_rules.pl
    if isinstance(counted_rules, str):
        counted_rules = cat(counted_rules)

    for counted_rule in counted_rules:
        rule, destress, right, partial, wrong = counted_rule.split(';')
        rule_score = int(right) + int(partial) * partial_score
        rule_score /= int(right) + int(partial) + int(wrong) + ballast
        if destress == "yes":
            rule_score -= destress_penalty
        yield '{};{};{:.5f}'.format(rule, destress, rule_score)


def dict_reverse_candidates(candidates):
    # reverse prefix/suffixes in candidate pronunciation list
    #
    # wsj/s5/local/dict/reverse_candidates.pl

    def reverse_str(x):
        return x[::-1]

    def reverse_pron(x):
        return ' '.join(x.split(' ')[::-1])

    if isinstance(candidates, str):
        candidates = cat(candidates)
    for candidate in candidates:
        word, pron, baseword, basepron, rule, rest = candidate.split(';', 5)
        word, pron = reverse_str(word), reverse_pron(pron)
        baseword, basepron = reverse_str(baseword), reverse_pron(basepron)
        r_suff, r_bsuff, r_pron, r_bpron = rule.split(',')
        r_suff, r_bsuff = reverse_str(r_suff), reverse_str(r_bsuff)
        r_pron, r_bpron = reverse_pron(r_pron), reverse_pron(r_bpron)
        rule = ','.join((r_suff, r_bsuff, r_pron, r_bpron))
        yield ';'.join((word, pron, baseword, basepron, rule, rest))


def dict_select_candidate_prons(candidates, max_prons=4, min_rule_score=0.35):
    # for a given word, sort its pronunciations by score and return up to
    # max_prons of the top-scored candidates, subject to the constraint
    # that all returned candidates have a score >= min_rule_score
    #
    # wsj/s5/local/dict/select_candidates.pl
    #
    # AFAICT Perl sort-keys-by-value introduces non-determinism in the order
    # of pronunciations of the same score. Unfortunately, this determines the
    # ultimate candidate pronunciations of some OOV words.
    # We pre-empt a bit of that non-determinism, but don't expect perfect
    # matching values with Kaldi.

    def process_word(cur_lines):
        pron2rule_score = dict()
        pron2line = dict()
        for line in cur_lines:
            word, pron, _, _, _, _, score = line.split(';')
            score = float(score)
            if (
                    score >= min_rule_score and
                    score > pron2rule_score.get(pron, -1)):
                pron2rule_score[pron] = score
                pron2line[pron] = line
        prons = sorted(
            pron2rule_score,
            key=lambda x: (-pron2rule_score[x], x),
            reverse=False
        )
        for pron, _ in zip(prons, range(max_prons)):
            yield pron2line[pron]

    cur_lines = []
    cur_word = None
    if isinstance(candidates, str):
        candidates = cat(candidates)
    for candidate in candidates:
        word, _ = candidate.split(';', 1)
        if word != cur_word:
            for line in process_word(cur_lines):
                yield line
            cur_word, cur_lines = word, []
        cur_lines.append(candidate)
    for line in process_word(cur_lines):
        yield line


def wsj_extend_dict(dir_13_32_1, data_root, src_dict_suffix, mincount=2):
    src_dict_dir = os.path.join(data_root, 'local', 'dict' + src_dict_suffix)
    dst_dict_dir = os.path.join(
        data_root, 'local', 'dict' + src_dict_suffix + '_larger')
    if os.path.isdir(dst_dict_dir):
        rmtree(dst_dict_dir)
    copytree(src_dict_dir, dst_dict_dir)

    # lexicon1_raw_nosil.txt is an unsorted (?) version of dict.cmu
    dict_cmu = os.path.join(dst_dict_dir, 'dict.cmu')
    pipe_to(
        sort(set(cat(os.path.join(src_dict_dir, 'lexicon1_raw_nosil.txt')))),
        dict_cmu
    )

    pipe_to(
        sort(set(
            x.split()[0]
            for x in cat(dict_cmu))),
        os.path.join(dst_dict_dir, 'wordlist.cmu')
    )

    cleaned_gz = os.path.join(dst_dict_dir, 'cleaned.gz')
    train_data_root = os.path.join(
        dir_13_32_1, 'wsj1', 'doc', 'lng_modl', 'lm_train', 'np_data')
    assert os.path.isdir(train_data_root)
    train_data_files = []
    for subdir in ('87', '88', '89'):
        train_data_files.extend(
            glob(os.path.join(train_data_root, subdir), r'*.z'))
    isword = set(
        x.strip() for x in cat(os.path.join(dst_dict_dir, 'wordlist.cmu')))
    with gzip.open(cleaned_gz, 'wt') as out:
        for train_data_file in train_data_files:
            with open(train_data_file, 'rb') as in_:
                compressed = in_.read()
            decompressed = unlzw(compressed)
            in_ = io.TextIOWrapper(io.BytesIO(decompressed))
            for line in in_:
                if line.startswith('<'):
                    continue
                A = line.strip().upper().split(' ')
                for n, a in enumerate(A):
                    if a not in isword and len(a) > 1 and a.endswith('.'):
                        out.write(a[:-1])
                        if n < len(A) - 1:
                            out.write("\n")
                    else:
                        out.write(a + " ")
                out.write("\n")
            del in_, compressed, decompressed

    counts = Counter()
    with gzip.open(cleaned_gz, 'rt') as cleaned:
        for line in cleaned:
            for token in line.strip().split():
                counts[token] += 1
    counts = sorted(((v, k) for (k, v) in counts.items()), reverse=True)
    digits = set(str(x) for x in range(10))
    oov_counts_path = os.path.join(dst_dict_dir, 'oov.counts')
    oovlist_path = os.path.join(dst_dict_dir, 'oovlist')
    with \
            open(os.path.join(dst_dict_dir, 'unigrams'), 'w') as unigrams, \
            open(oov_counts_path, 'w') as oov_cnts, \
            open(oovlist_path, 'w') as oov_lst:
        for count, word in counts:
            line = '{} {}\n'.format(count, word)
            unigrams.write(line)
            if word not in isword:
                oov_cnts.write(line)
                if not (set(word) & digits) and count >= mincount:
                    oov_lst.write(word + '\n')
    del counts

    dict_acronyms_path = os.path.join(dst_dict_dir, 'dict.acronyms')
    pipe_to(
        dict_get_acronym_prons(oovlist_path, dict_cmu),
        dict_acronyms_path,
    )

    f_dir = os.path.join(dst_dict_dir, 'f')
    b_dir = os.path.join(dst_dict_dir, 'b')
    mkdir(f_dir, b_dir)

    banned = set(',;')
    pipe_to(
        (x for x in cat(dict_cmu) if not (set(x.split()[0]) & banned)),
        os.path.join(f_dir, 'dict')
    )

    pipe_to(
        (
            x for x in cat(os.path.join(dst_dict_dir, 'oovlist'))
            if not (set(x.split()[0]) & banned)),
        os.path.join(f_dir, 'oovs')
    )

    pipe_to(
        (
            ' '.join([w[::-1]] + p.split()[::-1]) for (w, p) in (
                x.split(' ', 1) for x in cat(os.path.join(f_dir, 'dict')))),
        os.path.join(b_dir, 'dict')
    )

    pipe_to(
        (x[::-1] for x in cat(os.path.join(f_dir, 'oovs'))),
        os.path.join(b_dir, 'oovs')
    )

    for dir_ in (f_dir, b_dir):
        dict_path = os.path.join(dir_, 'dict')
        rules_path = os.path.join(dir_, 'rules')
        hierarchy_path = os.path.join(dir_, 'hierarchy')
        oovs_path = os.path.join(dir_, 'oovs')
        rule_counts_path = os.path.join(dir_, 'rule.counts')
        rules_with_scores_path = os.path.join(dir_, 'rules.with_scores')
        oov_candidates_path = os.path.join(dir_, 'oovs.candidates')
        pipe_to(dict_get_rules(cat(dict_path)), rules_path)
        pipe_to(dict_get_rule_hierarchy(rules_path), hierarchy_path)
        pipe_to(
            dict_count_rules(dict_score_prons(
                dict_path,
                dict_limit_candidate_prons(
                    hierarchy_path,
                    dict_get_candidate_prons(rules_path, dict_path, dict_path)
                ),
            )),
            rule_counts_path
        )
        pipe_to(
            sorted(
                dict_score_rules(rule_counts_path),
                key=lambda x: (float(x.split(';')[2]), x),
                reverse=True
            ),
            rules_with_scores_path
        )
        pipe_to(
            dict_limit_candidate_prons(
                hierarchy_path,
                dict_get_candidate_prons(
                    rules_with_scores_path, dict_path, oovs_path),
            ),
            oov_candidates_path
        )

    oov_candidates_path = os.path.join(dst_dict_dir, 'oovs.candidates')
    pipe_to(
        sorted(cat(
            dict_reverse_candidates(os.path.join(b_dir, 'oovs.candidates')),
            os.path.join(f_dir, 'oovs.candidates')
        )),
        oov_candidates_path
    )

    dict_oovs_path = os.path.join(dst_dict_dir, 'dict.oovs')
    pipe_to(
        (
            '{0}  {1}'.format(*x.split(';'))
            for x in dict_select_candidate_prons(oov_candidates_path)
        ),
        dict_oovs_path
    )

    dict_oovs_merged_path = os.path.join(dst_dict_dir, 'dict.oovs_merged')
    pipe_to(
        sorted(set(cat(dict_acronyms_path, dict_oovs_path))),
        dict_oovs_merged_path,
    )

    pipe_to(
        sorted(set(cat(
            ["!SIL SIL", "<SPOKEN_NOISE> SPN", "<UNK> SPN", "<NOISE> NSN"],
            dict_cmu, dict_oovs_merged_path
        ))),
        os.path.join(dst_dict_dir, 'lexicon.txt')
    )


def wsj_train_lms(data_root, src_dict_suffix, out_dir='local_lm', max_order=4):
    # Train a language model on WSJ lm training corpus
    # Here we don't do things the Kaldi way. Kaldi uses its own
    # derivative-based language modeling. We'll do modified Kneser-Ney
    # smoothing, which is a little more widespread.

    src_dict_dir = os.path.join(
        data_root, 'local', 'dict' + src_dict_suffix + '_larger')
    dst_lm_dir = os.path.join(data_root, 'local', out_dir)
    mkdir(dst_lm_dir)

    vocab = set(
        x.split()[0] for x in cat(os.path.join(src_dict_dir, 'lexicon.txt')))
    vocab.remove('!SIL')
    pipe_to(
        sorted(vocab),
        os.path.join(dst_lm_dir, 'wordlist.txt')
    )

    with gzip.open(os.path.join(src_dict_dir, 'cleaned.gz'), 'rt') as f:
        text = f.read()

    sents = ngram_lm.text_to_sents(
        text, sent_end_expr=r'\n', word_delim_expr=r' +')
    del text

    ngram_counts = ngram_lm.sents_to_ngram_counts(sents, max_order)
    ngram_counts[0]['<UNK>'] = 0  # add to vocab

    # find any ngrams that contain words that aren't part of the vocabulary.
    # we'll prune them. By pruning them we mean completely removing them.
    # Modified Kneser-Ney can use the frequency statistics before removing
    # them
    to_prune = set(ngram_counts[0]) - vocab
    to_prune.remove('<S>')
    for i, ngram_count in enumerate(ngram_counts[1:]):
        if i:
            to_prune.update(
                x for x in ngram_count
                if x[:-1] in to_prune or x[-1] in to_prune)
        else:
            to_prune.update(
                x for x in ngram_count
                if x[0] in to_prune or x[-1] in to_prune)

    prob_list = ngram_lm.ngram_counts_to_prob_list_kneser_ney(
        ngram_counts, sos='<S>', to_prune=to_prune)
    del ngram_counts

    lm = ngram_lm.BackoffNGramLM(prob_list, sos='<S>', eos='</S>', unk='<UNK>')
    # "pruning" here means removing the probability mass of the sos token and
    # redistributing to the other unigrams. "<S>" will remain in the
    # vocabulary
    lm.prune_by_name({'<S>'})
    del prob_list

    print('Corpus PPL:', lm.corpus_perplexity(sents))
