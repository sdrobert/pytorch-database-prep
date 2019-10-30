#! /usr/bin/env python

# Copyright 2019 Sean Robertson
#
# Adapted from kaldi/egs/wsj/s5/local/wsj_data_prep.sh:
#
# Copyright 2009-2012  Microsoft Corporation
#                      Johns Hopkins University (Author: Daniel Povey)
#
# and kaldi/egs/wsj/s5/local/{ndx2flist.pl,flist2scp.pl,find_transcripts.pl,
# normalize_transcript.pl}, kaldi/egs/wsj/s5/local/steps/utt2spk_to_spk2utt.pl:
#
# Copyright 2010-2011 Microsoft Corporation
#
# and kaldi/egs/wsj/s5/local/wsj_prepare_dict.sh:
#
# Copyright 2010-2012 Microsoft Corporation
#           2012-2014 Johns Hopkins University (Author: Daniel Povey)
#                2015 Guoguo Chen
#
# and kaldi/egs/wsj/s5/local/wsj_prepare_char_dict.sh
#
# Copyright 2017  Hossein Hadian
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import re
import warnings
import locale
import gzip
import itertools

from collections import OrderedDict
from shutil import copyfileobj

import ngram_lm

from common import glob, mkdir, sort, cat, pipe_to, wc_l
from pydrobert.torch.util import parse_arpa_lm

try:
    import urllib.request as request
except ImportError:
    import urllib2 as request

from unlzw import unlzw

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"


locale.setlocale(locale.LC_ALL, 'C')


def find_link_dir(wsj_subdirs, rel_path, required=True):
    '''find rel_path as suffix in wsj_subdirs, return it or None

    Kaldi makes soft links of all `wsj_subdirs` in a ``links/`` directory.
    `rel_path` is some path that would be in the run script as
    ``links/<rel_path>``, which means it should *really* exist as
    ``<wsj_subdir>/<rel_path>``, where ``wsj_subdir`` is an element of
    `wsj_subdirs` This removes the need to create symbolic links in a
    ``links/`` directory. Unix-style.
    '''
    for wsj_subdir in wsj_subdirs:
        dir_ = os.path.dirname(wsj_subdir)
        path = os.path.join(dir_, rel_path).replace(os.sep, '/')
        if (
                path.startswith(wsj_subdir.replace(os.sep, '/')) and
                os.path.exists(path)):
            return path
    if required:
        raise ValueError(
            '{} does not exist in {}'.format(rel_path).format(wsj_subdirs))
    return None


def ndx2flist(in_stream, wsj_subdirs):
    '''yields absolute file names from .ndx files'''

    dir_pattern = re.compile(r'.+/([0-9.-]+)/?$')
    disk2fn = dict()
    for fn in wsj_subdirs:
        fn = fn.replace(os.sep, '/')  # ensure unix-style file paths
        match = dir_pattern.match(fn)
        if match is None:
            raise ValueError(
                "Bad command-line argument {}".format(fn))
        disk_id = match.group(1)
        disk_id = disk_id.replace('.', '_').replace('-', '_')
        if fn.endswith(os.sep):
            fn = disk_id[:-1]
        disk2fn[disk_id] = fn

    line_pattern = re.compile(r'^([0-9_]+):\s*(\S+)$')
    for line_no, line in enumerate(in_stream):
        if line.startswith(';'):
            continue
        match = line_pattern.match(line)
        if match is None:
            raise ValueError("Could not parse line {}".format(line_no + 1))
        disk, filename = match.groups()
        if disk not in disk2fn:
            raise ValueError("Disk id {} not found".format(disk))
        yield "{}/{}".format(disk2fn[disk], filename)


def flist2scp(path):
    '''Yields <utt> <path> pairs from list of files'''
    line_pattern = re.compile(r'^\S+/(\w+)\.[wW][vV]1$')
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            match = line_pattern.match(line)
            if match is None:
                raise ValueError("Bad line {}".format(line))
            id_ = match.group(1).lower()
            yield "{} {}".format(id_, line)


def find_transcripts(in_stream, dot_flist):
    '''Yields <utt> <transcript> pairs from utts (stream) and transcripts'''

    spk2dot = dict()
    dot_pattern = re.compile(r'^\S+/(\w{6})00\.dot')
    with open(dot_flist) as f:
        for line in f:
            line = line.rstrip()
            match = dot_pattern.match(line)
            if match is None:
                raise ValueError('Bad line in dot file list {}'.format(line))
            spk = match.group(1)
            spk2dot[spk] = line

    utt_pattern = re.compile(r'^(\w{6})\w\w$')
    trans_pattern = re.compile(r'^(.+)\((\w{8})\)$')
    curspk = dotfile = None
    for uttid in in_stream:
        match = utt_pattern.match(uttid)
        if match is None:
            raise ValueError('Bad utterance id {}'.format(uttid))
        spk = match.group(1)
        if spk != curspk:
            utt2trans = dict()
            if spk not in spk2dot:
                raise ValueError('No dot file for speaker {}'.format(spk))
            dotfile = spk2dot[spk]
            with open(dotfile) as f:
                for line_no, line in enumerate(f):
                    line = line.rstrip()
                    match = trans_pattern.match(line)
                    if match is None:
                        raise ValueError(
                            'Bad line {} in dot file {} (line {})'
                            ''.format(line, dotfile, line_no + 1))
                    trans, utt = match.groups()
                    utt2trans[utt] = trans
            if uttid not in utt2trans:
                warnings.error(
                    'No transcript for utterance id {} (current dot file is '
                    '{})'.format(uttid, dotfile))
            yield "{} {}".format(uttid, utt2trans[uttid])


def normalize_transcript(in_stream, noiseword):
    '''Sanitize in_stream transcripts'''
    line_pattern = re.compile(r'^(\S+) (.+)$')
    del_pattern = re.compile(r'^([.~]|\[[</]\w+\]|\[\w+[>/]\])$')
    noise_pattern = re.compile(r'^\[\w+\]$')
    verbdel_pattern = re.compile(r"^<([\w']+)>$")
    for line in in_stream:
        match = line_pattern.match(line)
        if match is None:
            raise ValueError("Bad line {}".format(line))
        out, trans = match.groups()
        for w in trans.split(' '):
            w = w.upper().replace('\\', '')
            if del_pattern.match(w):
                continue
            elif w == '%PERCENT':
                w = 'PERCENT'
            elif w == '.POINT':
                w = 'POINT'
            elif w == '--DASH':
                w = '-DASH'
            elif noise_pattern.match(w):
                w = noiseword
            else:
                match = verbdel_pattern.match(w)
                if match:
                    w = match.group(1)
            out += " " + w
        yield out


def utt2spk_to_spk2utt(in_stream):
    spk_hash = OrderedDict()
    for line in in_stream:
        utt, spk = line.split(' ')
        spk_hash.setdefault(spk, []).append(utt)
    for spk, utts in spk_hash.items():
        yield "{} {}".format(spk, ' '.join(utts))


def wsj_data_prep(wsj_subdirs, data_root):

    lmdir = os.path.join(data_root, 'local', 'nist_lm')
    dir_ = os.path.join(data_root, 'local', 'data')
    mkdir(dir_, lmdir)

    wv1_pattern = re.compile(r'\.wv1$', flags=re.I)

    for rel_path in {'11-13.1', '13-34.1', '11-2.1'}:
        if find_link_dir(wsj_subdirs, rel_path, required=False) is None:
            raise ValueError('''\
wsj_data_prep: Spot check of command line arguments failed
Command line arguments must be absolute pathnames to WSJ directories
with names like 11-13.1
''')

    train_si84_flist = os.path.join(dir_, 'train_si84.flist')
    pipe_to(
        (
            x for x in sort(ndx2flist(
                cat(find_link_dir(
                    wsj_subdirs, '11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx',
                )),
                wsj_subdirs,
            ))
            if re.search(r'11-2.1/wsj0/si_tr_s/401', x, flags=re.I) is None
        ),
        train_si84_flist,
    )
    nl = wc_l(cat(train_si84_flist))
    if nl != 7138:
        warnings.warn(
            'expected 7138 lines in train_si84.flist, got {}'.format(nl))

    train_si284_flist = os.path.join(dir_, 'train_si284.flist')
    pipe_to(
        (
            x for x in sort(ndx2flist(
                cat(
                    find_link_dir(
                        wsj_subdirs,
                        '13-34.1/wsj1/doc/indices/si_tr_s.ndx',
                    ),
                    find_link_dir(
                        wsj_subdirs,
                        '11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx',
                    ),
                ),
                wsj_subdirs,
            ))
            if re.search(r'11-2.1/wsj0/si_tr_s/401', x, flags=re.I) is None
        ),
        train_si284_flist,
    )
    nl = wc_l(cat(train_si284_flist))
    if nl != 37416:
        warnings.warn(
            'expected 37416 lines in train_si284.flist, got {}'.format(nl))

    test_eval92_flist = os.path.join(dir_, 'test_eval92.flist')
    pipe_to(
        (
            x + '.wv1' for x in sort(ndx2flist(
                cat(find_link_dir(
                    wsj_subdirs,
                    '11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx',
                )),
                wsj_subdirs,
            ))
        ),
        test_eval92_flist,
    )
    nl = wc_l(cat(test_eval92_flist))
    if nl != 333:
        warnings.warn(
            'expected 333 lines in test_eval92.flist, got {}'.format(nl))

    test_eval92_5k_flist = os.path.join(dir_, 'test_eval92_5k.flist')
    pipe_to(
        (
            x + '.wv1' for x in sort(ndx2flist(
                cat(find_link_dir(
                    wsj_subdirs,
                    '11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx',
                )),
                wsj_subdirs,
            ))
        ),
        test_eval92_5k_flist,
    )
    nl = wc_l(cat(test_eval92_5k_flist))
    if nl != 330:
        warnings.warn(
            'expected 330 lines in test_eval92_5k.flist, got {}'.format(nl))

    test_eval93_flist = os.path.join(dir_, 'test_eval93.flist')
    pipe_to(
        sort(ndx2flist(
            (x.replace('13_32_1', '13_33_1') for x in cat(find_link_dir(
                wsj_subdirs,
                '13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx',
            ))),
            wsj_subdirs,
        )),
        test_eval93_flist,
    )
    nl = wc_l(cat(test_eval93_flist))
    if nl != 213:
        warnings.warn(
            'expected 213 lines in test_eval93.flist, got {}'.format(nl))

    test_eval93_5k_flist = os.path.join(dir_, 'test_eval93_5k.flist')
    pipe_to(
        sort(ndx2flist(
            (x.replace('13_32_1', '13_33_1') for x in cat(find_link_dir(
                wsj_subdirs,
                '13-32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx',
            ))),
            wsj_subdirs,
        )),
        test_eval93_5k_flist,
    )
    nl = wc_l(cat(test_eval93_5k_flist))
    if nl != 215:
        # I've found 215 entries, unlike the kaldi_data_prep.sh which suggests
        # 213. I've verified using kaldi_data_prep.sh
        warnings.warn(
            'expected 215 lines in test_eval93_5k.flist, got {}'.format(nl))

    test_dev93_flist = os.path.join(dir_, 'test_dev93.flist')
    pipe_to(
        sort(ndx2flist(
            cat(find_link_dir(
                wsj_subdirs,
                '13-34.1/wsj1/doc/indices/h1_p0.ndx',
            )),
            wsj_subdirs,
        )),
        test_dev93_flist,
    )
    nl = wc_l(cat(test_dev93_flist))
    if nl != 503:
        warnings.warn(
            'expected 503 lines in test_dev93.flist, got {}'.format(nl))

    test_dev93_5k_flist = os.path.join(dir_, 'test_dev93_5k.flist')
    pipe_to(
        sort(ndx2flist(
            cat(find_link_dir(
                wsj_subdirs,
                '13-34.1/wsj1/doc/indices/h2_p0.ndx',
                required=True,
            )),
            wsj_subdirs,
        )),
        test_dev93_5k_flist,
    )
    nl = wc_l(cat(test_dev93_5k_flist))
    if nl != 513:
        warnings.warn(
            'expected 513 lines in test_dev93_5k.flist, got {}'.format(nl))

    dir_13_16_1 = find_link_dir(wsj_subdirs, '13-16.1')
    dev_dt_20_flist = os.path.join(dir_, 'dev_dt_20.flist')
    dev_dt_05_flist = os.path.join(dir_, 'dev_dt_05.flist')
    pipe_to(
        sort((
            x for x in glob(dir_13_16_1, r'???1/??_??_20/**/*')
            if wv1_pattern.search(x)
        )),
        dev_dt_20_flist,
    )
    pipe_to(
        sort((
            x for x in glob(dir_13_16_1, r'???1/??_??_05/**/*')
            if wv1_pattern.search(x)
        )),
        dev_dt_05_flist,
    )

    dot_files_flist = os.path.join(dir_, 'dot_files.flist')
    pipe_to(
        itertools.chain(*(
            (
                y for y in glob(x, '**/*')
                if re.search(r'\.dot$', y, flags=re.I)
            ) for x in wsj_subdirs
        )),
        dot_files_flist,
    )

    noiseword = "<NOISE>"
    for x in {
            'train_si84', 'train_si284', 'test_eval92',
            'test_eval93', 'test_dev93', 'test_eval92_5k',
            'test_eval93_5k', 'test_dev93_5k', 'dev_dt_05',
            'dev_dt_20'}:
        src = os.path.join(dir_, x + '.flist')
        sph = os.path.join(dir_, x + '_sph.scp')
        trans1 = os.path.join(dir_, x + '.trans1')
        txt = os.path.join(dir_, x + '.txt')
        utt2spk = os.path.join(dir_, x + '.utt2spk')
        spk2utt = os.path.join(dir_, x + '.spk2utt')

        pipe_to(sort(flist2scp(src)), sph)
        pipe_to(
            find_transcripts(
                (x.split()[0] for x in cat(sph)), dot_files_flist),
            trans1
        )

        pipe_to(sort(normalize_transcript(cat(trans1), noiseword)), txt)

        # XXX(sdrobert): don't care about _wav.scp

        pipe_to(
            (
                '{} {}'.format(y, y[:3])
                for y in (x.split()[0] for x in cat(trans1))),
            utt2spk
        )

        pipe_to(utt2spk_to_spk2utt(cat(utt2spk)), spk2utt)

    # XXX(sdrobert): Here's why I don't copy wfl_64.lst
    # https://groups.google.com/forum/#!topic/kaldi-developers/Q2TZQMvDvEU

    # bcb20onp.z is not gzipped, it's lzw zipped. Unlike Kaldi, we do an
    # additional step where we uncompress and recompress as a gzipped file
    dir_13_32_1 = find_link_dir(wsj_subdirs, '13-32.1')
    base_lm_dir = os.path.join(
        dir_13_32_1, 'wsj1', 'doc', 'lng_modl', 'base_lm')
    with open(os.path.join(base_lm_dir, 'bcb20onp.z'), 'rb') as file_:
        compressed = file_.read()
    decompressed = unlzw(compressed)
    with gzip.open(os.path.join(lmdir, 'lm_bg.arpa.gz'), 'wb') as file_:
        file_.write(decompressed)
    del compressed, decompressed

    # tcb20onp.z is not actually compressed
    lm_tg_gz = os.path.join(lmdir, 'lm_tg.arpa.gz')
    with open(os.path.join(base_lm_dir, 'tcb20onp.z')) as src_, \
            gzip.open(lm_tg_gz, 'wt') as dst_:
        for line in src_:
            if line.startswith('\\data\\'):
                break
        if line.startswith('\\data\\'):
            dst_.write(line)
        copyfileobj(src_, dst_)

    # the baseline language models are hideously malformed. We're going to
    # filter most warnings about their probabilities.
    warnings.filterwarnings('ignore', 'Calculated backoff', UserWarning)
    # the criterion for pruning is a bit different from IRSTLM. The threshold
    # was modified to produce roughly the same size model as with IRSTLM at
    # threshold 1e-7.
    lm_tgpr_gz = os.path.join(lmdir, 'lm_tgpr.arpa.gz')
    with gzip.open(lm_tg_gz, 'rt') as file_:
        ngram_list = parse_arpa_lm(file_)
    lm = ngram_lm.BackoffNGramLM(ngram_list)
    lm.relative_entropy_pruning(15e-8)
    ngram_list = lm.to_ngram_list()
    with gzip.open(lm_tgpr_gz, 'wt') as file_:
        ngram_lm.write_arpa(ngram_list, file_)
    del lm, ngram_list

    with open(os.path.join(base_lm_dir, 'bcb05onp.z'), 'rb') as file_:
        compressed = file_.read()
    decompressed = unlzw(compressed)
    with gzip.open(os.path.join(lmdir, 'lm_bg_5k.arpa.gz'), 'wb') as file_:
        file_.write(decompressed)
    del compressed, decompressed

    # tcb05cnp.z *is* compressed
    lm_tg_5k_gz = os.path.join(lmdir, 'lm_tg_5k.arpa.gz')
    with open(os.path.join(base_lm_dir, 'tcb05cnp.z'), 'rb') as file_:
        compressed = file_.read()
    decompressed = unlzw(compressed)
    first_idx = decompressed.find(b'\\data\\')
    second_idx = decompressed.find(b'\\data\\', first_idx + 1)
    decompressed = decompressed[second_idx:]
    with gzip.open(lm_tg_5k_gz, 'wb') as file_:
        file_.write(decompressed)
    del compressed, decompressed

    lm_tgpr_5k_gz = os.path.join(lmdir, 'lm_tgpr_5k.arpa.gz')
    with gzip.open(lm_tg_5k_gz, 'rt') as file_:
        ngram_list = parse_arpa_lm(file_)
    lm = ngram_lm.BackoffNGramLM(ngram_list)
    lm.relative_entropy_pruning(15e-8)
    ngram_list = lm.to_ngram_list()
    with gzip.open(lm_tgpr_5k_gz, 'wt') as file_:
        ngram_lm.write_arpa(ngram_list, file_)
    del lm, ngram_list

    spkrinfo = os.path.join(dir_, 'wsj0-train-spkrinfo.txt')
    if not os.path.isfile(spkrinfo):
        request.urlretrieve(
            'https://catalog.ldc.upenn.edu/docs/LDC93S6A/'
            'wsj0-train-spkrinfo.txt',
            spkrinfo
        )

    spk2gender = os.path.join(dir_, 'spk2gender')
    pipe_to(
        sort(set(
            ' '.join(x.lower().split()[:2])
            for x in cat(
                find_link_dir(wsj_subdirs, '11-13.1/wsj0/doc/spkrinfo.txt'),
                find_link_dir(
                    wsj_subdirs, '13-32.1/wsj1/doc/evl_spok/spkrinfo.txt'),
                find_link_dir(
                    wsj_subdirs, '13-34.1/wsj1/doc/dev_spok/spkrinfo.txt'),
                find_link_dir(
                    wsj_subdirs, '13-34.1/wsj1/doc/train/spkrinfo.txt'),
                spkrinfo
            )
            if not x.startswith(';') and not x.startswith('--')
        )),
        spk2gender
    )


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


def prepare_char_dict(data_root, src_dict_suffix, dst_dict_suffix):
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


def main(args=None):
    '''Prepare WSJ data for end-to-end pytorch training'''

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        'wsj_roots', nargs='+', type=os.path.abspath,
        help='Location of WSJ data directories, corresponding to WSJ0 '
        '(LDC93S6A or LDC93S6B) and WSJ1 (LDC94S13A or LDC94S13B)'
    )
    parser.add_argument(
        'data_root', type=os.path.abspath,
        help='The root directory under which to store data. Typically '
        '``data/``'
    )
    options = parser.parse_args(args)

    wsj_subdirs = []
    for wsj_root in options.wsj_roots:
        wsj_subdirs.extend(glob(wsj_root, r'??-?.?'))
        wsj_subdirs.extend(glob(wsj_root, r'??-??.?'))

    wsj_data_prep(wsj_subdirs, options.data_root)

    # prepare_lang.sh converts the arpa models to FSTs, so we skip it

    wsj_prepare_dict(options.data_root, '_nosp')

    # do not extend dictionary with potential non-oovs

    prepare_char_dict(options.data_root, '_nosp', '_char')


if __name__ == '__main__':
    sys.exit(main())
