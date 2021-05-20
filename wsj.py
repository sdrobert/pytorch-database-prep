#! /usr/bin/env python

# Copyright 2021 Sean Robertson
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
#
# typo fixes taken from wav2letter/recipes/data/wsj/utils.py, which is
# BSD-Licensed:
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""Command-line interface to prepare the WSJ CSR corpus for end to end ASR"""

import io
import os
import sys
import argparse
import re
import warnings
import locale
import gzip
import itertools
import torch
import urllib.request as request

from collections import OrderedDict
from shutil import copy as copy_paths

import ngram_lm  # type: ignore (pylance might complain if in subdirectory)
import pydrobert.speech.command_line as speech_cmd
import pydrobert.torch.command_line as torch_cmd

from unlzw import unlzw  # type: ignore
from common import get_num_avail_cores, glob, mkdir, sort, cat, pipe_to, wc_l  # type: ignore

locale.setlocale(locale.LC_ALL, "C")


ALPHA = set(chr(x) for x in range(ord("A"), ord("Z") + 1))


def find_link_dir(wsj_subdirs, rel_path, required=True):
    """find rel_path as suffix in wsj_subdirs, return it or None

    Kaldi makes soft links of all `wsj_subdirs` in a ``links/`` directory.
    `rel_path` is some path that would be in the run script as
    ``links/<rel_path>``, which means it should *really* exist as
    ``<wsj_subdir>/<rel_path>``, where ``wsj_subdir`` is an element of
    `wsj_subdirs` This removes the need to create symbolic links in a
    ``links/`` directory. Unix-style.
    """
    for wsj_subdir in wsj_subdirs:
        dir_ = os.path.dirname(wsj_subdir)
        path = os.path.join(dir_, rel_path).replace(os.sep, "/")
        if path.startswith(wsj_subdir.replace(os.sep, "/")) and os.path.exists(path):
            return path
    if required:
        raise ValueError("{} does not exist in {}".format(rel_path).format(wsj_subdirs))
    return None


def ndx2flist(in_stream, wsj_subdirs):
    dir_pattern = re.compile(r".+/([0-9.-]+)/?$")
    disk2fn = dict()
    for fn in wsj_subdirs:
        fn = fn.replace(os.sep, "/")  # ensure unix-style file paths
        match = dir_pattern.match(fn)
        if match is None:
            raise ValueError("Bad command-line argument {}".format(fn))
        disk_id = match.group(1)
        disk_id = disk_id.replace(".", "_").replace("-", "_")
        if fn.endswith(os.sep):
            fn = disk_id[:-1]
        disk2fn[disk_id] = fn

    line_pattern = re.compile(r"^([0-9_]+):\s*(\S+)$")
    for line_no, line in enumerate(in_stream):
        if line.startswith(";"):
            continue
        match = line_pattern.match(line)
        if match is None:
            raise ValueError("Could not parse line {}".format(line_no + 1))
        disk, filename = match.groups()
        if disk not in disk2fn:
            raise ValueError("Disk id {} not found".format(disk))
        yield "{}/{}".format(disk2fn[disk], filename)


def flist2scp(path):
    line_pattern = re.compile(r"^\S+/(\w+)\.[wW][vV]1$")
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            match = line_pattern.match(line)
            if match is None:
                raise ValueError("Bad line {}".format(line))
            id_ = match.group(1).lower()
            yield "{} {}".format(id_, line)


def find_transcripts_many(in_stream, flist, flipped=False):
    spk2dot = dict()
    file_pattern = re.compile(r"^\S+/(\w{6})00\.(dot|lsn)")
    with open(flist) as f:
        for line in f:
            line = line.rstrip()
            match = file_pattern.match(line)
            if match is None:
                continue
            spk = match.group(1).lower()
            spk2dot[spk] = line

    utt_pattern = re.compile(r"^(\w{6})\w\w$")
    trans_pattern = re.compile(r"^(.+)\((\w{8})\)$")
    curspk = file_ = None
    for uttid in in_stream:
        match = utt_pattern.match(uttid)
        if match is None:
            raise ValueError("Bad utterance id {}".format(uttid))
        spk = match.group(1)
        if spk != curspk:
            utt2trans = dict()
            if spk not in spk2dot:
                raise ValueError("No file for speaker {}".format(spk))
            file_ = spk2dot[spk]
            with open(file_) as f:
                for line_no, line in enumerate(f):
                    line = line.rstrip()
                    match = trans_pattern.match(line)
                    if match is None:
                        raise ValueError(
                            "Bad line {} in file {} (line {})"
                            "".format(line, file_, line_no + 1)
                        )
                    trans, utt = match.groups()
                    if flipped:
                        utt = utt[:4] + utt[5] + utt[4] + utt[6:]
                    utt2trans[utt.lower()] = trans
            if uttid not in utt2trans:
                raise ValueError(
                    "No transcript for utterance id {} (current file is "
                    "{})".format(uttid, file_)
                )
            yield "{} {}".format(uttid, utt2trans[uttid])
            del utt2trans[uttid]  # no more need for it - free space


def find_transcripts_one(in_stream, lsn_path):
    """Yields <utt> <transcript> pairs from utts (stream) and transcripts"""

    # lsn test transcripts all come from the same master file
    utt2trans = dict()
    trans_pattern = re.compile(r"^(.+)\((\w{8})\)$")
    line_no = 0
    with open(lsn_path) as lsn_file:
        for uttid in in_stream:
            while uttid not in utt2trans:
                line = lsn_file.readline().rstrip()
                line_no += 1
                match = trans_pattern.match(line)
                if match is None:
                    raise ValueError(
                        "Bad line {} in lsn file {} (line {})"
                        "".format(line, lsn_path, line_no)
                    )
                trans, utt = match.groups()
                if utt == "4ODC0207":
                    # I listened to this recording. It's clearly "BUY OUT",
                    # not "BUY BACK"
                    trans = trans.replace("BUY BACK", "BUY OUT")
                utt = utt.lower()
                utt2trans[utt] = trans
            yield "{} {}".format(uttid, utt2trans[uttid])
            del utt2trans[uttid]


def normalize_transcript(in_stream, nsn, spn, lexical_equivs):
    """Sanitize in_stream transcripts"""
    # in addition to some typo fixes, we do a few things different from
    # Kaldi. We distinguish between vocalized/speech noise and non-vocalized
    # noise, for one. We also handle word fragments by removing the missing
    # part of the fragment (it is not vocalized)
    line_pattern = re.compile(r"^(\S+) (.+)$")
    del_pattern = re.compile(r"^([.~]|\[[</]\w+\]|\[\w+[>/]\])$")
    noise_pattern = re.compile(r"^\[\w+\]$")
    verbdel_pattern = re.compile(r"^<([\w']+)>$")
    missing_fragment_pattern = re.compile(r"^(.*)\([^)]+\)(.*)$")
    spoken_noise_tokens = {
        "[AH]",
        "[UH]",
        "[UM]",
        "[EH]",
        "[CROSS_TALK]",
        "[UNINTELLIGIBLE]",
        "<AH>",
    }
    skippable = {".", "~"}
    for line in in_stream:
        match = line_pattern.match(line)
        if match is None:
            raise ValueError("Bad line {}".format(line))
        out, trans = match.groups()
        if trans == "~~":
            # are 'null waveforms', and should be ignored
            continue
        for w in trans.split(" "):
            # typo fixes from wav2letter. The "(IN-PARENTHESIS" example isn't
            # a typo... the guy is saying "IN PARENTHESIS"
            w = w.upper().replace("\\", "").replace("CORP;", "CORP.").replace("`", "'")
            # a few of my own. We also remove emphasis ('*') here because there
            # isn't a lexical equivalence for it, but it interferes with some
            # of the other rules here
            w = (
                w.replace("*", "")
                .replace(")CLOSE_PAREN", ")CLOSE-PAREN")
                .replace(".PERCENT", "%PERCENT")
                .replace("<MR.>", "MI-")
            )
            if w in lexical_equivs:
                w = lexical_equivs[w]
            elif del_pattern.match(w) or w in skippable:
                continue
            elif w in spoken_noise_tokens:
                w = spn
            elif noise_pattern.match(w):
                w = nsn
            else:
                match = verbdel_pattern.match(w)
                if match:
                    # verbal deletions are often actually said, then repeated
                    w = match.group(1)
                match = missing_fragment_pattern.match(w)
                if match:
                    w = "".join(match.groups())
            w = w.replace(":", "").replace("!", "")
            out += " " + w
        yield out


def utt2spk_to_spk2utt(in_stream):
    spk_hash = OrderedDict()
    for line in in_stream:
        utt, spk = line.split(" ")
        spk_hash.setdefault(spk, []).append(utt)
    for spk, utts in spk_hash.items():
        yield "{} {}".format(spk, " ".join(utts))


def wsj_data_prep(wsj_subdirs, data_root):
    # this follows part of kaldi/egs/wsj/s5/local/wsj_data_prep.sh, but factors
    # out the language modelling stuff to wsj_word_lm in case we don't care
    # about words. We also make the following adjustments:
    # - base eval transcriptions off the gold-standard ones outlined in
    #   https://catalog.ldc.upenn.edu/docs/LDC93S6B/csrnov92.txt and
    #   https://catalog.ldc.upenn.edu/docs/LDC94S13A/csrnov93.html
    # - dev_dt_20 is the exact same set as dev93, *_05 the same as _5k.
    #   Skip former.
    # - uses lexical equivalence map in WSJ1 to convert verbal punctuations
    #   (and a few others) in training data to something within the vocabulary.
    #   Kaldi just calls these UNKs.
    # See wiki for more info on the corpus and tasks in general

    for rel_path in {"11-13.1", "13-34.1", "11-2.1"}:
        if find_link_dir(wsj_subdirs, rel_path, required=False) is None:
            raise ValueError(
                "wsj_data_prep: Spot check of command line arguments failed. "
                "Command line arguments must be absolute pathnames to WSJ "
                "directories with names like 11-13.1"
            )

    dir_ = os.path.join(data_root, "local", "data")
    train_si84_flist = os.path.join(dir_, "train_si84.flist")
    train_si284_flist = os.path.join(dir_, "train_si284.flist")
    test_eval92_flist = os.path.join(dir_, "test_eval92.flist")
    test_eval92_5k_flist = os.path.join(dir_, "test_eval92_5k.flist")
    test_eval93_flist = os.path.join(dir_, "test_eval93.flist")
    test_eval93_5k_flist = os.path.join(dir_, "test_eval93_5k.flist")
    test_dev93_flist = os.path.join(dir_, "test_dev93.flist")
    test_dev93_5k_flist = os.path.join(dir_, "test_dev93_5k.flist")
    dot_files_flist = os.path.join(dir_, "dot_files.flist")
    lsn_files_flist = os.path.join(dir_, "lsn_files.flist")
    spkrinfo = os.path.join(dir_, "wsj0-train-spkrinfo.txt")
    spk2gender = os.path.join(dir_, "spk2gender")
    lexical_equivs_csv = os.path.join(dir_, "lex_equivs.csv")
    vp_lexical_equivs_csv = os.path.join(dir_, "vp_lexical_equivs.csv")

    mkdir(dir_)

    lexical_equivs = dict()
    lex_equiv_pattern = re.compile(r"^\s*([^=]+)\s*=>\s*(.*)$")
    with open(find_link_dir(wsj_subdirs, "13-32.1/tranfilt/93map1.rls")) as f:
        for line in f:
            match = lex_equiv_pattern.match(line)
            if match is None:
                continue
            key, value = match.groups()
            lexical_equivs[key.strip()] = value.strip()
    # a problem with 93map1.rls is it maps "BUYOUT" to "BUY BACK".
    # it should be fixed in 93map1x.rls, as mentioned in tranfilt/readme.doc,
    # buit I'm not sure how widely distributed this is, so I just make the
    # change here
    lexical_equivs["BUYOUT"] = "BUY OUT"
    # Here are some additional lexical equivalents that I found looking at
    # the training transcripts. Since we don't include VPs in our vocabulary,
    # it won't hurt to include these here
    lexical_equivs.update(
        {
            "(PARENTHESES": "PARENTHESES",
            ")UN-PARENTHESES": "UN PARENTHESES",
            "(PAREN": "PAREN",
            ")PAREN": "PAREN",
            ")END-THE-PAREN": "END THE PAREN",
            ")CLOSE-PAREN": "CLOSE PAREN",
            "(BRACE": "BRACE",
            ")CLOSE-BRACE": "CLOSE BRACE",
            "(BEGIN-PARENS": "BEGIN PARENS",
            ")END-PARENS": "END PARENS",
            "(IN-PARENTHESES": "IN PARENTHESES",
            "(PARENTHETICALLY": "PARENTHETICALLY",
            ")END-OF-PAREN": "END OF PAREN",
            '"END-OF-QUOTE': "END OF QUOTE",
            '"IN-QUOTES': "IN QUOTES",
            "(IN-PARENTHESIS": "IN PARENTHESIS",
            "...ELLIPSIS": "ELLIPSIS",
            ".DOT": "DOT",
        }
    )
    with open(lexical_equivs_csv, "w") as file_:
        for key, value in sorted(lexical_equivs.items()):
            file_.write("{},{}\n".format(key, value))

    # determine which lexical equivalences are verbal punctuation. These are
    # the only terms we'll replace in the training data. The rest are fair
    # game in natural language
    vp_lexical_equivs = dict(
        (k, v) for (k, v) in lexical_equivs.items() if k[0] not in ALPHA
    )
    with open(vp_lexical_equivs_csv, "w") as file_:
        for key, value in sorted(vp_lexical_equivs.items()):
            file_.write("{},{}\n".format(key, value))

    # 11.2.1/si_tr_s/401 doesn't exist, which is why we filter it out
    missing_pattern = re.compile(r"11-2.1/wsj0/si_tr_s/401", flags=re.I)
    pipe_to(
        (
            x
            for x in sort(
                ndx2flist(
                    cat(
                        find_link_dir(
                            wsj_subdirs, "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx",
                        )
                    ),
                    wsj_subdirs,
                )
            )
            if missing_pattern.search(x) is None
        ),
        train_si84_flist,
    )
    nl = wc_l(cat(train_si84_flist))
    if nl != 7138:
        warnings.warn("expected 7138 lines in train_si84.flist, got {}".format(nl))

    # we also filter 47hc0418.wv1 and 46uc030b.wv1 because they're empty
    pipe_to(
        (
            x
            for x in sort(
                ndx2flist(
                    cat(
                        find_link_dir(
                            wsj_subdirs, "13-34.1/wsj1/doc/indices/si_tr_s.ndx",
                        ),
                        find_link_dir(
                            wsj_subdirs, "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx",
                        ),
                    ),
                    wsj_subdirs,
                )
            )
            if (
                missing_pattern.search(x) is None
                and not x.endswith("47hc0418.wv1")
                and not x.endswith("46uc030b.wv1")
            )
        ),
        train_si284_flist,
    )
    nl = wc_l(cat(train_si284_flist))
    if nl != 37414:  # two less than kaldi b/c of empty audio
        warnings.warn("expected 37414 lines in train_si284.flist, got {}".format(nl))

    pipe_to(
        (
            x + ".wv1"
            for x in sort(
                ndx2flist(
                    cat(
                        find_link_dir(
                            wsj_subdirs,
                            "11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx",
                        )
                    ),
                    wsj_subdirs,
                )
            )
        ),
        test_eval92_flist,
    )
    nl = wc_l(cat(test_eval92_flist))
    if nl != 333:
        warnings.warn("expected 333 lines in test_eval92.flist, got {}".format(nl))

    pipe_to(
        (
            x + ".wv1"
            for x in sort(
                ndx2flist(
                    cat(
                        find_link_dir(
                            wsj_subdirs,
                            "11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx",
                        )
                    ),
                    wsj_subdirs,
                )
            )
        ),
        test_eval92_5k_flist,
    )
    nl = wc_l(cat(test_eval92_5k_flist))
    if nl != 330:
        warnings.warn("expected 330 lines in test_eval92_5k.flist, got {}".format(nl))

    pipe_to(
        sort(
            ndx2flist(
                (
                    x.replace("13_32_1", "13_33_1")
                    for x in cat(
                        find_link_dir(
                            wsj_subdirs, "13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx",
                        )
                    )
                ),
                wsj_subdirs,
            )
        ),
        test_eval93_flist,
    )
    nl = wc_l(cat(test_eval93_flist))
    if nl != 213:
        warnings.warn("expected 213 lines in test_eval93.flist, got {}".format(nl))

    pipe_to(
        sort(
            ndx2flist(
                (
                    x.replace("13_32_1", "13_33_1")
                    for x in cat(
                        find_link_dir(
                            wsj_subdirs, "13-32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx",
                        )
                    )
                ),
                wsj_subdirs,
            )
        ),
        test_eval93_5k_flist,
    )
    nl = wc_l(cat(test_eval93_5k_flist))
    if nl != 215:
        # I've found 215 entries, unlike the kaldi_data_prep.sh which suggests
        # 213. I've verified using kaldi_data_prep.sh
        warnings.warn("expected 215 lines in test_eval93_5k.flist, got {}".format(nl))

    pipe_to(
        sort(
            ndx2flist(
                cat(find_link_dir(wsj_subdirs, "13-34.1/wsj1/doc/indices/h1_p0.ndx",)),
                wsj_subdirs,
            )
        ),
        test_dev93_flist,
    )
    nl = wc_l(cat(test_dev93_flist))
    if nl != 503:
        warnings.warn("expected 503 lines in test_dev93.flist, got {}".format(nl))

    pipe_to(
        sort(
            ndx2flist(
                cat(
                    find_link_dir(
                        wsj_subdirs,
                        "13-34.1/wsj1/doc/indices/h2_p0.ndx",
                        required=True,
                    )
                ),
                wsj_subdirs,
            )
        ),
        test_dev93_5k_flist,
    )
    nl = wc_l(cat(test_dev93_5k_flist))
    if nl != 513:
        warnings.warn("expected 513 lines in test_dev93_5k.flist, got {}".format(nl))

    pipe_to(
        itertools.chain(
            *(
                (y for y in glob(x, "**/*") if re.search(r"\.dot$", y, flags=re.I))
                for x in wsj_subdirs
            )
        ),
        dot_files_flist,
    )

    # My copy of WSJ0 does not have the appropriate score/ directory - it seems
    # to be a copy of WSJ1's. Fortunately, it looks like the NIST SCORE
    # package (https://www.nist.gov/document/score3-6-2tgz)
    # has the WSJ0 transcripts in all.snr, and those match the ones lying
    # around the wsj0 directories. So we scour them.

    pipe_to(
        itertools.chain(
            *(
                (y for y in glob(x, "**/*") if re.search(r"\.lsn$", y, flags=re.I))
                for x in wsj_subdirs
            )
        ),
        lsn_files_flist,
    )

    spn, nsn = "<SPOKEN_NOISE>", "<NOISE>"
    for x in {
        "train_si84",
        "train_si284",
        "test_eval92",
        "test_eval93",
        "test_dev93",
        "test_eval92_5k",
        "test_eval93_5k",
        "test_dev93_5k",
    }:
        src = os.path.join(dir_, x + ".flist")
        sph = os.path.join(dir_, x + "_sph.scp")
        raw = os.path.join(dir_, x + ".raw.txt")
        fragments = os.path.join(dir_, x + ".fragments.txt")
        cleaned = os.path.join(dir_, x + ".cleaned.txt")
        utt2spk = os.path.join(dir_, x + ".utt2spk")
        spk2utt = os.path.join(dir_, x + ".spk2utt")

        pipe_to(sort(flist2scp(src)), sph)

        if x in {"test_eval93", "test_eval93_5k"}:
            pipe_to(
                find_transcripts_many(
                    (x.split()[0] for x in cat(sph)), dot_files_flist
                ),
                raw,
            )
            pipe_to(
                sort(normalize_transcript(cat(raw), nsn, spn, vp_lexical_equivs)),
                fragments,
            )
            # use standard reference transcriptions for clean version
            lns_path = find_link_dir(wsj_subdirs, "13-32.1/score/lib/wsj/nov93wsj.ref")
            pipe_to(
                sort(find_transcripts_one((x.split()[0] for x in cat(sph)), lns_path)),
                cleaned,
            )
        elif x in {"test_eval92", "test_eval92_5k"}:
            pipe_to(
                find_transcripts_many(
                    (x.split()[0] for x in cat(sph)), dot_files_flist
                ),
                raw,
            )
            pipe_to(
                sort(normalize_transcript(cat(raw), nsn, spn, vp_lexical_equivs)),
                fragments,
            )
            pipe_to(
                sort(
                    find_transcripts_many(
                        (x.split()[0] for x in cat(sph)), lsn_files_flist, flipped=True
                    )
                ),
                cleaned,
            )
        else:
            pipe_to(
                find_transcripts_many(
                    (x.split()[0] for x in cat(sph)), dot_files_flist
                ),
                raw,
            )
            pipe_to(
                sort(normalize_transcript(cat(raw), nsn, spn, vp_lexical_equivs)),
                fragments,
            )
            pipe_to(
                (
                    " ".join(
                        spn if (w[0] == "-" or w[-1] == "-") else w
                        for w in line.split()
                    )
                    for line in cat(fragments)
                ),
                cleaned,
            )

        # XXX(sdrobert): don't care about _wav.scp

        pipe_to(
            ("{} {}".format(y, y[:3]) for y in (x.split()[0] for x in cat(raw))),
            utt2spk,
        )

        pipe_to(utt2spk_to_spk2utt(cat(utt2spk)), spk2utt)

    if not os.path.isfile(spkrinfo):
        request.urlretrieve(
            "https://catalog.ldc.upenn.edu/docs/LDC93S6A/" "wsj0-train-spkrinfo.txt",
            spkrinfo,
        )

    pipe_to(
        sort(
            set(
                " ".join(x.lower().split()[:2])
                for x in cat(
                    find_link_dir(wsj_subdirs, "11-13.1/wsj0/doc/spkrinfo.txt"),
                    find_link_dir(
                        wsj_subdirs, "13-32.1/wsj1/doc/evl_spok/spkrinfo.txt"
                    ),
                    find_link_dir(
                        wsj_subdirs, "13-34.1/wsj1/doc/dev_spok/spkrinfo.txt"
                    ),
                    find_link_dir(wsj_subdirs, "13-34.1/wsj1/doc/train/spkrinfo.txt"),
                    spkrinfo,
                )
                if not x.startswith(";") and not x.startswith("--")
            )
        ),
        spk2gender,
    )


def wsj_init_word_config(wsj_subdirs, data_root, config_dir, vocab_size):
    local_data_dir = os.path.join(data_root, "local", "data")
    token2id_txt = os.path.join(config_dir, "token2id.txt")
    id2token_txt = os.path.join(config_dir, "id2token.txt")
    dir_13_32_1 = find_link_dir(wsj_subdirs, "13-32.1")
    vocab_dir = os.path.join(dir_13_32_1, "wsj1", "doc", "lng_modl", "vocab")

    mkdir(config_dir)

    # determine the vocabulary based on whether we're in the 5k closed,
    # 20k open, or 64k closed condition. Note Kaldi uses the 5k open
    # vocabulary. Standard WSJ eval assumes a closed vocab @ 5k. We use
    # non-verbalized punctuation vocabulary versions since testing is all
    # non-verbalized. We are safe to remove verbalized punctuation from the
    # training data since there is no corresponding audio to worry about.

    # note that even though the 64k and 5k vocabularies are closed w.r.t the
    # test set, there are words outside of both in the training set. That's
    # why we always have to include the <UNK> character

    vocab = ["</s>", "<SPOKEN_NOISE>", "<NOISE>", "<UNK>", "<s>"]
    if vocab_size == 5:
        # though the test set vocabulary is closed, the training set will still
        # contain OOV words. We add the <UNK>
        vocab += sorted(
            x for x in cat(os.path.join(vocab_dir, "wlist5c.nvp")) if x[0] != "#"
        )
    elif vocab_size == 20:
        vocab += sorted(
            x for x in cat(os.path.join(vocab_dir, "wlist20o.nvp")) if x[0] != "#"
        )
    else:
        # we're allowed to use the full 64k vocabulary list to generate the
        # vocabulary, but not for language modelling:
        # 13-32.1\wsj1\doc\lng_modl\vocab\readme.txt
        vocab += sorted(
            x.split()[1]
            for x in cat(os.path.join(vocab_dir, "wfl_64.lst"))
            if x[0] != "#" and x.split()[1][0] in ALPHA
        )

    with open(token2id_txt, "w") as t2id, open(id2token_txt, "w") as id2t:
        for i, v in enumerate(vocab):
            t2id.write("{} {}\n".format(v, i))
            id2t.write("{} {}\n".format(i, v))

    # get the appropriate files for the vocabulary
    to_copy = {
        "lex_equivs.csv",
        "spk2gender",
        "vp_lexical_equivs.csv",
        "train_si84.trn",
        "train_si84_sph.scp",
        "train_si84.utt2spk",
        "train_si84.spk2utt",
        "train_si284.trn",
        "train_si284_sph.scp",
        "train_si284.utt2spk",
        "train_si284.spk2utt",
        "test_dev93_5k.trn",
        "test_dev93_5k_sph.scp",
        "test_dev93_5k.utt2spk",
        "test_eval92_5k.trn",
        "test_eval92_5k_sph.scp",
        "test_eval92_5k.utt2spk",
        "test_eval93_5k.trn",
        "test_eval93_5k_sph.scp",
        "test_eval93_5k.utt2spk",
    }
    for x in to_copy:
        if vocab_size != 5:
            x = x.replace("_5k", "")
        if x.endswith(".trn"):
            pipe_to(
                (
                    "{1} ({0})".format(*y.split(maxsplit=1))
                    for y in cat(os.path.join(local_data_dir, x[:-4] + ".cleaned.txt"))
                ),
                os.path.join(config_dir, x.replace("_5k", "")),
            )
        else:
            copy_paths(
                os.path.join(local_data_dir, x),
                os.path.join(config_dir, x.replace("_5k", "")),
            )

    # determine the OOVs in the training partition. Primarily for diagnostic
    # purposes
    vocab = set(vocab)
    oovs = set()
    with open(os.path.join(config_dir, "train_si284.trn")) as file_:
        for line in file_:
            trans = line.strip().split()
            trans.pop()
            oovs.update(set(trans) - vocab)
    pipe_to(sorted(oovs), os.path.join(config_dir, "train_oovs.txt"))


def write_cmu_vocab_to_path(path):
    url = (
        "https://raw.githubusercontent.com/Alexir/CMUdict/"
        "7a37de79f7e650fd6334333b1b5d2bcf0dee8ad3/cmudict-0.7b"
    )
    dict_path, _ = request.urlretrieve(url)
    with open(dict_path) as in_, open(path, "w") as out:
        for line in in_:
            line = line.strip()
            if not line or line[0] not in ALPHA:
                continue
            word = line.split(" ", maxsplit=1)[0]
            if word.endswith(")"):  # multiple pronunciations. Skip
                continue
            out.write(word)
            out.write("\n")


def wsj_train_lm(
    vocab, ngram_counts, max_order, toprune_txt_gz, lm_arpa_gz, deltas=None
):
    to_prune = set(ngram_counts[0]) - vocab
    # prune all out-of-vocabulary n-grams and hapax
    for i, ngram_count in enumerate(ngram_counts[1:]):
        if i:
            to_prune |= set(
                k
                for (k, v) in ngram_count.items()
                if k[:-1] in to_prune or k[-1] in to_prune or v == 1
            )
        else:
            to_prune |= set(
                k
                for (k, v) in ngram_count.items()
                if k[0] in to_prune or k[1] in to_prune or v == 1
            )
    assert not (to_prune & vocab)
    with gzip.open(toprune_txt_gz, "wt") as file_:
        for v in sorted(x if isinstance(x, str) else " ".join(x) for x in to_prune):
            file_.write(v)
            file_.write("\n")
    prob_list = ngram_lm.ngram_counts_to_prob_list_kneser_ney(
        ngram_counts, sos="<s>", to_prune=to_prune, delta=deltas
    )
    # remove start-of-sequence probability mass
    # (we don't necessarily have an UNK, so pretend it's something else)
    lm = ngram_lm.BackoffNGramLM(prob_list, sos="<s>", eos="</s>", unk="<s>")
    lm.prune_by_name({"<s>"})
    prob_list = lm.to_prob_list()
    with gzip.open(lm_arpa_gz, "wt") as file_:
        ngram_lm.write_arpa(prob_list, file_)


def wsj_word_lm(wsj_subdirs, config_dir, max_order):
    # We're doing things differently from Kaldi.
    # The NIST language model probabilities are really messed up. We train
    # up our own using Modified Kneser-Ney, but the same vocabulary that they
    # use.

    dir_13_32_1 = find_link_dir(wsj_subdirs, "13-32.1")
    lmdir = os.path.join(config_dir, "lm")
    cleaned_txt_gz = os.path.join(lmdir, "cleaned.txt.gz")
    train_data_root = os.path.join(
        dir_13_32_1, "wsj1", "doc", "lng_modl", "lm_train", "np_data"
    )
    wordlist_txt = os.path.join(lmdir, "wordlist.txt")
    vp_lexical_equivs_csv = os.path.join(config_dir, "vp_lexical_equivs.csv")
    token2id_txt = os.path.join(config_dir, "token2id.txt")
    toprune_txt_gz = os.path.join(lmdir, "toprune.txt.gz")
    lm_arpa_gz = os.path.join(lmdir, "lm.arpa.gz")

    mkdir(lmdir)

    write_cmu_vocab_to_path(wordlist_txt)

    isword = set(cat(wordlist_txt))
    vp = set(x.split()[0] for x in cat(vp_lexical_equivs_csv))
    vocab = set(x.split()[0] for x in cat(token2id_txt))

    # clean up training data. We do something similar to wsj_extend_dict.sh
    # we are safe to remove verbalized punctuation because that'll match the
    # non-verbalized punctuation sentences afterwards.
    assert os.path.isdir(train_data_root)
    train_data_files = []
    for subdir in ("87", "88", "89"):
        train_data_files.extend(glob(os.path.join(train_data_root, subdir), r"*.[Zz]"))
    with gzip.open(cleaned_txt_gz, "wt") as out:
        for train_data_file in train_data_files:
            with open(train_data_file, "rb") as in_:
                compressed = in_.read()
            decompressed = unlzw(compressed)
            in_ = io.TextIOWrapper(io.BytesIO(decompressed))
            for line in in_:
                if line.startswith("<"):
                    continue
                A = line.strip().upper().split(" ")
                for n, a in enumerate(A):
                    if a in vp:
                        continue
                    if a not in isword and len(a) > 1 and a.endswith("."):
                        out.write(a[:-1])
                        if n < len(A) - 1:
                            out.write("\n")
                    else:
                        out.write(a + " ")
                out.write("\n")
            del in_, compressed, decompressed
    del isword, train_data_files

    # convert cleaned data into sentences
    with gzip.open(cleaned_txt_gz, "rt") as file_:
        text = file_.read()
    sents = ngram_lm.text_to_sents(text, sent_end_expr="\n", word_delim_expr=" ")
    del text

    # count n-grams in sentences
    ngram_counts = ngram_lm.sents_to_ngram_counts(
        sents, max_order, sos="<s>", eos="</s>"
    )
    # ensure all vocab terms have unigram counts (even if 0) for zeroton
    # interpolation
    for v in vocab:
        ngram_counts[0].setdefault(v, 0)
    del sents

    with gzip.open(os.path.join(lmdir, "counts.txt.gz"), "wt") as file_:
        for ngram_count in ngram_counts:
            for ngram, count in ngram_count.items():
                if isinstance(ngram, str):
                    file_.write("{} {}\n".format(ngram, count))
                else:
                    file_.write("{} {}\n".format(" ".join(ngram), count))

    wsj_train_lm(vocab, ngram_counts, max_order, toprune_txt_gz, lm_arpa_gz)


def wsj_init_char_config(
    wsj_subdirs, data_root, config_dir, eval_vocab_size, ngraph_order
):
    local_data_dir = os.path.join(data_root, "local", "data")
    token2id_txt = os.path.join(config_dir, "token2id.txt")
    id2token_txt = os.path.join(config_dir, "id2token.txt")
    words_txt = os.path.join(config_dir, "words.txt")
    dir_13_32_1 = find_link_dir(wsj_subdirs, "13-32.1")
    vocab_dir = os.path.join(dir_13_32_1, "wsj1", "doc", "lng_modl", "vocab")

    mkdir(config_dir)

    # we save the (closed) list of words from the eval set in case anyone
    # cares to spellcheck. The primary purpose is to determine what characters
    # are inside this list
    if eval_vocab_size == 5:
        words = set(
            x for x in cat(os.path.join(vocab_dir, "wlist5c.nvp")) if x[0] != "#"
        )
    else:
        # we're allowed to use the full 64k vocabulary list to generate the
        # vocabulary, but not for language modelling:
        # 13-32.1\wsj1\doc\lng_modl\vocab\readme.txt
        words = set(
            x.split()[1]
            for x in cat(os.path.join(vocab_dir, "wfl_64.lst"))
            if x[0] != "#" and x.split()[1][0] in ALPHA
        )
    pipe_to(sorted(words), words_txt)

    chars = set("_".join(words))

    # note the character vocabulary is always closed if we take all the
    # characters from a closed word-level vocabulary.
    vocab = ["</s>", "<SPOKEN_NOISE>", "<NOISE>", "<s>"]
    vocab += sorted(
        "".join(ngraph) for ngraph in itertools.product(chars, repeat=ngraph_order)
    )
    with open(token2id_txt, "w") as t2id, open(id2token_txt, "w") as id2t:
        for i, v in enumerate(vocab):
            t2id.write("{} {}\n".format(v, i))
            id2t.write("{} {}\n".format(i, v))

    to_copy = {
        "lex_equivs.csv",
        "spk2gender",
        "vp_lexical_equivs.csv",
        "train_si84.trn",
        "train_si84_sph.scp",
        "train_si84.utt2spk",
        "train_si84.spk2utt",
        "train_si284.trn",
        "train_si284_sph.scp",
        "train_si284.utt2spk",
        "train_si284.spk2utt",
        "test_dev93_5k.trn",
        "test_dev93_5k_sph.scp",
        "test_dev93_5k.utt2spk",
        "test_eval92_5k.trn",
        "test_eval92_5k_sph.scp",
        "test_eval92_5k.utt2spk",
        "test_eval93_5k.trn",
        "test_eval93_5k_sph.scp",
        "test_eval93_5k.utt2spk",
    }
    for x in to_copy:
        if eval_vocab_size != 5:
            x = x.replace("_5k", "")
        if x.endswith(".trn"):
            y = x[:-4]
            y += ".cleaned.txt" if x.startswith("test_") else ".fragments.txt"
            word_txt_to_char_trn(
                os.path.join(local_data_dir, y),
                os.path.join(config_dir, x.replace("_5k", "")),
                ngraph_order,
            )
        else:
            copy_paths(
                os.path.join(local_data_dir, x),
                os.path.join(config_dir, x.replace("_5k", "")),
            )


def wsj_char_lm(wsj_subdirs, config_dir, max_order, ngraph_order):
    dir_13_32_1 = find_link_dir(wsj_subdirs, "13-32.1")
    lmdir = os.path.join(config_dir, "lm")
    cleaned_txt_gz = os.path.join(lmdir, "cleaned.txt.gz")
    train_data_root = os.path.join(
        dir_13_32_1, "wsj1", "doc", "lng_modl", "lm_train", "np_data"
    )
    wordlist_txt = os.path.join(lmdir, "wordlist.txt")
    vp_lexical_equivs_csv = os.path.join(config_dir, "vp_lexical_equivs.csv")
    token2id_txt = os.path.join(config_dir, "token2id.txt")
    toprune_txt_gz = os.path.join(lmdir, "toprune.txt.gz")
    lm_arpa_gz = os.path.join(lmdir, "lm.arpa.gz")

    mkdir(lmdir)

    write_cmu_vocab_to_path(wordlist_txt)

    isword = set(cat(wordlist_txt))
    vp = set(x.split()[0] for x in cat(vp_lexical_equivs_csv))
    vocab = set(x.split()[0] for x in cat(token2id_txt))

    def empty_buff(buff):
        for i in range(0, len(buff), ngraph_order):
            ngraph = buff[i : i + ngraph_order]
            ngraph += "_" * (ngraph_order - len(ngraph))
            out.write(ngraph)
            out.write(" " if i + ngraph_order < len(buff) else "\n")

    # clean up training data. We pretty much do the same thing as word-level,
    # but replace spaces with underscores and split on ngraph order,
    # post-pending with spaces if necessary
    assert os.path.isdir(train_data_root)
    train_data_files = []
    for subdir in ("87", "88", "89"):
        train_data_files.extend(glob(os.path.join(train_data_root, subdir), r"*.[Zz]"))
    with gzip.open(cleaned_txt_gz, "wt") as out:
        for train_data_file in train_data_files:
            with open(train_data_file, "rb") as in_:
                compressed = in_.read()
            decompressed = unlzw(compressed)
            in_ = io.TextIOWrapper(io.BytesIO(decompressed))
            for line in in_:
                if line.startswith("<"):
                    continue
                A = line.strip().upper().split(" ")
                buff = ""
                for n, a in enumerate(A):
                    if a in vp:
                        continue
                    if a not in isword and len(a) > 1 and a.endswith("."):
                        buff += a[:-1]
                        if n < len(A) - 1:
                            empty_buff(buff)
                            buff = ""
                    else:
                        buff += a
                        if n < len(A) - 1:
                            buff += "_"
                empty_buff(buff)
            del in_, compressed, decompressed
    del isword, train_data_files

    # split data into 'sentences'
    with gzip.open(cleaned_txt_gz, "rt") as file_:
        text = file_.read()
    sents = ngram_lm.text_to_sents(text, sent_end_expr="\n", word_delim_expr=" ")
    del text

    # count n-grams in sentences
    ngram_counts = ngram_lm.sents_to_ngram_counts(
        sents, max_order, sos="<s>", eos="</s>"
    )
    del sents
    # ensure all vocab terms have unigram counts (even if 0) for zeroton
    # interpolation
    for v in vocab:
        ngram_counts[0].setdefault(v, 0)
    with gzip.open(os.path.join(lmdir, "counts.txt.gz"), "wt") as file_:
        for ngram_count in ngram_counts:
            for ngram, count in ngram_count.items():
                if isinstance(ngram, str):
                    file_.write("{} {}\n".format(ngram, count))
                else:
                    file_.write("{} {}\n".format(" ".join(ngram), count))

    # learned deltas don't work for lower-order n-grams of characters because
    # the count of count 2 unigrams/bigrams is lower than that of the 3-count.
    # The higher-order deltas seem to be converging to 0.5, 1.0, 1.5, however,
    # so we use that here.
    # It turns out that 0.5, 1.0, 1.5 are the default fallbacks in KenLM, and
    # other character-based lms seem to be using that switch. Cool.
    wsj_train_lm(
        vocab,
        ngram_counts,
        max_order,
        toprune_txt_gz,
        lm_arpa_gz,
        [(0.5, 1.0, 1.5)] * max_order,
    )


def word_txt_to_char_trn(txt, trn, ngraph_order):
    char_token_pattern = re.compile(r"[^<]|<[^>]+>")
    with open(txt) as in_, open(trn, "w") as out:
        for line in in_:
            utt, trans = line.strip().split(maxsplit=1)
            # replace spaces with underscores to model spaces. Also, remove
            # '-' which indicates a word fragment. A subword speech
            # recognizer should be able to spell out the fragment
            trans = trans.replace(" ", "_").replace("-", "")
            chars = char_token_pattern.findall(trans)
            ngraph = ""
            # we can't just go writing n-graphs willy-nilly. If we run into
            # a control character (e.g. <NOISE>, <SPOKEN_NOISE>), that's its
            # on thing.
            while chars:
                next_ = chars.pop(0)
                if next_[0] == "<":
                    if ngraph:
                        out.write(ngraph + "_" * (ngraph_order - len(ngraph)))
                        out.write(" ")
                        ngraph = ""
                    out.write(next_)
                    out.write(" ")
                else:
                    ngraph += next_
                    if len(ngraph) == ngraph_order:
                        out.write(ngraph)
                        out.write(" ")
                        ngraph = ""
            if ngraph:
                ngraph += "_" * (ngraph_order - len(ngraph))
                out.write(ngraph)
                out.write(" ")
            out.write("(")
            out.write(utt)
            out.write(")\n")


def wsj_init_subword_config(
    wsj_subdirs, data_root, config_dir, subword_vocab_size, eval_vocab_size, algorithm
):
    local_data_dir = os.path.join(data_root, "local", "data")
    token2id_txt = os.path.join(config_dir, "token2id.txt")
    id2token_txt = os.path.join(config_dir, "id2token.txt")
    words_txt = os.path.join(config_dir, "words.txt")
    chars_txt = os.path.join(config_dir, "chars.txt")
    dir_13_32_1 = find_link_dir(wsj_subdirs, "13-32.1")
    vocab_dir = os.path.join(dir_13_32_1, "wsj1", "doc", "lng_modl", "vocab")
    subword_dir = os.path.join(config_dir, "subword")
    train_data_root = os.path.join(
        dir_13_32_1, "wsj1", "doc", "lng_modl", "lm_train", "np_data"
    )
    wordlist_txt = os.path.join(subword_dir, "wordlist.txt")
    vp_lexical_equivs_csv = os.path.join(local_data_dir, "vp_lexical_equivs.csv")
    cleaned_txt = os.path.join(subword_dir, "cleaned.txt")
    spm_prefix = os.path.join(subword_dir, "spm")

    import sentencepiece as spm

    mkdir(subword_dir)

    # we save the (closed) list of words from the eval set in case anyone
    # cares to spellcheck. The primary purpose is to determine what characters
    # are inside this list
    if eval_vocab_size == 5:
        words = set(
            x for x in cat(os.path.join(vocab_dir, "wlist5c.nvp")) if x[0] != "#"
        )
    else:
        # we're allowed to use the full 64k vocabulary list to generate the
        # vocabulary, but not for language modelling:
        # 13-32.1\wsj1\doc\lng_modl\vocab\readme.txt
        words = set(
            x.split()[1]
            for x in cat(os.path.join(vocab_dir, "wfl_64.lst"))
            if x[0] != "#" and x.split()[1][0] in ALPHA
        )
    pipe_to(sorted(words), words_txt)

    chars = set(" ".join(words))
    pipe_to(sorted(chars), chars_txt)

    write_cmu_vocab_to_path(wordlist_txt)

    isword = set(cat(wordlist_txt))
    vp = set(x.split()[0] for x in cat(vp_lexical_equivs_csv))

    # build our training data. It's pretty much the same as what we'd do for
    # word-level language modelling, but we have to save it in plain text for
    # sentencepiece. We skip any words that have characters outside of our
    # vocabulary.
    #
    # XXX(sdrobert): we're double-dipping on the training data for
    # both LM and subword selection. Will this be a problem?
    assert os.path.isdir(train_data_root)
    train_data_files = []
    for subdir in ("87", "88", "89"):
        train_data_files.extend(glob(os.path.join(train_data_root, subdir), r"*.[Zz]"))
    with open(cleaned_txt, "w") as out:
        for train_data_file in train_data_files:
            with open(train_data_file, "rb") as in_:
                compressed = in_.read()
            decompressed = unlzw(compressed)
            in_ = io.TextIOWrapper(io.BytesIO(decompressed))
            for line in in_:
                if line.startswith("<"):
                    continue
                A = line.strip().upper().split(" ")
                for n, a in enumerate(A):
                    if a in vp or any(c not in chars for c in a):
                        continue
                    if a not in isword and len(a) > 1 and a.endswith("."):
                        out.write(a[:-1])
                        if n < len(A) - 1:
                            out.write("\n")
                    else:
                        out.write(a + " ")
                out.write("\n")
            del in_, compressed, decompressed
    del isword, train_data_files

    unk_id = subword_vocab_size
    spm.SentencePieceTrainer.Train(
        "--input={} --model_prefix={} --vocab_size={} --model_type={} "
        "--user_defined_symbols=<NOISE>,<SPOKEN_NOISE> --unk_id={}".format(
            cleaned_txt,
            spm_prefix,
            subword_vocab_size + 1,  # add unk_id (we'll remove later)
            algorithm,
            unk_id,
        )
    )

    # convert the sentencepiece vocabulary into our vocabulary. There should
    # be no <unk> terms (closed vocabulary), so we map <unk> in sentencepiece
    # to <NOISE> in our vocab
    with open(token2id_txt, "w") as t2id, open(id2token_txt, "w") as id2t, open(
        spm_prefix + ".vocab", encoding="utf-8"
    ) as spm_vocab:
        for i, line in enumerate(spm_vocab):
            word, _ = line.strip().split()
            # replace the control character that sentencepiece uses with an
            # underscore
            word = word.replace("\u2581", "_")
            if word == "<unk>":
                break  # last word, but don't use it
            t2id.write("{} {}\n".format(word, i))
            id2t.write("{} {}\n".format(i, word))

    id2token = (x.strip().split() for x in cat(id2token_txt))
    id2token = dict((int(k), v) for (k, v) in id2token)
    sp = spm.SentencePieceProcessor()
    sp.load(spm_prefix + ".model")

    # similar to char stuff, but we write subwords intstead of characters
    to_copy = {
        "lex_equivs.csv",
        "spk2gender",
        "vp_lexical_equivs.csv",
        "train_si84.trn",
        "train_si84_sph.scp",
        "train_si84.utt2spk",
        "train_si84.spk2utt",
        "train_si284.trn",
        "train_si284_sph.scp",
        "train_si284.utt2spk",
        "train_si284.spk2utt",
        "test_dev93_5k.trn",
        "test_dev93_5k_sph.scp",
        "test_dev93_5k.utt2spk",
        "test_eval92_5k.trn",
        "test_eval92_5k_sph.scp",
        "test_eval92_5k.utt2spk",
        "test_eval93_5k.trn",
        "test_eval93_5k_sph.scp",
        "test_eval93_5k.utt2spk",
    }
    for x in to_copy:
        if eval_vocab_size != 5:
            x = x.replace("_5k", "")
        if x.endswith(".trn"):
            y = x[:-4]
            y += ".cleaned.txt" if x.startswith("test_") else ".fragments.txt"
            word_txt_to_subword_trn(
                os.path.join(local_data_dir, y),
                os.path.join(config_dir, x.replace("_5k", "")),
                sp,
                id2token,
            )
        else:
            copy_paths(
                os.path.join(local_data_dir, x),
                os.path.join(config_dir, x.replace("_5k", "")),
            )


def wsj_subword_lm(wsj_subdirs, config_dir, max_order):
    lmdir = os.path.join(config_dir, "lm")
    subword_dir = os.path.join(config_dir, "subword")
    cleaned_txt = os.path.join(subword_dir, "cleaned.txt")
    cleaned_txt_gz = os.path.join(lmdir, "cleaned.txt.gz")
    id2token_txt = os.path.join(config_dir, "id2token.txt")
    toprune_txt_gz = os.path.join(lmdir, "toprune.txt.gz")
    lm_arpa_gz = os.path.join(lmdir, "lm.arpa.gz")
    spm_model = os.path.join(subword_dir, "spm.model")

    import sentencepiece as spm

    mkdir(lmdir)

    id2token = (x.strip().split() for x in cat(id2token_txt))
    id2token = dict((int(k), v) for (k, v) in id2token)
    vocab = set(id2token.values())
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)

    # we've already cleaned up the LM training data in subword/cleaned.txt. Now
    # we have to convert it into subwords
    with open(cleaned_txt) as in_, gzip.open(cleaned_txt_gz, "wt") as out:
        for line in in_:
            line = line.strip()
            ids = sp.encode_as_ids(line)
            ids = " ".join(id2token[id_] for id_ in ids)
            out.write(ids)
            out.write("\n")

    # the rest proceeds in the same way as the word-level lm
    with gzip.open(cleaned_txt_gz, "rt") as file_:
        text = file_.read()
    sents = ngram_lm.text_to_sents(text, sent_end_expr="\n", word_delim_expr=" ")
    del text

    ngram_counts = ngram_lm.sents_to_ngram_counts(
        sents, max_order, sos="<s>", eos="</s>"
    )
    for v in vocab:
        ngram_counts[0].setdefault(v, 0)
    del sents

    with gzip.open(os.path.join(lmdir, "counts.txt.gz"), "wt") as file_:
        for ngram_count in ngram_counts:
            for ngram, count in ngram_count.items():
                if isinstance(ngram, str):
                    file_.write("{} {}\n".format(ngram, count))
                else:
                    file_.write("{} {}\n".format(" ".join(ngram), count))

    ngram_counts = [dict() for _ in range(max_order)]
    with gzip.open(os.path.join(lmdir, "counts.txt.gz"), "rt") as file_:
        for line in file_:
            line = line.strip().split(" ")
            count = int(line.pop())
            if len(line) == 1:
                ngram_counts[0][line[0]] = count
            else:
                ngram_counts[len(line) - 1][tuple(line)] = count

    # unless the subword vocabulary is really large, we'll have the same
    # problem inferring deltas as in the character LM case.
    wsj_train_lm(
        vocab,
        ngram_counts,
        max_order,
        toprune_txt_gz,
        lm_arpa_gz,
        [(0.5, 1.0, 1.5)] * max_order,
    )


def word_txt_to_subword_trn(txt, trn, sp, id2token):
    with open(txt) as in_, open(trn, "w") as out:
        for line in in_:
            utt, trans = line.strip().split(maxsplit=1)
            trans = trans.replace("-", "")
            trans = sp.encode_as_ids(trans)
            trans = " ".join(id2token[id_] for id_ in trans)
            out.write(trans)
            out.write(" (")
            out.write(utt)
            out.write(")\n")


def build_preamble_parser(subparsers):
    parser = subparsers.add_parser(
        "preamble", help="Do all pre-initialization setup. Needs to be done only once."
    )
    parser.add_argument(
        "wsj_roots",
        nargs="+",
        type=os.path.abspath,
        help="Location of WSJ data directories, corresponding to WSJ0 "
        "(LDC93S6A or LDC93S6B) and WSJ1 (LDC94S13A or LDC94S13B)",
    )


def build_init_word_parser(subparsers):
    parser = subparsers.add_parser(
        "init_word",
        help="Perform setup common to all word-based parsing. "
        "Needs to be done only once for a specific vocabulary size. "
        'Preceded by "preamble" command.',
    )
    parser.add_argument(
        "wsj_roots",
        nargs="+",
        type=os.path.abspath,
        help="Location of WSJ data directories, corresponding to WSJ0 "
        "(LDC93S6A or LDC93S6B) and WSJ1 (LDC94S13A or LDC94S13B)",
    )
    parser.add_argument(
        "--config-subdir",
        default=None,
        help="Name of sub directory in data/local/ under which to store "
        "setup specific to this vocabulary size. Defaults to "
        "``word<vocab_size>k``",
    )
    parser.add_argument(
        "--vocab-size",
        default=64,
        type=int,
        choices=[64, 20, 5],
        help="The size of the vocabulary, in thousands. One of: the 5k closed "
        "set (with corresponding closed vocab test and dev sets), the 20k "
        "open set (with 64k closed vocab test and dev sets), or the 64k "
        "closed set. The standard is either the 20k or 64k set.",
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        default=False,
        help="If set, will train a language model with Modified Kneser-Ney "
        "smoothing on the WSJ lm training data.",
    )
    parser.add_argument(
        "--lm-max-order",
        type=int,
        default=3,
        help="The maximum n-gram order to train the LM with when the --lm "
        "flag is set",
    )


def build_init_char_parser(subparsers):
    parser = subparsers.add_parser(
        "init_char",
        help="Perform setup common to all character-based parsing. "
        'Needs to be done only once. Preceded by "preamble" command.',
    )
    parser.add_argument(
        "wsj_roots",
        nargs="+",
        type=os.path.abspath,
        help="Location of WSJ data directories, corresponding to WSJ0 "
        "(LDC93S6A or LDC93S6B) and WSJ1 (LDC94S13A or LDC94S13B)",
    )
    parser.add_argument(
        "--config-subdir",
        default=None,
        help="Name of sub directory in data/local/ under which to store "
        "setup specific to this vocabulary size. Defaults to "
        "``char<ngraph_order>_<eval_vocab_size>k``",
    )
    parser.add_argument(
        "--ngraph-order",
        default=1,
        type=int,
        help="How many characters to consider in a token",
    )
    parser.add_argument(
        "--eval-vocab-size",
        default=64,
        type=int,
        choices=[5, 64],
        help="The size of the eval set (dev and test) vocabulary, in "
        "thousands of words. Choose between the 5k closed condition or the "
        "64k closed condition",
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        default=False,
        help="If set, will train a language model with Modified Kneser-Ney "
        "smoothing on the WSJ lm training data. Character-level.",
    )
    parser.add_argument(
        "--lm-max-order",
        type=int,
        default=5,
        help="The maximum n-gram order to train the LM with when the --lm "
        "flag is set. Character-level.",
    )


def build_init_subword_parser(subparsers):
    parser = subparsers.add_parser(
        "init_subword",
        help="Perform setup common to all subword-based parsing. "
        "Needs to be done once for a specific vocabulary size and subword "
        'algorithm pairing. Preceded by "preamble" command.',
    )
    parser.add_argument(
        "wsj_roots",
        nargs="+",
        type=os.path.abspath,
        help="Location of WSJ data directories, corresponding to WSJ0 "
        "(LDC93S6A or LDC93S6B) and WSJ1 (LDC94S13A or LDC94S13B)",
    )
    parser.add_argument(
        "--config-subdir",
        default=None,
        help="Name of sub directory in data/local/ under which to store "
        "setup specific to this algorithm and vocabulary size. Defaults to "
        "``<algorithm><subword_vocab_size>_<eval_vocab_size>k``",
    )
    parser.add_argument(
        "--subword-vocab-size",
        type=int,
        default=108,
        help="Total subword vocabulary size. Defaults to 108 (from "
        "https://arxiv.org/pdf/1811.04284.pdf)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["bpe", "unigram"],
        default="bpe",
        help="What algorithm to use to determine subwords",
    )
    parser.add_argument(
        "--eval-vocab-size",
        default=64,
        type=int,
        choices=[5, 64],
        help="The size of the eval set (dev and test) vocabulary, in "
        "thousands of words. Choose between the 5k closed condition or the "
        "64k closed condition",
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        default=False,
        help="If set, will train a language model with Modified Kneser-Ney "
        "smoothing on the WSJ lm training data. Subword-level.",
    )
    parser.add_argument(
        "--lm-max-order",
        type=int,
        default=4,
        help="The maximum n-gram order to train the LM with when the --lm "
        "flag is set. Subword-level.",
    )


def build_torch_dir_parser(subparsers):
    parser = subparsers.add_parser(
        "torch_dir",
        help="Write training, test, and extra data to subdirectories. Some "
        '"init_*" command must have been called. If more than one "init_*" '
        "call has been made, the next positional argument must be specified.",
    )
    parser.add_argument(
        "config_subdir",
        nargs="?",
        default=None,
        help="The configuration in data/local/ which to build the directories "
        'from. If "init_*" was called only once, it can be inferred from the '
        "contents fo data/local",
    )
    parser.add_argument(
        "data_subdir",
        nargs="?",
        default=".",
        help="What subdirectory in data/ to store training, test, and extra "
        "data subdirectories to. Defaults to directly in data/",
    )
    parser.add_argument(
        "--preprocess",
        default="[]",
        help="JSON list of configurations for "
        "``pydrobert.speech.pre.PreProcessor`` objects. Audio will be "
        "preprocessed in the same order as the list",
    )
    parser.add_argument(
        "--postprocess",
        default="[]",
        help="JSON List of configurations for "
        "``pydrobert.speech.post.PostProcessor`` objects. Features will be "
        "postprocessed in the same order as the list",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A random seed used for determinism. This affects operations "
        "like dithering. If unset, a seed will be generated at the moment",
    )
    parser.add_argument(
        "--si84",
        action="store_true",
        default=False,
        help="If set, will only train on the SI-84 (WSJ0) data rather than "
        "on both WSJ0 and WSJ1",
    )

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--eval93",
        action="store_true",
        default=False,
        help="If this flag is set, write the '93 eval set instead of the "
        "'92 eval set.",
    )
    test_group.add_argument(
        "--both-evals",
        action="store_true",
        default=False,
        help="If this flag is set, write *both* the '92 and '93 eval sets",
    )

    fbank_41_config = os.path.join(
        os.path.dirname(__file__), "conf", "feats", "fbank_41.json"
    )
    feat_group = parser.add_mutually_exclusive_group()
    feat_group.add_argument(
        "--computer-json",
        default=fbank_41_config,
        help="Path to JSON configuration of a feature computer for "
        "pydrobert-speech. Defaults to a 40-dimensional Mel-scaled triangular "
        "overlapping filter bank + 1 energy coefficient every 10ms.",
    )
    feat_group.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="If specified, tensors of raw audio of shape (S, 1) will be "
        "written instead of filter bank coefficients.",
    )
    feat_group.add_argument(
        "--feats-from",
        default=None,
        help="If specified, rather than computing features, will copy the "
        "feature folders from this subdirectory of data/",
    )


def build_filter_parser(subparsers):
    parser = subparsers.add_parser(
        "filter",
        help="Filter an input word-level hypothesis trn of special characters "
        "and apply allowed utterance- and word-level equivalences",
    )
    parser.add_argument(
        "raw_trn", type=argparse.FileType("r"), help="The input (unfiltered) trn file"
    )
    parser.add_argument(
        "filt_trn", type=argparse.FileType("w"), help="The output (filtered) trn file"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "data_root",
        type=os.path.abspath,
        help="The root directory under which to store data. Typically " "``data/``",
    )
    subparsers = parser.add_subparsers(title="commands", required=True, dest="command")
    build_preamble_parser(subparsers)
    build_init_word_parser(subparsers)
    build_init_char_parser(subparsers)
    build_init_subword_parser(subparsers)
    build_torch_dir_parser(subparsers)
    build_filter_parser(subparsers)
    return parser


def preamble(options):

    wsj_subdirs = []
    for wsj_root in options.wsj_roots:
        wsj_subdirs.extend(glob(wsj_root, r"??-?.?"))
        wsj_subdirs.extend(glob(wsj_root, r"??-??.?"))

    wsj_data_prep(wsj_subdirs, options.data_root)


def init_word(options):

    wsj_subdirs = []
    for wsj_root in options.wsj_roots:
        wsj_subdirs.extend(glob(wsj_root, r"??-?.?"))
        wsj_subdirs.extend(glob(wsj_root, r"??-??.?"))

    if options.config_subdir is None:
        config_dir = os.path.join(
            options.data_root, "local", "word{}k".format(options.vocab_size)
        )
    else:
        config_dir = os.path.join(options.data_root, "local", options.config_subdir)

    wsj_init_word_config(wsj_subdirs, options.data_root, config_dir, options.vocab_size)

    if options.lm:
        wsj_word_lm(wsj_subdirs, config_dir, options.lm_max_order)


def init_char(options):

    wsj_subdirs = []
    for wsj_root in options.wsj_roots:
        wsj_subdirs.extend(glob(wsj_root, r"??-?.?"))
        wsj_subdirs.extend(glob(wsj_root, r"??-??.?"))

    if options.config_subdir is None:
        config_dir = os.path.join(
            options.data_root,
            "local",
            "char{}_{}k".format(options.ngraph_order, options.eval_vocab_size),
        )
    else:
        config_dir = os.path.join(options.data_root, "local", options.config_subdir)

    wsj_init_char_config(
        wsj_subdirs,
        options.data_root,
        config_dir,
        options.eval_vocab_size,
        options.ngraph_order,
    )

    if options.lm:
        wsj_char_lm(wsj_subdirs, config_dir, options.lm_max_order, options.ngraph_order)


def init_subword(options):

    wsj_subdirs = []
    for wsj_root in options.wsj_roots:
        wsj_subdirs.extend(glob(wsj_root, r"??-?.?"))
        wsj_subdirs.extend(glob(wsj_root, r"??-??.?"))

    if options.config_subdir is None:
        config_dir = os.path.join(
            options.data_root,
            "local",
            "{}{}_{}k".format(
                options.algorithm, options.subword_vocab_size, options.eval_vocab_size
            ),
        )
    else:
        config_dir = os.path.join(options.data_root, "local", options.config_subdir)

    wsj_init_subword_config(
        wsj_subdirs,
        options.data_root,
        config_dir,
        options.subword_vocab_size,
        options.eval_vocab_size,
        options.algorithm,
    )

    if options.lm:
        wsj_subword_lm(wsj_subdirs, config_dir, options.lm_max_order)


def torch_dir(options):

    local_dir = os.path.join(options.data_root, "local")
    if options.config_subdir is None:
        dirs = os.listdir(local_dir)
        try:
            dirs.remove("data")
        except ValueError:
            pass
        if len(dirs) == 1:
            config_dir = os.path.join(local_dir, dirs[0])
        else:
            raise ValueError(
                'More than one directory ({}) besides "data" exists in "{}". '
                "Cannot infer configuration. Please specify as a positional "
                "argument".format(", ".join(dirs), local_dir)
            )
    else:
        config_dir = os.path.join(local_dir, options.config_subdir)
        if not os.path.isdir(config_dir):
            raise ValueError('"{}" is not a directory'.format(config_dir))

    dir_ = os.path.join(options.data_root, options.data_subdir)
    ext = os.path.join(dir_, "ext")
    mkdir(ext)

    for x in {"spk2gender", "id2token.txt", "token2id.txt"}:
        copy_paths(os.path.join(config_dir, x), ext)

    lm_arpa_gz = os.path.join(config_dir, "lm", "lm.arpa.gz")
    if os.path.exists(lm_arpa_gz):
        copy_paths(lm_arpa_gz, ext)

    train = "si84" if options.si84 else "si284"
    dev = "dev93"
    if options.both_evals:
        tests = ("eval92", "eval93")
    elif options.eval93:
        tests = ("eval93",)
    else:
        tests = ("eval92",)

    feat_optional_args = [
        "--channel",
        "-1",
        "--num-workers",
        str(get_num_avail_cores()),
        "--force-as",
        "sph",
        "--preprocess",
        options.preprocess,
        "--postprocess",
        options.postprocess,
    ]
    if options.seed is not None:
        feat_optional_args.extend(["--seed", str(options.seed)])

    token2id_txt = os.path.join(config_dir, "token2id.txt")
    token2id = (line.strip().split() for line in cat(token2id_txt))
    token2id = dict((k, int(v)) for (k, v) in token2id)
    if "<UNK>" in token2id:
        unk = "<UNK>"
    elif "<unk>" in token2id:
        unk = "<unk>"
    else:
        unk = None

    for is_test, partition in enumerate((train, dev) + tests):
        prefix = "test_" if is_test else "train_"
        trn_src = os.path.join(config_dir, prefix + partition + ".trn")
        trn_dest = os.path.join(ext, partition + ".ref.trn")
        map_path = os.path.join(config_dir, prefix + partition + "_sph.scp")
        part_dir = os.path.join(dir_, partition)
        feat_dir = os.path.join(part_dir, "feat")
        ref_dir = os.path.join(part_dir, "ref")

        copy_paths(trn_src, trn_dest)

        mkdir(feat_dir)

        if options.feats_from is None:
            args = [map_path, feat_dir] + feat_optional_args
            if not options.raw:
                args.insert(1, options.computer_json)
            speech_cmd.signals_to_torch_feat_dir(args)
        else:
            feat_src = os.path.join(
                options.data_root, options.feats_from, partition, "feat"
            )
            if not os.path.isdir(feat_src):
                raise ValueError(
                    'Specified --feats-from, but "{}" is not a directory'
                    "".format(feat_src)
                )
            for filename in os.listdir(feat_src):
                src = os.path.join(feat_src, filename)
                dest = os.path.join(feat_dir, filename)
                copy_paths(src, dest)

        if is_test:
            # before writing the ref/ dir, we need to check if there's any new
            # vocabulary in the test set. We write special token2id and
            # id2token files for the test set. This ensures the stored
            # reference ids won't be considered <UNK>
            cur_token2id = dict(token2id)
            for line in cat(trn_src):
                trans = line.split()
                trans.pop()  # remove utt id
                for word in trans:
                    cur_token2id.setdefault(word, len(cur_token2id))
            assert len(cur_token2id) == len(set(cur_token2id.values()))
            cur_token2id_txt = os.path.join(ext, "token2id." + partition + ".txt")
            cur_id2token_txt = os.path.join(ext, "id2token." + partition + ".txt")
            with open(cur_id2token_txt, "w") as id2t, open(
                cur_token2id_txt, "w"
            ) as t2id:
                for t, id_ in sorted(cur_token2id.items(), key=lambda x: x[1]):
                    id2t.write("{} {}\n".format(id_, t))
                    t2id.write("{} {}\n".format(t, id_))
        else:
            cur_token2id_txt = token2id_txt

        args = [
            trn_src,
            cur_token2id_txt,
            ref_dir,
            "--num-workers",
            str(get_num_avail_cores()),
        ]
        if not is_test and unk is not None:  # never write <UNK> for test
            args += ["--unk-symbol", unk]
        torch_cmd.trn_to_torch_token_data_dir(args)

        # verify correctness (while storing info as a bonus)
        args = [
            part_dir,
            os.path.join(ext, f"{partition}.info.ark"),
            "--strict",
        ]
        assert not torch_cmd.get_torch_spect_data_dir_info(args)


def filter_(options):

    dir_ = os.path.join(options.data_root, "local", "data")
    lex_equivs_csv = os.path.join(dir_, "lex_equivs.csv")

    # The utterance mappings in 93uttmap.rls are garbage. Using NIST's own
    # program, tranfilt, we're just as likely to duplicate words as fix the
    # transcription. e.g.
    # ./nov93flt.sh < ~/WSJ1/13-32.1/score/lib/wsj/nov93wsj.ref > a.trn
    # diff ~/WSJ1/13-32.1/score/lib/wsj/nov93wsj.ref a.trn
    # 1140c1140
    # < SEOUL YUK STATION HE REPEATS PRACTICING HIS NEW ENGLISH WORD (4OBC020E)
    # ---
    # > SEOUL SEOUL YUK STATION HE REPEATS PRACTICING HIS NEW ENGLISH ...
    # 4447c4447
    # < THE STORM HAS CREATED A NATIONAL DISASTER AND COMMUNITIES WITH ...
    # ---
    # > THE STORM HAS CREATED A NATIONAL DISASTER AND COMMUNITIES WITH WITH ...

    lex_equivs = dict(x.rsplit(",", maxsplit=1) for x in cat(lex_equivs_csv))

    # determine utterances from '93 sets (can only apply lexical equivalences
    # to these
    utts_from_93 = set()
    for x in ("train_si284", "test_dev93", "test_eval93"):
        Y = ("",) if x.startswith("train_") else ("", "_5k")
        for y in Y:
            path = os.path.join(dir_, x + y + ".raw.txt")
            utts_from_93.update(x.split()[0] for x in cat(path))
    path = os.path.join(dir_, "train_si84.raw.txt")
    utts_from_93.difference_update(x.split()[0] for x in cat(path))

    for line in options.raw_trn:
        toks = line.strip().split(" ")
        utt = toks.pop()
        if utt[0] != "(" or utt[-1] != ")":
            raise ValueError("{} is not a trn".format(options.raw_trn.name))
        utt = utt[1:-1]
        toks = (tok for tok in toks if tok and tok[0] != "<")
        if utt in utts_from_93:
            toks = (lex_equivs.get(tok, tok) for tok in toks)
        options.filt_trn.write(" ".join(toks))
        options.filt_trn.write(" (")
        options.filt_trn.write(utt)
        options.filt_trn.write(")\n")


def main(args=None):
    """Prepare WSJ data for end-to-end pytorch training"""

    parser = build_parser()
    options = parser.parse_args(args)

    if options.command == "preamble":
        preamble(options)
    elif options.command == "init_word":
        init_word(options)
    elif options.command == "init_char":
        init_char(options)
    elif options.command == "init_subword":
        init_subword(options)
    elif options.command == "torch_dir":
        torch_dir(options)
    elif options.command == "filter":
        filter_(options)


if __name__ == "__main__":
    sys.exit(main())
