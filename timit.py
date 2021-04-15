#! /usr/bin/env python

# Copyright 2021 Sean Robertson
#
# Adapted from kaldi/egs/timit/s5/local/timit_data_prep.sh (and associated resource
# files conf/{{dev,test}_spk.lst,phones.60-48-39.map}:
#
# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
#           2014   Brno University of Technology (Author: Karel Vesely)
#
# phones.map has been slightly adjusted from phones.60-48-39.map
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

"""Command-line interface to prepare the TIMIT speech corpus for end-to-end ASR"""

import argparse
import glob
import gzip
import locale
import os
import sys
import torch
import json

from shutil import copy as copy_paths
from tempfile import SpooledTemporaryFile

import ngram_lm

import pydrobert.torch.data
import pydrobert.speech.util as speech_util
import pydrobert.speech.command_line as speech_cmd
import pydrobert.torch.command_line as torch_cmd

from pydrobert.speech.compute import FrameComputer
from pydrobert.speech.util import alias_factory_subclass_from_arg

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2020 Sean Robertson"


locale.setlocale(locale.LC_ALL, "C")

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources", "timit")
if not os.path.isdir(RESOURCE_DIR):
    raise ValueError('"{}" is not a directory'.format(RESOURCE_DIR))

ALPHA = set(chr(x) for x in range(ord("a"), ord("z") + 1))


def get_phone_map():
    phone_map = dict()
    p39s = set()
    with open(os.path.join(RESOURCE_DIR, "phones.map")) as file_:
        for line in file_:
            line = line.strip().split()
            if not line:
                continue
            p61, p48, p39 = line[0], None, None
            if len(line) > 1:
                p48 = line[1]
                if len(line) == 3:
                    p39 = line[2]
                    p39s.add(p39)
            phone_map[p61] = (p61, p48, p39)
            # p61->p48 and p48->p39 are surjective mappings. We extend the composition
            # p61->p48 to (p61 U p48)->p48 with the inclusion map. We only add the
            # mapping if it doesn't already exist and allow it to be clobbered if some
            # p61 == p48.
            if p48 is not None and p48 not in phone_map:
                phone_map[p48] = (None, p48, p39)
    # any p39s that don't have entries already we extend with the inclusion map.
    for p39 in p39s:
        phone_map.setdefault(p39, (None, None, p39))
    return phone_map


def timit_data_prep(timit, data_root):
    # this function is only loosely based on timit_data_prep.sh. We do both less
    # and more in this function
    #
    # - utterance ids will match Kaldi format (<spk><prompt>)
    # - we do not partition the data into separate data files according to train/test
    #   partitions. We handle that in init_phn. Instead, we create a number of maps for
    #   where speakers *will* belong according to whatever partitioning strategy

    if not os.path.isdir(timit):
        raise ValueError('"{}" is not a directory'.format(timit))

    test_dir = train_dir = None
    for dir_ in glob.iglob(glob.escape(timit) + "/**/faks0", recursive=True):
        test_dir = os.path.dirname(os.path.dirname(dir_))
        break
    if test_dir is None:
        raise ValueError('Could not find "FAKS0" in {}'.format(timit))
    for dir_ in glob.iglob(glob.escape(timit) + "/**/fcjf0", recursive=True):
        train_dir = os.path.dirname(os.path.dirname(dir_))
        break
    if train_dir is None:
        raise ValueError('Could not find "FCJF0" in {}'.format(timit))

    utt2type = dict()
    spk2part = dict()
    utt2sph = dict()
    spk2dialect = dict()
    spk2gender = dict()
    utt2wc = dict()
    utt2prompt = dict()
    utt2spk = dict()
    ctm = []
    stm = []
    for part, part_path in (("train", train_dir), ("test", test_dir)):
        for path in glob.iglob(glob.escape(part_path) + "/**/*.WAV", recursive=True):
            dur = len(speech_util.read_signal(path, force_as="sph")) / 16000
            dname, bname = os.path.split(path.lower())
            prompt = bname.split(".")[0]
            dname, spk = os.path.split(dname)
            dr = os.path.basename(dname)
            gender = spk[0].upper()
            utt = spk + prompt
            type_ = prompt[:2].upper()
            utt2type[utt] = type_
            spk2part[spk] = part
            utt2sph[utt] = path
            spk2dialect[spk] = dr.upper()
            spk2gender[spk] = gender
            utt2wc[utt] = utt + " A"
            utt2spk[utt] = spk
            no_ext = path.rsplit(".", maxsplit=1)[0]
            if os.path.isfile(no_ext + ".phn"):
                phn_path = no_ext + ".phn"
            elif os.path.isfile(no_ext + ".PHN"):
                phn_path = no_ext + ".PHN"
            else:
                raise ValueError(
                    "Could not find .PHN file associated with {}".format(path)
                )
            with open(phn_path) as file_:
                stm_start = stm_end = None
                phns = []
                for line in file_:
                    start, end, phn = line.strip().split()
                    start = max(float(start) / 16000, 0.0)
                    end = min(float(end) / 16000, dur)
                    if stm_start is None:
                        stm_start = start
                    stm_end = end
                    ctm.append((utt, "A", start, end - start, phn))
                    phns.append(phn)
                stm.append((utt, "A", spk, stm_start, stm_end, " ".join(phns)))
            if os.path.isfile(no_ext + ".txt"):
                txt_path = no_ext + ".txt"
            elif os.path.isfile(no_ext + ".TXT"):
                txt_path = no_ext + ".TXT"
            else:
                raise ValueError(
                    "Could not find .TXT file associated with {}".format(path)
                )
            with open(txt_path) as file_:
                line = file_.readline()
                _, _, prompt = line.strip().split(maxsplit=2)
                prompt = (
                    prompt.lower()
                    .replace("?", " ")
                    .replace(",", " ")
                    .replace("-", " ")
                    .replace(".", " ")
                    .replace(":", " ")
                    .replace("  ", " ")
                    .replace("!", " ")
                    .replace(";", " ")
                    .replace('"', " ")
                    .strip()
                )
                assert not (set(prompt) - (ALPHA | {" ", "'"})), set(prompt) - ALPHA
                utt2prompt[utt] = prompt

    if len(spk2part) != 630:
        raise ValueError("Expected 630 speakers, got {}".format(len(spk2part)))

    if len(utt2type) != 6300:
        raise ValueError("Expected 6300 utterances, got {}".format(len(utt2type)))

    stm.sort(key=lambda x: (x[0], x[3]))  # x[1] is always "A"
    ctm.sort()

    dir_ = os.path.join(data_root, "local", "data")
    os.makedirs(dir_, exist_ok=True)

    with open(os.path.join(dir_, "all.trn"), "w") as file_:
        for line in stm:
            file_.write("{5} ({0})\n".format(*line))

    with open(os.path.join(dir_, "all.stm"), "w") as file_:
        for line in stm:
            file_.write("{} {} {} {:.7f} {:.7f} {}\n".format(*line))

    with open(os.path.join(dir_, "all.ctm"), "w") as file_:
        for line in ctm:
            file_.write("{} {} {:.7f} {:.7f} {}\n".format(*line))

    for name, dict_ in (
        ("spk2dialect", spk2dialect),
        ("spk2gender", spk2gender),
        ("spk2part", spk2part),
        ("utt2prompt", utt2prompt),
        ("utt2sph", utt2sph),
        ("utt2spk", utt2spk),
        ("utt2type", utt2type),
        ("utt2wc", utt2wc),
    ):
        with open(os.path.join(dir_, name), "w") as file_:
            for key in sorted(dict_):
                file_.write("{} {}\n".format(key, dict_[key]))


def preamble(options):
    timit_data_prep(options.timit_root, options.data_root)


def write_mapped_trn(src, dst, map_, utts=None):

    if isinstance(src, str):
        if isinstance(dst, str):
            with open(src) as in_trn, open(dst, "w") as out_trn:
                return write_mapped_trn(in_trn, out_trn, map_, utts)
        else:
            with open(src) as in_trn:
                return write_mapped_trn(in_trn, dst, map_, utts)
    elif isinstance(dst, str):
        with open(dst) as out_trn:
            return write_mapped_trn(src, out_trn, map_, utts)

    for line_no, line in enumerate(src):
        phns = line.strip().split()
        if not phns:
            continue
        utt = phns.pop()
        if utt[0] != "(" or utt[-1] != ")":
            raise RuntimeError(f"{src.name} line {line_no + 1}: not a trn line")
        if utts is not None and utt[1:-1] not in utts:
            continue
        idx = 0
        while idx < len(phns):
            to = map_[phns[idx]]
            if to is None:
                phns.pop(idx)
            else:
                phns[idx] = to
                idx += 1
        phns.append(utt)
        dst.write(" ".join(phns))
        dst.write("\n")


def write_mapped_stm(src, dst, map_, wcinfo=None):

    if isinstance(src, str):
        if isinstance(dst, str):
            with open(src) as in_stm, open(dst, "w") as out_stm:
                return write_mapped_stm(in_stm, out_stm, map_, wcinfo)
        else:
            with open(src) as in_stm:
                return write_mapped_stm(in_stm, dst, map_, wcinfo)
    elif isinstance(dst, str):
        with open(dst) as out_stm:
            return write_mapped_stm(src, out_stm, map_, wcinfo)

    if wcinfo is not None:
        dst.write(';; LABEL "F" "Female" "Female speaker"\n')
        dst.write(';; LABEL "M" "Male" "Male speaker"\n')
        dst.write(';; LABEL "DR1" "Dialect 1" "Speaker from New England"\n')
        dst.write(';; LABEL "DR2" "Dialect 2" "Speaker from Northern U.S."\n')
        dst.write(';; LABEL "DR3" "Dialect 3" "Speaker from North Midland U.S."\n')
        dst.write(';; LABEL "DR4" "Dialect 4" "Speaker from South Midland U.S."\n')
        dst.write(';; LABEL "DR5" "Dialect 5" "Speaker from Southern U.S."\n')
        dst.write(';; LABEL "DR6" "Dialect 6" "Speaker from New York City"\n')
        dst.write(';; LABEL "DR7" "Dialect 7" "Speaker from Western U.S."\n')
        dst.write(';; LABEL "DR8" "Dialect 8" "Speaker was army brat in U.S."\n')
        dst.write(';; LABEL "SA" "Dialect Prompt" "Prompt was a shibboleth"\n')
        dst.write(
            ';; LABEL "SI" "Diverse Prompt" "Prompt extracted from existing text"\n'
        )
        dst.write(
            ';; LABEL "SX" "Compact Prompt" "Prompt designed to elicit biphones"\n'
        )

    for line_no, line in enumerate(src):
        line = line.strip().split(";;")[0].split()
        if not line:
            continue
        if len(line) < 5:
            raise RuntimeError(f"{src.name} line {line_no + 1}: invalid stm line")
        prefix, phns = line[:5], line[5:]
        if phns and phns[0][0] == "<":
            prefix.append(phns.pop(0))  # it's actually wcinfo
        if wcinfo is not None:
            wc = " ".join(prefix[:2])
            if wc not in wcinfo:
                continue
            wcinfo_ = "<" + ",".join(wcinfo[wc]) + ">"
            if len(prefix) == 5:
                prefix.append(wcinfo_)
            elif prefix[-1] != wcinfo_:
                raise RuntimeError(
                    f"{src.name} line {line_no + 1}: waveform/channel '{wc}' has info "
                    f"string {prefix[-1]} but expected {wcinfo_}. Did you modify it?"
                )
        idx = 0
        while idx < len(phns):
            phn = phns[idx]
            if phn not in map_:
                raise RuntimeError(
                    f"{src.name} line {line_no + 1}: unknown phone {phn}"
                )
            to = map_[phn]
            if to is None:
                phns.pop(idx)
            else:
                phns[idx] = to
                idx += 1
        dst.write(" ".join(prefix + phns))
        dst.write("\n")


def write_mapped_ctm(src, dst, map_, wclist=None):

    if isinstance(src, str):
        if isinstance(dst, str):
            with open(src) as in_ctm, open(dst, "w") as out_ctm:
                return write_mapped_ctm(in_ctm, out_ctm, map_, wclist)
        else:
            with open(src) as in_ctm:
                return write_mapped_ctm(in_ctm, dst, map_, wclist)
    elif isinstance(dst, str):
        with open(dst) as out_ctm:
            return write_mapped_ctm(src, out_ctm, map_, wclist)

    last_wc = last_start = last_dur = last_phn = None
    for line_no, line in enumerate(src):
        line = line.strip().rsplit(maxsplit=3)
        if len(line) != 4:
            raise RuntimeError(f"{src.name} line {line_no + 1}: not a ctm line")
        wc, start, dur, phn = line
        if wclist is not None and wc not in wclist:
            continue
        try:
            start, dur = float(start), float(dur)
        except ValueError as e:
            raise RuntimeError(f"{src.name} line {line_no + 1}") from e
        if wc != last_wc:
            if last_phn is not None:
                dst.write(
                    "{} {:.7f} {:.7f} {}\n".format(
                        last_wc, last_start, last_dur, last_phn
                    )
                )
            last_wc = wc
            last_start = last_dur = 0.0
            last_phn = None
        if phn not in map_:
            raise RuntimeError(f"{src.name} line {line_no + 1}: unknown phone {phn}")
        to = map_[phn]
        if to is not None:
            if last_phn is not None:
                dst.write(
                    "{} {:.7f} {:.7f} {}\n".format(
                        last_wc, last_start, last_dur, last_phn
                    )
                )
            last_wc, last_start, last_dur, last_phn = wc, start, dur, to
        else:
            # collapse this phone into the previous phone
            last_dur += dur
    if last_phn is not None:
        dst.write(
            "{} {:.7f} {:.7f} {}\n".format(last_wc, last_start, last_dur, last_phn)
        )


def init_phn(options):

    local_dir = os.path.join(options.data_root, "local")
    data_dir = os.path.join(local_dir, "data")
    if not os.path.isdir(data_dir):
        raise ValueError("{} does not exist; call preamble first!".format(data_dir))

    if options.config_subdir is None:
        config_dir = os.path.join(local_dir, "phn{}".format(options.vocab_size))
    else:
        config_dir = os.path.join(local_dir, options.config_subdir)

    phone_map = get_phone_map()
    for key in set(phone_map):
        if options.vocab_size == 61:
            phone_map[key] = phone_map[key][0]
        elif options.vocab_size == 48:
            phone_map[key] = phone_map[key][1]
        else:
            phone_map[key] = phone_map[key][2]
    phone_set = sorted(set(val for val in phone_map.values() if val is not None))
    phone_set += ["</s>", "<s>"]

    os.makedirs(config_dir, exist_ok=True)

    for fn in (
        "spk2dialect",
        "spk2gender",
        "spk2part",
        "utt2prompt",
        "utt2sph",
        "utt2spk",
        "utt2type",
        "utt2wc",
    ):
        copy_paths(os.path.join(data_dir, fn), os.path.join(config_dir, fn))

    with open(os.path.join(config_dir, "token2id.txt"), "w") as token2id, open(
        os.path.join(config_dir, "id2token.txt"), "w"
    ) as id2token:
        for id_, token in enumerate(phone_set):
            token2id.write("{} {}\n".format(token, id_))
            id2token.write("{} {}\n".format(id_, token))

    write_mapped_trn(
        os.path.join(data_dir, "all.trn"),
        os.path.join(config_dir, "all.trn"),
        phone_map,
    )

    write_mapped_stm(
        os.path.join(data_dir, "all.stm"),
        os.path.join(config_dir, "all.stm"),
        phone_map,
    )

    write_mapped_ctm(
        os.path.join(data_dir, "all.ctm"),
        os.path.join(config_dir, "all.ctm"),
        phone_map,
    )

    if options.lm:
        train_phn_lm(config_dir, options.lm_max_order)


def train_phn_lm(config_dir, max_order):

    spk2utts = dict()
    with open(os.path.join(config_dir, "utt2spk")) as utt2spk:
        for line in utt2spk:
            utt, spk = line.strip().split()
            spk2utts.setdefault(spk, []).append(utt)

    train_utt = set()
    with open(os.path.join(config_dir, "spk2part")) as utt2part:
        for line in utt2part:
            spk, part = line.strip().split()
            if part != "train":
                continue
            for utt in spk2utts[spk]:
                if not utt.endswith("sa1") and not utt.endswith("sa2"):
                    train_utt.add(utt)

    text = ""
    with open(os.path.join(config_dir, "all.trn")) as trn:
        for line in trn:
            phns, utt = line.strip().rsplit(maxsplit=1)
            assert utt[0] == "(" and utt[-1] == ")"
            if utt[1:-1] not in train_utt:
                continue
            text += phns + "\n"

    sents = ngram_lm.text_to_sents(
        text, sent_end_expr="\n", word_delim_expr=" ", to_case=None
    )
    del text

    # the number of times a given prompt occurred was controlled by the corpus
    # creators, so there's no reason a sentence that occurred three times is more
    # valuable than one that occurred only once. Drop duplicate entries
    sents = set(sents)

    ngram_counts = ngram_lm.sents_to_ngram_counts(
        sents, max_order, sos="<s>", eos="</s>"
    )
    # 4-fold cross-validation said K&M with delta=0.5 lead to the best perplexity for
    # both 2-grams and 3-grams. Too small for Katz. Perplexity for add k was higher
    # for a variety of k.
    # I suspect the primary benefit of the LM is to forbid invalid phone sequences.
    prob_list = ngram_lm.ngram_counts_to_prob_list_kneser_ney(ngram_counts, delta=0.5)
    lm = ngram_lm.BackoffNGramLM(prob_list, sos="<s>", eos="</s>", unk="<s>")
    lm.prune_by_name({"<s>"})
    prob_list = lm.to_prob_list()
    with gzip.open(os.path.join(config_dir, "lm.arpa.gz"), "wt") as file_:
        ngram_lm.write_arpa(prob_list, file_)


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
    os.makedirs(ext, exist_ok=True)

    phone_map = get_phone_map()
    phone_map = dict((k, v[2]) for (k, v) in phone_map.items())

    for fn in (
        "id2token.txt",
        "spk2dialect",
        "spk2gender",
        "token2id.txt",
        "utt2prompt",
        "utt2spk",
        "utt2type",
        "utt2wc",
    ):
        copy_paths(os.path.join(config_dir, fn), os.path.join(ext, fn))

    lm_arpa_gz = os.path.join(config_dir, "lm", "lm.arpa.gz")
    if os.path.exists(lm_arpa_gz):
        copy_paths(lm_arpa_gz, ext)

    spk2info = dict()
    for idx, fn in enumerate(("spk2part", "spk2gender", "spk2dialect")):
        with open(os.path.join(config_dir, fn)) as file_:
            for line in file_:
                spk, tidbit = line.strip().split(" ", maxsplit=1)
                spk2info.setdefault(spk, [None] * 3)[idx] = tidbit

    utt2info = dict()
    for idx, fn in enumerate(("utt2type", "utt2wc", "utt2spk")):
        with open(os.path.join(config_dir, fn)) as file_:
            for line in file_:
                utt, tidbit = line.strip().split(" ", maxsplit=1)
                utt2info.setdefault(utt, [None] * 3)[idx] = tidbit

    train_types = test_types = ("SX", "SI")
    if options.include_sa:
        train_types += ("SA",)
    train_uttids = set()
    all_test_uttids = set()
    for utt, (type_, wc, spk) in utt2info.items():
        part = spk2info[spk][0]
        if part == "train" and type_ in train_types:
            train_uttids.add(utt)
        elif part == "test" and type_ in test_types:
            all_test_uttids.add(utt)
    if len(all_test_uttids) != 168 * 8:
        raise ValueError(
            "Expected {} test utterance ids in complete test set, got {}".format(
                168 * 8, len(all_test_uttids)
            )
        )

    if options.complete_test:
        test_uttids = all_test_uttids
        dev_uttids = set()
    else:
        with open(os.path.join(RESOURCE_DIR, "core_spk.lst")) as lst:
            test_spk = set()
            for line in lst:
                line = line.strip()
                if line:
                    test_spk.add(line)
        test_uttids = set()
        left_uttids = set()
        for utt in all_test_uttids:
            if utt2info[utt][2] in test_spk:
                test_uttids.add(utt)
            else:
                left_uttids.add(utt)
        if len(test_uttids) != 24 * 8:
            raise ValueError(
                "Expected {} core test utterance ids in complete test set, "
                "got {}".format(24 * 8, len(test_uttids))
            )
        if options.large_dev:
            dev_uttids = left_uttids
        else:
            with open(os.path.join(RESOURCE_DIR, "dev50_spk.lst")) as lst:
                dev_spk = set()
                for line in lst:
                    line = line.strip()
                    if line:
                        dev_spk.add(line)
            dev_uttids = set()
            for utt in left_uttids:
                if utt2info[utt][2] in dev_spk:
                    dev_uttids.add(utt)
            if len(dev_uttids) != 50 * 8:
                raise ValueError(
                    "Expected {} dev utterance ids in complete test set, got {}".format(
                        50 * 8, len(dev_uttids)
                    )
                )

    parts = (("train", train_uttids), ("test", test_uttids))
    if dev_uttids:
        parts += (("dev", dev_uttids),)

    feat_optional_args = [
        "--channel",
        "-1",
        "--num-workers",
        str(torch.multiprocessing.cpu_count()),
        "--force-as",
        "sph",
        "--preprocess",
        options.preprocess,
        "--postprocess",
        options.postprocess,
    ]
    if options.seed is not None:
        feat_optional_args.extend(["--seed", str(options.seed)])

    if options.raw:
        # 16 samps per ms = 1 / 16 ms per samp
        frame_shift_ms = 1 / 16
    else:
        # more complicated. Have to build our feature computer
        with open(options.computer_json) as file_:
            json_ = json.load(file_)
        computer: FrameComputer = alias_factory_subclass_from_arg(FrameComputer, json_)
        frame_shift_ms = computer.frame_shift_ms
        del computer, json_

    ctm_src = os.path.join(config_dir, "all.ctm")
    stm_src = os.path.join(config_dir, "all.stm")
    trn_src = os.path.join(config_dir, "all.trn")
    utt2sph_src = os.path.join(config_dir, "utt2sph")
    token2id_txt = os.path.join(config_dir, "token2id.txt")
    id_map = dict((k, k) for k in phone_map)
    for part, uttids in parts:
        unmapped_ctm_dst = os.path.join(ext, part + ".ref_unmapped.ctm")
        ctm_dst = os.path.join(ext, part + ".ref.ctm")
        unmapped_stm_dst = os.path.join(ext, part + ".ref_unmapped.stm")
        stm_dst = os.path.join(ext, part + ".ref.stm")
        unmapped_trn_dst = os.path.join(ext, part + ".ref_unmapped.trn")
        trn_dst = os.path.join(ext, part + ".ref.trn")
        map_path = os.path.join(ext, part + ".scp")
        part_dir = os.path.join(dir_, part)
        feat_dir = os.path.join(part_dir, "feat")
        ref_dir = os.path.join(part_dir, "ref")
        wcinfo = dict(
            (utt2info[utt][1], spk2info[utt2info[utt][2]][1:] + utt2info[utt][:1])
            for utt in uttids
        )

        with open(utt2sph_src) as in_, open(map_path, "w") as out_:
            for line in in_:
                utt, sph = line.strip().split()
                if utt in uttids:
                    out_.write("{} {}\n".format(utt, sph))

        write_mapped_ctm(ctm_src, unmapped_ctm_dst, id_map, wcinfo)
        write_mapped_ctm(ctm_src, ctm_dst, phone_map, wcinfo)

        write_mapped_stm(stm_src, unmapped_stm_dst, id_map, wcinfo)
        write_mapped_stm(stm_src, stm_dst, phone_map, wcinfo)

        write_mapped_trn(trn_src, unmapped_trn_dst, id_map, uttids)
        write_mapped_trn(trn_src, trn_dst, phone_map, uttids)

        os.makedirs(feat_dir, exist_ok=True)

        if options.feats_from is None:
            args = [map_path, feat_dir] + feat_optional_args
            if not options.raw:
                args.insert(1, options.computer_json)
            assert not speech_cmd.signals_to_torch_feat_dir(args)
        else:
            feat_src = os.path.join(options.data_root, options.feats_from, part, "feat")
            if not os.path.isdir(feat_src):
                raise ValueError(
                    'Specified --feats-from, but "{}" is not a directory'
                    "".format(feat_src)
                )
            for filename in os.listdir(feat_src):
                src = os.path.join(feat_src, filename)
                dest = os.path.join(feat_dir, filename)
                copy_paths(src, dest)

        args = [
            unmapped_ctm_dst,
            token2id_txt,
            ref_dir,
            "--frame-shift-ms",
            f"{frame_shift_ms:e}",
        ]
        assert not torch_cmd.ctm_to_torch_token_data_dir(args)

        # verify correctness (while storing info as a bonus)
        args = [
            part_dir,
            os.path.join(ext, f"{part}.info.ark"),
            "--strict",
        ]
        assert not torch_cmd.get_torch_spect_data_dir_info(args)


def filter(options):
    phone_map = get_phone_map()
    # always drop down to the p39 set
    phone_map = dict((k, v[2]) for (k, v) in phone_map.items())

    if options.both_ctm:
        write_mapped_ctm(options.raw_trn, options.filt_trn, phone_map)
        return

    if options.both_stm:

        # these files shouldn't change for specific subdirectory configurations, so
        # we won't bother the user with 'em.
        config_dir = os.path.join(options.data_root, "local", "data")

        spk2info = dict()
        for idx, fn in enumerate(("spk2part", "spk2gender", "spk2dialect")):
            with open(os.path.join(config_dir, fn)) as file_:
                for line in file_:
                    spk, tidbit = line.strip().split(" ", maxsplit=1)
                    spk2info.setdefault(spk, [None] * 3)[idx] = tidbit
        utt2info = dict()
        for idx, fn in enumerate(("utt2type", "utt2wc", "utt2spk")):
            with open(os.path.join(config_dir, fn)) as file_:
                for line in file_:
                    utt, tidbit = line.strip().split(" ", maxsplit=1)
                    utt2info.setdefault(utt, [None] * 3)[idx] = tidbit
        wcinfo = dict(
            (utt2info[utt][1], spk2info[utt2info[utt][2]][1:] + utt2info[utt][:1])
            for utt in utt2info
        )

        write_mapped_stm(options.raw_trn, options.filt_trn, phone_map, wcinfo)
        return

    in_ = options.raw_trn

    # we'll always be writing to a trn file. If our source is a CTM or an STM file,
    # we'll first convert to a trn file
    if options.in_ctm:
        transcripts = pydrobert.torch.data.read_ctm(in_)
        tmp = SpooledTemporaryFile(mode="w+")
        for utt, transcript in transcripts:
            transcript = " ".join(x[0] for x in transcript)
            tmp.write(transcript)
            tmp.write(f" ({utt})\n")
        tmp.seek(0)
        in_ = tmp
        del transcripts, tmp
    elif options.in_stm:
        tmp = SpooledTemporaryFile(mode="w+")
        for line_no, line in enumerate(in_):
            line = line.strip().split(";;")[0].split(maxsplit=6)
            if not line:
                # keep it around so as not to mess with the line numbering later
                tmp.write("\n")
                continue
            if len(line) < 5:
                raise RuntimeError(f"{in_.name} line {line_no + 1}: not stm entry")
            if len(line) > 5 and line[5][0] == "<":
                line = [line[6], f"({line[0]})"]
            else:
                line = line[5:] + [f"({line[0]})"]
            tmp.write(" ".join(line))
            tmp.write("\n")
        tmp.seek(0)
        in_ = tmp

    write_mapped_trn(in_, options.filt_trn, phone_map)


def build_parser():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "data_root",
        type=os.path.abspath,
        help="The root directory under which to store data. Typically " "``data/``",
    )
    subparsers = parser.add_subparsers(title="commands", required=True, dest="command")
    build_preamble_parser(subparsers)
    build_init_phn_parser(subparsers)
    build_torch_dir_parser(subparsers)
    build_filter_parser(subparsers)

    return parser


def build_preamble_parser(subparsers):
    parser = subparsers.add_parser(
        "preamble", help="Do all pre-initialization setup. Needs to be done only once"
    )
    parser.add_argument(
        "timit_root",
        type=os.path.abspath,
        help="Location of TIMIT data directory (LDC935S1)",
    )


def build_init_phn_parser(subparsers):
    parser = subparsers.add_parser(
        "init_phn",
        help="Perform setup common to all phone-based parsing. "
        "Needs to be done only once for a specific vocabulary size",
    )
    parser.add_argument(
        "--config-subdir",
        default=None,
        help="Name of sub directory in data/local/ under which to store setup "
        "specific to this vocabulary size. Defaults to "
        "``phn<vocab_size>``",
    )
    parser.add_argument(
        "--vocab-size",
        default=48,
        type=int,
        choices=[48, 61, 39],
        help="The number of phones to train against. For smaller phone sets, a "
        "surjective mapping is applied. WARNING: stored labels for test data will "
        "have the same vocabulary size as the training data. You should NOT report "
        "error rates against these. Please consult the repo wiki!",
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        default=False,
        help="If set, will train a language model with Modified Kneser-Ney "
        "smoothing on the training data labels",
    )
    parser.add_argument(
        "--lm-max-order",
        type=int,
        default=3,
        help="The maximum n-gram order to train the LM with when the --lm "
        "flag is set",
    )


def build_torch_dir_parser(subparsers):
    parser = subparsers.add_parser(
        "torch_dir",
        help="Write training, test, and extra data to subdirectories. The init_phn "
        "command must have been called previously. If more than one init_phn call has "
        "been made, the next positional argument must be specified.",
    )
    parser.add_argument(
        "config_subdir",
        nargs="?",
        default=None,
        help="The configuration in data/local/ which to build the directories "
        "from. If init_phn was called only once, it can be inferred from the "
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
        "--include-sa",
        action="store_true",
        default=False,
        help="If specified, SA utterances (dialect utterances) from speakers not in "
        "the test set will be included in the training set. Please check the wiki "
        "before using this flag.",
    )

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--complete-test",
        action="store_true",
        default=False,
        help='Use the "complete" TIMIT test set rather than just the core. Specifying '
        "this flag means no development set will be generated. Please check the wiki "
        "before using this flag.",
    )
    test_group.add_argument(
        "--large-dev",
        action="store_true",
        default=False,
        help="Expand the dev set from 50 speakers to all speakers not in the core test "
        "set nor the training set. Please check the wiki before using this flag.",
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
        "filter", help="Filter an input phone-level hypothesis trn file to"
    )
    parser.add_argument(
        "raw_trn", type=argparse.FileType("r"), help="The input (unfiltered) trn file"
    )
    parser.add_argument(
        "filt_trn", type=argparse.FileType("w"), help="The output (filtered) trn file"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--in-ctm",
        action="store_true",
        default=False,
        help="The input is a CTM file, not a trn file. Will still output the filtered "
        "trn file",
    )
    group.add_argument(
        "--both-ctm",
        action="store_true",
        default=False,
        help="The input is a CTM file, not a trn file. Will output a filtered CTM file "
        "as well",
    )
    group.add_argument(
        "--in-stm",
        action="store_true",
        default=False,
        help="The input is an STM file, not a trn file. Will still output the filtered "
        "trn file",
    )
    group.add_argument(
        "--both-stm",
        action="store_true",
        default=False,
        help="The input is an STM file, not a trn file. Will output a filtered STM "
        "file as well",
    )


def main(args=None):
    """Prepare TIMIT data for end-to-end pytorch training"""

    parser = build_parser()
    options = parser.parse_args(args)

    if options.command == "preamble":
        preamble(options)
    elif options.command == "init_phn":
        init_phn(options)
    elif options.command == "torch_dir":
        torch_dir(options)
    elif options.command == "filter":
        filter(options)


if __name__ == "__main__":
    sys.exit(main())
