#! /usr/bin/env python

# Copyright 2020 Sean Robertson
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

import argparse
import locale
import os
import sys
import glob

from shutil import copy as copy_paths

import pydrobert.speech.util as speech_util

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
    with open(os.path.join(RESOURCE_DIR, "phones.map")) as file_:
        for line in file_:
            line = line.strip().split()
            if not line:
                continue
            p61, p48, p39 = line[0], None, None
            if len(line) > 1:
                p48 = line[1]
                if len(line) == 2:
                    p39 = line[2]
            phone_map[p61] = (p61, p48, p39)

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
            spk2dialect[spk] = dr
            spk2gender[spk] = gender
            utt2wc[utt] = (utt, "A")
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

    stm.sort(key=lambda x: (x[0], x[1], x[3]))
    ctm.sort()

    dir_ = os.path.join(data_root, "local", "data")
    os.makedirs(dir_, exist_ok=True)

    with open(os.path.join(dir_, "all.stm"), "w") as file_:
        for line in stm:
            file_.write("{} {} {} {:.3f} {:.3f} {}\n".format(*line))

    with open(os.path.join(dir_, "all.ctm"), "w") as file_:
        for line in ctm:
            file_.write("{} {} {:.3f} {:.3f} {}\n".format(*line))

    for name, dict_ in (
        ("spk2dialect", spk2dialect),
        ("spk2gender", spk2gender),
        ("spk2part", spk2part),
        ("utt2prompt", utt2prompt),
        ("utt2sph", utt2sph),
        ("utt2type", utt2type),
        ("utt2wc", utt2wc),
    ):
        with open(os.path.join(dir_, name), "w") as file_:
            for key in sorted(dict_):
                file_.write("{} {}\n".format(key, dict_[key]))


def preamble(options):
    timit_data_prep(options.timit_root, options.data_root)


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
            phone_map[key] = phone_map[key][1]
    phone_set = sorted(set(val for val in phone_map.values() if val is not None))

    os.makedirs(config_dir, exist_ok=True)

    for fn in (
        "spk2dialect",
        "spk2gender",
        "spk2part",
        "utt2prompt",
        "utt2sph",
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

    with open(os.path.join(data_dir, "all.stm")) as in_stm, open(
        os.path.join(config_dir, "all.stm"), "w"
    ) as out_stm:
        for line in in_stm:
            line = line.strip().split()
            prefix, phns = line[:5], line[5:]
            idx = 0
            while idx < len(phns):
                to = phone_map[phns[idx]]
                if to is None:
                    phns.pop(idx)
                else:
                    phns[idx] = to
                    idx += 1
            out_stm.write(" ".join(prefix + phns))
            out_stm.write("\n")

    with open(os.path.join(data_dir, "all.ctm")) as in_ctm, open(
        os.path.join(config_dir, "all.ctm"), "w"
    ) as out_ctm:
        last_wc = last_start = last_dur = last_phn = None
        for line in in_ctm:
            wc, start, dur, phn = line.strip().rsplit(maxsplit=3)
            start, dur = float(start), float(dur)
            if wc != last_wc:
                if last_phn is not None:
                    out_ctm.write(
                        "{} {:.3f} {:.3f} {}\n".format(
                            last_wc, last_start, last_dur, last_phn
                        )
                    )
                last_wc = wc
                last_start = last_dur = 0.0
                last_phn = None
            to = phone_map[phn]
            if to is not None:
                if last_phn is not None:
                    out_ctm.write(
                        "{} {:.3f} {:.3f} {}\n".format(
                            last_wc, last_start, last_dur, last_phn
                        )
                    )
                last_wc, last_start, last_dur, last_phn = wc, start, dur, to
            else:
                # collapse this phone into the previous phone
                last_dur += dur
        if last_phn is not None:
            out_ctm.write(
                "{} {:.3f} {:.3f} {}\n".format(last_wc, last_start, last_dur, last_phn)
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
    build_init_phn_parser(subparsers)

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
        default=2,
        help="The maximum n-gram order to train the LM with when the --lm "
        "flag is set",
    )


def main(args=None):
    """Prepare TIMIT data for end-to-end pytorch training"""

    parser = build_parser()
    options = parser.parse_args(args)

    if options.command == "preamble":
        preamble(options)
    if options.command == "init_phn":
        init_phn(options)


if __name__ == "__main__":
    sys.exit(main())
