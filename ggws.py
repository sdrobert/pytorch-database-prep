#! /usr/bin/env python

# Copyright 2020 Sean Robertson
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

"""Command-line interface to prepare the Gigaword Summarization Corpus"""

# we now assume python3. Py2.7 has reached EOL. Yay.

import os
import sys
import argparse
import locale
import warnings
import xml.etree.ElementTree as et

from collections import Counter
from shutil import copy as copy_paths

import pydrobert.torch.command_line as torch_cmd

from .common import mkdir

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2020 Sean Robertson"


locale.setlocale(locale.LC_ALL, "C")


def preamble(options):

    dir_ = os.path.join(options.data_root, "local", "data")
    src = os.path.join(options.ggws_root, "org_data")
    word2freq_txt = os.path.join(dir_, "word2freq.txt")

    if not os.path.exists(src):
        raise ValueError(
            '"{}" does not exist. Are you sure "{}" points to the UniLM '
            "version of the Gigaword Summarization Corpus?".format(
                src, options.ggws_root
            )
        )

    mkdir(dir_)

    word2freq = Counter()

    for fn_nosuf in ("train", "dev", "test"):
        is_train = fn_nosuf == "train"
        for suffix_from, suffix_to in ((".src", ".sent"), (".tgt", ".head")):
            with open(
                os.path.join(src, fn_nosuf + suffix_from + ".txt"), encoding="utf-8"
            ) as in_, open(
                os.path.join(dir_, fn_nosuf + suffix_to + ".trn"), "w"
            ) as out_:
                for idx, line in enumerate(in_):
                    # the non-breaking space sometimes comes up instead of a
                    # space. We also replace the underscore with its html code
                    # so that it doesn't muck with our use of underscore in
                    # subwords (the database hasn't been entirely sanitized of
                    # these anyway)
                    line = line.replace(u"\u00A0", " ").replace("_", "&#95;")
                    # replace 'UNK' in the test set with '<unk>' to be
                    # consistent with the training set. I prefer '<unk>'
                    # because it's quite clearly a control character
                    tokens = line.strip().split(" ")
                    if is_train:
                        word2freq.update(tokens)
                    else:
                        tokens = ["<unk>" if x == "UNK" else x for x in tokens]
                    out_.write(" ".join(tokens))
                    # we add sentence ids to each transcription and store as a
                    # NIST .trn file. This will allow us to disambiguate the
                    # pairs when we read them from file.
                    out_.write(" (sent_{}_{:07d})\n".format(fn_nosuf, idx))

    with open(word2freq_txt, "w") as file_:
        for word, freq in sorted(word2freq.items()):
            file_.write("{} {}\n".format(word, freq))


def init_word(options):

    local_dir = os.path.join(options.data_root, "local")
    local_data_dir = os.path.join(local_dir, "data")
    word2freq_txt = os.path.join(local_data_dir, "word2freq.txt")
    vocab_size = 1000 if options.vocab_size is None else options.vocab_size
    prune_thresh = 0 if options.prune_thresh is None else options.prune_thresh
    if options.config_subdir is None:
        config_dir = os.path.join(local_dir, "word")
        if options.vocab_size is not None:
            config_dir += "{}k".format(vocab_size)
        elif options.prune_thresh is not None:
            config_dir += "{}p".format(prune_thresh)
    else:
        config_dir = os.path.join(local_dir, options.config_subdir)
    token2id_txt = os.path.join(config_dir, "token2id.txt")
    id2token_txt = os.path.join(config_dir, "id2token.txt")
    train_oovs_txt = os.path.join(config_dir, "train_oovs.txt")

    with open(word2freq_txt) as file_:
        freq_word = (line.strip().split() for line in file_)
        freq_word = ((int(x[1]), x[0]) for x in freq_word)
        freq_word = sorted(freq_word, reverse=True)

    oovs = {x[1] for x in freq_word[vocab_size * 1000 :]}
    freq_word = freq_word[: vocab_size * 1000]
    while freq_word and freq_word[-1][0] <= prune_thresh:
        oovs.add(freq_word.pop(-1)[1])

    if not freq_word:
        warnings.warn(
            "No words are left after pruning + vocab size. All tokens will be "
            "<unk> in training"
        )
    vocab = set(x[1] for x in freq_word) | {"<unk>", "<s>", "</s>"}
    vocab = sorted(vocab)
    del freq_word

    mkdir(config_dir)

    with open(token2id_txt, "w") as t2id, open(id2token_txt, "w") as id2t:
        for i, v in enumerate(vocab):
            t2id.write("{} {}\n".format(v, i))
            id2t.write("{} {}\n".format(i, v))

    to_copy = {
        "train.sent.trn",
        "train.head.trn",
        "dev.sent.trn",
        "dev.head.trn",
        "test.sent.trn",
        "test.head.trn",
    }
    for x in to_copy:
        copy_paths(os.path.join(local_data_dir, x), os.path.join(config_dir, x))

    # determine the OOVs in the training partition. Primarily for diagnostic
    # purposes
    oovs -= {"<unk>"}
    oovs = sorted(oovs)
    with open(train_oovs_txt, "w") as file_:
        file_.write("\n".join(oovs))
        file_.write("\n")


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
    token2id_txt = os.path.join(ext, "token2id.txt")
    mkdir(ext)

    for x in {"id2token.txt", "token2id.txt"}:
        copy_paths(os.path.join(config_dir, x), ext)

    token2id = dict()
    with open(token2id_txt) as file_:
        for line in file_:
            token, id_ = line.strip().split()
            token2id[token] = int(id_)
    assert "<unk>" in token2id

    for is_test, partition in enumerate(("train", "dev", "test")):
        both_args = []

        if is_test:
            cur_token2id = dict(token2id)
            cur_token2id_txt = os.path.join(
                ext, ".".join(("token2id", partition, "txt"))
            )
            cur_id2token_txt = os.path.join(
                ext, ".".join(("id2token", partition, "txt"))
            )
        else:
            cur_token2id_txt = token2id_txt

        for type_ in ("sent", "head"):
            trn_src = os.path.join(config_dir, ".".join((partition, type_, "trn")))
            trn_dest = os.path.join(ext, ".".join((partition, type_, "trn")))
            part_dir = os.path.join(dir_, partition)
            torch_dir_ = os.path.join(part_dir, "feat" if type_ == "sent" else "ref")

            copy_paths(trn_src, trn_dest)

            args = [trn_src, cur_token2id_txt, torch_dir_]
            both_args.append(args)
            if is_test:
                with open(trn_src) as file_:
                    for line in file_:
                        trans = line.split()
                        trans.pop()  # remove sent id
                        for word in trans:
                            cur_token2id.setdefault(word, len(cur_token2id))
                args += ["--skip-frame-times"]
            else:
                args += ["--unk-symbol", "<unk>", "--feat-sizing"]

        if is_test:
            with open(cur_id2token_txt, "w") as id2t, open(
                cur_token2id_txt, "w"
            ) as t2id:
                for t, id_ in sorted(cur_token2id.items(), key=lambda x: x[1]):
                    id2t.write("{} {}\n".format(id_, t))
                    t2id.write("{} {}\n".format(t, id_))

        for args in both_args:
            torch_cmd.trn_to_torch_token_data_dir(args)


def prefix_baseline(options):

    if options.cutoff < 1:
        raise ValueError("Cutoff needs to be >= 1")

    ext = os.path.join(options.data_root, "ext")

    parts = ("test",)
    if not options.exclude_dev:
        parts += ("dev",)
    if options.include_train:
        parts += ("train",)

    for part in parts:
        with open(os.path.join(ext, part + ".sent.trn")) as trn:
            for line in trn:
                toks = line.strip().split(" ")
                utt = toks.pop()
                if utt[0] != "(" or utt[-1] != ")":
                    raise ValueError("{} is not a trn".format(trn.name))
                utt = utt[1:-1]
                sent = " ".join(tok for tok in toks if tok != ",")
                co = min(options.cutoff, len(sent))
                if options.drop_on_end:
                    while 0 < co < len(sent) and sent[co : co + 1] != " ":
                        co -= 1
                elif not options.split_on_end:
                    while co < len(sent) and sent[co : co + 1] != " ":
                        co += 1
                sent = sent[:co]
                options.out_trn.write(sent)
                options.out_trn.write(" (")
                options.out_trn.write(utt)
                options.out_trn.write(")\n")


def rouge_dir(options):

    ext_dir = os.path.join(options.data_root, "ext")

    # the ids we assigned to each sentence contain the partition id,
    # making them unambiguous w.r.t. the initial document

    # read reference
    part2utts = dict()
    utt2ref = dict()
    utt2part = dict()
    for part in ("train", "dev", "test"):
        with open(os.path.join(ext_dir, part + ".head.trn")) as trn:
            for line in trn:
                toks = line.strip().split(" ")
                utt = toks.pop()
                if utt[0] != "(" or utt[-1] != ")":
                    raise ValueError("{} is not a trn".format(trn.name))
                utt = utt[1:-1]
                utt2ref[utt] = " ".join(toks) + " ."
                utt2part[utt] = part
                part2utts.setdefault(part, set()).add(utt)

    # read hypotheses
    part2present = dict((p, False) for p in part2utts)
    utt2present = dict((u, False) for u in utt2part)
    utt2peer2hyp = dict()
    for trn in options.trns:
        peer = os.path.splitext(os.path.basename(trn.name))[0]
        for line in trn:
            toks = line.strip().split(" ")
            utt = toks.pop()
            if utt[0] != "(" or utt[-1] != ")":
                raise ValueError("{} is not a trn".format(trn.name))
            utt = utt[1:-1]
            if utt not in utt2part:
                warnings.warn(
                    "utterance id {} from '{}' does not have a reference."
                    " Skipping".format(utt, options.trn.name)
                )
                continue
            peer2hyp = utt2peer2hyp.setdefault(utt, dict())
            if peer in peer2hyp:
                raise ValueError("Duplicate peer {}".format())
            if not toks or toks[-1] != ".":
                toks.append(".")
            part2present[utt2part[utt]] = utt2present[utt] = True
            peer2hyp[peer] = " ".join(toks)

    # write each partition to an independent subdirectory
    # (only if some hypothesis mentioning it exists)
    for part, present in part2present.items():

        if not present:
            continue

        subdir = os.path.join(options.project_dir, part)
        model_dir = os.path.abspath(os.path.join(subdir, "model"))
        peer_dir = os.path.abspath(os.path.join(subdir, "peer"))
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(peer_dir, exist_ok=True)

        utts = sorted(utt for utt in part2utts[part] if utt2present[utt])

        # Warning! The "model" is the gold standard/ref and the "peers" are the things
        # we've generated (hypotheses). Ugh
        root = et.Element("ROUGE_EVAL", version="1.5.5")
        for utt in utts:
            peer2hyp = utt2peer2hyp[utt]
            eval_ = et.SubElement(root, "EVAL", ID=utt)
            et.SubElement(eval_, "MODEL-ROOT").text = model_dir
            et.SubElement(eval_, "PEER-ROOT").text = peer_dir
            et.SubElement(eval_, "INPUT-FORMAT", TYPE="SPL")

            models = et.SubElement(eval_, "MODELS")
            et.SubElement(models, "M", ID="1").text = utt + ".1.txt"
            with open(os.path.join(model_dir, utt + ".1.txt"), "w") as txt:
                txt.write(utt2ref[utt])
                txt.write("\n")

            peers = et.SubElement(eval_, "PEERS")
            for peer in sorted(peer2hyp):
                bn = utt + "." + peer + ".txt"
                et.SubElement(peers, "P", ID=peer).text = bn
                with open(os.path.join(peer_dir, bn), "w") as txt:
                    txt.write(peer2hyp[peer])
                    txt.write("\n")

        tree = et.ElementTree(root)
        tree.write(os.path.join(subdir, "settings.xml"))


def build_preamble_parser(subparsers):
    parser = subparsers.add_parser(
        "preamble", help="Do all pre-initializaiton setup. Needs to be done only once"
    )
    parser.add_argument(
        "ggws_root",
        type=os.path.abspath,
        help="Location of the GGWS data directory, downloaded from the UniLM "
        'project. Contains files like "dev.src" and "train.tgt"',
    )


def build_init_word_parser(subparsers):
    parser = subparsers.add_parser(
        "init_word",
        help="Perform setup common to full word-level parsing. "
        "Needs to be done only once for a specific vocabulary size. "
        'Preceded by "preamble" command.',
    )
    parser.add_argument(
        "--config-subdir",
        default=None,
        help="Name of sub directory in data/local/ under which to store "
        "setup specific to this vocabulary size. Defaults to "
        "``word(<vocab_size>k|<prune_thresh>p|)``, depending on whether the "
        "full vocabulary was used (~124k words), the top ``<vocab_size>k`` "
        "words in terms of frequency, or the words remaining after pruning "
        "those with less than or equal to ``<prune_thresh>`` tokens.",
    )

    vocab_group = parser.add_mutually_exclusive_group()
    vocab_group.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Limit the vocabulary size to this many words (in thousands). "
        "The vocabulary will be chosen from the most frequent word types in "
        "the training set.",
    )
    vocab_group.add_argument(
        "--prune-thresh",
        type=int,
        default=None,
        help="Limit the vocabulary size by pruning all word types with equal "
        "or fewer than this number of tokens in the training set.",
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


def build_prefix_baseline_parser(subparsers):
    parser = subparsers.add_parser(
        "prefix_baseline",
        help="Construct a baseline model that builds headers from a prefix of the "
        "input sentences",
    )
    parser.add_argument(
        "out_trn",
        type=argparse.FileType("w"),
        help="Where to store the baseline's hypotheses/results",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=75,
        help="The number of characters from the input sentence to use",
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        default=False,
        help="Include baseline results for the training partition as well",
    )
    parser.add_argument(
        "--exclude-dev",
        action="store_true",
        default=False,
        help="Exclude baseline results for the development partition",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--split-on-end",
        action="store_true",
        default=False,
        help="If the cutoff is reached within a word, split the word (instead of "
        "keeping it)",
    )
    group.add_argument(
        "--drop-on-end",
        action="store_true",
        default=False,
        help="If the cutoff is reached within a word, drop that word (instead of "
        "keeping it)",
    )


def build_rouge_dir_parser(subparsers):
    parser = subparsers.add_parser(
        "rouge_dir",
        help="Produce a rouge-style project directory out of one or more "
        "hypothesis/peer TRN files",
    )
    parser.add_argument("project_dir", help="Where to save the project directory to")
    parser.add_argument(
        "trns",
        nargs="+",
        type=argparse.FileType("r"),
        help="Hypothesis/peer summaries of the GGWS data set (what you generated "
        "using your fancy neural networks and whatnot) in TRN format",
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
    build_torch_dir_parser(subparsers)
    build_prefix_baseline_parser(subparsers)
    build_rouge_dir_parser(subparsers)
    return parser


def main(args=None):
    """Prepare GGWS data for pytorch training"""

    parser = build_parser()
    options = parser.parse_args(args)

    if options.command == "preamble":
        preamble(options)
    elif options.command == "init_word":
        init_word(options)
    elif options.command == "torch_dir":
        torch_dir(options)
    elif options.command == "prefix_baseline":
        prefix_baseline(options)
    elif options.command == "rouge_dir":
        rouge_dir(options)


if __name__ == "__main__":
    sys.exit(main())
