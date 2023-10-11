#!/usr/bin/env python

# Copyright 2023 Sean Robertson
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

from pathlib import Path
from typing import Optional, Sequence, TextIO
import warnings

import torch

from pyctcdecode import build_ctcdecoder
from pyctcdecode.alphabet import BLANK_TOKEN_PTN
from pyctcdecode.constants import DEFAULT_BEAM_WIDTH, DEFAULT_ALPHA, DEFAULT_BETA

BLANK_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

DESCRIPTION = """Decode logits in folder using pyctcdecode

Intended to be used in concert with "asr-baseline.py decode --write-logits", this
script decodes all PyTorch data files in a flat directory into hypothesis transcriptions
using pyctcdecode (https://github.com/kensho-technologies/pyctcdecode).

pyctcdecode is intended for character or subword-level ASR systems. The type of system
must be specified with either the "--char" or "--bpe" flag. Unlike the vanilla
prefix search of asr-baseline.py in which the mixed LM is assumed to be of the same
token granularity as the ASR system (word LM for word vocab, sub-word LM for sub-word
vocab, etc.), pyctcdecode's LM is word-level. In addition, pyctdecode always outputs
words, not subwords.

In addition to the beam width hyperparameter "--width", pyctcdecode has two additional
hyperparameters influencing the search: "--alpha" and "--beta". "--alpha" controls the
weight of the LM prediction (higher = more) while "--beta" is a "length score
adjustment". Documentation is minimal
(https://github.com/kensho-technologies/pyctcdecode/issues/16).
"""


def main(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Parse a folder containing logits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--file-prefix", default="")
    parser.add_argument("--file-suffix", default=".pt")
    parser.add_argument(
        "--blank-first",
        action="store_true",
        default=False,
        help="If nothing looking like a blank token can be found in the vocabulary and "
        "this flag is set, ",
    )
    lm_arg = parser.add_argument(
        "--lm", type=Path, default=None, help="Path to KenLM ARPA or bin file"
    )
    parser.add_argument(
        "--words",
        type=argparse.FileType("r"),
        default=None,
        help="Path to list of words, one per line",
    )
    width_arg = parser.add_argument("--width", type=int, default=DEFAULT_BEAM_WIDTH)
    beta_arg = parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    batch_arg = parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Number of utterances to process simultaneously. Default (0) is serial",
    )

    style_group = parser.add_mutually_exclusive_group(required=True)
    style_group.add_argument("--bpe", action="store_true", default=False)
    style_group.add_argument("--char", action="store_true", default=False)

    alpha_group = parser.add_mutually_exclusive_group()
    alpha_arg = alpha_group.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ainv_arg = alpha_group.add_argument(
        "--alpha-inv",
        type=float,
        default=None,
        help="1 over --alpha, if a preferable input",
    )

    tk2id_group = parser.add_mutually_exclusive_group(required=True)
    tk2id_group.add_argument(
        "--id2token",
        type=argparse.FileType("r"),
        default=None,
        help="Path to id2token.txt file, mapping integer ids to tokens",
    )
    tk2id_group.add_argument(
        "--token2id",
        type=argparse.FileType("r"),
        default=None,
        help="Path to token2id.txt file, mapping tokens to integer ids",
    )

    ldir_arg = parser.add_argument("logit_dir", type=Path, help="Logit directory")
    parser.add_argument(
        "trn",
        nargs="?",
        type=argparse.FileType("w"),
        default=argparse.FileType("w")("-"),
        help="The output TRN file. Defaults to stdout",
    )

    options = parser.parse_args(args)
    lm: Optional[Path] = options.lm
    fp: str = options.file_prefix
    fs: str = options.file_suffix
    blank_first: bool = options.blank_first
    logit_dir: Path = options.logit_dir
    width: int = options.width
    alpha: float = options.alpha
    beta: float = options.beta
    word_file: Optional[TextIO] = options.words
    trn: TextIO = options.trn
    batch_size: int = options.batch_size

    id2token_file: TextIO
    swap: bool
    if options.id2token is not None:
        id2token_file, swap = options.id2token, False
    else:
        assert options.token2id is not None
        id2token_file, swap = options.token2id, True
    id2tname = id2token_file.name

    if width <= 0:
        raise argparse.ArgumentError(width_arg, "is negative")

    if options.alpha_inv is not None:
        if options.alpha_inv <= 0:
            raise argparse.ArgumentError(ainv_arg, "is non-positive")
        alpha = 1 / options.alpha_inv
    elif alpha < 0:
        raise argparse.ArgumentError(alpha_arg, "is negative")

    if beta < 0:
        raise argparse.ArgumentError(beta_arg, "is non-positive")

    if batch_size < 0:
        raise argparse.ArgumentError(batch_arg, "is negative")

    if not logit_dir.is_dir():
        raise argparse.ArgumentError(ldir_arg, "not a directory")

    if lm is not None:
        if not lm.is_file():
            raise argparse.ArgumentError(lm_arg, "is not a file")
        lm = str(lm)

    id2token = dict()
    token2id = dict()
    blank_token = None
    if swap:
        expected = "<token> <integer-id>"
    else:
        expected = "<integer-id> <token>"
    for no, line in enumerate(id2token_file):
        pair = line.strip().split()
        try:
            if swap:
                token, id = pair
            else:
                id, token = pair
            id = int(id)
            assert id >= 0
        except:
            raise ValueError(
                f"could not parse {id2tname} line {no + 1}: expected "
                f"'{expected}', got '{line}'"
            )
        if BLANK_TOKEN_PTN.match(token):
            if blank_token is not None:
                raise ValueError(
                    f"Two tokens {blank_token} and {token} from {id2tname} "
                    "look like blank tokens. pyctcdecode can't handle this"
                )
            blank_token = token
        if options.bpe:
            token = token.replace("_", "\u2581")
        else:
            assert options.char
            token = token.replace("_", " ")
        if id in id2token:
            raise ValueError(f"Duplicate id {id} in {id2tname}")
        if token in token2id:
            raise ValueError(f"Duplicate token {token} in {id2tname}")
        token2id[token], id2token[id] = id, token

    if not id2token:
        raise ValueError(f"{id2tname} is empty!")
    max_vocab = max(id2token)
    assert max_vocab >= 0
    for id in range(max_vocab):
        if id not in id2token:
            raise ValueError(
                f"found id {max_vocab} in {id2tname} but not {id}. Ids need to be "
                "contiguous"
            )
    vocab = [x[1] for x in sorted(id2token.items())]
    if blank_token is None:
        assert BLANK_TOKEN not in vocab
        blank_token = BLANK_TOKEN
        if blank_first:
            warnings.warn(
                f"Adding {blank_token} to the beginning of the vocabulary because "
                "--blank-first was set"
            )
            vocab.insert(0, blank_token)
        else:
            warnings.warn(
                f"Adding {blank_token} to the end of the vocabulary. If it should be "
                "first, set --blank-first"
            )
            vocab.append(blank_token)

    words = None
    if word_file is not None:
        words = set()
        wname = word_file.name
        for no, line in enumerate(word_file):
            try:
                (word,) = line.strip().split()
            except:
                raise ValueError(
                    f"could not parse {wname} line {no + 1}: expected <word>; "
                    f"got '{line}'"
                )
            if word in words:
                raise ValueError(f"Duplicate entries of {word} in {wname}")
            words.add(word)

    decoder = build_ctcdecoder(
        vocab, kenlm_model_path=lm, unigrams=words, alpha=alpha, beta=beta
    )

    pool = None
    if batch_size:
        try:
            pool = torch.multiprocessing.get_context("fork").Pool(processes=batch_size)
        except:
            pass

    utt_batch: list[str] = []
    logit_batch: list[torch.Tensor] = []

    def decode():
        if pool and len(utt_batch):
            hyp_batch = decoder.decode_batch(pool, logit_batch, width)
            assert len(utt_batch) == len(hyp_batch)
            for utt, hyp in zip(utt_batch, hyp_batch):
                trn.write(f"{hyp.strip()} ({utt})\n")
        else:
            assert len(utt_batch) == len(logit_batch)
            for utt, logits in zip(utt_batch, logit_batch):
                hyp = decoder.decode(logits, width)
                trn.write(f"{hyp.strip()} ({utt})\n")
        utt_batch.clear()
        logit_batch.clear()

    for logit_pth in sorted(logit_dir.iterdir()):
        utt = logit_pth.name
        if not logit_pth.is_file() or not utt.startswith(fp) or not utt.endswith(fs):
            continue
        utt = utt[len(fp) :]
        if fs:
            utt = utt[: -len(fs)]
        logits = torch.load(logit_pth)
        utt_batch.append(utt)
        logit_batch.append(logits.detach().numpy())
        if len(utt_batch) > batch_size:
            decode()
    decode()


if __name__ == "__main__":
    main()
