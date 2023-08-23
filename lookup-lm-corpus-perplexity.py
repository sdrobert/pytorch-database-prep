#! /usr/bin/env python

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

import sys
import argparse
import os
import logging
import gzip
import io
import itertools

from typing import Optional, Mapping, List, Union

import torch
import numpy as np
import ngram_lm
import pydrobert.torch.config as config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from pydrobert.torch.data import parse_arpa_lm
from pydrobert.torch.modules import LookupLanguageModel
from pydrobert.torch.argcheck import as_file, as_nat


SOS_DEFTS = ("<s>", "<S>")
EOS_DEFTS = ("</s>", "</S>")
UNK_DEFTS = ("<unk>", "<UNK>")
VOCAB_SIZE_KEY = "vocab_size"
SOS_KEY = "sos"
EOS_KEY = "eos"
UNK_KEY = "unk"
FN = os.path.basename(__file__)

DESCRIPTION = f"""\
Compute perplexity of a corpus using an arpa LM

Treating a corpus C as a sequence of independent sentences s^(1), s^(2), ..., s^(S),

    P(C) = prod_(i=1 to S) P(s^(i)),

each sentence as a sequence of tokens s = (s_1, s_2, ... s_W), |s^(i)| = W_i, and the
probability of a sentence s determined by an N-gram/lookup LM as

    P(s) = prod_(j=1 to W_i) P(s_j|s_(j - 1), s_(j - 2), s_(j - (N - 1))),

the perplexity of a corpus is just the inverse of the M-th root of the corpus
probability,

    PP(C) = P(C)^(-1/M),

where M is the total number of tokens in the corpus,

    M = sum_(i=1 to S) W_s

It may be interpreted as the "average compressed vocabulary size." For actual vocabulary
size V, PP(C) << V in general. The perplexity of the corpus will change with the choice
of LM.

This script takes in a corpus as an (optionally gzipped) text file, one line per
sentence, and an n-gram/lookup LM, and prints the perplexity of the corpus to stdout.
For example, assuming that a gzipped ARPA lm is saved to "lm.arpa.gz" and the text file
is "text.gz":

   {FN} --arpa lm.arpa.gz text.gz

While fast and serviceable for small LMs, this pure-Python implementation isn't very
efficient memory-wise. 
"""


class CorpusDataset(torch.utils.data.IterableDataset):
    filename: str
    token2id: Optional[Mapping[str, int]]
    eos: Optional[Union[str, int]]
    unk: Optional[int]

    def __init__(
        self,
        filename: str,
        token2id: Optional[Mapping[str, int]] = None,
        eos: Optional[Union[str, int]] = None,
        unk: Optional[int] = None,
    ):
        assert os.path.isfile(filename)
        if isinstance(eos, str) and token2id is not None:
            eos = token2id[eos]
        self.filename, self.token2id, self.eos, self.unk = filename, token2id, eos, unk

    def process_line(self, line: str):
        line_ = line.strip().split()
        if self.eos is not None:
            line_.append(self.eos)
        if self.token2id is None:
            return line_
        elif self.unk is None:
            return torch.tensor([self.token2id[tok] for tok in line_])
        else:
            return torch.tensor([self.token2id.get(tok, self.unk) for tok in line_])

    def __iter__(self):
        with open(self.filename, "rb") as fb:
            if fb.peek()[:2] == b"\x1f\x8b":
                ft = gzip.open(fb, mode="rt")
            else:
                ft = io.TextIOWrapper(fb)
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                ft = itertools.islice(ft, worker_info.id, None, worker_info.num_workers)
            yield from (self.process_line(line) for line in ft)


def main_arpa(options: argparse.Namespace):
    logging.info("Parsing lm...")
    with open(options.arpa, "rb") as arpa:
        if arpa.peek()[:2] == b"\x1f\x8b":
            arpa = gzip.open(arpa, mode="rt")
        else:
            arpa = io.TextIOWrapper(arpa)
        prob_dicts = parse_arpa_lm(
            arpa, ftype=np.float32, to_base_e=False, logger=logging.getLogger()
        )
    logging.info("Parsed lm")

    logging.info("Building LM")
    lm = ngram_lm.BackoffNGramLM(
        prob_dicts,
        options.sos_token,
        options.eos_token,
        options.unk_token,
        destructive=True,
    )
    del prob_dicts
    logging.info("Built LM")

    logging.info("Computing perplexity")
    corpus = CorpusDataset(options.corpus, eos=options.eos_token)
    if options.verbose:
        corpus = tqdm(corpus)
    pp = lm.corpus_perplexity(corpus)
    logging.info("Computed perplexity")

    print(f"{pp:.10f}", file=options.output)


def main_state(options: argparse.Namespace):
    state_dict_fn, token2id_fn = options.states_and_token2id

    if options.device is None:
        if torch.cuda.is_available():
            options.device = torch.device(torch.cuda.current_device())
        else:
            options.device = torch.device("cpu")

        logging.info(f"Inferred --device: {options.device}")

    if options.num_workers is None:
        try:
            options.num_workers = len(os.sched_getaffinity(0))
        except:
            options.num_workers = torch.multiprocessing.cpu_count()
        logging.info(f"Inferred --num-workers: {options.num_workers}")

    logging.info("Parsing token2id...")
    with open(token2id_fn, "rb") as token2id_f:
        tname = os.path.basename(token2id_fn)
        if token2id_f.peek()[:2] == b"\x1f\x8b":
            token2id_f = gzip.open(token2id_f, mode="rt")
        else:
            token2id_f = io.TextIOWrapper(token2id_f)

        token2id, id2token = dict(), dict()
        max_id, min_id = -np.inf, np.inf
        for idx, line in enumerate(token2id_f):
            emsg = f"'{tname}' line {idx+1}:"
            line = line.strip()
            if not line:
                continue
            try:
                token, id_ = line.split()
                id_ = int(id_)
            except:
                raise ValueError(f"{emsg} not <token> <id>")
            if token in token2id:
                raise ValueError(f"{emsg} token '{token}' seen twice")
            if id_ in id2token:
                raise ValueError(f"{emsg} id '{id_}' seen twice")
            if id_ < 0:
                logging.warn(
                    f"token '{token}' has negative id {id_}, which will end up excluded "
                    "from the vocabulary"
                )
            max_id, min_id = max(id_, max_id), min(id_, min_id)
            token2id[token] = id_
            id2token[id_] = token
    if not token2id:
        raise ValueError(f"'{tname}' was empty")
    logging.info(f"Parsed token2id with {len(token2id)} entries")
    assert np.isfinite(max_id) and np.isfinite(min_id)

    logging.info(f"Loading state dict from file...")
    state_dict = torch.load(state_dict_fn)
    logging.info(f"Loaded state dict from file")

    token2id_vocab = set(token2id)

    sos_token, sos_id = options.sos_token, options.sos_id
    if SOS_KEY in state_dict:
        sos_id_ = state_dict.pop(SOS_KEY)
        if sos_id is not None and int(sos_id_) != sos_id:
            raise ValueError(
                f"--sos-id was set to {sos_id}, but state dict file contained "
                f"key '{SOS_KEY}' with value {sos_id_}"
            )
        else:
            logging.info(f"'{SOS_KEY}' read from state dict as {sos_id_}")
            sos_id = int(sos_id_)
    if sos_token is None:
        if sos_id in id2token:
            sos_token = id2token[sos_id]
            logging.info(
                f"sos token ('{sos_token}') inferred from sos id ({sos_id}) and "
                "token2id"
            )
        else:
            matches = set(SOS_DEFTS) & token2id_vocab
            if len(matches) > 1:
                raise ValueError(
                    f"Found multiple matching default sos labels in token2id: {matches}"
                )
            elif len(matches):
                sos_token = matches.pop()
                logging.info(f"sos token inferred from token2id: '{sos_token}'")
                sos_id = token2id[sos_token]
            else:
                logging.info("Could not infer sos token")
    if sos_id is None:
        if sos_token in token2id:
            sos_id = token2id[sos_token]
            logging.info(
                f"sos id ({sos_id}) inferred from sos token ('{sos_token}') and "
                "token2id"
            )
        else:
            raise ValueError("Could not infer sos id")
    assert sos_id is not None
    if (
        token2id.get(sos_token, sos_id) != sos_id
        or id2token.get(sos_id, sos_token) != sos_token
    ):
        raise ValueError(
            f"sos token ('{sos_token}') does not match id ({sos_id}) listed in token2id"
        )
    if sos_id is not None and sos_token is not None:
        token2id_vocab.add(sos_token)
        token2id[sos_token] = sos_id
        id2token[sos_id] = sos_token

    eos_token, eos_id = options.eos_token, options.eos_id
    if EOS_KEY in state_dict:
        eos_id_ = state_dict.pop(EOS_KEY)
        if eos_id is not None and int(eos_id_) != eos_id:
            raise ValueError(
                f"--eos-id was set to {eos_id}, but state dict file contained "
                f"key '{EOS_KEY}' with value {eos_id_}"
            )
        else:
            logging.info(f"'{EOS_KEY}' read from state dict as {eos_id_}")
            eos_id = int(eos_id_)
    if eos_token is None:
        if eos_id in id2token:
            eos_token = id2token[eos_id]
            logging.info(
                f"eos token ('{eos_token}') inferred from eos id ({eos_id}) and "
                "token2id"
            )
        else:
            matches = set(EOS_DEFTS) & token2id_vocab
            if len(matches) > 1:
                raise ValueError(
                    f"Found multiple matching default eos labels in token2id: {matches}"
                )
            elif len(matches):
                eos_token = matches.pop()
                logging.info(f"eos token inferred from token2id: '{eos_token}'")
                if eos_id is not None:
                    logging.info(f"--eos-id was set to {eos_id}, so mapping to that")
            else:
                logging.info("Could not infer eos token")
    if eos_id is None:
        if eos_token in token2id:
            eos_id = token2id[eos_token]
            logging.info(
                f"eos id ({eos_id}) inferred from eos token ('{eos_token}') and "
                "token2id"
            )
        else:
            logging.warning("Could not infer eos id (but might not exist)")
    if (
        token2id.get(eos_token, eos_id) != eos_id
        or id2token.get(eos_id, eos_token) != eos_token
    ):
        raise ValueError(
            f"eos token ('{eos_token}') does not match id ({eos_id}) listed in token2id"
        )
    if eos_token is not None and eos_id is not None:
        token2id_vocab.add(eos_token)
        token2id[eos_token] = eos_id
        id2token[eos_id] = eos_token

    unk_token, unk_id = options.unk_token, options.unk_id
    if UNK_KEY in state_dict:
        unk_id_ = state_dict.pop(UNK_KEY)
        if unk_id is not None and int(unk_id_) != unk_id:
            raise ValueError(
                f"--unk-id was set to {unk_id}, but state dict file contained "
                f"key '{UNK_KEY}' with value {unk_id_}"
            )
        else:
            logging.info(f"'{UNK_KEY}' read from state dict as {unk_id_}")
            unk_id = int(unk_id_)
    if unk_token is None:
        if unk_id in id2token:
            unk_token = id2token[unk_id]
            logging.info(
                f"unk token ('{unk_token}') inferred from unk id ({unk_id}) and "
                "token2id"
            )
        else:
            matches = set(UNK_DEFTS) & token2id_vocab
            if len(matches) > 1:
                raise ValueError(
                    f"Found multiple matching default unk labels in token2id: {matches}"
                )
            elif len(matches):
                unk_token = matches.pop()
                logging.info(f"unk token inferred from id2token: '{unk_token}'")
            else:
                logging.info("Could not infer unk token")
    if unk_id is None:
        if unk_token in token2id:
            unk_id = token2id[unk_token]
            logging.info(
                f"unk id ({unk_id}) inferred from unk token ('{unk_token}') and "
                "token2id"
            )
        else:
            logging.warning("Could not infer unk id (but might not exist)")
    if (
        token2id.get(unk_token, unk_id) != unk_id
        or id2token.get(unk_id, unk_token) != unk_token
    ):
        raise ValueError(
            f"unk token ('{unk_token}') does not match id ({unk_id}) listed in token2id"
        )
    if unk_token is not None and unk_id is not None:
        token2id_vocab.add(unk_token)
        token2id[unk_token] = unk_id
        id2token[unk_id] = unk_token

    pad_idx = config.INDEX_PAD_VALUE
    while pad_idx in token2id_vocab:
        pad_idx -= 1

    if VOCAB_SIZE_KEY in state_dict:
        vocab_size = state_dict.pop(VOCAB_SIZE_KEY)
        logging.info(f"'{VOCAB_SIZE_KEY}' read from state dict as {vocab_size}")
    else:
        vocab_size = max(token2id[token] for token in token2id_vocab) + 1
        logging.info(f"vocab size inferred from token2id to be {vocab_size}")
    del id2token

    logging.info("Initializing lm...")
    lm = LookupLanguageModel(vocab_size, sos_id)
    lm.load_state_dict(state_dict)
    lm = lm.to(options.device)
    del state_dict
    logging.info("Initialized lm")

    def collate_fn(hyps: List[torch.Tensor]) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(hyps, padding_value=pad_idx)

    logging.info("Computing perplexity...")
    corpus = torch.utils.data.DataLoader(
        CorpusDataset(options.corpus, token2id, eos_id, unk_id),
        batch_size=options.batch_size,
        collate_fn=collate_fn,
        pin_memory=options.device.type == "cuda",
        num_workers=options.num_workers,
    )

    if options.verbose:
        corpus = tqdm(corpus)
    total_logp = torch.zeros(1, dtype=torch.double, device=options.device)
    total_tokens = torch.zeros(1, dtype=torch.double, device=options.device)
    with torch.no_grad():
        for hyps in corpus:
            hyps = hyps.to(options.device)
            logps = lm.calc_full_log_probs_chunked(
                hyps[:-1], dict(), options.chunk_size
            )
            non_pad_mask = hyps != pad_idx
            logp = logps.gather(2, hyps.clamp_(0, vocab_size - 1).unsqueeze(2))
            logp = logp.squeeze(2).masked_select(non_pad_mask)
            total_tokens += logp.numel()
            total_logp += logp.sum()
    logging.info("Computed perplexity")

    print(f"{(-total_logp / total_tokens).exp().item():.10f}", file=options.output)


def main(args: Optional[str] = None):
    logging.captureWarnings(True)

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    lm_grp = parser.add_mutually_exclusive_group(required=True)
    lm_grp.add_argument(
        "--arpa",
        type=as_file,
        metavar="PTH",
        default=None,
        help="Path to a(n optionally gzipped) ARPA file",
    )
    lm_grp.add_argument(
        "--states-and-token2id",
        nargs=2,
        type=as_file,
        metavar=("STATE_PTH", "TOKEN2ID_PTH"),
        default=None,
        help="Path to a(n optionally gzipped) state dict file (from "
        "arpa-lm-to-state-dict.py) and a token2id file for PyTorch decoding",
    )
    parser.add_argument(
        "corpus",
        metavar="PTH",
        type=as_file,
        help="Path to a(n optionally gzipped) text corpus",
    )
    parser.add_argument(
        "output",
        type=argparse.FileType("w"),
        nargs="?",
        default=sys.stdout,
        help="File to write perplexity to. Defaults to stdout",
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    parser.add_argument(
        "--sos-token",
        metavar="TOK",
        default=None,
        help="Token used to demarcate the start of a token sequence",
    )
    parser.add_argument(
        "--sos-id",
        type=np.int64,
        metavar="INT",
        default=None,
        help="Integer id associated with start-of-sequence tokens (PyTorch only)",
    )
    parser.add_argument(
        "--eos-token",
        default=None,
        metavar="TOK",
        help="Token used to demarcate the end of a token sequence",
    )
    parser.add_argument(
        "--eos-id",
        type=np.int64,
        metavar="INT",
        default=None,
        help="Integer id associated with end-of-sequence tokens (PyTorch only)",
    )
    parser.add_argument(
        "--unk-token",
        default=None,
        metavar="TOK",
        help="Token replacing those missing from LM",
    )
    parser.add_argument(
        "--unk-id",
        type=np.int64,
        default=None,
        metavar="INT",
        help="Integer id associated with unknown/oov LM (PyTorch only)",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=None,
        help="Device to perform PyTorch decoding on. Defaults to CUDA if available",
    )
    parser.add_argument(
        "--batch-size",
        type=as_nat,
        default=1,
        help="Number of sentences to process simultaneously (PyTorch only)",
    )
    parser.add_argument(
        "--chunk-size",
        type=as_nat,
        default=1,
        help="Number of tokens in a sentence to process simultaneously (PyTorch only)",
    )
    parser.add_argument(
        "--num-workers",
        type=as_nat,
        default=None,
        help="Number of workers to ready data (PyTorch only). Defaults to number of "
        "cores on the machine",
    )

    options = parser.parse_args(args)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO if options.verbose else logging.WARNING,
    )

    if options.arpa is not None:
        main_arpa(options)
    else:
        main_state(options)


if __name__ == "__main__":
    main()
