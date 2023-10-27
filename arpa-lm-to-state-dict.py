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

import io
import sys
import argparse
import gzip
import logging

import numpy as np
import torch
import ngram_lm

from pydrobert.torch.modules import LookupLanguageModel
from pydrobert.torch.data import parse_arpa_lm

SOS_DEFTS = ("<s>", "<S>")
EOS_DEFTS = ("</s>", "</S>")
UNK_DEFTS = ("<unk>", "<UNK>")
VOCAB_SIZE_KEY = "vocab_size"
SOS_KEY = "sos"
EOS_KEY = "eos"
UNK_KEY = "unk"

DESCRIPTION = f"""\
Convert an arpa lm file to a state dictionary for a LookupLanguageModel

Some recipes generate arpa n-gram language models (e.g. 'lm.arpa.gz'). If you are using
pydrobert-pytorch's (https://pydrobert-pytorch.readthedocs.io) LookupLanguageModel to
process these in PyTorch, this script can simplify some code and reduce run times. It
saves the state dictionary of the LookupLanguage model to file, which can be loaded
as follows:

    state_dict = torch.load('path/to/lm.pt')
    lm = LookupLanguageModel(vocab_size, sos)
    lm.load_state_dict(state_dict)

Additionally, if the flags --save-vocab-size and --save-sos are passed, the
associated LookupLanguageModel parameters will be stored in the state dict. They can
be recovered with:

    vocab_size = state_dict.pop('{VOCAB_SIZE_KEY}')
    sos = state_dict.pop('{SOS_KEY}')

See the epilogue for more details on handling special tokens.
"""

EPILOGUE = f"""\
Special tokens
--------------
This script considers three token types specially: the start-of-sequence token (sos),
the end-of-sequence token (eos), and the unknown/out-of-vocabulary token (unk). Of the
three, only the sos token mut be specified or inferred in order to build the
LookupLanguage model. The unk token can be paired with --on-missing to turn an
open-vocabulary LM into a closed-vocabulary one.

It is possible that some or even all of the special tokens do not have entries in
token2id. For example: a closed-vocabulary task (e.g. character-level ASR) has no need
for an unk token; Connectionist Temporal Classification has no need for an eos token;
and, while all transcripts implicitly start with an sos token, there is never any need
to generate one as the next token. Hence, the ids of these tokens may be arbitrary
integers, including negative values. The LookupLanguageModel excludes such tokens
from its next-token distributions.

It is safest to manually specify all the special token types/labels and ids with the
--*-token and --*-id flags, respectively. If the arguments passed with those flags
exist in token2id, then they must agree with token2id. However, if either the token
or id can be found in token2id, there is no need to specify both values: the other
may be inferred from token2id. If neither value is specified, the following standard
labels are checked for in token2id:

sos: {', '.join(SOS_DEFTS)}
eos: {', '.join(EOS_DEFTS)}
unk: {', '.join(UNK_DEFTS)}

If this fails, we may search for the default labels in the LM, but we may no longer
infer appropriate ids. This doesn't matter for the unk or eos tokens (unless either
--save-unk or --save-eos was specified), but we do need an id for the start-of-sequence
token to initialize the LookupLanguageModel. Thus, in this situation, it would be
necessary to specify at least --sos-id in order to avoid an error.

Missing and extra tokens
------------------------
Except in the case of the special tokens mentioned above, token2id should be an
exhaustive, bijective map which matches the vocabulary of the LM exactly. Nonetheless,
we provide rudimentary means for handling mismatches.

The --on-missing flag handles the case where a token is in the vocabulary file
(token2id) but is missing from the LM. We also include here the case where token2id is
missing contiguous values. The default behaviour ("error") is to raise an
error. Setting it to "zero" will instead assign those tokens a near-zero probability
(exp({ngram_lm.DEFT_EPS_LPROB})). Setting it to "unk" redistributes the probability mass
of the unk token to all the missing tokens, effectively turning an open-vocabulary task
into a closed-vocabulary one.

The --on-extra flag handles when a token can be foud in the LM but not in the vocabulary
file. The default behaviour ("error") is to raise an error. "drop" simply drops the
entry without renormalization. This is fast and approximately correct when the
associated probabilities are already near zero (a warning is issued if not). "prune"
is like "drop", but first redistributes the probability mass evenly over the remaining
tokens. Since it requires renormalization, it is much slower.
"""


class MyDict(dict):
    def __missing__(self, key):
        return key


def main(args=None):
    logging.captureWarnings(True)

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOGUE,
    )
    parser.add_argument("arpa", type=argparse.FileType("rb"), help="Path to arpa file")
    parser.add_argument(
        "token2id", type=argparse.FileType("r"), help="Path to token2id file"
    )
    parser.add_argument("lm_pt", help="Path to save state dict to")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    parser.add_argument(
        "--save-vocab-size",
        action="store_true",
        default=False,
        help=f"Save the vocabulary size in the state dict with key '{VOCAB_SIZE_KEY}'",
    )

    parser.add_argument(
        "--sos-token",
        metavar="TOK",
        default=None,
        help="Token used to demarcate the start of a token sequence. See epilogue",
    )
    parser.add_argument(
        "--sos-id",
        type=np.int64,
        metavar="INT",
        default=None,
        help="Integer id associated with start-of-sequence tokens. See epilogue",
    )
    parser.add_argument(
        "--save-sos",
        action="store_true",
        default=False,
        help="Save the start-of-sequence id in the state dict with the key "
        f"'{SOS_KEY}'",
    )

    parser.add_argument(
        "--eos-token",
        default=None,
        metavar="TOK",
        help="Token used to demarcate the end of a token sequence. See epilogue",
    )
    parser.add_argument(
        "--eos-id",
        type=np.int64,
        metavar="INT",
        default=None,
        help="Integer id associated with end-of-sequence tokens. See epilogue",
    )
    parser.add_argument(
        "--save-eos",
        action="store_true",
        default=False,
        help=f"Save the end-of-sequence id in the state dict with the key '{EOS_KEY}'",
    )

    parser.add_argument(
        "--unk-token",
        default=None,
        metavar="TOK",
        help="Token replacing those missing from token2id. See epilogue",
    )
    parser.add_argument(
        "--unk-id",
        type=np.int64,
        default=None,
        metavar="INT",
        help="Integer id associated with unknown/oov tokens. See epilogue",
    )
    parser.add_argument(
        "--save-unk",
        action="store_true",
        default=False,
        help=f"Save the unknown/oov id in the sate dict with the key '{UNK_KEY}'",
    )

    parser.add_argument(
        "--on-missing",
        choices=["error", "zero", "unk"],
        default="error",
        help="What to do if a token is in the vocab but not in the lm. See epilogue",
    )
    parser.add_argument(
        "--on-extra",
        choices=["error", "drop", "prune"],
        default="error",
        help="What to do if a token is in the lm but not in the vocab. See epilogue",
    )

    options = parser.parse_args(args)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO if options.verbose else logging.WARNING,
    )

    logging.info("Parsing token2id...")
    token2id, id2token = dict(), dict()
    max_id, min_id = -np.inf, np.inf
    for idx, line in enumerate(options.token2id):
        emsg = f"'{options.token2id.name}' line {idx+1}:"
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
        raise ValueError(f"'{options.token2id.name}' was empty")
    logging.info(f"Parsed token2id with {len(token2id)} entries")
    assert np.isfinite(max_id) and np.isfinite(min_id)
    # use the smallest integer we can for ids to save memory
    id_type = None
    for id_type in (np.int8, np.int16, np.int32, np.int64, int):
        if np.iinfo(id_type).min < min_id + 1 and np.iinfo(id_type).max > max_id + 1:
            break
    assert id_type is not None
    id2token = MyDict((id_type(k), v) for (k, v) in id2token.items())
    token2id = MyDict((k, id_type(v)) for (k, v) in token2id.items())

    logging.info("Parsing lm...")
    if options.arpa.peek()[:2] == b"\x1f\x8b":
        options.arpa = gzip.open(options.arpa, mode="rt")
    else:
        options.arpa = io.TextIOWrapper(options.arpa)
    # why try to convert tokens to ids now if we're going to do so again later?
    # it's (probably) cheaper to store an integer id than a string.
    # Also, we don't yet convert to log base e in case we are doing pruning later
    prob_dicts = parse_arpa_lm(
        options.arpa, token2id, False, np.float32, logging.getLogger()
    )
    logging.info("Parsed lm")

    max_order = len(prob_dicts)
    token2id_vocab = set(token2id)
    lm_vocab = set(id2token.get(x, x) for x in prob_dicts[0])

    sos_token, sos_id = options.sos_token, options.sos_id
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
                matches = set(SOS_DEFTS) & lm_vocab
                if len(matches) > 1:
                    raise ValueError(
                        f"Found multiple matching default sos ids in lm: {matches}"
                    )
                elif len(matches):
                    sos_token = matches.pop()
                    logging.info(f"sos token inferred from lm: '{sos_token}'")
                    if sos_id is not None:
                        logging.info(
                            f"--sos-id was set to {sos_id}, so mapping to that"
                        )
                else:
                    raise ValueError("Could not infer sos token")
    if sos_id is None:
        if sos_token in token2id:
            sos_id = token2id[sos_token]
            logging.info(
                f"sos id ({sos_id}) inferred from sos token ('{sos_token}') and "
                "token2id"
            )
        else:
            raise ValueError("Could not infer sos id")
    assert sos_token is not None and sos_id is not None
    if (
        token2id.get(sos_token, sos_id) != sos_id
        or id2token.get(sos_id, sos_token) != sos_token
    ):
        raise ValueError(
            f"sos token ('{sos_token}') does not match id ({sos_id}) listed in token2id"
        )
    token2id_vocab.add(sos_token)
    token2id[sos_token] = sos_id
    id2token[sos_id] = sos_token

    eos_token, eos_id = options.eos_token, options.eos_id
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
                matches = set(EOS_DEFTS) & lm_vocab
                if len(matches) > 1:
                    raise ValueError(
                        f"Found multiple matching default eos ids in lm: {matches}"
                    )
                elif len(matches):
                    eos_token = matches.pop()
                    logging.info(f"eos token inferred from lm: '{eos_token}'")
                else:
                    logging.warning("Could not infer eos token (but might not exist)")
    if eos_id is None:
        if eos_token in token2id:
            eos_id = token2id[eos_token]
            logging.info(
                f"eos id ({eos_id}) inferred from eos token ('{eos_token}') and "
                "token2id"
            )
        elif options.save_eos:
            raise ValueError("Could not infer eos id and --save-eos was specified")
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
                matches = set(UNK_DEFTS) & lm_vocab
                if len(matches) > 1:
                    raise ValueError(
                        f"Found multiple matching default unk ids in lm: {matches}"
                    )
                elif len(matches):
                    unk_token = matches.pop()
                    logging.info(f"unk token inferred from lm: '{unk_token}'")
                    if unk_id is not None:
                        logging.info(
                            f"--unk-id was set to {unk_id}, so mapping to that"
                        )
                else:
                    logging.warning("Could not infer unk token (but might not exist)")
    if unk_id is None:
        if unk_token in token2id:
            unk_id = token2id[unk_token]
            logging.info(
                f"unk id ({unk_id}) inferred from unk token ('{unk_token}') and "
                "token2id"
            )
        elif options.save_unk:
            raise ValueError("Could not infer unk id and --save-unk was specified")
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

    # extra goes first as it simplifies the case when the vocab needs to be extended
    # with missing values
    extra = lm_vocab - token2id_vocab
    if extra:
        if options.on_extra == "error":
            raise ValueError(f"Extra tokens {extra} in lm")
        elif options.on_extra == "prune":
            logging.warning(f"extra tokens {extra} will be pruned")
            lm = ngram_lm.BackoffNGramLM(
                prob_dicts, sos_token, eos_token, unk_token, True
            )
            lm.prune_by_name(extra)
            prob_dicts = lm.to_prob_list()
            del lm
        else:
            logging.warning(f"Extra tokens {extra} will be dropped")
        # "drop" and the end of "prune" are handled in the token2id map below
    del extra

    missing = token2id_vocab - lm_vocab
    if missing:
        if options.on_missing == "zero":
            for token in missing:
                logging.warning(f"Missing token '{token}'. Assigning near-zero prob")
                prob_dicts[0][token] = ngram_lm.DEFT_EPS_LPROB
        elif options.on_missing == "unk":
            raise NotImplementedError(
                "missing tokens but --on-missing unk not yet implemented"
            )
        else:  # error
            raise ValueError(f"Missing tokens {missing} in lm")
    del missing

    vocab_size = max(token2id[token] for token in token2id_vocab) + 1
    if vocab_size <= len(token2id_vocab) - int(sos_id < 0):
        logging.info(f"vocab_size is {vocab_size}")
    else:
        logging.warning(
            f"the maximum token id + 1 ({vocab_size}) is greater than the number of "
            f"vocabulary items ({len(token2id_vocab)}), possibly excluding a negative "
            "sos. This means ids were not contiguous. The resulting vocabulary will be "
            "inflated with tokens with probability 0"
        )

    new_prob_dicts = []
    norm = np.float32(np.log10(np.e))
    for n, prob_dict in enumerate(prob_dicts):
        logging.info(f"Mapping tokens to ids and converting to base e (order {n + 1})")
        new_prob_dict = dict()
        while len(prob_dict):
            ngram, lprob = prob_dict.popitem()
            if n == max_order - 1:
                lprob = lprob / norm
            else:
                lprob = (lprob[0] / norm, lprob[1] / norm)
            if n:
                ngram = tuple(token2id[tok] for tok in ngram)
                if all(isinstance(id_, (int, np.integer)) for id_ in ngram):
                    new_prob_dict[ngram] = lprob
                else:
                    assert options.on_extra != "error"
            else:
                ngram = token2id[ngram]
                if isinstance(ngram, (int, np.integer)):
                    new_prob_dict[ngram] = lprob
                else:
                    assert options.on_extra != "error"
        new_prob_dicts.append(new_prob_dict)
    prob_dicts = new_prob_dicts
    del new_prob_dicts

    sos_lprob = prob_dicts[0][sos_id]
    sos_lprob = sos_lprob * norm if max_order == 1 else sos_lprob[0] * norm
    if sos_lprob > ngram_lm.DEFT_EPS_LPROB and 0 <= sos_id < vocab_size:
        logging.warning(
            f"sos ('{sos_token}') has non-negligible unigram probability ({sos_lprob}) "
            "and is in the output vocabulary. You probably don't want this (why "
            "predict the start of sequence?). It is recommended that you prune it "
            "and/or remove it from the vocabulary by setting --sos-id to something "
            "less than 0"
        )

    logging.info("Creating state dictionary")
    lm = LookupLanguageModel(
        vocab_size, int(sos_id), prob_dicts, True, logging.getLogger()
    )
    del prob_dicts
    state_dict = lm.state_dict()

    if options.save_vocab_size:
        state_dict[VOCAB_SIZE_KEY] = int(vocab_size)

    if options.save_sos:
        state_dict[SOS_KEY] = int(sos_id)

    if options.save_eos:
        state_dict[EOS_KEY] = int(eos_id)

    if options.save_unk:
        state_dict[UNK_KEY] = int(unk_id)

    logging.info("Saving state dictionary")
    torch.save(state_dict, options.lm_pt)
    logging.info("Done")


if __name__ == "__main__":
    sys.exit(main())
