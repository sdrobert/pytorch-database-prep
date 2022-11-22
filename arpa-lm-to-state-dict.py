#! /usr/bin/env python

import io
import sys
import argparse
import gzip

import torch

from pydrobert.torch.data import parse_arpa_lm
from pydrobert.torch.modules import LookupLanguageModel

SOS_DEFTS = ("<s>", "<S>")
EOS_DEFTS = ("</s>", "</S>")


def main(args=None):
    """\
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

    vocab_size = state_dict.pop('vocab_size')
    sos = state_dict.pop('sos')

"""
    parser = argparse.ArgumentParser(
        description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("arpa", type=argparse.FileType("rb"), help="Path to arpa file")
    parser.add_argument(
        "token2id", type=argparse.FileType("r"), help="Path to token2id file"
    )
    parser.add_argument(
        "lm_pt", type=argparse.FileType("wb"), help="Path to save state dict to"
    )
    parser.add_argument(
        "--sos-token",
        default=None,
        help=f"Token used to indicate a start-of-sequence. The set '{SOS_DEFTS}' is "
        "checked if the token is needed to determine the sos id",
    )
    parser.add_argument(
        "--sos-id",
        type=int,
        default=None,
        help="Integer id associated with start-of-sequence tokens. If unspecified and "
        "the sos token cannot be found in token2id, it is assigned the id one less "
        "than the greatest id in the mapping",
    )
    parser.add_argument(
        "--eos-token",
        default=None,
        help=f"Token used to indicate an end-of-sequence. The set '{EOS_DEFTS}' is "
        "checked if the token is needed to determine the eos id.",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=None,
        help="Integer id associated with end-of-sequence tokens. If unspecified and "
        "the eos token cannot be found in token2id, it is assigned the id one greater "
        "than the greatest id in the mapping",
    )
    parser.add_argument(
        "--remove-eos",
        action="store_true",
        default=False,
        help="Delete any occurrence of the eos from the language model. Does not "
        "renormalize the probabilities, but can affect the vocabulary size",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="The size of the vocabulary. Defaults to one greater than any id from "
        "the unigram list (excluding eos if --remove-eos was set)",
    )
    parser.add_argument(
        "--save-vocab-size",
        action="store_true",
        default=False,
        help="Save the vocabulary size in the state dict with key 'vocab_size'",
    )
    parser.add_argument(
        "--save-sos",
        action="store_true",
        default=False,
        help="Save the start-of-sequence id (--sos-id) in the state dict with the key "
        "'sos'",
    )

    options = parser.parse_args(args)

    token2id = dict()
    id2token = dict()
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
        token2id[token] = id_
        id2token[id_] = token
    if not token2id:
        raise ValueError(f"'{options.token2id.name}' was empty")

    if options.sos_id is None:
        if options.sos_token is None:
            for candidate in SOS_DEFTS:
                options.sos_id = token2id.get(candidate, None)
                if options.sos_id is not None:
                    options.sos_token = candidate
                    break
        else:
            options.sos_id = token2id.get(options.sos_token, None)
        if options.sos_id is None:
            options.sos_id = min(id2token) - 1
    if options.sos_token is None:
        for candidate in SOS_DEFTS:
            token2id[candidate] = options.sos_id
    elif token2id.setdefault(options.sos_token, options.sos_id) != options.sos_id:
        raise ValueError(
            f"Entry for --sos-token '{options.sos_token}' in token2id does not match "
            f"--sos-id '{options.sos_id}'"
        )
    if id2token.setdefault(options.sos_id, options.sos_token) != options.sos_token:
        raise ValueError(
            f"Entry for --sos-id '{options.sos_id}' in token2id does not match "
            f"--sos-token '{options.sos_token}'"
        )

    if options.eos_id is None:
        if options.eos_token is None:
            for candidate in EOS_DEFTS:
                options.eos_id = token2id.get(candidate, None)
                if options.eos_id is not None:
                    options.eos_token = candidate
                    break
        else:
            options.eos_id = token2id.get(options.eos_token, None)
        if options.eos_id is None:
            options.eos_id = max(id2token) + 1
    if options.eos_token is None:
        for candidate in EOS_DEFTS:
            token2id[candidate] = options.eos_id
    elif token2id.setdefault(options.eos_token, options.eos_id) != options.eos_id:
        raise ValueError(
            f"Entry for --eos-token '{options.eos_token}' in token2id does not match "
            f"--eos-id '{options.eos_id}'"
        )
    if id2token.setdefault(options.eos_id, options.eos_token) != options.eos_token:
        raise ValueError(
            f"Entry for --eos-id '{options.eos_id}' in token2id does not match "
            f"--eos-token '{options.eos_token}'"
        )

    if options.arpa.name.endswith(".gz"):
        options.arpa = gzip.GzipFile(mode="r", fileobj=options.arpa)
    options.arpa = io.TextIOWrapper(options.arpa)

    prob_list = parse_arpa_lm(options.arpa, token2id)

    if options.remove_eos:
        if options.eos_id in prob_list[0]:
            prob_list[0].pop(options.eos_id)
        for ngram in prob_list[1:]:
            for context in list(ngram):
                if options.eos_id in context:
                    ngram.pop(context)

    if options.vocab_size is None:
        options.vocab_size = max(prob_list[0]) + 1

    lm = LookupLanguageModel(options.vocab_size, options.sos_id, prob_list)
    state_dict = lm.state_dict()

    if options.save_vocab_size:
        state_dict["vocab_size"] = options.vocab_size

    if options.save_sos:
        state_dict["sos"] = options.sos_id

    torch.save(state_dict, options.lm_pt)


if __name__ == "__main__":
    sys.exit(main())
