#! /usr/bin/env python

import sys
import argparse
import warnings

import jiwer

from pydrobert.torch.data import read_trn_iter


def main(args=None):
    """\
    Determine error rates between two or more trn files

An error rate measures the difference between reference (gold-standard) and
hypothesis (machine-generated) transcriptions by the number of single-token
insertions, deletions, and substitutions necessary to convert the hypothesis
transcription into the reference one.

A "trn" file is the standard transcription file without alignment information used
in the sclite (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm)
toolkit. It has the format

    here is a transcription (utterance_a)
    here is another (utterance_b)

WARNING! this command uses jiwer (https://github.com/jitsi/jiwer) as a backend,
which assumes a uniform cost for instertions, deletions, and substitutions. This is
not suited to certain corpora. Consult the corpus-specific page on the wiki
(https://github.com/sdrobert/pytorch-database-prep/wiki) for more details."""

    parser = argparse.ArgumentParser(
        description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "ref_file", type=argparse.FileType("r"), help="The reference trn file"
    )
    parser.add_argument(
        "hyp_files",
        nargs="+",
        type=argparse.FileType("r"),
        help="One or more hypothesis trn files",
    )
    parser.add_argument(
        "--suppress-warning",
        action="store_true",
        default=False,
        help="Suppress the warning about the backend",
    )
    options = parser.parse_args(args)

    if not options.suppress_warning:
        warnings.warn(
            "Not all corpora compute error rates the same way. Look at this command's "
            "documentation. To suppress this warning, use the flag '--suppress-warning'"
        )

    ref_dict = dict(read_trn_iter(options.ref_file, not options.suppress_warning))
    rname = options.ref_file.name
    print(f"ref '{rname}'")
    keys = sorted(ref_dict)
    ref = [" ".join(ref_dict[x]) for x in keys]
    del ref_dict

    best_name = None
    best_er = float("inf")
    for hyp_file in options.hyp_files:
        hname = hyp_file.name
        hyp_dict = dict(read_trn_iter(hyp_file, not options.suppress_warning))
        if sorted(hyp_dict) != keys:
            keys_, keys = set(hyp_dict), set(keys)
            print(
                f"ref and hyp file '{hname}' have different utterances!",
                file=sys.stderr,
            )
            diff = sorted(keys - keys_)
            if diff:
                print(f"Missing from hyp: " + " ".join(diff), file=sys.stderr)
            diff = sorted(keys - keys_)
            if diff:
                print(f"Missing from ref: " + " ".join(diff), file=sys.stderr)
            return 1
        hyp = [" ".join(hyp_dict[x]) for x in keys]
        er = jiwer.wer(ref, hyp)
        print(f"hyp '{hname}': {er:.1%}")
        if er < best_er:
            best_name = hname
            best_er = er
    print(f"best hyp '{best_name}': {best_er:.1%}")


if __name__ == "__main__":
    sys.exit(main())
