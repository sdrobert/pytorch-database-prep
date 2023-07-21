#! /usr/bin/env python

import os
import re
import argparse

from collections import Counter
from itertools import chain
from heapq import heappush, heappop
from tempfile import TemporaryDirectory

import ngram_lm


def pos_int(val: str) -> int:
    val = int(val)
    if val < 1:
        raise ValueError(f"{val} is not positive")
    return val


def is_type(val: str) -> str:
    if not val:
        raise ValueError(f"is empty")
    if len(val.split()) != 1:
        raise ValueError(f"'{val}' contains whitespace")
    return val


def main(args=None):
    parser = argparse.ArgumentParser(
        "Iterate through raw text, constructing a vocabulary of types, and writing "
        "that vocabulary out, one type per line"
    )
    parser.add_argument(
        "text",
        metavar="IN",
        type=argparse.FileType("r"),
        nargs="?",
        default=argparse.FileType("r")("-"),
        help="File to read. Defaults to stdin",
    )
    parser.add_argument(
        "vocab",
        metavar="OUT",
        type=argparse.FileType("w"),
        nargs="?",
        default=argparse.FileType("w")("-"),
        help="File to write. Defaults to stdout",
    )
    parser.add_argument(
        "--word-delim-expr",
        metavar="REGEX",
        type=re.compile,
        default=ngram_lm.DEFT_WORD_DELIM_EXPR,
        help="Delimiter to split sentences into tokens",
    )
    parser.add_argument(
        "--to-case",
        choices=("upper", "lower"),
        default=None,
        help="Convert all types to either upper or lower case. Not applied to "
        "--unk or --ensure",
    )
    parser.add_argument(
        "--write-counts",
        action="store_true",
        default=False,
        help="Write total count of unique tokens in text after their type",
    )
    parser.add_argument(
        "--max-size",
        type=pos_int,
        default=None,
        help="If set, ensures the vocabulary grows no larger than this size. "
        "The least frequent types are pruned first (excluding those passed by "
        "--ensure). In this case, the unk type (--unk will be added to the "
        "vocabulary ",
    )
    parser.add_argument(
        "--unk",
        type=is_type,
        default="<unk>",
        help="The unknown type. Added to the vocabulary when the number of types "
        "exceeds --max-size",
    )
    parser.add_argument(
        "--ensure",
        type=is_type,
        nargs="+",
        default=[],
        help="Types to always add to the vocabulary, whether or not they exist in the "
        "text file",
    )
    parser.add_argument(
        "--sort", action="store_true", default=False, help="Sort vocab before saving"
    )
    parser.add_argument(
        "--count-dir",
        "-T",
        nargs="?",
        const=1,
        type=os.abspath,
        default=None,
        help="Store rare vocab items on disk to reduce memory pressure while "
        "collecting. If no arg, constructs temporary directory. If arg, saves "
        "cache files in dir with prefix 'counts'",
    )

    options = parser.parse_args(args)

    ensure = set(options.ensure)

    unk = options.unk
    max_size = float("inf") if options.max_size is None else options.max_size

    sents = (x.rstrip("\n") for x in options.text)
    sents = ngram_lm.titer_to_siter(
        sents, options.word_delim_expr, options.to_case, True
    )
    tokens = chain.from_iterable(sents)

    if options.count_dir is None:
        type2count = Counter(tokens)
    else:
        if options.count_dir == 1:
            count_dir_ = TemporaryDirectory()
            count_dir = os.fspath(count_dir_.name)
        else:
            count_dir = options.count_dir
        os.makedirs(count_dir, exist_ok=True)
        count_prefix = os.path.join(count_dir, "counts")
        type2count = ngram_lm.open_count_dict(count_prefix, "n")
        for token in tokens:
            type2count[token] = type2count.get(token, 0) + 1

    if len(type2count) > max_size:
        ensure.add(unk)
        if max_size < len(ensure):
            raise ValueError(
                f"max_size {max_size} too small to contain --ensure and --unk"
            )

        type2count_ = dict((e, 0) for e in ensure)
        max_heap_size = max_size - len(type2count_)
        count_type_heap = []
        while len(type2count):
            type_, count = type2count.popitem()
            if type_ in ensure:
                type2count_[type_] += count
            else:
                heappush(count_type_heap, (-count, type_))
                if len(count_type_heap) > max_heap_size:
                    count = -heappop(count_type_heap)[0]
                    assert len(count_type_heap) == max_heap_size
                    type2count_[unk] += count

        while len(count_type_heap):
            ncount, type_ = count_type_heap.pop()
            type2count_[type_] = -ncount

        assert len(type2count_) == max_size
        type2count = type2count_

    type2count = type2count.items()
    if options.sort:
        type2count = sorted(type2count)

    for type_, count in type2count:
        options.vocab.write(type_)
        if options.write_counts:
            options.vocab.write(f" {count}")
        options.vocab.write("\n")


if __name__ == "__main__":
    main()
