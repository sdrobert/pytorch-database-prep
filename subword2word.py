#! /usr/bin/env python

# Copyright 2021 Sean Robertson
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
import sys

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2021 Sean Robertson"


def main(args=None):

    parser = argparse.ArgumentParser(
        description="Convert a subword or character-level transcription to a "
        "word-level one. Supposed to be used on hypothesis transcriptions; "
        'alternates (e.g. "{ foo / bar / @ }") are not permitted'
    )
    parser.add_argument(
        "subword_trn",
        type=argparse.FileType("r"),
        help="A subword or character trn file to read",
    )
    parser.add_argument(
        "word_trn", type=argparse.FileType("w"), help="A word trn file to write"
    )
    parser.add_argument(
        "--space-char",
        default="_",
        help="The character used in the character-level transcript that "
        "substitutes spaces",
    )
    options = parser.parse_args(args)

    for line in options.subword_trn:
        trans, utt = line.strip().rsplit(" ", maxsplit=1)
        trans = (
            trans.replace(" ", "")
            .replace(options.space_char, " ")
            .replace("  ", " ")
            .strip()
        )
        options.word_trn.write(trans)
        options.word_trn.write(" ")
        options.word_trn.write(utt)
        options.word_trn.write("\n")


if __name__ == "__main__":
    sys.exit(main())
