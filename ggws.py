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

'''Command-line interface to prepare the Gigaword Summarization Corpus'''

# we now assume python3. Py2.7 has reached EOL. Yay.

import os
import sys
import argparse
import locale

from collections import Counter

from common import mkdir

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2020 Sean Robertson"


locale.setlocale(locale.LC_ALL, 'C')


def preamble(options):

    dir_ = os.path.join(options.data_root, 'local', 'data')
    src = os.path.join(options.ggws_root, 'org_data')
    word2freq_txt = os.path.join(dir_, 'word2freq.txt')

    if not os.path.exists(src):
        raise ValueError(
            '"{}" does not exist. Are you sure "{}" points to the UniLM '
            'version of the Gigaword Summarization Corpus?'.format(
                src, options.ggws_root))

    mkdir(dir_)

    word2freq = Counter()

    for fn_nosuf in ('train', 'dev', 'test'):
        is_train = fn_nosuf == 'train'
        for suffix in ('.src.txt', '.tgt.txt'):
            with \
                    open(
                        os.path.join(src, fn_nosuf + suffix),
                        encoding='utf-8') as in_, \
                    open(os.path.join(dir_, fn_nosuf + suffix), 'w') as out_:
                for line in in_:
                    # the non-breaking space sometimes comes up instead of a
                    # space. We also replace the underscore with its html code
                    # so that it doesn't muck with our use of underscore in
                    # subwords (the database hasn't been entirely sanitized of
                    # these anyway)
                    line = line.replace(u'\u00A0', ' ').replace('_', '&#95;')
                    # replace 'UNK' in the test set with '<unk>' to be
                    # consistent with the training set. I prefer '<unk>'
                    # because it's quite clearly a control character
                    tokens = line.strip().split(' ')
                    if is_train:
                        word2freq.update(tokens)
                    else:
                        tokens = ['<unk>' if x == 'UNK' else x for x in tokens]
                    out_.write(' '.join(tokens))
                    out_.write('\n')

    with open(word2freq_txt, 'w') as file_:
        for word, freq in sorted(word2freq.items()):
            file_.write('{} {}\n'.format(word, freq))


def build_preamble_parser(subparsers):
    parser = subparsers.add_parser(
        'preamble',
        help='Do all pre-initializaiton setup. Needs to be done only once'
    )
    parser.add_argument(
        'ggws_root', type=os.path.abspath,
        help='Location of the GGWS data directory, downloaded from the UniLM '
        'project. Contains files like "dev.src" and "train.tgt"'
    )


def build_parser():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        'data_root', type=os.path.abspath,
        help='The root directory under which to store data. Typically '
        '``data/``'
    )
    subparsers = parser.add_subparsers(
        title='commands', required=True, dest='command')
    build_preamble_parser(subparsers)
    return parser


def main(args=None):
    '''Prepare GGWS data for pytorch training'''

    parser = build_parser()
    options = parser.parse_args(args)

    if options.command == 'preamble':
        preamble(options)


if __name__ == '__main__':
    sys.exit(main())
