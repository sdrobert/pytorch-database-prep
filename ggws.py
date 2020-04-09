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
import warnings

from collections import Counter
from shutil import copy as copy_paths

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


def init_word(options):

    local_dir = os.path.join(options.data_root, 'local')
    local_data_dir = os.path.join(local_dir, 'data')
    word2freq_txt = os.path.join(local_data_dir, 'word2freq.txt')
    vocab_size = 1000 if options.vocab_size is None else options.vocab_size
    prune_thresh = 0 if options.prune_thresh is None else options.prune_thresh
    if options.config_subdir is None:
        config_dir = os.path.join(local_dir, 'word')
        if options.vocab_size is not None:
            config_dir += '{}k'.format(vocab_size)
        elif options.prune_thresh is not None:
            config_dir += '{}p'.format(prune_thresh)
    else:
        config_dir = os.path.join(local_dir, options.config_subdir)
    token2id_txt = os.path.join(config_dir, 'token2id.txt')
    id2token_txt = os.path.join(config_dir, 'id2token.txt')
    train_oovs_txt = os.path.join(config_dir, 'train_oovs.txt')

    with open(word2freq_txt) as file_:
        freq_word = (line.strip().split() for line in file_)
        freq_word = ((int(x[1]), x[0]) for x in freq_word)
        freq_word = sorted(freq_word, reverse=True)

    oovs = {x[1] for x in freq_word[vocab_size * 1000:]}
    freq_word = freq_word[:vocab_size * 1000]
    while freq_word and freq_word[-1][0] <= prune_thresh:
        oovs.add(freq_word.pop(-1)[1])

    if not freq_word:
        warnings.warn(
            'No words are left after pruning + vocab size. All tokens will be '
            '<unk> in training')
    vocab = set(x[1] for x in freq_word) | {'<unk>', '<s>', '</s>'}
    vocab = sorted(vocab)
    del freq_word

    mkdir(config_dir)

    with open(token2id_txt, 'w') as t2id, open(id2token_txt, 'w') as id2t:
        for i, v in enumerate(vocab):
            t2id.write('{} {}\n'.format(v, i))
            id2t.write('{} {}\n'.format(i, v))

    to_copy = {
        'train.src.txt', 'train.tgt.txt', 'dev.src.txt', 'dev.tgt.txt',
        'test.src.txt', 'test.tgt.txt'
    }
    for x in to_copy:
        copy_paths(
            os.path.join(local_data_dir, x),
            os.path.join(config_dir, x)
        )

    # determine the OOVs in the training partition. Primarily for diagnostic
    # purposes
    oovs -= {'<unk>'}
    oovs = sorted(oovs)
    with open(train_oovs_txt, 'w') as file_:
        file_.write('\n'.join(oovs))
        file_.write('\n')


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


def build_init_word_parser(subparsers):
    parser = subparsers.add_parser(
        'init_word',
        help='Perform setup common to full word-level parsing. '
        'Needs to be done only once for a specific vocabulary size. '
        'Preceded by "preamble" command.'
    )
    parser.add_argument(
        '--config-subdir', default=None,
        help='Name of sub directory in data/local/ under which to store '
        'setup specific to this vocabulary size. Defaults to '
        '``word(<vocab_size>k|<prune_thresh>p|)``, depending on whether the '
        'full vocabulary was used (~124k words), the top ``<vocab_size>k`` '
        'words in terms of frequency, or the words remaining after pruning '
        'those with less than or equal to ``<prune_thresh>`` tokens.'
    )

    vocab_group = parser.add_mutually_exclusive_group()
    vocab_group.add_argument(
        '--vocab-size', type=int, default=None,
        help='Limit the vocabulary size to this many words (in thousands). '
        'The vocabulary will be chosen from the most frequent word types in '
        'the training set.'
    )
    vocab_group.add_argument(
        '--prune-thresh', type=int, default=None,
        help='Limit the vocabulary size by pruning all word types with equal '
        'or fewer than this number of tokens in the training set.'
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
    build_init_word_parser(subparsers)
    return parser


def main(args=None):
    '''Prepare GGWS data for pytorch training'''

    parser = build_parser()
    options = parser.parse_args(args)

    if options.command == 'preamble':
        preamble(options)
    elif options.command == 'init_word':
        init_word(options)


if __name__ == '__main__':
    sys.exit(main())
