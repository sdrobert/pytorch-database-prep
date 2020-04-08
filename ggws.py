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

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2020 Sean Robertson"


def preamble(options):
    pass


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
