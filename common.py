# Copyright 2019 Sean Robertson
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

'''Common utilities for multiple setup scripts'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import locale
import stat

try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'glob',
    'mkdir',
    'sort',
    'cat',
    'pipe_to',
    'wc_l',
    'chmod_u_plus_w',
    'uniq',
]


locale.setlocale(locale.LC_ALL, 'C')


def glob(root, pattern):
    path = pathlib.Path(root)
    for match in path.glob(pattern):
        yield match.as_posix()


def mkdir(*dirs):
    '''make a directory structures, ok if exists'''
    for dir_ in dirs:
        if not os.path.isdir(dir_):
            os.makedirs(dir_)


def sort(in_stream):
    '''yields sorted input stream'''
    for line in sorted(in_stream):
        yield line


def uniq(in_stream):
    '''yields input stream with duplicate subssequent lines suppressed'''
    last = None
    for line in in_stream:
        if line != last:
            yield line
            last = line


def cat(*paths):
    '''yields lines of files in order'''
    for path in paths:
        if isinstance(path, str):
            with open(path) as f:
                for line in f:
                    yield line.rstrip()
        else:  # pretend this is stdin
            for line in path:
                yield line.rstrip()


def pipe_to(in_stream, file_, append=False):
    '''Write in_stream to file'''
    if isinstance(file_, str):
        with open(file_, 'a' if append else 'w') as f:
            pipe_to(in_stream, f)
    else:
        for line in in_stream:
            # unix-style line ends
            print(line, file=file_, end='\n')


def wc_l(inp):
    '''Output number of lines in file or stream'''
    if isinstance(inp, str):
        with open(inp) as f:
            v = wc_l(f)
        return '{} {}'.format(v, inp)
    else:
        return sum(1 for x in inp)


def chmod_u_plus_w(*files):
    for file_ in files:
        os.chmod(file_, os.stat(file_).st_mode | stat.S_IWUSR)
