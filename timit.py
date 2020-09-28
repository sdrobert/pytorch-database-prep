#! /usr/bin/env python

# Copyright 2020 Sean Robertson
#
# Adapted from kaldi/egs/timit/s5/local/timit_data_prep.sh (and associated resource
# files conf/{{dev,test}_spk.lst,phones.60-48-39.map}:
#
# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
#           2014   Brno University of Technology (Author: Karel Vesely)
#
# phones.map has been slightly adjusted from phones.60-48-39.map
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
import locale

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2020 Sean Robertson"


locale.setlocale(locale.LC_ALL, "C")


def preamble(options):
    pass


def main(args=None):
    pass


if __name__ == "__main__":
    sys.exit(main())
