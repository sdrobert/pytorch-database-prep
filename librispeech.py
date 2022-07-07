#! /usr/bin/env python

# Copyright 2022 Sean Robertson
#
# Adapted from kaldi/egs/librispeech/s5/local/
#                                   {data_prep,download_and_untar,download_lm}.sh
#
# Copyright 2014 Vassil Panayotov
#                Johns Hopkins University (author: Daniel Povey)
#           2021 Xuechen Liu
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
import os
import sys
import hashlib
import urllib.parse
import requests
import shutil
import tarfile
import glob

AM_FNAMES = (
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)

ADDITIONAL_FILES = (
    "BOOKS.TXT",
    "CHAPTERS.TXT",
    "LICENSE.TXT",
    "README.TXT",
    "SPEAKERS.TXT",
)

LM_FILES = (
    "3-gram.arpa.gz",
    "3-gram.pruned.1e-7.arpa.gz",
    "3-gram.pruned.3e-7.arpa.gz",
    "4-gram.arpa.gz",
    "librispeech-lexicon.txt",
    "librispeech-lm-norm.txt.gz",
)

FILE2MD5 = {
    "3-gram.arpa.gz": "74cbb837e582378c26c602e1c8cbcb9c",
    "3-gram.pruned.1e-7.arpa.gz": "9a0f1325bcaf3a0f7b7be4be10823c2a",
    "3-gram.pruned.3e-7.arpa.gz": "8f67ec741827742eee05799ab7605b50",
    "4-gram.arpa.gz": "6abdcd5b055bd482e4f9defe91d40408",
    "dev-clean.tar.gz": "42e2234ba48799c1f50f24a7926300a1",
    "dev-other.tar.gz": "c8d0bcc9cca99d4f8b62fcc847357931",
    "intro-disclaimers.tar.gz": "92ba57a9611a70fd7d34b73249ae48cf",
    "librispeech-lexicon.txt": "28f72663f6bfed7cd346283c064c315b",
    "librispeech-lm-norm.txt.gz": "c83c64c726a1aedfe65f80aa311de402",
    "librispeech-vocab.txt": "bf3cc7a50112831ed8b0afa40cae96a5",
    "original-books.tar.gz": "9da96b465573c8d1ee1d5ad3d01c08e3",
    "original-mp3.tar.gz": "7e14b6df14f1c04a852a50ba5f7be915",
    "raw-metadata.tar.gz": "25eced105e10f4081585af89b8d27cd2",
    "test-clean.tar.gz": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other.tar.gz": "fb5a50374b501bb3bac4815ee91d3135",
    "train-clean-100.tar.gz": "2a93770f6d5c6c964bc36631d331a522",
    "train-clean-360.tar.gz": "c0e676e450a7ff2f54aeade5171606fa",
    "train-other-500.tar.gz": "d1a0fd59409feb2c614ce4d30c387708",
}

DEFT_HOST = "https://www.openslr.org"
REGION_MIRRORS = {
    "us": "https://us.openslr.org",
    "eu": "https://openslr.elda.org",
    "cn": "https://openslr.magicdatatech.com",
}

AM_URL_PATH = "resources/12"
LM_URL_PATH = "resources/11"

CHUNK_SIZE = 10240


# https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
def get_file_md5(path, chunk_size):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def download_file(url, path, chunk_size):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            if path.endswith(".txt"):
                for chunk in r.iter_content(chunk_size):
                    f.write(chunk)
            else:
                shutil.copyfileobj(r.raw, f, chunk_size)


def download(options):

    if options.download_dir is None:
        download_dir = os.path.join(options.data_root, "local", "data")
    else:
        download_dir = options.download_dir

    os.makedirs(download_dir, exist_ok=True)

    if options.mirror_region is not None:
        host = REGION_MIRRORS[options.mirror_region]
    else:
        host = options.host

    chunk_size = options.chunk_size

    for fname in AM_FNAMES:
        subdir = os.path.join(download_dir, fname)
        complete_file = os.path.join(subdir, ".complete")
        if os.path.exists(complete_file) and all(
            os.path.exists(os.path.join(download_dir, x)) for x in ADDITIONAL_FILES
        ):
            print(f"Files appear to have already been extracted to '{subdir}'. Skip")
            continue
        path = os.path.join(download_dir, fname + ".tar.gz")
        if os.path.exists(path) and FILE2MD5[fname + ".tar.gz"] == get_file_md5(
            path, chunk_size
        ):
            print(f"'{path}' already exists and has correct md5sum. Not redownloading")
        else:
            url = urllib.parse.urljoin(host + "/", f"{AM_URL_PATH}/{fname}.tar.gz")
            print(
                f"'{path}' either does not exist or is corrupted. Downloading '{url}'"
            )
            download_file(url, path, chunk_size)
            if FILE2MD5[fname + ".tar.gz"] != get_file_md5(path, chunk_size):
                raise ValueError(f"Downloaded '{path}' does not match md5sum!")
            print("Downloaded")
        os.makedirs(subdir, exist_ok=True)
        print(f"Extracting '{path}")
        with tarfile.open(path, "r|gz", bufsize=chunk_size) as f:
            for member in f:
                if member.name.endswith(".TXT"):
                    idx = member.name.find("LibriSpeech/")
                    assert idx > -1
                    member.name = member.name[idx + 12 :]
                    f.extract(member, download_dir)
                    continue
                idx = member.name.find(fname)
                if idx == -1 or member.isdir():
                    continue
                member.name = member.name[idx + len(fname) + 1 :]
                member_path = os.path.join(subdir, member.name)
                if (
                    os.path.exists(member_path)
                    and os.path.getsize(member_path) == member.size
                ):
                    print(f"Skipping '{member_path}'. Exists and correct size")
                    continue
                f.extract(member, subdir)
        print("Extracted")
        with open(os.path.join(subdir, ".complete"), "w") as f:
            pass
        if not options.dirty:
            print(f"Deleting '{path}'")
            os.remove(path)

    lmdir = os.path.join(download_dir, "lm")
    os.makedirs(lmdir, exist_ok=True)

    for file_ in ("librispeech-vocab.txt",) + (LM_FILES if options.lm else tuple()):
        path = os.path.join(lmdir, file_)
        if os.path.exists(path) and FILE2MD5[file_] == get_file_md5(path, chunk_size):
            print(f"'{path}' already exists and has correct md5sum. Not redownloading")
        else:
            url = urllib.parse.urljoin(host + "/", f"{LM_URL_PATH}/{file_}")
            print(
                f"'{path}' either does not exist or is corrupted. Downloading '{url}'"
            )
            download_file(url, path, chunk_size)
            if FILE2MD5[file_] != get_file_md5(path, chunk_size):
                raise ValueError(f"Downloaded '{path}' does not match md5sum!")


def find_file(root, file_name):
    paths = glob.glob(f"{glob.escape(root)}/**/{file_name}", recursive=True)
    if not len(paths):
        raise ValueError(f"'{file_name}' not in '{root}'")
    elif len(paths) > 1:
        raise ValueError(f"More than one instance of '{file_name}' exists in '{root}'")
    return paths.pop()


def data_prep(libri_dir, data_dir, reader2gender):
    # FIXME(sdrobert): allow for reader to be speaker, not reader/chapter
    # combo. Kaldi uses the latter for practical reasons, but the former is
    # more accurate.
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "wav.scp"), "w") as wav_scp, open(
        os.path.join(data_dir, "text"), "w"
    ) as trans, open(os.path.join(data_dir, "utt2spk"), "w") as utt2spk, open(
        os.path.join(data_dir, "spk2gender"), "w"
    ) as spk2gender:
        for reader in sorted(os.listdir(libri_dir)):
            reader_dir = os.path.join(libri_dir, reader)

            if not os.path.isdir(reader_dir):
                continue

            try:
                int(reader)
            except ValueError:
                raise ValueError(f"Unexpected subdirectory name '{reader}'")

            reader_gender = reader2gender[reader]

            for chapter in sorted(os.listdir(reader_dir)):
                chapter_dir = os.path.join(reader_dir, chapter)

                if not os.path.isdir(chapter_dir):
                    continue

                try:
                    int(chapter)
                except ValueError:
                    raise ValueError(
                        f"Unexpected chapter-subdirectory name '{chapter}'"
                    )

                for flac in sorted(os.listdir(chapter_dir)):
                    if not flac.endswith(".flac"):
                        continue
                    utt_id = "lbi-" + flac.rsplit(".", 1)[0]
                    wav_scp.write(f"{utt_id} {os.path.join(chapter_dir, flac)}\n")

                chapter_trans = os.path.join(
                    chapter_dir, f"{reader}-{chapter}.trans.txt"
                )

                if not os.path.isfile(chapter_trans):
                    raise ValueError(f"Expected file '{chapter_trans}' to exist")

                with open(chapter_trans) as f:
                    for line in f:
                        trans.write("lbi-" + line)
                        uttid = "lbi-" + line.split(" ", 1)[0]
                        utt2spk.write(f"{uttid} lbi-{reader}-{chapter}\n")

                spk2gender.write(f"lbi-{reader}-{chapter} {reader_gender}\n")


def preamble(options):
    local_dir = os.path.join(options.data_root, "local")
    if options.librispeech_root is None:
        root_dir = os.path.join(local_dir, "data")
        if not os.path.isdir(root_dir):
            raise ValueError(
                f"'{root_dir}' does not exist or is not a directory. If you did not "
                "download librispeech via the 'download' command but manually, specify "
                "the download directory as a command-line argument."
            )
    else:
        root_dir = options.librispeech_root
        if not os.path.isdir(root_dir):
            raise ValueError(f"'{root_dir}' does not exist or is not a directory")

    speakers_txt = find_file(root_dir, "SPEAKERS.TXT")

    for fname in AM_FNAMES:
        libri_dir = find_file(root_dir, fname)
        data_dir = os.path.join(local_dir, fname)
        data_prep(libri_dir, data_dir, speakers_txt)

    vocab_txt = find_file(root_dir, "librispeech-vocab.txt")
    shutil.copy(vocab_txt, os.path.join(local_dir, "librispeech-vocab.txt"))

    for fname in LM_FILES:
        try:
            path = find_file(root_dir, fname)
        except:
            continue
        shutil.copy(path, os.path.join(local_dir, fname))


def build_parser():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "data_root",
        type=os.path.abspath,
        help="The root directory under which to store data. Typically 'data/'",
    )
    subparsers = parser.add_subparsers(title="commands", required=True, dest="command")
    build_download_parser(subparsers)
    build_preamble_parser(subparsers)

    return parser


def build_download_parser(subparsers):
    parser = subparsers.add_parser(
        "download", help="Download corpus from the web. Needs to be done only once."
    )
    parser.add_argument(
        "download_dir",
        nargs="?",
        type=os.path.abspath,
        default=None,
        help="Where to download files to. If unset, stores in the subfolder "
        "'local/data/' of the data folder",
    )
    parser.add_argument(
        "--dirty", help="If set, do not clean up tar files when done with them."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="The size of the buffer to work with when downloading/extracting. Larger "
        "means more memory usage, but possibly faster.",
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        default=False,
        help="Also download language model resources.",
    )

    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument(
        "--host",
        default=DEFT_HOST,
        help="Set schema + host of the mirror of openSLR to download from. Defaults to "
        f"'{DEFT_HOST}'",
    )
    mirror_group.add_argument(
        "--mirror-region",
        choices=sorted(REGION_MIRRORS),
        default=None,
        help="If set, picks an openSLR mirror to download from based on the region.",
    )


def build_preamble_parser(subparsers):
    parser = subparsers.add_parser(
        "preamble",
        help="Do all the pre-initialization setup. Need to be done only once.",
    )
    parser.add_argument(
        "librispeech_root",
        nargs="?",
        type=os.path.abspath,
        default=None,
        help="The root of the librispeech data directory. Contains 'dev-clean', "
        "'train-clean-360', etc. If unset, will check the 'local/data/' subfolder of "
        "the data folder (the default storage location of the 'download' command).",
    )


def main(args=None):
    """Prepare Librispeech data for end-to-end pytorch training"""

    parser = build_parser()
    options = parser.parse_args(args)

    if options.command == "download":
        download(options)
    elif options.command == "preamble":
        preamble(options)


if __name__ == "__main__":
    sys.exit(main())
