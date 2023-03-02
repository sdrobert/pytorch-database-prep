#! /usr/bin/env python

# Copyright 2022 Sean Robertson
#
# The download and preamble steps were adapted from kaldi/egs/librispeech/s5/local/
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
import warnings
import requests
import shutil
import tarfile
import glob
import json

import pydrobert.speech.command_line as speech_cmd
import pydrobert.torch.command_line as torch_cmd

from pydrobert.speech.compute import FrameComputer
from pydrobert.speech.util import alias_factory_subclass_from_arg
from pydrobert.speech.post import PostProcessor, Stack

from common import get_num_avail_cores, utt2spk_to_spk2utt

# XXX(sdrobert): order important for torch_dir
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
    # "librispeech-lexicon.txt",
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

TRAIN_SUBSETS = ("train_2kshort", "train_5k", "train_10k")

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources", "librispeech")
if not os.path.isdir(RESOURCE_DIR):
    raise ValueError(f"'{RESOURCE_DIR}' is not a directory")


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

    if options.files is not None:
        am_fnames = sorted(set(x for x in AM_FNAMES if x + ".tar.gz" in options.files))
        lm_files = sorted(
            set(x for x in LM_FILES + ("librispeech-vocab.txt",) if x in options.files)
        )
    else:
        am_fnames = AM_FNAMES
        lm_files = ("librispeech-vocab.txt",)
        if options.lm:
            lm_files = lm_files + LM_FILES

    for fname in am_fnames:
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

    for file_ in lm_files:
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


def find_file(root, file_name, allow_missing=False):
    paths = glob.glob(f"{glob.escape(root)}/**/{file_name}", recursive=True)
    if not len(paths):
        if allow_missing:
            return None
        raise ValueError(f"'{file_name}' not in '{root}'")
    elif len(paths) > 1:
        raise ValueError(f"More than one instance of '{file_name}' exists in '{root}'")
    return paths.pop()


def data_prep(libri_dir, data_prefix, reader2gender, speakers_are_readers):
    with open(data_prefix + ".wav.scp", "w") as wav_scp, open(
        os.path.join(data_prefix + ".text"), "w"
    ) as trans, open(data_prefix + ".utt2spk", "w") as utt2spk, open(
        os.path.join(data_prefix + ".spk2gender"), "w"
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
            if speakers_are_readers:
                spk2gender.write(f"lbi-{reader} {reader_gender}\n")

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
                        if speakers_are_readers:
                            utt2spk.write(f"{uttid} lbi-{reader}\n")
                        else:
                            utt2spk.write(f"{uttid} lbi-{reader}-{chapter}\n")

                if not speakers_are_readers:
                    spk2gender.write(f"lbi-{reader}-{chapter} {reader_gender}\n")


def preamble(options):
    data_dir = os.path.join(options.data_root, "local", "data")
    if options.librispeech_root is None:
        libri_dir = data_dir
    else:
        libri_dir = options.librispeech_root
    if not os.path.isdir(libri_dir):
        raise ValueError(f"'{libri_dir}' does not exist or is not a directory")
    find_file(libri_dir, "84-121123-0000.flac")

    speakers_txt = find_file(libri_dir, "SPEAKERS.TXT")
    reader2gender = dict()
    with open(speakers_txt) as f:
        for line in f:
            line = line.strip()
            if line.startswith(";"):
                continue
            reader, gender, _ = line.split("|", 2)
            reader, gender = reader.strip(), gender.strip().lower()
            if gender not in {"m", "f"}:
                raise ValueError(f"Unexpected gender '{gender}'")
            reader2gender[reader] = gender

    os.makedirs(data_dir, exist_ok=True)

    for fname in AM_FNAMES:
        libri_subdir = find_file(libri_dir, fname)
        data_prefix = os.path.join(data_dir, fname.replace("-", "_"))
        data_prep(
            libri_subdir, data_prefix, reader2gender, options.speakers_are_readers
        )

    # don't be coy about it - these lists always end up aggregated
    for file_ in ("spk2gender", "utt2spk"):
        aggr = dict()
        for fname in AM_FNAMES:
            with open(
                os.path.join(data_dir, fname.replace("-", "_") + "." + file_)
            ) as in_:
                for line in in_:
                    key, value = line.strip().split(" ", maxsplit=1)
                    assert aggr.setdefault(key, value) == value
        with open(os.path.join(data_dir, file_), "w") as out:
            for key, value in sorted(aggr.items()):
                out.write(f"{key} {value}\n")
    with open(os.path.join(data_dir, "spk2utt"), "w") as out:
        for line in utt2spk_to_spk2utt(os.path.join(data_dir, "utt2spk")):
            out.write(line + "\n")

    if not options.exclude_subsets:
        clean_100_prefix = os.path.join(data_dir, "train_clean_100")
        for subset_name in TRAIN_SUBSETS:
            lst_file = os.path.join(RESOURCE_DIR, subset_name + ".lst")
            if not os.path.exists(lst_file):
                raise ValueError(f"Could not find '{lst_file}'")
            subset_ids = set()
            with open(lst_file) as f:
                for line in f:
                    subset_ids.add(line.strip())
            subset_prefix = os.path.join(data_dir, subset_name)
            for file_ in (".text", ".wav.scp"):
                with open(clean_100_prefix + file_) as src, open(
                    os.path.join(subset_prefix + file_), "w"
                ) as dst:
                    for line in src:
                        utt_id = line.split(" ", maxsplit=1)[0]
                        if utt_id in subset_ids:
                            dst.write(line)


def init_word(options):
    local_dir = os.path.join(options.data_root, "local")
    data_dir = os.path.join(local_dir, "data")
    if not os.path.isdir(data_dir):
        raise ValueError("{} does not exist; call preamble first!".format(data_dir))
    if options.librispeech_root is None:
        libri_dir = data_dir
    else:
        libri_dir = options.librispeech_root

    config_dir = os.path.join(local_dir, options.config_subdir)

    vocab_txt = find_file(libri_dir, "librispeech-vocab.txt")
    vocab = {"<s>", "</s>", "<UNK>"}
    with open(vocab_txt) as f:
        for line in f:
            vocab.add(line.strip())
    vocab = sorted(vocab)
    assert len(vocab) == 200_003

    os.makedirs(config_dir, exist_ok=True)

    with open(os.path.join(config_dir, "token2id.txt"), "w") as token2id, open(
        os.path.join(config_dir, "id2token.txt"), "w"
    ) as id2token:
        for id_, token in enumerate(vocab):
            token2id.write(f"{token} {id_}\n")
            id2token.write(f"{id_} {token}\n")

    for file_ in ("utt2spk", "spk2utt", "spk2gender"):
        shutil.copy(os.path.join(data_dir, file_), os.path.join(config_dir, file_))

    lm_path = find_file(libri_dir, options.lm_name + ".arpa.gz", True)
    if lm_path is not None and os.path.isfile(lm_path):
        # we copy in the next step
        os.link(lm_path, os.path.join(config_dir, "lm.arpa.gz"))

    for fname in AM_FNAMES + TRAIN_SUBSETS:
        fname = fname.replace("-", "_")
        config_prefix = os.path.join(config_dir, fname)
        data_prefix = os.path.join(data_dir, fname)
        text_file = data_prefix + ".text"
        if not os.path.isfile(text_file):
            if fname not in TRAIN_SUBSETS:
                raise ValueError(
                    f"'{text_file}' does not exist (did you finish preamble?)"
                )
            continue

        with open(text_file) as in_, open(config_prefix + ".ref.trn", "w") as out:
            for line in in_:
                utt_id, transcript = line.strip().split(" ", maxsplit=1)
                out.write(transcript + f" ({utt_id})\n")

        shutil.copy(data_prefix + ".wav.scp", config_prefix + ".wav.scp")


def torch_dir(options):
    local_dir = os.path.join(options.data_root, "local")
    if options.config_subdir is None:
        dirs = os.listdir(local_dir)
        try:
            dirs.remove("data")
        except ValueError:
            pass
        if len(dirs) == 1:
            config_dir = os.path.join(local_dir, dirs[0])
        else:
            raise ValueError(
                'More than one directory ({}) besides "data" exists in "{}". '
                "Cannot infer configuration. Please specify as a positional "
                "argument".format(", ".join(dirs), local_dir)
            )
    else:
        config_dir = os.path.join(local_dir, options.config_subdir)
        if not os.path.isdir(config_dir):
            raise ValueError('"{}" is not a directory'.format(config_dir))

    dir_ = os.path.join(options.data_root, options.data_subdir)
    ext = os.path.join(dir_, "ext")
    os.makedirs(ext, exist_ok=True)

    for file_ in ("utt2spk", "spk2utt", "spk2gender", "id2token.txt", "token2id.txt"):
        shutil.copy(os.path.join(config_dir, file_), os.path.join(ext, file_))

    lm_arpa_gz = os.path.join(config_dir, "lm.arpa.gz")
    if os.path.exists(lm_arpa_gz):
        shutil.copy(lm_arpa_gz, ext)

    fnames = tuple(x.replace("-", "_") for x in AM_FNAMES)
    if options.force_compute_subsets:
        fnames = TRAIN_SUBSETS + fnames

    feat_optional_args = [
        "--channel",
        "-1",
        "--num-workers",
        str(get_num_avail_cores() - 1),
        "--preprocess",
        options.preprocess,
        "--postprocess",
        options.postprocess,
    ]
    if options.seed is not None:
        feat_optional_args.extend(["--seed", str(options.seed)])

    if options.raw:
        # 16 samps per ms = 1 / 16 ms per samp
        frame_shift_ms = 1 / 16
    else:
        # more complicated. Have to build our feature computer
        with open(options.computer_json) as file_:
            json_ = json.load(file_)
        computer: FrameComputer = alias_factory_subclass_from_arg(FrameComputer, json_)
        frame_shift_ms = computer.frame_shift_ms
        del computer, json_

    # FIXME(sdrobert): brittle. Needs to be manually updated with new postprocessors
    # and preprocessors
    do_strict = True
    try:
        with open(options.postprocess) as f:
            json_ = json.load(f)
    except IOError:
        json_ = json.loads(options.postprocess)
    postprocessors = []
    if isinstance(json_, dict):
        postprocessors.append(alias_factory_subclass_from_arg(PostProcessor, json_))
    else:
        for element in json_:
            postprocessors.append(
                alias_factory_subclass_from_arg(PostProcessor, element)
            )
    for postprocessor in postprocessors:
        if isinstance(postprocessor, Stack):
            if postprocessor._pad_mode is None:
                warnings.warn(
                    "Found a stack postprocessor with no pad_mode. This will likely "
                    "mess up the segment boundaries. Disabling --strict check."
                )
                do_strict = False
            frame_shift_ms *= postprocessor.num_vectors
    del postprocessors, json_

    for fname in fnames:
        wav_scp = os.path.join(config_dir, fname + ".wav.scp")
        if not os.path.isfile(wav_scp):
            raise ValueError(f"'{wav_scp}' not a file (did you finish init_*)?")

        feat_dir = os.path.join(dir_, fname, "feat")
        os.makedirs(feat_dir, exist_ok=True)

        if options.feats_from is None:
            args = [wav_scp, feat_dir] + feat_optional_args
            if not options.raw:
                args.insert(1, options.computer_json)
            assert not speech_cmd.signals_to_torch_feat_dir(args)
        else:
            feat_src = os.path.join(
                options.data_root, options.feats_from, fname, "feat"
            )
            if not os.path.isdir(feat_src):
                raise ValueError(
                    f"Specified --feats-from, but '{feat_src}' is not a directory"
                )
            for filename in os.listdir(feat_src):
                src = os.path.join(feat_src, filename)
                dest = os.path.join(feat_dir, filename)
                shutil.copy(src, dest)

        if fname.endswith(options.compute_up_to):
            break

    if not options.force_compute_subsets:
        clean_dir = os.path.join(dir_, "train_clean_100", "feat")
        for fname in TRAIN_SUBSETS:
            wav_scp = os.path.join(config_dir, fname + ".wav.scp")
            if not os.path.isfile(wav_scp):
                warnings.warn(f"'{wav_scp}' does not exist. Skipping {fname}")
                continue

            feat_dir = os.path.join(dir_, fname, "feat")
            os.makedirs(feat_dir, exist_ok=True)

            with open(wav_scp) as in_:
                for line in in_:
                    utt_id = line.split(" ", maxsplit=1)[0]
                    file_ = utt_id + ".pt"
                    os.link(
                        os.path.join(clean_dir, file_), os.path.join(feat_dir, file_)
                    )
        fnames = TRAIN_SUBSETS + fnames

    token2id_txt = os.path.join(config_dir, "token2id.txt")
    for fname in fnames:
        ref_trn = os.path.join(config_dir, fname + ".ref.trn")
        if not os.path.isfile(ref_trn):
            if fname in TRAIN_SUBSETS and not os.path.isfile(
                os.path.join(config_dir, fname + ".wav.scp")
            ):
                continue
            else:
                raise ValueError(f"'{ref_trn}' not a file (did you finish init_*)?")

        fname_dir = os.path.join(dir_, fname)
        ref_dir = os.path.join(fname_dir, "ref")
        os.makedirs(ref_dir, exist_ok=True)

        args = [ref_trn, token2id_txt, ref_dir, "--unk-symbol=<UNK>"]
        assert not torch_cmd.trn_to_torch_token_data_dir(args)

        # verify correctness (while storing info as a bonus)
        args = [fname_dir, os.path.join(ext, f"{fname}.info.ark")]
        if do_strict:
            args.append("--strict")
        assert not torch_cmd.get_torch_spect_data_dir_info(args)

        if fname.endswith(options.compute_up_to):
            break


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
    build_init_word_parser(subparsers)
    build_torch_dir_parser(subparsers)

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
    dl_grp = parser.add_mutually_exclusive_group()
    dl_grp.add_argument(
        "--files",
        nargs="+",
        metavar="FNAME",
        choices=sorted(FILE2MD5),
        default=None,
        help="If passed, download (and extract, where appropriate) only the files "
        f"specified here. Permitted: {', '.join(FILE2MD5)}",
    )
    dl_grp.add_argument(
        "--lm",
        action="store_true",
        default=False,
        help="If set, download all language model resources in addition to the "
        "audio corpus",
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
    parser.add_argument(
        "--speakers-are-readers",
        action="store_true",
        default=False,
        help="Kaldi (and us by default) treats each reader,chapter pair as a speaker. "
        "Setting this flag overrides this behaviour, equating speakers with readers.",
    )
    parser.add_argument(
        "--exclude-subsets",
        action="store_true",
        default=False,
        help="Kaldi (and us by default) creates subsets of the train-clean-100 tranch "
        "of the 2k shortest, 5k uniformly-spaced, and 10k unifomly-spaced samples. "
        "These can be gradually introduced to the model to simplify training, though "
        "they contain no new training data. Setting this flag excludes these subsets "
        "from being created.",
    )


def build_init_word_parser(subparsers):
    parser = subparsers.add_parser(
        "init_word",
        help="Perform setup common to all word-based parsing. "
        "Needs to be done only once for a specific language model.",
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
    parser.add_argument(
        "--config-subdir",
        default="wrd",
        help="Name of sub directory in data/local/ under which to store setup "
        "specific to this lm. Defaults to 'wrd'.",
    )
    parser.add_argument(
        "--lm-name",
        default="4-gram",
        choices=["4-gram", "3-gram", "3-gram.pruned.1e-7", "3-gram.pruned.3e-7"],
        help="The LM to save in the configuration",
    )
    # TODO(sdrobert): Custom n-gram LM training, if desired.


def build_torch_dir_parser(subparsers):
    parser = subparsers.add_parser(
        "torch_dir",
        help="Write training, test, and extra data to subdirectories. The init_* "
        "command must have been called previously. If more than one init_* call has "
        "been made, the next positional argument must be specified.",
    )
    parser.add_argument(
        "config_subdir",
        nargs="?",
        default=None,
        help="The configuration in data/local/ which to build the directories "
        "from. If init_* was called only once, it can be inferred from the "
        "contents fo data/local",
    )
    parser.add_argument(
        "data_subdir",
        nargs="?",
        default=".",
        help="What subdirectory in data/ to store training, test, and extra "
        "data subdirectories to. Defaults to directly in data/",
    )
    parser.add_argument(
        "--preprocess",
        default="[]",
        help="JSON list of configurations for "
        "``pydrobert.speech.pre.PreProcessor`` objects. Audio will be "
        "preprocessed in the same order as the list",
    )
    parser.add_argument(
        "--postprocess",
        default="[]",
        help="JSON List of configurations for "
        "``pydrobert.speech.post.PostProcessor`` objects. Features will be "
        "postprocessed in the same order as the list",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A random seed used for determinism. This affects operations "
        "like dithering. If unset, a seed will be generated at the moment",
    )
    parser.add_argument(
        "--force-compute-subsets",
        action="store_true",
        default=False,
        help="Compute features of subsets rather than linking them from 100h partition",
    )
    parser.add_argument(
        "--compute-up-to",
        default="500",
        choices=["100", "360", "500"],
        help="Compute features for up to the XXXh training partition. '100' is "
        "'train-clean-100' only. '360' is 'train-clean-100' + 'train-clean-360'. "
        "'500' is 'train-clean-100', 'train-clean-360', and 'train-other-500'. All "
        "dev and test partitions are always computed.",
    )

    fbank_41_config = os.path.join(
        os.path.dirname(__file__), "conf", "feats", "fbank_41.json"
    )
    feat_group = parser.add_mutually_exclusive_group()
    feat_group.add_argument(
        "--computer-json",
        default=fbank_41_config,
        help="Path to JSON configuration of a feature computer for "
        "pydrobert-speech. Defaults to a 40-dimensional Mel-scaled triangular "
        "overlapping filter bank + 1 energy coefficient every 10ms.",
    )
    feat_group.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="If specified, tensors of raw audio of shape (S, 1) will be "
        "written instead of filter bank coefficients.",
    )
    feat_group.add_argument(
        "--feats-from",
        default=None,
        help="If specified, rather than computing features, will copy the "
        "feature folders from this subdirectory of data/",
    )


def main(args=None):
    """Prepare Librispeech data for end-to-end pytorch training"""

    parser = build_parser()
    options = parser.parse_args(args)

    if options.command == "download":
        download(options)
    elif options.command == "preamble":
        preamble(options)
    elif options.command == "init_word":
        init_word(options)
    elif options.command == "torch_dir":
        torch_dir(options)
    else:
        raise NotImplementedError(f"Command {options.command} not implemented")


if __name__ == "__main__":
    sys.exit(main())
