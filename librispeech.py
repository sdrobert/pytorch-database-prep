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
import zlib
import gzip

import ngram_lm
import pydrobert.speech.command_line as speech_cmd
import pydrobert.torch.command_line as torch_cmd

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

    found_fnames = []
    for fname in AM_FNAMES:
        libri_subdir = find_file(
            libri_dir, fname, fname in {"train-clean-360", "train-other-500"}
        )
        if libri_subdir is None:
            warnings.warn(
                f"Could not find folder '{fname}' in '{libri_dir}'. Skipping partition"
            )
            continue
        found_fnames.append(fname)
        data_prefix = os.path.join(data_dir, fname.replace("-", "_"))
        data_prep(
            libri_subdir, data_prefix, reader2gender, options.speakers_are_readers
        )

    # don't be coy about it - these lists always end up aggregated
    for file_ in ("spk2gender", "utt2spk"):
        aggr = dict()
        for fname in found_fnames:
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
        dst = os.path.join(config_dir, "lm.arpa.gz")
        if os.path.exists(dst):
            os.unlink(dst)
        os.link(lm_path, dst)

    for fname in AM_FNAMES + TRAIN_SUBSETS:
        fname = fname.replace("-", "_")
        config_prefix = os.path.join(config_dir, fname)
        data_prefix = os.path.join(data_dir, fname)
        text_file = data_prefix + ".text"
        if not os.path.isfile(text_file):
            if fname not in TRAIN_SUBSETS:
                if fname in {"train_clean_360", "train_other_500"}:
                    warnings.warn(f"'{text_file}' does not exist. Skipping partition")
                else:
                    raise ValueError(
                        f"'{text_file}' does not exist (did you finish preamble?)"
                    )
            continue

        with open(text_file) as in_, open(config_prefix + ".ref.trn", "w") as out:
            for line in in_:
                utt_id, transcript = line.strip().split(" ", maxsplit=1)
                out.write(transcript + f" ({utt_id})\n")

        shutil.copy(data_prefix + ".wav.scp", config_prefix + ".wav.scp")


def train_custom_lm(config_dir, vocab, max_order, prune_counts, delta=None):
    sents = []
    for fname in AM_FNAMES:
        fname = fname.replace("-", "_")
        trn_path = os.path.join(config_dir, fname + ".ref.trn")
        if not os.path.exists(trn_path):
            warnings.warn(
                f"'{trn_path}' does not exist, so not using to train LM. If the file "
                "is added later, you'll get a different LM if you rerun this stage. "
                "You've been warned!"
            )
            continue
        with open(trn_path) as f:
            for line in f:
                sent = line.strip().split()
                sent.pop()  # utterance id
                sents.append(sent)

    # count n-grams in sentences
    ngram_counts = ngram_lm.sents_to_ngram_counts(
        sents, max_order, sos="<s>", eos="</s>"
    )
    # ensure all vocab terms have unigram counts (even if 0) for zeroton
    # interpolation
    for v in vocab:
        ngram_counts[0].setdefault(v, 0)
    del sents

    to_prune = set(ngram_counts[0]) - vocab
    for i, ngram_count in enumerate(ngram_counts[1:]):
        if i >= len(prune_counts):
            prune_count = prune_counts[-1]
        else:
            prune_count = prune_counts[i]
        if i:
            to_prune |= set(
                k
                for (k, v) in ngram_count.items()
                if k[:-1] in to_prune or k[-1] in to_prune or v <= prune_count
            )
        else:
            to_prune |= set(
                k
                for (k, v) in ngram_count.items()
                if k[0] in to_prune or k[1] in to_prune or v <= prune_count
            )

    prob_list = ngram_lm.ngram_counts_to_prob_list_kneser_ney(
        ngram_counts, sos="<s>", to_prune=to_prune, delta=delta
    )

    # remove start-of-sequence probability mass
    lm = ngram_lm.BackoffNGramLM(prob_list, sos="<s>", eos="</s>", unk="<s>")
    lm.prune_by_name({"<s>"})
    prob_list = lm.to_prob_list()

    # save it
    with gzip.open(os.path.join(config_dir, "lm.arpa.gz"), "wt") as file_:
        ngram_lm.write_arpa(prob_list, file_)


def init_char(options):
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
    vocab = {"<s>", "</s>", "_", "<UNK>"}
    with open(vocab_txt) as f:
        for line in f:
            vocab.update(line.strip())
    vocab = sorted(vocab)

    os.makedirs(config_dir, exist_ok=True)

    with open(os.path.join(config_dir, "token2id.txt"), "w") as token2id, open(
        os.path.join(config_dir, "id2token.txt"), "w"
    ) as id2token:
        for id_, token in enumerate(vocab):
            token2id.write(f"{token} {id_}\n")
            id2token.write(f"{id_} {token}\n")

    for file_ in ("utt2spk", "spk2utt", "spk2gender"):
        shutil.copy(os.path.join(data_dir, file_), os.path.join(config_dir, file_))

    for fname in AM_FNAMES + TRAIN_SUBSETS:
        fname = fname.replace("-", "_")
        config_prefix = os.path.join(config_dir, fname)
        data_prefix = os.path.join(data_dir, fname)
        text_file = data_prefix + ".text"
        if not os.path.isfile(text_file):
            if fname not in TRAIN_SUBSETS:
                if fname in {"train_clean_360", "train_other_500"}:
                    warnings.warn(f"'{text_file}' does not exist. Skipping partition")
                else:
                    raise ValueError(
                        f"'{text_file}' does not exist (did you finish preamble?)"
                    )
            continue

        with open(text_file) as in_, open(config_prefix + ".ref.trn", "w") as out:
            for line in in_:
                utt_id, transcript = line.strip().split(" ", maxsplit=1)
                transcript = transcript.replace(" ", "_")
                transcript = " ".join(transcript)
                out.write(transcript + f" ({utt_id})\n")

        shutil.copy(data_prefix + ".wav.scp", config_prefix + ".wav.scp")

    if options.custom_lm_max_order > 0:
        # XXX(sdrobert): see wsj.py for details on deltas
        train_custom_lm(
            config_dir,
            set(vocab),
            options.custom_lm_max_order,
            options.custom_lm_prune_counts,
            [(0.5, 1.0, 1.5)] * options.custom_lm_max_order,
        )


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

    print("Copying files to ext...")
    for file_ in ("utt2spk", "spk2utt", "spk2gender", "id2token.txt", "token2id.txt"):
        shutil.copy(os.path.join(config_dir, file_), os.path.join(ext, file_))

    lm_arpa_gz = os.path.join(config_dir, "lm.arpa.gz")
    if os.path.exists(lm_arpa_gz):
        shutil.copy(lm_arpa_gz, ext)

    fnames = tuple(x.replace("-", "_") for x in AM_FNAMES)
    if options.force_compute_subsets:
        fnames = TRAIN_SUBSETS + fnames

    num_workers = str(get_num_avail_cores() - 1)
    feat_optional_args = [
        "--channel",
        "-1",
        "--num-workers",
        num_workers,
        "--preprocess",
        options.preprocess,
        "--postprocess",
        options.postprocess,
    ]
    if options.seed is not None:
        feat_optional_args.extend(["--seed", str(options.seed)])

    for fname in fnames:
        wav_scp = os.path.join(config_dir, fname + ".wav.scp")
        if not os.path.isfile(wav_scp):
            raise ValueError(f"'{wav_scp}' not a file (did you finish init_*)?")

        feat_dir = os.path.join(dir_, fname, "feat")
        os.makedirs(feat_dir, exist_ok=True)

        if options.feats_from is None:
            args = [wav_scp, feat_dir] + feat_optional_args
            hash_ = zlib.adler32("\0".join(args).encode("utf-8"))
            manifest_path = os.path.join(config_dir, f"{fname}.{hash_}.manifest")
            args += ["--manifest", manifest_path]
            if not options.raw:
                args.insert(1, options.computer_json)
            print(f"Generating features in {fname}...")
            assert not speech_cmd.signals_to_torch_feat_dir(args)
        else:
            print(f"Copying features into {fname}")
            feat_src = os.path.join(
                options.data_root, options.feats_from, fname, "feat"
            )
            if not os.path.isdir(feat_src):
                raise ValueError(
                    f"Specified --feats-from, but '{feat_src}' is not a directory"
                )
            if os.path.normpath(os.path.abspath(feat_src)) == os.path.normpath(
                os.path.abspath(feat_dir)
            ):
                warnings.warn(
                    f"feature source ('{feat_src}') and dest ('{feat_dir}') "
                    "are the same. Not moving anything"
                )
            else:
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

            print(f"linking features in {fname}...")
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

        print(f"building refs for {fname}...")
        args = [
            ref_trn,
            token2id_txt,
            ref_dir,
            "--unk-symbol=<UNK>",
            "--num-workers",
            num_workers,
            "--skip-frame-times",
        ]
        assert not torch_cmd.trn_to_torch_token_data_dir(args)

        if not options.skip_verify:
            # verify correctness (while storing info as a bonus)
            print(f"Verifying {fname} is correct...")
            args = [fname_dir, os.path.join(ext, f"{fname}.info.ark"), "--strict"]
            assert not torch_cmd.get_torch_spect_data_dir_info(args)

            if fname.endswith(options.compute_up_to):
                break

    if options.aggregate_by_copy:
        cp = shutil.copy2
    elif options.aggregate_by_symlink:
        cp = os.symlink
    elif options.aggregate_by_link:
        cp = os.link
    else:
        return
    src_subdirs = ["train_clean_100", "train_clean_360"]
    if options.compute_up_to == "500":
        dest_subdir = "train_all_960"
        src_subdirs.append("train_other_500")
    elif options.compute_up_to == "360":
        dest_subdir = "train_clean_460"
    else:
        warnings.warn(
            "'--aggregate-by-*' flag was specified, but so was '--compute-up-to 100'. "
            "There is only one training partition to aggregate, so skipping"
        )
        return
    dst_dir = os.path.join(dir_, dest_subdir)
    os.makedirs(dst_dir, exist_ok=True)
    for src_subdir in src_subdirs:
        print(f"Aggregating {src_subdir}...")
        src_dir = os.path.join(dir_, src_subdir)
        shutil.copytree(
            src_dir,
            dst_dir,
            copy_function=cp,
            dirs_exist_ok=True,
            ignore_dangling_symlinks=True,
            ignore=shutil.ignore_patterns("ali*"),
        )


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
    build_init_char_parser(subparsers)
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


def build_init_char_parser(subparsers):
    parser = subparsers.add_parser(
        "init_char",
        help="Perform setup common to all char-based parsing. "
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
        default="char",
        help="Name of sub directory in data/local/ under which to store setup "
        "specific to this lm. Defaults to 'char'.",
    )
    parser.add_argument(
        "--custom-lm-max-order",
        type=int,
        default=0,
        help="If > 0, an n-gram LM with Modified Kneser-Ney smoothing will be created "
        "from whatever training partition transcripts we have. NOTE: the n-gram LMs "
        "available from OpenSLR train on much more text!",
    )
    parser.add_argument(
        "--custom-lm-prune-counts",
        type=int,
        nargs="+",
        default=[0],
        help="Applies when --custom-lm-max_order is greater than 0. Prunes n-grams "
        "less than or equal to this counts before saving the custom lm. The first "
        "value passed is the threshold for bigrams, the second for trigrams, and so "
        "on (unigrams are not pruned). If there are fewer counts than "
        "--custom-lm-max-order, the last count is duplicated for the higher-order "
        "n-grams. E.g. '--custom-lm-prune-counts 1 2 3' prunes bigrams with a count in "
        "0-1, trigrams with a count in 0-2, and everything higher with a count of 3 or "
        "less",
    )


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
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        default=False,
        help="Skip the (very slow) verification/info step for each partition. This "
        "may be done later using the command 'get-torch-spect-data-dir-info' "
        "command on each partition",
    )
    aggregate_group = parser.add_mutually_exclusive_group()
    aggregate_common_text = (
        "Aggregates all computed training partitions into a single partition. "
        "Usually this will produce the partition 'train_all_960', but, if used in "
        "concert with '--compute-up-to 360', this will produce the partition "
        "'train_clean_460'. This version of '--aggregate-*' {} all files from "
        "the source partition"
    )
    aggregate_group.add_argument(
        "--aggregate-by-copy",
        action="store_true",
        default=False,
        help=aggregate_common_text.format("copies"),
    )
    aggregate_group.add_argument(
        "--aggregate-by-symlink",
        action="store_true",
        default=False,
        help=aggregate_common_text.format("symbolically links"),
    )
    aggregate_group.add_argument(
        "--aggregate-by-link",
        action="store_true",
        default=False,
        help=aggregate_common_text.format("(hard) links"),
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
    elif options.command == "init_char":
        init_char(options)
    elif options.command == "torch_dir":
        torch_dir(options)
    else:
        raise NotImplementedError(f"Command {options.command} not implemented")


if __name__ == "__main__":
    sys.exit(main())
