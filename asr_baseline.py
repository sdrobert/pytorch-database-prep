#! /usr/bin/env python
#
# Copyright 2023 Sean Robertson
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

import abc
import argparse
import os

from typing import Any, Callable, Collection, Dict, Mapping, Optional, Sequence, Tuple
from typing_extensions import Literal
from itertools import chain

import param
import torch

from tqdm import tqdm
import pydrobert.torch.config as config
from pydrobert.torch.modules import (
    BeamSearch,
    ConcatSoftAttention,
    CTCPrefixSearch,
    CTCGreedySearch,
    ErrorRate,
    ExtractableSequentialLanguageModel,
    ExtractableShallowFusionLanguageModel,
    LookupLanguageModel,
    MixableSequentialLanguageModel,
    SpecAugment,
)
from pydrobert.param.argparse import (
    add_serialization_group_to_parser,
    add_deserialization_group_to_parser,
)
from pydrobert.torch.training import TrainingStateParams
from pydrobert.torch.training import TrainingStateController
from pydrobert.torch.data import (
    SpectDataLoader,
    SpectDataLoaderParams,
    SpectDataParams,
    SpectDataSet,
)

__all__ = [
    "BaselineParams",
    "construct_baseline",
    "CTCSpeechRecognizer",
    "Encoder",
    "EncoderDecoderSpeechRecognizer",
    "FeedForwardEncoder",
    "FeedForwardEncoderParams",
    "RecurrentDecoderWithAttention",
    "RecurrentDecoderWithAttentionParams",
    "RecurrentEncoder",
    "RecurrentEncoderParams",
    "SpeechRecognizer",
]


TRAIN_DATA_LOADER_PARAMS_SUBSET = {
    "batch_size",
    "drop_last",
    "num_length_buckets",
    "size_batch_by_length",
    "subset_ids",
    "eos",
    "delta_order",
    "do_mvn",
}

DECODE_DATA_SET_PARAMS_SUBSET = {"subset_ids", "delta_order", "do_mvn", "eos"}


def check_in(name: str, val: str, choices: Collection[str]):
    if val not in choices:
        choices = "', '".join(sorted(choices))
        raise ValueError(f"Expected {name} to be one of '{choices}'; got '{val}'")


def check_positive(name: str, val, nonnegative=False):
    pos = "non-negative" if nonnegative else "positive"
    if val < 0 or (val == 0 and not nonnegative):
        raise ValueError(f"Expected {name} to be {pos}; got {val}")


def check_equals(name: str, val, other):
    if val != other:
        raise ValueError(f"Expected {name} to be equal to {other}; got {val}")


def check_less_than(name: str, val, other, equal=False):
    lt = "less than or equal to" if equal else "less than"
    if val > other or (val == other and not equal):
        raise ValueError(f"Expected {name} to be {lt} {other}; got {val}")


class Encoder(torch.nn.Module, metaclass=abc.ABCMeta):
    """Encoder module

    The encoder is the part of the baseline which, given some input, always performs
    the same computations in the same order. It contains no auto-regressive connections.

    This class serves both as a base class for encoders and

    Call Parameters
    ---------------
    input : torch.Tensor
        A tensor of shape ``(max_seq_len, batch_size, input_size)`` of sequences
        which have been right-padded to ``max_seq_len``.
    lens : torch.Tensor
        A tensor of shape ``(batch_size,)`` containing the actual lengths of sequences
        such that, for each batch element ``n``, only ``input[:lens[n], n]`` are valid
        coefficients.

    Returns
    -------
    output, lens _: torch.Tensor
        `output` is a  tensor of shape ``(max_seq_len', batch_size, hidden_size)`` with a
        similar interpretation to `input`. `lens_` similar to `lens`
    """

    __constants__ = "input_size", "hidden_size"

    input_size: int
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int) -> None:
        check_positive("input_size", input_size)
        check_positive("hidden_size", hidden_size)
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    def reset_parameters(self):
        pass

    @abc.abstractmethod
    def forward(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    __call__: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class FeedForwardEncoder(Encoder):
    """Encoder is a simple feed-forward network"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: Literal["relu", "tanh", "sigmoid"] = "relu",
        dropout: float = 0.0,
    ):
        check_positive("num_layers", num_layers)
        check_positive("dropout", dropout, nonnegative=True)
        check_in("nonlinearity", nonlinearity, {"relu", "tanh", "sigmoid"})
        super().__init__(input_size, hidden_size)
        drop = torch.nn.Dropout(dropout)
        if nonlinearity == "relu":
            nonlin = torch.nn.ReLU()
        elif nonlinearity == "tanh":
            nonlin = torch.nn.Tanh()
        else:
            nonlin = torch.nn.Sigmoid()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias),
            nonlin,
            drop,
            *chain(
                *(
                    (torch.nn.Linear(hidden_size, hidden_size, bias), nonlin, drop)
                    for _ in range(num_layers - 1)
                )
            ),
        )

    def reset_parameters(self):
        super().reset_parameters()
        for n in range(0, len(self.stack), 3):
            self.stack[n].reset_parameters()

    def forward(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.stack(input), lens


class RecurrentEncoder(Encoder):
    """Recurrent encoder

    Warnings
    --------
    If `bidirectional` is :obj:`True`, `hidden_size` must be divisible by 2. Each
    direction's hidden_state
    """

    rnn: torch.nn.RNNBase

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        recurrent_type: Literal["lstm", "rnn", "gru"] = "lstm",
        dropout: float = 0.0,
    ) -> None:
        check_positive("num_layers", num_layers)
        check_in("recurrent_type", recurrent_type, {"lstm", "rnn", "gru"})
        check_positive("dropout", dropout, nonnegative=True)
        check_positive("hidden_size", hidden_size)
        if bidirectional:
            if hidden_size % 2:
                raise ValueError(
                    "for bidirectional encoder, hidden_size must be divisible by 2; "
                    f"got {hidden_size}"
                )
        super().__init__(input_size, hidden_size)
        if recurrent_type == "lstm":
            class_ = torch.nn.LSTM
        elif recurrent_type == "gru":
            class_ = torch.nn.GRU
        else:
            class_ = torch.nn.RNN
        self.rnn = class_(
            input_size,
            hidden_size // 2 if bidirectional else hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.rnn.reset_parameters()

    def forward(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = input.size(0)
        input = torch.nn.utils.rnn.pack_padded_sequence(
            input, lens.cpu(), enforce_sorted=False
        )
        output, lens_ = torch.nn.utils.rnn.pad_packed_sequence(
            self.rnn(input)[0], total_length=T
        )
        return output, lens_.to(lens)


class RecurrentDecoderWithAttention(MixableSequentialLanguageModel):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        embed_size: int = 128,
        decoder_hidden_size: Optional[int] = None,
        dropout: float = 0.0,
    ):
        check_positive("hidden_size", hidden_size)
        check_positive("vocab_size", vocab_size)
        check_positive("embed_size", embed_size)
        check_positive("dropout", dropout, nonnegative=True)
        if decoder_hidden_size is None:
            decoder_hidden_size = hidden_size
        else:
            check_positive("decoder_hidden_size", decoder_hidden_size)
        super().__init__(vocab_size)
        self.embed = torch.nn.Embedding(
            vocab_size + 1, embed_size, padding_idx=vocab_size
        )
        self.attn = ConcatSoftAttention(
            decoder_hidden_size, hidden_size, hidden_size=decoder_hidden_size
        )
        self.cell = torch.nn.LSTMCell(hidden_size + embed_size, decoder_hidden_size)
        self.ff = torch.nn.Linear(decoder_hidden_size, vocab_size)

    def reset_parameters(self):
        self.embed.reset_parameters()
        self.attn.reset_parameters()
        self.cell.reset_parameters()
        self.ff.reset_parameters()

    @torch.jit.export
    def extract_by_src(
        self, prev: Dict[str, torch.Tensor], src: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "input": prev["input"].index_select(1, src),
            "mask": prev["mask"].index_select(1, src),
            "hidden": prev["hidden"].index_select(0, src),
            "cell": prev["cell"].index_select(0, src),
        }

    @torch.jit.export
    def mix_by_mask(
        self,
        prev_true: Dict[str, torch.Tensor],
        prev_false: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mask = mask.unsqueeze(1)
        return {
            "input": prev_true["input"],
            "mask": prev_true["mask"],
            "hidden": torch.where(mask, prev_true["hidden"], prev_false["hidden"]),
            "cell": torch.where(mask, prev_true["cell"], prev_false["cell"]),
        }

    @torch.jit.export
    def update_input(
        self, prev: Dict[str, torch.Tensor], hist: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if "input" not in prev:
            raise RuntimeError("'input' must be in prev")
        input = prev["input"]
        if "mask" not in prev:
            if "lens" in prev:
                prev["mask"] = (
                    torch.arange(input.size(0), device=input.device).unsqueeze(1)
                    < prev["lens"]
                )
            else:
                prev["mask"] = input.new_ones(input.shape[:-1], dtype=torch.bool)
        if "hidden" in prev and "cell" in prev:
            return prev
        N = hist.size(1)
        zeros = self.ff.weight.new_zeros((N, self.attn.query_size))
        return {
            "input": input,
            "mask": prev["mask"],
            "hidden": zeros,
            "cell": zeros,
        }

    @torch.jit.export
    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input, mask, h_0 = prev["input"], prev["mask"], prev["hidden"]
        i = self.attn(h_0, input, input, mask)  # (N, I)
        idx_zero = idx == 0
        if idx_zero.all():
            x = torch.full(
                (hist.size(1),), self.vocab_size, dtype=hist.dtype, device=hist.device
            )
        else:
            x = hist.gather(
                0, (idx - 1).expand(hist.shape[1:]).clamp_min(0).unsqueeze(0)
            ).squeeze(
                0
            )  # (N,)
            x = x.masked_fill(idx_zero.expand(x.shape), self.vocab_size)
        x = self.embed(x)  # (N, E)
        x = torch.cat([i, x], 1)  # (N, I + E)
        h_1, c_1 = self.cell(x, (h_0, prev["cell"]))
        logits = self.ff(h_1)
        return (
            torch.nn.functional.log_softmax(logits, -1),
            {"input": input, "mask": mask, "hidden": h_1, "cell": c_1},
        )


class SpeechRecognizer(torch.nn.Module, metaclass=abc.ABCMeta):
    """ABC for ASR"""

    encoder: Encoder

    def __init__(self, encoder: Encoder, stack: int = 1) -> None:
        check_positive("stack", stack)
        super().__init__()
        self.stack, self.encoder = stack, encoder

    def reset_parameters(self):
        self.encoder.reset_parameters()

    @abc.abstractmethod
    def train_loss_from_encoded(
        self,
        input: torch.Tensor,
        lens: torch.Tensor,
        refs: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def encode(
        self, feats: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.stack > 1:
            T, N, C = feats.shape
            Tp = T - (T % self.stack)
            feats = (
                feats.transpose(0, 1)[:, :Tp]
                .reshape(N, Tp // self.stack, C * self.stack)
                .transpose(0, 1)
            )
            lens = lens // self.stack
        return self.encoder(feats, lens)

    def train_loss(
        self,
        feats: torch.Tensor,
        lens: torch.Tensor,
        refs: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        input, lens = self.encode(feats, lens)
        return self.train_loss_from_encoded(input, lens, refs, ref_lens)

    @abc.abstractmethod
    def decode_from_encoded(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def decode(
        self, feats: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input, lens = self.encode(feats, lens)
        return self.decode_from_encoded(input, lens)

    def forward(
        self, feats: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode(feats, lens)

    __call__: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class MyCTCGreedySearch(CTCGreedySearch):
    def __init__(self):
        super().__init__(-1, False, True)

    def reset_parameters(self):
        pass

    def forward(
        self, logits: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_, paths, out_lens = super().forward(logits, lens)
        return paths.unsqueeze(2), out_lens.unsqueeze(1), max_.unsqueeze(1)


class CTCSpeechRecognizer(SpeechRecognizer):
    """ASR + CTC transduction

    Warning
    -------
    The blank token will be taken to be the index `vocab_size`, not :obj:`0`.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder: Encoder,
        beam_width: int = 8,
        lm: Optional[MixableSequentialLanguageModel] = None,
        beta: float = 0.0,
        stack: int = 1,
    ) -> None:
        check_positive("vocab_size", vocab_size)
        check_positive("beam_width", beam_width, nonnegative=True)
        if lm is not None:
            check_equals("lm.vocab_size", lm.vocab_size, vocab_size)
        super().__init__(encoder, stack)
        self.ff = torch.nn.Linear(encoder.hidden_size, vocab_size + 1)
        if beam_width:
            self.search = CTCPrefixSearch(beam_width, beta, lm)
        else:
            self.search = MyCTCGreedySearch()
        self.ctc_loss = torch.nn.CTCLoss(vocab_size, zero_infinity=True)

    def reset_parameters(self):
        super().reset_parameters()
        self.ff.reset_parameters()
        self.search.reset_parameters()

    def train_loss_from_encoded(
        self,
        input: torch.Tensor,
        lens: torch.Tensor,
        refs: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(self.ff(input), 2)
        return self.ctc_loss(log_probs, refs.T, lens, ref_lens)

    def decode_from_encoded(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.ff(input)
        beam, lens, _ = self.search(logits, lens)
        return beam[..., 0], lens[..., 0]  # most probable


class EncoderDecoderSpeechRecognizer(SpeechRecognizer):
    """Encoder/decoder speech recognizer"""

    __constants__ = (
        "encoded_name_train",
        "encoded_lens_name_train",
        "encoded_name_decode",
        "encoded_lens_name_decode",
        "max_iters",
    )
    encoded_name_train: str
    encoded_lens_name_train: str
    encoded_name_decode: str
    encoded_lens_name_decode: str
    max_iters: int

    decoder: ExtractableSequentialLanguageModel

    def __init__(
        self,
        encoder: Encoder,
        decoder: ExtractableSequentialLanguageModel,
        beam_width: int = 8,
        lm: Optional[ExtractableSequentialLanguageModel] = None,
        beta: float = 0.0,
        encoded_name: str = "input",
        encoded_lens_name: str = "lens",
        pad_value: int = config.INDEX_PAD_VALUE,
        eos: Optional[int] = None,
        max_iters: int = 200,
        stack: int = 1,
    ) -> None:
        check_positive("beam_width", beam_width)
        if lm is not None:
            check_equals("lm.vocab_size", lm.vocab_size, decoder.vocab_size)
        check_positive("max_iters", max_iters)
        if eos is not None:
            check_less_than("eos", eos, decoder.vocab_size)
        super().__init__(encoder, stack)
        self.decoder = decoder
        self.encoded_name_train = self.encoded_name_decode = encoded_name
        self.encoded_lens_name_train = self.encoded_lens_name_decode = encoded_lens_name
        self.max_iters = max_iters
        if lm is None:
            lm = decoder
        else:
            lm = ExtractableShallowFusionLanguageModel(
                decoder, lm, beta, first_prefix="first."
            )
            self.encoded_name_decode = "first." + encoded_name
            self.encoded_lens_name_decode = "first." + encoded_lens_name

        self.search = BeamSearch(
            lm,
            beam_width,
            eos=eos,
            pad_value=pad_value,
        )
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=pad_value)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self.decoder, "reset_parameters"):
            self.decoder.reset_parameters()

    def train_loss_from_encoded(
        self,
        input: torch.Tensor,
        lens: torch.Tensor,
        refs: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.arange(refs.size(0), device=refs.device).unsqueeze(1) >= ref_lens
        refs = refs.masked_fill(mask, self.search.pad_value)
        prev = {self.encoded_name_train: input, self.encoded_lens_name_train: lens}
        log_probs = self.decoder(refs[:-1].clamp_min(0), prev)
        assert log_probs.shape[:-1] == refs.shape
        return self.ce_loss(log_probs.flatten(0, 1), refs.flatten())

    def decode_from_encoded(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prev = {self.encoded_name_decode: input, self.encoded_lens_name_decode: lens}
        hyps, lens, _ = self.search(prev, input.size(1), self.max_iters)
        return hyps[..., 0], lens[..., 0]  # most probable


class FeedForwardEncoderParams(param.Parameterized):
    num_layers: int = param.Integer(2, bounds=(1, None), doc="Number of layers")
    bias: bool = param.Boolean(True, doc="Whether to add a bias vector")
    nonlinearity: Literal["relu", "tanh", "sigmoid"] = param.ObjectSelector(
        "relu", ["relu", "tanh", "sigmoid"], doc="Nonlinearities after linear layers"
    )


class RecurrentEncoderParams(param.Parameterized):
    num_layers: int = param.Integer(2, bounds=(1, None), doc="Number of layers")
    recurrent_type: Literal["lstm", "gru", "rnn"] = param.ObjectSelector(
        "lstm", ["lstm", "gru", "rnn"], doc="Type of recurrent cell"
    )
    bidirectional: bool = param.Boolean(
        True,
        doc="Whether layers are bidirectional. If so, each direction has "
        "a hidden size of half the parameterized hidden_size. In this case, the "
        "parameter must be divisible by 2",
    )


class RecurrentDecoderWithAttentionParams(param.Parameterized):
    embed_size: int = param.Integer(
        128, bounds=(1, None), doc="Size of token embedding vectors"
    )
    hidden_size: int = param.Integer(
        512, bounds=(1, None), doc="Size of decoder hidden states"
    )


class BaselineParams(param.Parameterized):
    input_size: int = param.Integer(
        41, bounds=(1, None), doc="Size of input feature vectors (pre stacking)"
    )
    stack: int = param.Integer(
        1, bounds=(1, None), doc="Number of feature vectors to stack together"
    )
    hidden_size: int = param.Integer(
        512, bounds=(1, None), doc="Size of encoder hidden states/output"
    )
    vocab_size: int = param.Integer(
        32, bounds=(1, None), doc="Size of output vocabulary (excluding eos and blank)"
    )

    encoder_type: Literal["recur", "ff"] = param.ObjectSelector(
        "recur", ["recur", "ff"], doc="What encoder structure to use"
    )
    transducer_type: Literal["ctc", "encdec"] = param.ObjectSelector(
        "ctc", ["ctc", "encdec"], doc="How to perform audio -> text transduction"
    )
    decoder_type: Literal["rwa"] = param.ObjectSelector(
        "rwa",
        ["rwa"],
        doc="What decoder structure to use (if transducer_type = 'encdec')",
    )

    recur_encoder: Optional[RecurrentEncoderParams] = param.ClassSelector(
        RecurrentEncoderParams,
        instantiate=False,
        doc="Parameters for recurrent encoded (if encoder_type = 'recur')",
    )
    ff_encoder: Optional[FeedForwardEncoderParams] = param.ClassSelector(
        FeedForwardEncoderParams,
        instantiate=False,
        doc="Parameters for feed-forward encoder (if encoder_type = 'ff')",
    )

    rwa_decoder: Optional[RecurrentDecoderWithAttentionParams] = param.ClassSelector(
        RecurrentDecoderWithAttentionParams,
        instantiate=False,
        doc="Parameters for recurrent decoder w/ attention (if transducer_type = "
        "'encdec' and decoder_type = 'rwa')",
    )

    def initialize_missing(self):
        if self.recur_encoder is None:
            self.recur_encoder = RecurrentEncoderParams(name="recur_encoder")
        if self.ff_encoder is None:
            self.ff_encoder = FeedForwardEncoderParams(name="ff_encoder")
        if self.rwa_decoder is None:
            self.rwa_decoder = RecurrentDecoderWithAttentionParams(name="rwa_decoder")
        return self


def construct_baseline(
    params: BaselineParams,
    lm: Optional[ExtractableSequentialLanguageModel] = None,
    beta: float = 0.0,
    beam_width: int = 8,
    dropout: float = 0.0,
    eos: Optional[int] = None,
    max_iters: int = 200,
) -> SpeechRecognizer:
    if params.encoder_type == "ff":
        if params.ff_encoder is None:
            raise ValueError(
                "encoder_type is 'ff' but ff_encoder has not been initialized"
            )
        encoder = FeedForwardEncoder(
            params.input_size * params.stack,
            params.hidden_size,
            params.ff_encoder.num_layers,
            params.ff_encoder.bias,
            params.ff_encoder.nonlinearity,
            dropout,
        )
    elif params.encoder_type == "recur":
        if params.recur_encoder is None:
            raise ValueError(
                "encoder_type is 'recur' but recur_encoder has not been initialized"
            )
        encoder = RecurrentEncoder(
            params.input_size * params.stack,
            params.hidden_size,
            params.recur_encoder.num_layers,
            params.recur_encoder.bidirectional,
            params.recur_encoder.recurrent_type,
            dropout,
        )
    else:
        raise NotImplementedError(
            f"encoder_type '{params.encoder_type}' not implemented"
        )
    if params.transducer_type == "ctc":
        if lm is not None and not isinstance(lm, MixableSequentialLanguageModel):
            raise ValueError(
                "ctc transducers require MixableSequentialLanguageModel instances "
                "for shallow fusion"
            )
        recognizer = CTCSpeechRecognizer(
            params.vocab_size, encoder, beam_width, lm, beta, params.stack
        )
    elif params.transducer_type == "encdec":
        if params.decoder_type == "rwa":
            if params.rwa_decoder is None:
                raise ValueError(
                    "decoder_type = 'rwa' but rwa_decoder is not initialized"
                )
            decoder = RecurrentDecoderWithAttention(
                params.hidden_size,
                params.vocab_size,
                params.rwa_decoder.embed_size,
                params.rwa_decoder.hidden_size,
                dropout,
            )
        else:
            raise NotImplementedError(
                f"decoder_type '{params.decoder_type}' not implemented"
            )
        recognizer = EncoderDecoderSpeechRecognizer(
            encoder,
            decoder,
            max(beam_width, 1),
            lm,
            beta,
            eos=eos,
            max_iters=max_iters,
            stack=params.stack,
        )
    else:
        raise NotImplementedError(
            f"transducer_type = '{params.transducer_type}' is not implemented"
        )
    return recognizer


class MyTrainingStateParams(TrainingStateParams):
    dropout: float = param.Magnitude(0.0, doc="Dropout probability")
    optimizer: Literal["adamw", "adam", "sgd"] = param.ObjectSelector(
        "adamw", ["adamw", "adam", "sgd"], doc="Which optimizer to train with"
    )

    spec_aug_max_time_warp: float = param.Number(
        80.0, bounds=(0.0, None), doc="SpecAugment: maximum frames warping can shift"
    )
    spec_aug_max_freq_warp: float = param.Number(
        0.0,
        bounds=(0.0, None),
        doc="SpecAugment: maximum feat coeffs warping can shift",
    )
    spec_aug_max_time_mask: int = param.Integer(
        100,
        bounds=(0, None),
        doc="SpecAugment: absolute max number of consecutive frames to mask",
    )
    spec_aug_max_freq_mask: int = param.Integer(
        27,
        bounds=(0, None),
        doc="SpecAugment: absolute max number of consecutive feat coeffs to mask",
    )
    spec_aug_max_time_mask_proportion: float = param.Magnitude(
        0.04, doc="SpecAugment: relative max number of consecutive frames to mask"
    )
    spec_aug_max_freq_mask_proportion: float = param.Magnitude(
        0.04, doc="SpecAugment: relative max number of consecutive feat coeffs to mask"
    )
    spec_aug_num_time_mask: int = param.Integer(
        20,
        bounds=(0, None),
        doc="SpecAugment: absolute max number of frame masks to apply",
    )
    spec_aug_num_time_mask_proportion: float = param.Magnitude(
        0.04, doc="SpecAgument: relative max number of frame masks to apply"
    )
    spec_aug_num_freq_mask: int = param.Integer(
        2,
        bounds=(0, None),
        doc="SpecAugment: absolute max number of feat coeff masks to apply",
    )
    spec_aug_interpolation_order: int = param.Integer(
        1, bounds=(1, None), doc="SpecAugment: warping interpolation order"
    )


def existing_dir(val: str) -> str:
    if not os.path.isdir(val):
        raise ValueError(f"'{val}' is not a directory")
    return os.path.normpath(val)


def _train_for_epoch(
    recognizer: SpeechRecognizer,
    optimizer: torch.optim.Optimizer,
    spec_aug: SpecAugment,
    dl: SpectDataLoader,
    device: torch.device,
) -> float:
    recognizer.train()

    total_loss = 0.0
    total_elems = 0
    for feats, refs, lens, ref_lens in dl:
        # batch first = True for spec_aug
        N = ref_lens.size(0)
        refs, ref_lens = refs.transpose(0, 1).to(device), ref_lens.to(device)
        with torch.no_grad():
            feats = spec_aug(feats, lens).transpose(0, 1).to(device)
            lens = lens.to(device)

        optimizer.zero_grad()

        loss = recognizer.train_loss(feats, lens, refs, ref_lens)
        loss.backward()

        optimizer.step()

        total_loss += N * loss.item()
        total_elems += N
        del feats, lens, refs, ref_lens, loss

    return total_loss / total_elems


def _val_for_epoch(
    recognizer: SpeechRecognizer, dl: SpectDataLoader, eos: int, device: torch.device
) -> float:
    recognizer.eval()

    total_er = 0
    total_ref_tokens = 0
    if eos is None:
        eos = config.INDEX_PAD_VALUE
    er_func = ErrorRate(eos, norm=False)
    with torch.no_grad():
        for feats, refs, lens, ref_lens in dl:
            # batch_first = False
            feats, lens = feats.to(device), lens.to(device)
            refs, ref_lens = refs.to(device), ref_lens.to(device)

            hyps, hyp_lens = recognizer.decode(feats, lens)
            del feats, lens

            mask = torch.arange(hyps.size(0), device=device).unsqueeze(1) >= hyp_lens
            hyps = hyps.masked_fill_(mask, eos)
            mask = torch.arange(refs.size(0), device=device).unsqueeze(1) >= ref_lens
            refs = refs.masked_fill_(mask, eos)

            er = er_func(refs, hyps).sum().item()
            ref_tokens = ref_lens.sum().item()
            total_er += er
            total_ref_tokens += ref_tokens
            del hyps, hyp_lens, refs, ref_lens
        # print("refs", refs[: ref_lens[0], 0])
        # print("hyps", hyps[: hyp_lens[0], 0])

    return total_er / total_ref_tokens


def train(options: argparse.Namespace):
    bparams: BaselineParams = options.bparams
    tparams: MyTrainingStateParams = options.tparams
    dparams: SpectDataLoaderParams = options.dparams
    bparams.initialize_missing()

    torch.autograd.set_detect_anomaly(True)

    if options.quiet:
        print_ = lambda x: None
    else:
        print_ = print

    print_("Constructing model...")
    recognizer = construct_baseline(
        bparams, beam_width=0, dropout=tparams.dropout, eos=dparams.eos
    )
    if options.init_state_dict:
        print_("loading initial state dict...")
        recognizer.load_state_dict(torch.load(options.init_state_dict, "cpu"))
        # don't let controller reset the parameters at epoch 0
        recognizer.reset_parameters = lambda *args, **kwargs: None
    recognizer.to(options.device)

    if options.distributed_backend is not None:
        print_("initializing backend...")
        torch.distributed.init_process_group(options.distributed_backend)
        local_rank = os.environ.get("LOCAL_RANK")
        recognizer = torch.nn.parallel.DistributedDataParallel(
            recognizer, [local_rank], local_rank
        )

    if tparams.optimizer == "adamw":
        Optimizer = torch.optim.AdamW
    elif tparams.optimizer == "adam":
        Optimizer = torch.optim.Adam
    elif tparams.optimizer == "sgd":
        Optimizer = torch.optim.SGD
    else:
        raise NotImplementedError(f"optimizer '{tparams.optimizer}' not implemented")
    if tparams.log10_learning_rate is None:
        optimizer = Optimizer(recognizer.parameters())
    else:
        optimizer = Optimizer(
            recognizer.parameters(), lr=10**tparams.log10_learning_rate
        )
    print_("Creating controller and possibly loading model/optimizer...")
    controller = TrainingStateController(
        tparams, options.state_csv_path, options.state_dir
    )
    controller.load_model_and_optimizer_for_epoch(recognizer, optimizer)

    init_epoch = controller.get_last_epoch() + 1
    total_epochs = (tparams.num_epochs + 1) if tparams.num_epochs else 10_000

    print_("Creating data loaders...")
    tdl = SpectDataLoader(
        options.train_dir,
        dparams,
        shuffle=True,
        batch_first=True,
        pin_memory=options.device.type == "cuda",
        num_workers=options.num_workers,
        init_epoch=init_epoch,
        suppress_alis=True,
        seed=tparams.seed,
        feat_mean=options.feat_mean,
        feat_std=options.feat_std,
    )
    vdl = SpectDataLoader(
        options.val_dir,
        dparams,
        shuffle=False,
        suppress_alis=True,
        batch_first=False,
        feat_mean=options.feat_mean,
        feat_std=options.feat_std,
    )
    if options.quiet:
        get_tdl, get_vdl = (lambda: tdl), (lambda: vdl)
    else:
        get_tdl, get_vdl = (lambda: tqdm(tdl)), (lambda: tqdm(vdl))

    spec_aug = SpecAugment(
        tparams.spec_aug_max_time_warp,
        tparams.spec_aug_max_freq_warp,
        tparams.spec_aug_max_time_mask,
        tparams.spec_aug_max_freq_mask,
        tparams.spec_aug_max_time_mask_proportion,
        tparams.spec_aug_num_time_mask,
        tparams.spec_aug_num_time_mask_proportion,
        tparams.spec_aug_num_freq_mask,
        tparams.spec_aug_interpolation_order,
    )
    spec_aug.train()

    for epoch in range(init_epoch, total_epochs):
        print_(f"Training for epoch {epoch}...")
        train_loss = _train_for_epoch(
            recognizer, optimizer, spec_aug, get_tdl(), options.device
        )
        print_(f"Validating for epoch {epoch}...")
        val_err = _val_for_epoch(recognizer, get_vdl(), dparams.eos, options.device)
        print_(
            f"Epoch {epoch} training loss was {train_loss:.02f}, "
            f"validation error was {val_err:.02%}"
        )
        print_("Saving and checking if continuing...")
        if not controller.update_for_epoch(
            recognizer, optimizer, train_loss, val_err, epoch
        ):
            print_("Controller says we're done")
            break
        print_("Continuing training")

    recognizer.cpu()
    if options.last:
        print_(f"Saving last model to '{options.best}'...")
    else:
        print_(f"Saving best model to '{options.best}'...")
        controller.load_model_for_epoch(recognizer, controller.get_best_epoch())
    state_dict = recognizer.state_dict()
    torch.save(state_dict, options.best)
    print_("Done saving")


def decode(options: argparse.Namespace):
    bparams: BaselineParams = options.bparams
    dparams: SpectDataParams = options.dparams
    bparams.initialize_missing()

    if options.quiet:
        print_ = lambda x: None
    else:
        print_ = print

    if options.lookup_lm_state_dict is not None:
        print_("Loading LM...")
        lm_state_dict: dict = torch.load(options.lookup_lm_state_dict)
        sos = lm_state_dict.pop("sos")
        vocab_size = lm_state_dict.pop("vocab_size", None)
        if vocab_size is None:
            vocab_size = bparams.vocab_size
        else:
            check_equals(
                "state dict in --lookup-lm-state-dict", vocab_size, bparams.vocab_size
            )
        lm = LookupLanguageModel(vocab_size, sos)
        lm.load_state_dict(lm_state_dict)
        del lm_state_dict
    else:
        lm = None

    print_("Constructing model...")
    recognizer = construct_baseline(
        bparams,
        lm,
        options.beta,
        options.beam_width,
        0,
        dparams.eos,
        options.max_hyp_len,
    )
    # FIXME(sdrobert): lm + search shouldn't be included in the state dict.
    print_("Loading the model state dict...")
    recognizer.load_state_dict(torch.load(options.model, "cpu"), False)
    recognizer.to(options.device)
    recognizer.eval()

    print_("Creating dataset...")
    ds = SpectDataSet(
        options.data_dir,
        params=dparams,
        feat_mean=options.feat_mean,
        feat_std=options.feat_std,
        suppress_alis=True,
        suppress_uttids=False,
        tokens_only=True,
    )

    if not options.quiet:
        ds = tqdm(ds)

    os.makedirs(options.hyp_dir, exist_ok=True)

    print_("Beginning decoding...")
    for feat, _, uttid in ds:
        feats = feat.unsqueeze(1).to(options.device)
        lens = torch.tensor([feats.size(0)], device=options.device)
        hyps, lens = recognizer.decode(feats, lens)
        hyp = hyps[: lens[0], 0].cpu()
        if dparams.eos is not None:
            eos_mask = (hyp == dparams.eos).cumsum(0).bool()
            hyp = hyp.masked_select(~eos_mask)
        torch.save(hyp, os.path.join(options.hyp_dir, uttid + ".pt"))
    print("Done decoding")


def main(args: Optional[Sequence[str]] = None):
    """Train and decode baseline, supervised speech recognizers"""

    parser = argparse.ArgumentParser(
        description=main.__doc__, fromfile_prefix_chars="@"
    )
    add_serialization_group_to_parser(
        parser,
        BaselineParams(name="baseline").initialize_missing(),
        flag_format_str="--print-model-{file_format}",
        reckless=True,
    )
    add_deserialization_group_to_parser(
        parser,
        BaselineParams(name="baseline").initialize_missing(),
        "bparams",
        flag_format_str="--read-model-{file_format}",
        reckless=True,
    )
    parser.add_argument(
        "--device",
        default=None,
        type=torch.device,
        help="Device to perform operations on. Default is to use cuda if available",
    )
    parser.add_argument(
        "--mvn-path",
        type=argparse.FileType("rb"),
        default=None,
        help="Path to corpus-level mean-variance stats for normalization",
    )

    subparsers = parser.add_subparsers(
        dest="cmd", description="Whether to train or decode", required=True
    )

    train_parser = subparsers.add_parser("train", description="Train a model")
    add_serialization_group_to_parser(
        train_parser,
        MyTrainingStateParams(name="training"),
        flag_format_str="--print-training-{file_format}",
    )
    add_deserialization_group_to_parser(
        train_parser,
        MyTrainingStateParams(name="training"),
        "tparams",
        flag_format_str="--read-training-{file_format}",
    )
    add_serialization_group_to_parser(
        train_parser,
        SpectDataLoaderParams(name="data"),
        subset=TRAIN_DATA_LOADER_PARAMS_SUBSET,
        flag_format_str="--print-data-{file_format}",
    )
    add_deserialization_group_to_parser(
        train_parser,
        SpectDataLoaderParams(name="data"),
        "dparams",
        subset=TRAIN_DATA_LOADER_PARAMS_SUBSET,
        flag_format_str="--read-data-{file_format}",
    )
    train_parser.add_argument(
        "--last",
        action="store_true",
        default=False,
        help="Save last epoch's state dict (rather than best)",
    )
    train_parser.add_argument(
        "--state-dir", default=None, help="Where to store intermediary state dicts"
    )
    train_parser.add_argument(
        "--state-csv-path", default=None, help="Path to CSV to log per-epoch stats"
    )
    train_parser.add_argument(
        "--distributed-backend",
        default=None,
        choices=["mpi", "gloo", "nccl", "ucc"],
        help="Backend for torch.distributed. If set, script should be called with "
        "torchrun. If unset, will be run in serial",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data workers. Default (0) is on main thread",
    )
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Whether to perform training quietly",
    )
    train_parser.add_argument(
        "--init-state-dict",
        type=argparse.FileType("rb"),
        default=None,
        help="If set, the model will be loaded first with this state dict. ",
    )
    train_parser.add_argument(
        "train_dir", type=existing_dir, help="Training SpectDataSet"
    )
    train_parser.add_argument(
        "val_dir", type=existing_dir, help="Validation/dev SpectDataSet"
    )
    train_parser.add_argument(
        "best",
        help="Where to save the state dict frome the epoch with the lowest error rate",
    )

    decode_parser = subparsers.add_parser(
        "decode", description="Use a trained model to decode features"
    )
    add_serialization_group_to_parser(
        decode_parser,
        SpectDataParams(name="data"),
        subset=DECODE_DATA_SET_PARAMS_SUBSET,
        flag_format_str="--print-data-{file_format}",
    )
    add_deserialization_group_to_parser(
        decode_parser,
        SpectDataParams(name="data"),
        "dparams",
        subset=DECODE_DATA_SET_PARAMS_SUBSET,
        flag_format_str="--read-data-{file_format}",
    )
    decode_parser.add_argument(
        "--beam-width",
        type=int,
        default=8,
        help="Number of prefixes to keep track of in search",
    )
    decode_parser.add_argument(
        "--max-hyp-len",
        type=int,
        default=1000,
        help="Maximum length of hypothesis to output (ensures termination)",
    )
    decode_parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Shallow fusion mixing coefficient (only applies if "
        "--lookup-lm-state-dict is set)",
    )
    decode_parser.add_argument(
        "--lookup-lm-state-dict",
        default=None,
        type=argparse.FileType("rb"),
        help="The resulting state file of calling arpa-lm-to-state-dict.py on an "
        "arpa LM. The start-of-sequence id should be saved in the state dict, i.e. "
        "the '--save-sos' flag should have been set",
    )
    decode_parser.add_argument(
        "model",
        type=argparse.FileType("rb"),
        help="Path to the state dictionary of the model to load",
    )
    decode_parser.add_argument(
        "data_dir", type=existing_dir, help="SpectDataSet directory of input"
    )
    decode_parser.add_argument(
        "hyp_dir", help="Directory to store hypothesis transcriptions"
    )
    decode_parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Whether to perform decoding quietly",
    )

    options = parser.parse_args(args)
    if options.device is None:
        if torch.cuda.is_available():
            options.device = torch.device(torch.cuda.current_device())
        else:
            options.device = torch.device("cpu")

    if options.mvn_path is None:
        options.feat_mean = options.feat_std = None
    else:
        dict_ = torch.load(options.mvn_path)
        options.feat_mean, options.feat_std = dict_["mean"], dict_["std"]

    if options.cmd == "train":
        train(options)
    elif options.cmd == "decode":
        decode(options)
    else:
        raise NotImplementedError(f"command 'cmd' not implemented")


if __name__ == "__main__":
    main()
