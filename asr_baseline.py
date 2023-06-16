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

from typing import Callable, Collection, Optional, Sequence, Tuple
from typing_extensions import Literal

import param
import torch


def check_in(name: str, val: str, choices: Collection[str]):
    if val not in choices:
        choices = "', '".join(sorted(choices))
        raise ValueError(f"Expected {name} to be one of '{choices}'; got '{val}'")


def check_positive(name: str, val, nonnegative=False):
    pos = "non-negative" if nonnegative else "positive"
    if val < 0 or (val == 0 and not nonnegative):
        raise ValueError(f"Expected {name} to be {pos}; got {val}")


class Frontend(torch.nn.Module, metaclass=abc.ABCMeta):
    """Frontend ABC

    The frontend is the part of the baseline which, given some input, always performs
    the same computations in the same order. That is, it is not influenced by the
    transduction process.

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

    @abc.abstractmethod
    def forward(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    __call__: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class RecurrentFrontend(Frontend):
    """Frontend with a recurrent architecture"""

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
        super().__init__(input_size, hidden_size)
        if recurrent_type == "lstm":
            class_ = torch.nn.LSTM
        elif recurrent_type == "gru":
            class_ = torch.nn.GRU
        else:
            class_ = torch.nn.RNN
        self.rnn = class_(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

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


class BaselineParameters(param.Parameterized):
    """All parameters for a baseline model"""

    input_size: int = param.Integer(
        41, bounds=(1, None), doc="Size of input feature vectors"
    )
    vocab_size: int = param.Integer(
        32, bounds=(1, None), doc="Size of output vocabulary"
    )

    transducer_type: Literal["ctc", "encdec"] = param.ObjectSelector(
        "ctc", ["ctc", "encdec"], doc="How to perform audio -> text transduction"
    )


def main(args: Optional[Sequence[str]] = None):
    pass
