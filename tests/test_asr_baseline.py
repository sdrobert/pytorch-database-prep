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

from typing import Dict, Tuple
import torch
import pytest

from pydrobert.torch.modules import MixableSequentialLanguageModel
from asr_baseline import *


@pytest.mark.parametrize("encoder_type", ["recur", "ff"])
def test_encoder_batched_matches_full(encoder_type, device):
    T, N, I, H = 100, 10, 30, 4
    input = torch.randn(T, N, I, device=device)
    lens = torch.randint(1, T + 1, (N,), device=device)
    if encoder_type == "recur":
        encoder = RecurrentEncoder(I, H)
    elif encoder_type == "ff":
        encoder = FeedForwardEncoder(I, H, 2)
    else:
        assert False, f"no encoder of type {encoder_type}"
    encoder.to(device)
    output_act, lens_act = encoder(input, lens)
    assert output_act.shape[1:] == (N, H)
    for n in range(N):
        lens_n = lens[n : n + 1]
        input_n = input[: lens[n], n : n + 1]
        output_exp_n, lens_exp_n = encoder(input_n, lens_n)
        assert lens_exp_n == lens_act[n], n
        assert torch.allclose(
            output_exp_n, output_act[: lens_act[n], n : n + 1], atol=1e-5
        ), n


def test_recurrent_decoder_with_attention(device):
    T, S, N, V, I, H = 10, 20, 30, 40, 50, 60
    input = torch.randn(T, N, I, device=device)
    lens = torch.randint(1, T + 1, (N,), device=device)
    hist = torch.randint(0, V, (S, N), device=device)
    decoder = RecurrentDecoderWithAttention(I, V, decoder_hidden_size=H).to(device)
    log_probs_exp = []
    for n in range(N):
        input_n, hist_n = input[: lens[n], n : n + 1], hist[:, n : n + 1]
        log_probs_exp.append(decoder(hist_n, {"input": input_n}))
    log_probs_exp = torch.cat(log_probs_exp, 1)
    assert log_probs_exp.shape == (S + 1, N, V)
    log_probs_act = decoder(hist, {"input": input, "lens": lens})
    assert log_probs_exp.shape == log_probs_act.shape
    assert torch.allclose(log_probs_exp, log_probs_act, atol=1e-5)


class DummyLM(MixableSequentialLanguageModel):
    def __init__(self, vocab_size: int):
        super().__init__(vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size + 1, vocab_size, vocab_size)

    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        x = torch.nn.functional.log_softmax(self.embedding(x), 1)
        return x, prev

    def extract_by_src(
        self, prev: Dict[str, torch.Tensor], src: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return prev

    def mix_by_mask(
        self,
        prev_true: Dict[str, torch.Tensor],
        prev_false: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return prev_true


@pytest.mark.parametrize("with_lm", [True, False], ids=["w/ lm", "w/o lm"])
@pytest.mark.parametrize("recognizer", ["ctc", "ctc-greedy", "encdec"])
def test_recognizer(device, recognizer, with_lm):
    T, N, V, I, H = 50, 20, 20, 10, 5
    feats = torch.randn(T, N, I, device=device)
    lens = torch.randint(1, T + 1, (N,), device=device)
    refs = torch.randint(0, V, (T, N), device=device)
    ref_lens = (torch.rand(N, device=device) * lens).clamp_min_(1).long()
    lm = DummyLM(V) if with_lm else None
    encoder = FeedForwardEncoder(I, H)
    if recognizer.startswith("ctc"):
        recognizer = CTCSpeechRecognizer(
            V, encoder, 0 if recognizer == "ctc-greedy" else 8, lm=lm
        ).to(device)
        # CTC loss averages over reference lengths per batch element, THEN batch
        # elements
        nums, denom = [1] * N, N
    elif recognizer == "encdec":
        decoder = RecurrentDecoderWithAttention(H, V)
        recognizer = EncoderDecoderSpeechRecognizer(
            encoder, decoder, lm=lm, max_iters=2 * T
        ).to(device)
        nums, denom = ref_lens, ref_lens.sum()
    else:
        assert False, f"unknown recognizer {recognizer}"
    loss_exp = recognizer.train_loss(feats, lens, refs, ref_lens)
    hyps_exp, hyp_lens_exp = recognizer.decode(feats, lens)
    loss_act = 0.0
    for n in range(N):
        feats_n, lens_n = feats[: lens[n], n : n + 1], lens[n : n + 1]
        refs_n, ref_lens_n = refs[: ref_lens[n], n : n + 1], ref_lens[n : n + 1]
        hyps_n_exp = hyps_exp[: hyp_lens_exp[n], n : n + 1]
        hyp_lens_n_exp = hyp_lens_exp[n : n + 1]
        loss_n_act = recognizer.train_loss(feats_n, lens_n, refs_n, ref_lens_n)
        loss_act = loss_act + nums[n] * loss_n_act
        hyps_n_act, hyp_lens_n_act = recognizer.decode(feats_n, lens_n)
        assert hyp_lens_n_exp == hyp_lens_n_act, n
        hyps_n_act = hyps_n_act[: hyps_n_exp.size(0)]
        assert (hyps_n_exp == hyps_n_act).all(), n

    loss_act = loss_act / denom
    assert torch.allclose(loss_exp, loss_act)


@pytest.mark.parametrize("with_lm", [True, False], ids=["w/ lm", "w/o lm"])
@pytest.mark.parametrize("encoder_type", ["recur", "ff"])
@pytest.mark.parametrize("transducer_type", ["ctc", "encdec"])
def test_construct_baseline(with_lm, encoder_type, transducer_type):
    T, N, V, I = 5, 10, 15, 20
    feats = torch.rand(T, N, I)
    lens = torch.randint(1, T + 1, (N,))
    lm = DummyLM(V) if with_lm else None
    params = BaselineParams(
        encoder_type=encoder_type,
        transducer_type=transducer_type,
        vocab_size=V,
        input_size=I,
    )
    params.initialize_missing()
    recognizer = construct_baseline(params, lm)
    hyps, hyp_lens = recognizer(feats, lens)
    assert hyp_lens.shape == lens.shape
    assert hyps.size(0) >= hyp_lens.max()
    assert hyps.size(1) == N
