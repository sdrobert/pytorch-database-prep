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

import torch
import pytest
import asr_baseline


@pytest.mark.parametrize("encoder_type", ["recur", "ff"])
def test_encoder_batched_matches_full(encoder_type, device):
    T, N, I, H = 100, 10, 30, 4
    input = torch.randn(T, N, I, device=device)
    lens = torch.randint(1, T + 1, (N,), device=device)
    if encoder_type == "recur":
        encoder = asr_baseline.RecurrentEncoder(I, H)
    elif encoder_type == "ff":
        encoder = asr_baseline.FeedForwardEncoder(I, H, 2)
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
    T, S, N, V, I = 10, 20, 30, 40, 50
    input = torch.randn(T, N, I, device=device)
    lens = torch.randint(1, T + 1, (N,), device=device)
    hist = torch.randint(0, V, (S, N), device=device)
    decoder = asr_baseline.RecurrentDecoderWithAttention(I, V).to(device)
    log_probs_exp = []
    for n in range(N):
        input_n, hist_n = input[: lens[n], n : n + 1], hist[:, n : n + 1]
        log_probs_exp.append(decoder(hist_n, {"input": input_n}))
    log_probs_exp = torch.cat(log_probs_exp, 1)
    assert log_probs_exp.shape == (S + 1, N, V)
    log_probs_act = decoder(hist, {"input": input, "lens": lens})
    assert log_probs_exp.shape == log_probs_act.shape
    assert torch.allclose(log_probs_exp, log_probs_act, atol=1e-5)
