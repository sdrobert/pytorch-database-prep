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


@pytest.mark.parametrize("frontend_type", ["recur"])
def test_frontend_batched_matches_full(frontend_type, device):
    T, N, I, H = 100, 10, 30, 5
    input = torch.randn(T, N, I, device=device)
    lens = torch.randint(1, T + 1, (N,), device=device)
    if frontend_type == "recur":
        frontend = asr_baseline.RecurrentFrontend(I, H, recurrent_type="rnn")
    else:
        assert False, f"no frontend of type {frontend_type}"
    frontend.to(device)
    output_act, lens_act = frontend(input, lens)
    for n in range(N):
        lens_n = lens[n : n + 1]
        input_n = input[: lens[n], n : n + 1]
        output_exp_n, lens_exp_n = frontend(input_n, lens_n)
        assert lens_exp_n == lens_act[n], n
        assert torch.allclose(
            output_exp_n, output_act[: lens_act[n], n : n + 1], atol=1e-5
        ), n
