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

import os
import sys
from zlib import adler32

import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(
    params=["cpu", "cuda"],
    scope="session",
)
def device(request):
    if request.param == "cuda":
        return torch.device(torch.cuda.current_device())
    else:
        return torch.device(request.param)


CUDA_AVAIL = torch.cuda.is_available()


def pytest_runtest_setup(item):
    if any(mark.name == "gpu" for mark in item.iter_markers()):
        if not CUDA_AVAIL:
            pytest.skip("cuda is not available")
    # implicitly seeds all tests for the sake of reproducibility
    torch.manual_seed(abs(adler32(bytes(item.name, "utf-8"))))
