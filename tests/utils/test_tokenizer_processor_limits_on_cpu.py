# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

from verl.utils.tokenizer import _apply_image_processor_limits, _max_pixels_from_image_tokens


def test_max_pixels_from_image_tokens_for_square_patch():
    image_processor = SimpleNamespace(patch_size=16, merge_size=2)
    max_pixels = _max_pixels_from_image_tokens(image_processor, max_image_tokens=10000)
    assert max_pixels == 16 * 16 * 2 * 2 * 10000


def test_max_pixels_from_image_tokens_for_rectangular_patch():
    image_processor = SimpleNamespace(patch_size=(14, 16), merge_size=2)
    max_pixels = _max_pixels_from_image_tokens(image_processor, max_image_tokens=10000)
    assert max_pixels == 14 * 16 * 2 * 2 * 10000


def test_apply_image_processor_limits_updates_size_and_attrs():
    image_processor = SimpleNamespace(size={"shortest_edge": 1024, "longest_edge": 4096})
    processor = SimpleNamespace(image_processor=image_processor)

    _apply_image_processor_limits(processor, min_pixels=2048, max_pixels=8192)

    assert processor.image_processor.size["shortest_edge"] == 2048
    assert processor.image_processor.size["longest_edge"] == 8192
    assert processor.image_processor.min_pixels == 2048
    assert processor.image_processor.max_pixels == 8192
