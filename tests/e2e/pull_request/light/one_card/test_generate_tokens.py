#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""E2E test for the vLLM "tokens in <> tokens out" endpoint on Ascend NPU.

Mirrors upstream vLLM ``tests/entrypoints/serve/disagg/test_serving_tokens.py``
but exercises the Ascend NPU code path through ``RemoteOpenAIServer``.

``/inference/v1/generate`` accepts raw ``token_ids`` (no server-side
tokenization) and returns the generated ``token_ids`` directly, i.e. the
token-in-token-out contract used by Disaggregated-Everything coordinators.

The test validates that:

1. A request carrying ``token_ids`` returns generated ``token_ids`` and
   respects ``max_tokens``.
2. Omitting ``max_tokens`` does not silently truncate at the dataclass
   default of 16 (server-side defaulting from ``max_model_len``).
3. Streaming yields the same number of tokens as the non-streaming path
   for a deterministic (greedy) request.
4. The decoded tokens-out match the text returned by ``/v1/chat/completions``
   for the identical token prompt, proving the tokens path is consistent
   with the standard OpenAI-compatible path.
"""

import json

import requests
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

MODEL_NAME = "Qwen/Qwen3-0.6B"
GENERATE_PATH = ("inference", "v1", "generate")
CHAT_COMPLETIONS_PATH = ("v1", "chat", "completions")
REQUEST_TIMEOUT_SECONDS = 600

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How many countries are in the EU?"},
]


def _decode_generate_choice(tokenizer: AutoTokenizer, choice: dict) -> str:
    return tokenizer.decode(choice["token_ids"], skip_special_tokens=True)


@wait_until_npu_memory_free()
def test_generate_tokens_in_tokens_out():
    """Verify ``/inference/v1/generate`` token-in-token-out on NPU."""
    port = get_open_port()
    server_args = [
        "--max-model-len",
        "1024",
        "--enforce-eager",
        # Prefix caching can make cache-hit vs cache-miss prefills take
        # different GEMM shapes and flip argmax tokens, which would break
        # the streaming-vs-non-streaming and chat-completions equality
        # assertions below. Disable it so greedy decoding is reproducible.
        "--no-enable-prefix-caching",
        "--port",
        str(port),
    ]

    with RemoteOpenAIServer(MODEL_NAME, server_args, server_port=port, auto_port=False) as server:
        generate_url = server.url_for(*GENERATE_PATH)
        chat_url = server.url_for(*CHAT_COMPLETIONS_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # 1) Basic token-in-token-out: raw token_ids -> generated token_ids.
        basic_resp = requests.post(
            generate_url,
            json={
                "model": MODEL_NAME,
                "token_ids": [1, 2, 3],
                "sampling_params": {"max_tokens": 5, "temperature": 0.0},
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        basic_resp.raise_for_status()
        basic_data = basic_resp.json()
        assert "choices" in basic_data and len(basic_data["choices"]) == 1
        basic_token_ids = basic_data["choices"][0]["token_ids"]
        assert basic_token_ids is not None
        assert 0 < len(basic_token_ids) <= 5

        # 2) Omitting max_tokens must not silently cap at the dataclass
        #    default of 16; the server fills it from max_model_len.
        default_resp = requests.post(
            generate_url,
            json={
                "model": MODEL_NAME,
                "token_ids": [1, 2, 3],
                "sampling_params": {"temperature": 0.0, "ignore_eos": True},
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        default_resp.raise_for_status()
        default_token_ids = default_resp.json()["choices"][0]["token_ids"]
        assert len(default_token_ids) > 16, (
            f"expected server-side max_tokens default to exceed the legacy 16-token cap, got {len(default_token_ids)}"
        )

        # Tokenize a realistic chat prompt once for the remaining checks.
        prompt_token_ids = tokenizer.apply_chat_template(
            MESSAGES,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 3) Streaming must reconstruct the same tokens as non-streaming for
        #    a deterministic (greedy) request.
        non_stream_resp = requests.post(
            generate_url,
            json={
                "model": MODEL_NAME,
                "token_ids": prompt_token_ids,
                "sampling_params": {"max_tokens": 16, "temperature": 0.0},
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        non_stream_resp.raise_for_status()
        non_stream_token_ids = non_stream_resp.json()["choices"][0]["token_ids"]

        streamed_token_ids: list[int] = []
        last_finish_reason = None
        with requests.post(
            generate_url,
            json={
                "model": MODEL_NAME,
                "token_ids": prompt_token_ids,
                "sampling_params": {"max_tokens": 16, "temperature": 0.0},
                "stream": True,
            },
            stream=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as stream_resp:
            stream_resp.raise_for_status()
            for raw_line in stream_resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                choice = chunk["choices"][0]
                if choice.get("token_ids"):
                    streamed_token_ids.extend(choice["token_ids"])
                if choice.get("finish_reason") is not None:
                    last_finish_reason = choice["finish_reason"]

        assert last_finish_reason is not None
        assert streamed_token_ids == non_stream_token_ids, (
            "streaming and non-streaming token-out must match for greedy decoding"
        )

        # 4) Tokens-out decoded text must match the standard chat-completions
        #    path for the identical token prompt.
        generate_resp = requests.post(
            generate_url,
            json={
                "model": MODEL_NAME,
                "token_ids": prompt_token_ids,
                "sampling_params": {
                    "max_tokens": 24,
                    "temperature": 0.0,
                    "detokenize": False,
                    "ignore_eos": False,
                },
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        generate_resp.raise_for_status()
        generate_text = _decode_generate_choice(tokenizer, generate_resp.json()["choices"][0])

        chat_resp = requests.post(
            chat_url,
            json={
                "model": MODEL_NAME,
                "messages": MESSAGES,
                "max_tokens": 24,
                "temperature": 0.0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        chat_resp.raise_for_status()
        chat_text = chat_resp.json()["choices"][0]["message"]["content"]

        assert generate_text == chat_text, "tokens-out decoded text must match /v1/chat/completions output"
