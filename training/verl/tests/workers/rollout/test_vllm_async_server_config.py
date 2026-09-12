from types import SimpleNamespace

import pytest

from verl.workers.rollout.vllm_rollout.vllm_async_server import _resolve_max_model_len
from verl.workers.rollout.vllm_rollout.utils import _resolve_validation_max_tokens


def test_explicit_async_vllm_max_model_len_is_preserved() -> None:
    config = SimpleNamespace(max_model_len=32768, prompt_length=2048, response_length=2048)

    assert _resolve_max_model_len(config) == 32768


def test_async_vllm_max_model_len_falls_back_to_train_dimensions() -> None:
    config = SimpleNamespace(max_model_len=None, prompt_length=2048, response_length=7168)

    assert _resolve_max_model_len(config) == 9216


@pytest.mark.parametrize(
    ("val_kwargs", "response_length", "expected"),
    [
        (SimpleNamespace(max_tokens=None), 15360, 15360),
        ({"max_tokens": None}, 15360, 15360),
        (SimpleNamespace(max_tokens=4096), 15360, 4096),
        ({"max_tokens": 4096}, 15360, 4096),
    ],
)
def test_validation_max_tokens_never_falls_back_to_worker_local_shape(
    val_kwargs, response_length: int, expected: int
) -> None:
    assert _resolve_validation_max_tokens(val_kwargs, response_length) == expected


@pytest.mark.parametrize("value", [0, -1])
def test_async_vllm_max_model_len_must_be_positive(value: int) -> None:
    config = SimpleNamespace(max_model_len=value, prompt_length=2048, response_length=2048)

    with pytest.raises(ValueError, match="max_model_len must be positive"):
        _resolve_max_model_len(config)
