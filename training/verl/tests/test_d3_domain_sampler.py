from __future__ import annotations

import json
from collections import Counter
from itertools import islice

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.dataset.d3_domain_sampler import D3DomainSampler, hamilton_quotas


class _Frame:
    column_names = ["domain"]

    def __init__(self, domains):
        self.domains = domains

    def __getitem__(self, key):
        assert key == "domain"
        return self.domains


class _Dataset:
    def __init__(self, per_domain=20):
        self.domains = [domain for domain in ("math", "code", "if") for _ in range(per_domain)]
        self.dataframe = _Frame(self.domains)

    def __len__(self):
        return len(self.domains)


def _config(**sampler_values):
    return OmegaConf.create(
        {
            "train_batch_size": 12,
            "gen_batch_size": 12,
            "seed": 7,
            "sampler": sampler_values,
        }
    )


def _batch(values, domains=("math", "code", "if")) -> DataProto:
    domain_values = np.array(domains, dtype=object)
    rm_scores = torch.tensor(values, dtype=torch.float32)
    if rm_scores.dim() == 2:
        response_mask = torch.ones_like(rm_scores, dtype=torch.bool)
    else:
        response_mask = torch.ones(rm_scores.shape[:2], dtype=torch.bool)
    return DataProto.from_dict(
        tensors={"rm_scores": rm_scores, "response_mask": response_mask},
        non_tensors={"domain": domain_values},
    )


def test_hamilton_quotas_sum_to_batch_and_respect_floor() -> None:
    quotas = hamilton_quotas(128, {"math": 0.1, "code": 0.2, "if": 0.7}, ("math", "code", "if"))
    assert sum(quotas.values()) == 128
    assert quotas == {"math": 13, "code": 26, "if": 89}


def test_sampler_emits_exact_dynamic_batches_and_jitter_is_deterministic() -> None:
    config = _config(d3_batch_jitter=0.30)
    first = D3DomainSampler(_Dataset(), config)
    second = D3DomainSampler(_Dataset(), config)
    first_indices = list(islice(iter(first), 24))
    second_indices = list(islice(iter(second), 24))
    assert first_indices == second_indices
    assert len(first_indices) == 24
    assert Counter(first.data_source.domains[index] for index in first_indices[:12]).total() == 12
    assert Counter(first.data_source.domains[index] for index in first_indices[12:]).total() == 12


def test_sampler_warmup_then_composite_ratio_and_status_file(tmp_path) -> None:
    status_file = tmp_path / "scheduler.json"
    config = _config(
        status_file=str(status_file),
        d3_update_interval=10,
        d3_velocity_window=10,
        d3_velocity_windows=3,
        d3_initial_kl_steps=5,
        d3_ema_window=10,
        d3_kl_floor=0.15,
        d3_temperature=0.5,
        d3_mixture_floor=0.10,
        d3_batch_jitter=0.0,
    )
    sampler = D3DomainSampler(_Dataset(), config)

    for step in range(20):
        # Math descends much faster, code steadily descends, IF is flat.
        math_value = max(0.2, 2.0 - step * 0.15)
        code_value = max(0.2, 1.5 - step * 0.03)
        if_value = 1.0
        sampler.update(_batch([[-math_value], [-code_value], [-if_value]]))

    assert sampler.step == 20
    assert sampler.mixture["code"] > sampler.mixture["if"]
    assert sampler.mixture["math"] > sampler.mixture["if"]
    assert sum(sampler.mixture.values()) == pytest.approx(1.0)
    assert all(value >= 0.10 for value in sampler.mixture.values())
    payload = json.loads(status_file.read_text())
    assert payload["step"] == 20
    assert payload["config"]["update_interval"] == 10
    assert set(payload["mixture"]) == {"math", "code", "if"}


def test_velocity_clips_when_kl_increases_and_all_plateau_is_uniform() -> None:
    sampler = D3DomainSampler(
        _Dataset(),
        _config(
            d3_update_interval=1,
            d3_velocity_window=1,
            d3_velocity_windows=1,
            d3_initial_kl_steps=1,
            d3_ema_window=1,
            d3_batch_jitter=0.0,
        ),
    )
    for value in (1.0, 2.0):
        sampler.update(_batch([[-value], [-value], [-value]]))
    assert sampler.velocity["math"] == 0.0
    assert sampler.mixture == {"math": pytest.approx(1 / 3), "code": pytest.approx(1 / 3), "if": pytest.approx(1 / 3)}


def test_sampler_accepts_2d_and_3d_rm_scores() -> None:
    sampler_2d = D3DomainSampler(_Dataset(), _config(d3_update_interval=100))
    sampler_2d.update(_batch([[-1.0], [-2.0], [-3.0]]))

    sampler_3d = D3DomainSampler(_Dataset(), _config(d3_update_interval=100))
    sampler_3d.update(_batch([[[-1.0, 0.0]], [[-2.0, 0.0]], [[-3.0, 0.0]]]))
    assert sampler_2d.raw_kl_history["code"] == [2.0]
    assert sampler_3d.raw_kl_history["if"] == [3.0]


def test_sampler_rejects_missing_domain_or_signal() -> None:
    sampler = D3DomainSampler(_Dataset(), _config())
    with pytest.raises(ValueError, match="every domain"):
        sampler.update(_batch([[-1.0], [-2.0]], domains=("math", "code")))

    missing_signal = DataProto.from_dict(
        tensors={"response_mask": torch.ones(3, 1, dtype=torch.bool)},
        non_tensors={"domain": np.array(["math", "code", "if"], dtype=object)},
    )
    with pytest.raises(ValueError, match="rm_scores"):
        sampler.update(missing_signal)
