"""D^3-MOPD dynamic domain sampler.

This module deliberately lives behind the external ``data.sampler`` hook.  It
does not change the default sampler or the MOPD loss; a launcher must opt in by
loading :class:`D3DomainSampler` explicitly.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sized
from typing import Any

from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


DEFAULT_DOMAIN_ORDER = ("math", "code", "if")


def hamilton_quotas(total: int, weights: dict[str, float], domain_order: tuple[str, ...]) -> dict[str, int]:
    """Allocate an integer batch quota using the largest-remainder method."""

    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if not domain_order:
        raise ValueError("domain_order must not be empty")
    values = {domain: float(weights[domain]) for domain in domain_order}
    if any(not math.isfinite(weight) or weight <= 0 for weight in values.values()):
        raise ValueError(f"domain weights must be finite and positive: {values}")

    denominator = sum(values.values())
    raw = {domain: total * weight / denominator for domain, weight in values.items()}
    quotas = {domain: math.floor(value) for domain, value in raw.items()}
    remainder = total - sum(quotas.values())
    priority = sorted(
        domain_order,
        key=lambda domain: (-(raw[domain] - quotas[domain]), domain_order.index(domain)),
    )
    for domain in priority[:remainder]:
        quotas[domain] += 1
    return quotas


def _as_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


class D3DomainSampler(AbstractCurriculumSampler):
    """Stratified sampler with D^3-MOPD gap/velocity scheduling.

    The trainer invokes ``update(batch)`` after each optimization step through
    the existing ``AbstractCurriculumSampler`` hook.  The sampler reads the
    already-computed OPD reward tensor and treats ``-rm_scores`` as the sampled
    reverse-KL estimate.  No additional teacher or student forward pass is
    introduced.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        sampler_config = data_config.get("sampler", {}) or {}

        configured_order = sampler_config.get("domain_order", DEFAULT_DOMAIN_ORDER)
        if OmegaConf.is_config(configured_order):
            configured_order = OmegaConf.to_container(configured_order, resolve=True)
        self.domain_order = tuple(str(domain) for domain in configured_order)
        if not self.domain_order or len(set(self.domain_order)) != len(self.domain_order):
            raise ValueError(f"domain_order must contain unique domains, got {self.domain_order}")

        self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))
        self.seed = int(data_config.get("seed", 0) or 0)
        self.num_samples = (len(data_source) // self.batch_size) * self.batch_size
        if self.num_samples <= 0:
            raise ValueError(
                f"dataset length {len(data_source)} is smaller than batch size {self.batch_size}"
            )

        self.update_interval = int(sampler_config.get("d3_update_interval", 10))
        self.velocity_window = int(sampler_config.get("d3_velocity_window", 10))
        self.velocity_windows = int(sampler_config.get("d3_velocity_windows", 3))
        self.initial_kl_steps = int(sampler_config.get("d3_initial_kl_steps", 5))
        self.ema_window = int(sampler_config.get("d3_ema_window", 10))
        self.kl_floor = _as_float(sampler_config.get("d3_kl_floor", 0.15), "d3_kl_floor")
        self.temperature = _as_float(sampler_config.get("d3_temperature", 0.5), "d3_temperature")
        self.mixture_floor = _as_float(sampler_config.get("d3_mixture_floor", 0.10), "d3_mixture_floor")
        self.batch_jitter = _as_float(sampler_config.get("d3_batch_jitter", 0.30), "d3_batch_jitter")
        self.status_file = sampler_config.get("status_file", None)
        self.status_file = str(self.status_file) if self.status_file else None

        if self.update_interval <= 0:
            raise ValueError("d3_update_interval must be positive")
        if self.velocity_window <= 0 or self.velocity_windows <= 0:
            raise ValueError("D^3 velocity windows must be positive")
        if self.initial_kl_steps <= 0 or self.ema_window <= 0:
            raise ValueError("D^3 initial_kl_steps and ema_window must be positive")
        if self.kl_floor <= 0 or self.temperature <= 0:
            raise ValueError("d3_kl_floor and d3_temperature must be positive")
        if not 0 <= self.batch_jitter < 1:
            raise ValueError("d3_batch_jitter must be in [0, 1)")
        if not 0 <= self.mixture_floor and len(self.domain_order) * self.mixture_floor < 1:
            raise ValueError(
                "d3_mixture_floor must be non-negative and satisfy K * floor < 1"
            )

        dataframe = getattr(data_source, "dataframe", None)
        if dataframe is None or "domain" not in dataframe.column_names:
            raise ValueError("D3DomainSampler requires a top-level dataset 'domain' column")
        pools: dict[str, list[int]] = defaultdict(list)
        for index, domain_value in enumerate(dataframe["domain"]):
            pools[str(domain_value)].append(index)
        missing = [domain for domain in self.domain_order if not pools[domain]]
        if missing:
            raise ValueError(f"D3DomainSampler has empty domain pools: {missing}")
        self.pools = {domain: pools[domain] for domain in self.domain_order}

        uniform = 1.0 / len(self.domain_order)
        self.mixture = {domain: uniform for domain in self.domain_order}
        self.step = 0
        self.epoch = 0
        self.raw_kl_history = {domain: [] for domain in self.domain_order}
        self.ema_kl_history = {domain: [] for domain in self.domain_order}
        self.initial_kl: dict[str, float | None] = {domain: None for domain in self.domain_order}
        self.remaining_gap = {domain: 0.0 for domain in self.domain_order}
        self.velocity = {domain: 0.0 for domain in self.domain_order}
        self.signal = {domain: 0.0 for domain in self.domain_order}
        self.last_quota = hamilton_quotas(self.batch_size, self.mixture, self.domain_order)
        self._write_status()

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools = {domain: list(indices) for domain, indices in self.pools.items()}
        cursors = {domain: 0 for domain in self.domain_order}
        for indices in pools.values():
            rng.shuffle(indices)

        def take(domain: str, count: int) -> list[int]:
            selected: list[int] = []
            while len(selected) < count:
                pool = pools[domain]
                cursor = cursors[domain]
                remaining = len(pool) - cursor
                amount = min(count - len(selected), remaining)
                selected.extend(pool[cursor : cursor + amount])
                cursors[domain] += amount
                if cursors[domain] == len(pool):
                    rng.shuffle(pool)
                    cursors[domain] = 0
            return selected

        num_batches = self.num_samples // self.batch_size
        for _ in range(num_batches):
            batch_weights = self._jittered_mixture(rng)
            quotas = hamilton_quotas(self.batch_size, batch_weights, self.domain_order)
            self.last_quota = quotas
            batch: list[int] = []
            for domain in self.domain_order:
                batch.extend(take(domain, quotas[domain]))
            rng.shuffle(batch)
            yield from batch

    def _jittered_mixture(self, rng: random.Random) -> dict[str, float]:
        if self.batch_jitter == 0:
            return dict(self.mixture)
        jittered = {
            domain: weight * (1.0 + rng.uniform(-self.batch_jitter, self.batch_jitter))
            for domain, weight in self.mixture.items()
        }
        total = sum(jittered.values())
        return {domain: weight / total for domain, weight in jittered.items()}

    def update(self, batch: DataProto) -> None:
        """Consume one training batch and update the next mixture if due."""

        domains = batch.non_tensor_batch.get("domain")
        if domains is None:
            raise ValueError("D3DomainSampler.update requires batch.non_tensor_batch['domain']")
        response_mask = batch.batch.get("response_mask")
        rm_scores = batch.batch.get("rm_scores")
        if response_mask is None or rm_scores is None:
            raise ValueError("D3DomainSampler.update requires response_mask and rm_scores")

        labels = [str(domain) for domain in domains]
        if len(labels) != response_mask.shape[0]:
            raise ValueError(
                f"domain count {len(labels)} does not match response batch {response_mask.shape[0]}"
            )
        missing = [domain for domain in self.domain_order if domain not in labels]
        if missing:
            raise ValueError(
                f"D3 batch must contain every domain so its KL trajectory remains defined: {missing}"
            )

        reward = rm_scores.detach()
        if reward.dim() == 3:
            reward = reward.sum(dim=-1)
        if reward.dim() != 2 or reward.shape != response_mask.shape:
            raise ValueError(
                "D3DomainSampler expects rm_scores with shape [batch, response] or "
                f"[batch, response, topk], got {tuple(rm_scores.shape)} for mask {tuple(response_mask.shape)}"
            )

        mask = response_mask.detach().bool()
        kl_estimate = -reward.float()
        for domain in self.domain_order:
            indices = [index for index, label in enumerate(labels) if label == domain]
            domain_mask = mask[indices]
            denominator = domain_mask.sum().item()
            if denominator <= 0:
                raise ValueError(f"D3 domain {domain!r} has no valid response tokens")
            value = (kl_estimate[indices] * domain_mask).sum().item() / denominator
            if not math.isfinite(value):
                raise ValueError(f"non-finite reverse-KL estimate for domain {domain}: {value}")
            # The top-k estimator can have tiny negative numerical noise even
            # though the full reverse-KL is non-negative.
            value = max(0.0, value)
            self.raw_kl_history[domain].append(value)
            previous = self.ema_kl_history[domain][-1] if self.ema_kl_history[domain] else None
            alpha = 2.0 / (self.ema_window + 1.0)
            ema = value if previous is None else alpha * value + (1.0 - alpha) * previous
            self.ema_kl_history[domain].append(ema)
            if len(self.raw_kl_history[domain]) == self.initial_kl_steps:
                self.initial_kl[domain] = sum(self.raw_kl_history[domain]) / self.initial_kl_steps

        self.step += 1
        if self.step % self.update_interval == 0:
            self._update_mixture()
        else:
            self._write_status()

    def _update_mixture(self) -> None:
        warmup_steps = 2 * self.velocity_window
        if self.step < warmup_steps or any(value is None for value in self.initial_kl.values()):
            self.mixture = {domain: 1.0 / len(self.domain_order) for domain in self.domain_order}
            self._write_status()
            return

        available_windows = min(self.velocity_windows, self.step // self.velocity_window - 1)
        if available_windows <= 0:
            self.mixture = {domain: 1.0 / len(self.domain_order) for domain in self.domain_order}
            self._write_status()
            return

        for domain in self.domain_order:
            ema_history = self.ema_kl_history[domain]
            current = ema_history[-1]
            initial = max(float(self.initial_kl[domain]), self.kl_floor)
            self.remaining_gap[domain] = max(0.0, current / initial)
            deltas = []
            current_index = self.step - 1
            for window_index in range(available_windows):
                newer = ema_history[current_index - window_index * self.velocity_window]
                older = ema_history[current_index - (window_index + 1) * self.velocity_window]
                deltas.append((newer - older) / max(older, self.kl_floor))
            self.velocity[domain] = max(0.0, -sum(deltas) / len(deltas))
            self.signal[domain] = self.remaining_gap[domain] * self.velocity[domain]

        max_signal = max(self.signal.values())
        if max_signal <= 0:
            normalized = {domain: 0.0 for domain in self.domain_order}
        else:
            normalized = {domain: self.signal[domain] / max_signal for domain in self.domain_order}

        logits = [normalized[domain] / self.temperature for domain in self.domain_order]
        max_logit = max(logits)
        exponentials = [math.exp(logit - max_logit) for logit in logits]
        denominator = sum(exponentials)
        softmax = [value / denominator for value in exponentials]
        residual = 1.0 - len(self.domain_order) * self.mixture_floor
        self.mixture = {
            domain: self.mixture_floor + residual * probability
            for domain, probability in zip(self.domain_order, softmax, strict=True)
        }
        self._write_status()

    def _status_payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "domain_order": list(self.domain_order),
            "mixture": self.mixture,
            "last_quota": self.last_quota,
            "raw_kl": self.raw_kl_history,
            "ema_kl": self.ema_kl_history,
            "initial_kl": self.initial_kl,
            "remaining_gap": self.remaining_gap,
            "velocity": self.velocity,
            "signal": self.signal,
            "config": {
                "update_interval": self.update_interval,
                "velocity_window": self.velocity_window,
                "velocity_windows": self.velocity_windows,
                "initial_kl_steps": self.initial_kl_steps,
                "ema_window": self.ema_window,
                "kl_floor": self.kl_floor,
                "temperature": self.temperature,
                "mixture_floor": self.mixture_floor,
                "batch_jitter": self.batch_jitter,
            },
        }

    def _write_status(self) -> None:
        if not self.status_file:
            return
        directory = os.path.dirname(os.path.abspath(self.status_file))
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".d3-status-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self._status_payload(), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.status_file)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
