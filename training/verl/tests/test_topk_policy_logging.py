"""Regression tests for PPO metrics with dense top-k log probabilities."""

import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla


def test_topk_clip_metrics_are_normalized_over_topk_entries():
    """Top-k clip fractions must be probabilities, not K times probabilities."""

    # The first token is upper-clipped for both top-k entries.  The second
    # token triggers dual-clip's lower fraction for both entries.
    old_log_prob = torch.zeros(1, 2, 2)
    log_prob = torch.log(torch.tensor([[[1.5, 1.5], [4.0, 4.0]]]))
    advantages = torch.tensor([[[1.0], [-1.0]]])
    response_mask = torch.ones(1, 2)
    config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "clip_ratio_c": 3.0,
        }
    )

    _, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    assert metrics["actor/pg_clipfrac"] == 0.5
    assert metrics["actor/pg_clipfrac_lower"] == 0.5
    assert 0.0 <= metrics["actor/pg_clipfrac"] <= 1.0
    assert 0.0 <= metrics["actor/pg_clipfrac_lower"] <= 1.0
