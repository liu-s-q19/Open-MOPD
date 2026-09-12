from __future__ import annotations

import torch

from verl.workers.fsdp_workers import RewardModelWorker


def test_entropy_accepts_noncontiguous_logits() -> None:
    worker = RewardModelWorker.__new__(RewardModelWorker)
    contiguous = torch.randn(2, 5, 17)
    # The production path creates this kind of strided tensor by slicing the
    # model output to the response positions.
    noncontiguous = contiguous.transpose(0, 1)
    assert not noncontiguous.is_contiguous()

    actual = worker._compute_entropy_safe(noncontiguous, chunk_size=3)
    expected = torch.special.entr(torch.softmax(noncontiguous, dim=-1)).sum(dim=-1)

    assert actual.shape == noncontiguous.shape[:-1]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_entropy_accepts_two_dimensional_logits() -> None:
    worker = RewardModelWorker.__new__(RewardModelWorker)
    logits = torch.randn(7, 19)

    actual = worker._compute_entropy_safe(logits, chunk_size=2)
    expected = torch.special.entr(torch.softmax(logits, dim=-1)).sum(dim=-1)

    assert actual.shape == (7,)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
