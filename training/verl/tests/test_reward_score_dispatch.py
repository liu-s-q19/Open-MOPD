from __future__ import annotations

from verl.utils.reward_score import default_compute_score


def test_math_dapo_boxed_uses_boxed_math_scorer() -> None:
    result = default_compute_score(
        "math_dapo_boxed",
        "We derive the result.\\n\\boxed{540}",
        "540",
    )

    assert result == 1.0


def test_math_dapo_keeps_answer_line_scorer() -> None:
    result = default_compute_score(
        "math_dapo",
        "We derive the result.\\nAnswer: 540",
        "540",
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
