from __future__ import annotations

from verl.utils.reward_score import default_compute_score


def test_code_reward_falls_back_without_pyext() -> None:
    result = default_compute_score(
        "codecontests",
        "```python\nimport sys\nprint(sys.stdin.read().strip())\n```",
        {"inputs": ["42\n"], "outputs": ["42\n"]},
    )

    assert result["acc"] == 1.0
    assert result["base_score"] == 1.0


def test_codeforces_dict_tests_are_normalized_for_lcb_scorer() -> None:
    result = default_compute_score(
        "codeforces",
        "```python\nimport sys\nprint(sys.stdin.read().strip())\n```",
        {"inputs": ["42\n"], "outputs": ["42\n"]},
    )

    assert result["acc"] == 1.0
    assert result["base_score"] == 1.0
