"""Smoke tests for the supported local shell entry points.

These tests intentionally use ``/bin/echo`` instead of starting a trainer.  They
verify the user-facing contract (shell syntax, local-path validation, and command
construction) without requiring GPUs or model/data downloads.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = (
    "opd",
    "rl",
    "sft",
    "eval",
    "mopd/vanilla_mopd_local",
    "mopd/default_mopd",
    "mopd/vanilla_mopd",
    "mopd/open_mopd",
    "mopd/d3_mopd",
)
SINGLE_NODE_LAUNCHERS = ("opd", "rl", "sft", "eval")


def _run(name: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ROOT / "scripts/local" / f"{name}.sh"), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_launchers_are_valid_shell() -> None:
    for name in LAUNCHERS:
        result = subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/local" / f"{name}.sh")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_local_launchers_print_commands_without_running(tmp_path: Path) -> None:
    model = tmp_path / "model"
    teacher = tmp_path / "teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    model.mkdir()
    teacher.mkdir()
    train.touch()
    val.touch()

    base = (
        "--model",
        str(model),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--torchrun",
        "/bin/echo",
        "--dry-run",
    )
    for name in SINGLE_NODE_LAUNCHERS:
        result = _run(name, *base)
        assert result.returncode == 0, (name, result.stderr)
        assert "[local]" in result.stdout
        assert "://" not in result.stdout
        assert "remote_submit" not in result.stdout

    result = _run(
        "mopd/vanilla_mopd_local",
        *base,
        "--teacher",
        str(teacher),
        "--teacher",
        str(teacher),
        "--domains",
        "math,code",
    )
    assert result.returncode == 0, result.stderr
    assert "://" not in result.stdout
    assert "remote_submit" not in result.stdout


def test_local_launchers_print_repo_verl_pythonpath(tmp_path: Path) -> None:
    model = tmp_path / "model"
    teacher = tmp_path / "teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    model.mkdir()
    teacher.mkdir()
    train.touch()
    val.touch()
    previous_verl = tmp_path / "previous-verl"
    env = {"PYTHONPATH": str(previous_verl)}

    result = _run(
        "mopd/vanilla_mopd_local",
        "--model",
        str(model),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--teacher",
        str(teacher),
        "--teacher",
        str(teacher),
        "--domains",
        "math,code",
        "--dry-run",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"{ROOT / 'training' / 'verl'}" in result.stdout
    assert str(previous_verl) in result.stdout


def test_local_mopd_two_gpu_defaults_are_configured(tmp_path: Path) -> None:
    model = tmp_path / "model"
    teacher = tmp_path / "teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    model.mkdir()
    teacher.mkdir()
    train.touch()
    val.touch()

    result = _run(
        "mopd/vanilla_mopd_local",
        "--model",
        str(model),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--teacher",
        str(teacher),
        "--teacher",
        str(teacher),
        "--gpus",
        "2",
        "--domains",
        "math,code",
        "--dry-run",
        env={"RAY_NUM_CPUS": "2"},
    )

    assert result.returncode == 0, result.stderr
    assert "data.train_batch_size=2" in result.stdout
    assert "reward_model.micro_batch_size_per_gpu=1" in result.stdout
    assert "ray_kwargs.ray_init.num_cpus=2" in result.stdout
    assert "+ray_kwargs.ray_init._node_ip_address=127.0.0.1" in result.stdout
    assert "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE=\\\"1\\\"" in result.stdout


def test_run_mode_rejects_remote_uri(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    result = _run(
        "rl",
        "--model",
        "https://example.invalid/model",
        "--train",
        str(tmp_path / "train.parquet"),
        "--val",
        str(tmp_path / "val.parquet"),
        "--run",
    )
    assert result.returncode != 0
    assert "local filesystem path" in result.stderr


def test_run_mode_executes_with_local_paths_and_fake_binaries(tmp_path: Path) -> None:
    model = tmp_path / "model"
    teacher = tmp_path / "teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    model.mkdir()
    teacher.mkdir()
    train.touch()
    val.touch()

    base = (
        "--model",
        str(model),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--torchrun",
        "/bin/echo",
        "--run",
    )
    for name in SINGLE_NODE_LAUNCHERS:
        result = _run(name, *base)
        assert result.returncode == 0, (name, result.stderr)
        assert "[local]" in result.stdout

    result = _run(
        "mopd/vanilla_mopd_local",
        *base,
        "--teacher",
        str(teacher),
        "--teacher",
        str(teacher),
        "--domains",
        "math,code",
    )
    assert result.returncode == 0, result.stderr


def test_adaptive_mt_opd_prints_dynamic_cluster_configuration(tmp_path: Path) -> None:
    model = tmp_path / "model"
    math_teacher = tmp_path / "math-teacher"
    code_teacher = tmp_path / "code-teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    for path in (model, math_teacher, code_teacher):
        path.mkdir()
    train.touch()
    val.touch()

    result = _run(
        "mopd/default_mopd",
        "--model",
        str(model),
        "--math-teacher",
        str(math_teacher),
        "--code-teacher",
        str(code_teacher),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--gpus",
        "0,1",
        "--dry-run",
        env={
            "NNODES": "2",
            "NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "PYTHONPATH": str(tmp_path / "previous-verl"),
            "RAY_BIN": "/bin/echo",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "nnodes=2" in result.stdout
    assert "visible_gpus=2 expected_gpus=4" in result.stdout
    assert "trainer.nnodes=2" in result.stdout
    assert "trainer.n_gpus_per_node=2" in result.stdout
    assert "reward_mode=mt_opd" in result.stdout
    assert "mt_opd.teacher_domains=" in result.stdout
    assert "ray_init.address=auto" in result.stdout
    assert "reward_model.model.input_tokenizer=null" in result.stdout
    assert "reward_model.reward_kwargs.compute_true_reward=False" in result.stdout
    assert "trainer.total_epochs=3" in result.stdout
    assert str(ROOT / "training" / "verl") in result.stdout
    assert str(tmp_path / "previous-verl") in result.stdout


def test_matched_baseline_mopd_aligns_budget_without_open_mechanisms(tmp_path: Path) -> None:
    model = tmp_path / "model"
    math_teacher = tmp_path / "math-teacher"
    code_teacher = tmp_path / "code-teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    for path in (model, math_teacher, code_teacher):
        path.mkdir()
    train.touch()
    val.touch()

    result = _run(
        "mopd/vanilla_mopd",
        "--model",
        str(model),
        "--math-teacher",
        str(math_teacher),
        "--code-teacher",
        str(code_teacher),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--gpus",
        "0,1",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "ppo_mini=32" in result.stdout
    assert "log_prob_top_k=16" in result.stdout
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=32" in result.stdout
    assert "actor_rollout_ref.actor.opd_refresh_advantage" not in result.stdout
    assert "mt_opd.target_share_domains" not in result.stdout
    assert "mt_opd.normalize_reward_scale" not in result.stdout
    assert "vanilla train_batch=128" in result.stdout
    assert "trainer.project_name=Qwen3-4B" in result.stdout
    assert "trainer.experiment_name=vanilla_mopd" in result.stdout


def test_open_mopd_qwen3_enables_budget_mechanisms(tmp_path: Path) -> None:
    model = tmp_path / "model"
    math_teacher = tmp_path / "math-teacher"
    code_teacher = tmp_path / "code-teacher"
    if_teacher = tmp_path / "if-teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    for path in (model, math_teacher, code_teacher, if_teacher):
        path.mkdir()
    train.touch()
    val.touch()

    result = _run(
        "mopd/open_mopd",
        "--model",
        str(model),
        "--math-teacher",
        str(math_teacher),
        "--code-teacher",
        str(code_teacher),
        "--if-teacher",
        str(if_teacher),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--ray",
        "/bin/echo",
        "--gpus",
        "0,1",
        "--dry-run",
        env={
            "NNODES": "2",
            "NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "PYTHONPATH": str(tmp_path / "previous-verl"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "nnodes=2" in result.stdout
    assert "visible_gpus=2 expected_gpus=4" in result.stdout
    assert f"if:{if_teacher}" in result.stdout
    assert "reward_mode=mt_opd" in result.stdout
    assert "log_prob_top_k=16" in result.stdout
    assert "ppo_mini_batch_size=32" in result.stdout
    assert "actor_rollout_ref.actor.opd_refresh_advantage=True" in result.stdout
    assert "actor_rollout_ref.actor.opd_reward_weight_mode=student_p" in result.stdout
    assert r"+mt_opd.target_share_domains=\[math\,code\,if\]" in result.stdout
    assert r"+mt_opd.target_share_values=\[1\,1\,1\]" in result.stdout
    assert "+mt_opd.normalize_reward_scale=True" in result.stdout
    assert "+mt_opd.reward_scale_direction=multiply" in result.stdout
    assert "+mt_opd.reward_scale_stat=mean" in result.stdout
    assert "inner_updates=4" in result.stdout
    assert str(ROOT / "training" / "verl") in result.stdout
    assert str(tmp_path / "previous-verl") in result.stdout


def test_open_mopd_qwen3_defaults_to_math_if_fallback(tmp_path: Path) -> None:
    model = tmp_path / "model"
    math_teacher = tmp_path / "math-teacher"
    code_teacher = tmp_path / "code-teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    for path in (model, math_teacher, code_teacher):
        path.mkdir()
    train.touch()
    val.touch()

    result = _run(
        "mopd/open_mopd",
        "--model",
        str(model),
        "--math-teacher",
        str(math_teacher),
        "--code-teacher",
        str(code_teacher),
        "--train",
        str(train),
        "--val",
        str(val),
        "--python",
        "/bin/echo",
        "--gpus",
        "0,1",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert f"if:{math_teacher}" in result.stdout
    assert "if_teacher_fallback=1" in result.stdout


def test_d3_mopd_enables_only_dynamic_domain_sampler(tmp_path: Path) -> None:
    model = tmp_path / "model"
    math_teacher = tmp_path / "math-teacher"
    code_teacher = tmp_path / "code-teacher"
    if_teacher = tmp_path / "if-teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    for path in (model, math_teacher, code_teacher, if_teacher):
        path.mkdir()
    train.touch()
    val.touch()

    result = _run(
        "mopd/d3_mopd",
        "--model",
        str(model),
        "--math-teacher",
        str(math_teacher),
        "--code-teacher",
        str(code_teacher),
        "--if-teacher",
        str(if_teacher),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--gpus",
        "0,1",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "[d3-mopd] scheduler=gap*velocity" in result.stdout
    assert "data.sampler.class_name=D3DomainSampler" in result.stdout
    assert "d3_update_interval=10" in result.stdout
    assert "d3_velocity_window=10" in result.stdout
    assert "d3_velocity_windows=3" in result.stdout
    assert "d3_initial_kl_steps=5" in result.stdout
    assert "d3_ema_window=10" in result.stdout
    assert "d3_kl_floor=0.15" in result.stdout
    assert "d3_temperature=0.5" in result.stdout
    assert "d3_mixture_floor=0.10" in result.stdout
    assert "d3_batch_jitter=0.30" in result.stdout
    assert "data.dataloader_num_workers=0" in result.stdout
    assert "actor_rollout_ref.actor.opd_refresh_advantage" not in result.stdout
    assert "mt_opd.target_share_domains" not in result.stdout
    assert "mt_opd.normalize_reward_scale" not in result.stdout
    assert "trainer.experiment_name=d3_mopd" in result.stdout


def test_open_mopd_qwen3_fake_binaries_complete_single_node_run(tmp_path: Path) -> None:
    model = tmp_path / "model"
    math_teacher = tmp_path / "math-teacher"
    code_teacher = tmp_path / "code-teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    for path in (model, math_teacher, code_teacher):
        path.mkdir()
    train.touch()
    val.touch()

    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if (($# == 0)) || [[ \"$1\" == \"-\" ]]; then\n"
        "  cat >/dev/null\n"
        "  echo '2 2'\n"
        "fi\n"
        "exit 0\n"
    )
    fake_python.chmod(0o755)

    result = _run(
        "mopd/open_mopd",
        "--model",
        str(model),
        "--math-teacher",
        str(math_teacher),
        "--code-teacher",
        str(code_teacher),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        str(fake_python),
        "--ray",
        "/bin/echo",
        "--gpus",
        "0,1",
        "--run",
        env={
            "NNODES": "1",
            "NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MAX_WAIT_SECONDS": "5",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "cluster=2/2 available=2/2" in result.stdout
    assert "dry-run only" not in result.stdout
