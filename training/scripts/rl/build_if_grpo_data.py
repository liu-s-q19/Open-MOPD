#!/usr/bin/env python3
"""Build the Open-MOPD IF-GRPO train and validation parquets.

The Nemotron instruction-following JSONL is converted to the training schema
already used by Open-MOPD. The aligned IFEval/IFBench parquets are converted
to the same schema while retaining official-evaluation metadata in
``extra_info`` for the custom reward function.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ifrl_family import IF_RL_SYSTEM_PROMPT, classify_family


DATA_SOURCE = "nemotron_if_rl"
ABILITY = "instruction_following"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--ifeval-parquet", type=Path, required=True)
    parser.add_argument("--ifbench-parquet", type=Path, default=None)
    parser.add_argument("--output-train-parquet", type=Path, required=True)
    parser.add_argument("--output-val-parquet", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_schema() -> pa.Schema:
    message_struct = pa.struct([("role", pa.string()), ("content", pa.string())])
    extra_info_struct = pa.struct(
        [
            ("sample_id", pa.int64()),
            ("raw_prompt", pa.string()),
            ("instruction_id_list", pa.list_(pa.string())),
            ("instruction_kwargs_json", pa.list_(pa.string())),
            ("dataset", pa.string()),
            ("agent_name", pa.string()),
            ("family", pa.string()),
            ("metadata", pa.string()),
            ("evaluator", pa.string()),
            ("eval_prompt_for_scorer", pa.string()),
        ]
    )
    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(message_struct)),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            ("extra_info", extra_info_struct),
        ]
    )


def _json_kwargs(value: Any, instruction_ids: list[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized_items = []
    for instruction_id, item in zip(instruction_ids or [], value, strict=False):
        if instruction_id == "count:count_increment_word" and isinstance(item, dict):
            item = dict(item)
            for key in ("keyword1", "keyword2"):
                candidate = item.get(key)
                if isinstance(candidate, list) and len(candidate) == 1:
                    item[key] = candidate[0]
        normalized_items.append(item)
    if instruction_ids is None:
        normalized_items = value
    return [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in normalized_items]


def _sample_id(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _train_record(obj: dict[str, Any]) -> dict[str, Any]:
    prompt = str(obj["prompt"]).strip()
    instruction_ids = [str(item) for item in obj["instruction_id_list"]]
    kwargs = obj["kwargs"]
    if not prompt or not instruction_ids or not isinstance(kwargs, list) or len(kwargs) != len(instruction_ids):
        raise ValueError("invalid instruction-following train row")
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": IF_RL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "sample_id": _sample_id(obj.get("id"), 0),
            "raw_prompt": prompt,
            "instruction_id_list": instruction_ids,
            "instruction_kwargs_json": _json_kwargs(kwargs, instruction_ids),
            "dataset": str(obj.get("dataset", "")),
            "agent_name": str((obj.get("agent_ref") or {}).get("name", "")),
            "family": classify_family(prompt),
            "metadata": "",
            "evaluator": "",
            "eval_prompt_for_scorer": "",
        },
    }


def _parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("validation row metadata must be a JSON object")


def _eval_record(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    messages = row.get("prompt")
    if not isinstance(messages, list) or not messages:
        raise ValueError("validation row prompt must be a non-empty message list")
    metadata = _parse_metadata(row.get("metadata"))
    prompt = str(row.get("original_prompt") or metadata.get("prompt") or _user_prompt(messages)).strip()
    dataset = str(row.get("dataset") or "")
    evaluator = str(row.get("evaluator") or "")
    instruction_ids = [str(item) for item in metadata.get("instruction_id_list", [])]
    kwargs = metadata.get("kwargs", [])
    if (
        not prompt
        or not dataset
        or not instruction_ids
        or not isinstance(kwargs, list)
        or len(kwargs) != len(instruction_ids)
    ):
        raise ValueError("invalid aligned IF validation row")
    return {
        "data_source": dataset,
        "prompt": messages,
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "sample_id": _sample_id(row.get("sample_id"), row_index),
            "raw_prompt": prompt,
            "instruction_id_list": instruction_ids,
            "instruction_kwargs_json": _json_kwargs(kwargs, instruction_ids),
            "dataset": dataset,
            "agent_name": "",
            "family": classify_family(prompt),
            "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            "evaluator": evaluator,
            "eval_prompt_for_scorer": prompt,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_eval(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _write_records(path: Path, records: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path)


def _ensure_inputs(args: argparse.Namespace) -> None:
    input_paths = [args.train_jsonl, args.ifeval_parquet]
    if args.ifbench_parquet is not None:
        input_paths.append(args.ifbench_parquet)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
    existing = [
        path for path in (args.output_train_parquet, args.output_val_parquet, args.summary_path) if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError(
            "output already exists; pass --force to rebuild: " + ", ".join(str(path) for path in existing)
        )


def main() -> None:
    args = parse_args()
    _ensure_inputs(args)

    train_records = [_train_record(obj) for obj in _read_jsonl(args.train_jsonl)]
    random.Random(args.seed).shuffle(train_records)

    val_records: list[dict[str, Any]] = []
    val_counts: dict[str, int] = {}
    validation_paths = [args.ifeval_parquet]
    if args.ifbench_parquet is not None:
        validation_paths.append(args.ifbench_parquet)
    for path in validation_paths:
        converted = [_eval_record(row, index) for index, row in enumerate(_read_eval(path))]
        val_records.extend(converted)
        val_counts[str(path)] = len(converted)

    schema = build_schema()
    _write_records(args.output_train_parquet, train_records, schema)
    _write_records(args.output_val_parquet, val_records, schema)
    summary = {
        "train_jsonl": str(args.train_jsonl),
        "ifeval_parquet": str(args.ifeval_parquet),
        "ifbench_parquet": str(args.ifbench_parquet) if args.ifbench_parquet is not None else None,
        "output_train_parquet": str(args.output_train_parquet),
        "output_val_parquet": str(args.output_val_parquet),
        "seed": args.seed,
        "system_prompt": IF_RL_SYSTEM_PROMPT,
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "val_counts": val_counts,
        "data_sources": sorted({record["data_source"] for record in val_records}),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
