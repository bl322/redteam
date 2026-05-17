from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd

from .core import AgentConfig
from .dataset import DatasetLoader
from .graph import build_graph


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "main" / "data" / "dataset.csv.csv"
DEFAULT_BATCH_DIR = ROOT / "results" / "redteam_batch"


def _invoke_with_thread(app, state: Dict[str, Any], thread_id: str):
    return app.invoke(state, config={"configurable": {"thread_id": thread_id}})


def _summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    history = result.get("history", [])
    refusals = sum(1 for row in history if row.get("judge", {}).get("is_refusal_template"))
    rounds = len(history)
    return {
        "rounds": rounds,
        "refusals": refusals,
        "final_status": history[-1]["judge"]["status"] if history else "empty",
        "last_prompt": result.get("current_text", ""),
    }


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "model"


def _default_batch_paths(dataset_path: Path, model_name: str, population_size: int, max_rounds: int) -> tuple[Path, Path, Path]:
    stem = dataset_path.stem
    model_slug = _slugify(model_name)
    suffix = f"p{population_size}_r{max_rounds}"
    output_path = DEFAULT_BATCH_DIR / f"{stem}_{model_slug}_{suffix}_batch_eval_results.jsonl"
    summary_path = DEFAULT_BATCH_DIR / f"{stem}_{model_slug}_{suffix}_batch_eval_summary.json"
    checkpoint_path = DEFAULT_BATCH_DIR / f"{stem}_{model_slug}_{suffix}_batch_eval_checkpoint.json"
    return output_path, summary_path, checkpoint_path


def _load_existing_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    status = getattr(exc, "status", None)
    if status == 429:
        return True
    message = f"{type(exc).__name__}: {exc}".lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


def run_single(
    seed_text: str,
    model_name: str,
    base_url: str,
    api_key: str,
    population_size: int,
    max_rounds: int,
    use_mock: bool,
) -> tuple[Dict[str, Any], pd.DataFrame, str]:
    config = AgentConfig(
        model_name=model_name,
        base_url=base_url or None,
        api_key=api_key or None,
        use_mock=use_mock,
        population_size=population_size,
        max_rounds=max_rounds,
    )
    app = build_graph(config)
    result = _invoke_with_thread(
        app,
        {
            "seed_text": seed_text,
            "current_text": seed_text,
            "round_index": 0,
            "max_rounds": max_rounds,
            "trace": [],
            "history": [],
        },
        "single-sample",
    )
    history = result.get("history", [])
    df = pd.DataFrame(history)
    return _summarize(result), df, json.dumps(result, ensure_ascii=False, indent=2)


def run_batch(
    csv_file: str,
    limit: int,
    model_name: str,
    base_url: str,
    api_key: str,
    population_size: int,
    max_rounds: int,
    use_mock: bool,
    resume: bool = True,
) -> tuple[pd.DataFrame, str]:
    dataset_path = Path(csv_file)
    rows = DatasetLoader(dataset_path).load_records(limit=limit)
    config = AgentConfig(
        model_name=model_name,
        base_url=base_url or None,
        api_key=api_key or None,
        use_mock=use_mock,
        population_size=population_size,
        max_rounds=max_rounds,
    )
    app = build_graph(config)
    output_path, summary_path, checkpoint_path = _default_batch_paths(dataset_path, model_name, population_size, max_rounds)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_records = _load_existing_records(output_path) if resume else []
    processed_ids = {str(row.get("id", "")).strip() for row in existing_records if str(row.get("id", "")).strip()}
    records: List[Dict[str, Any]] = list(existing_records)
    total = len(existing_records)
    refusals = sum(1 for row in existing_records if int(row.get("refusals", 0)) > 0)
    avg_rounds = [float(row.get("rounds", 0.0)) for row in existing_records]
    by_column: Dict[str, Dict[str, float]] = {}
    by_primary_domain: Dict[str, Dict[str, float]] = {}
    for row in existing_records:
        source_column = row.get("source_column", "unknown")
        primary_domain = row.get("primary_domain", "unknown")
        rounds = float(row.get("rounds", 0.0))
        refusal = int(row.get("refusals", 0)) > 0
        by_column.setdefault(source_column, {"num_samples": 0, "num_refusals": 0, "style_score_sum": 0.0})
        by_primary_domain.setdefault(primary_domain, {"num_samples": 0, "num_refusals": 0, "style_score_sum": 0.0})
        by_column[source_column]["num_samples"] += 1
        by_column[source_column]["num_refusals"] += int(refusal)
        by_column[source_column]["style_score_sum"] += rounds
        by_primary_domain[primary_domain]["num_samples"] += 1
        by_primary_domain[primary_domain]["num_refusals"] += int(refusal)
        by_primary_domain[primary_domain]["style_score_sum"] += rounds

    write_mode = "a" if resume else "w"
    with output_path.open(write_mode, encoding="utf-8") as sink:
        for row in rows:
            row_id = str(row["id"])
            if row_id in processed_ids:
                continue

            result: Optional[Dict[str, Any]] = None
            error_text: Optional[str] = None
            delay = 1.0
            for attempt in range(1, 6):
                try:
                    result = _invoke_with_thread(
                        app,
                        {
                            "seed_text": row["goal"],
                            "current_text": row["goal"],
                            "round_index": 0,
                            "max_rounds": max_rounds,
                            "trace": [],
                            "history": [],
                        },
                        f"batch-{row_id}",
                    )
                    break
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    if not _is_retryable_error(exc) or attempt >= 5:
                        break
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)

            summary = _summarize(result) if result is not None else {
                "rounds": 0,
                "refusals": 0,
                "final_status": "error",
                "last_prompt": row["goal"],
            }
            record = {
                "id": row_id,
                "source_column": row.get("source_column", "unknown"),
                "prompt": row["goal"],
                "primary_domain": row["primary_domain"],
                "secondary_domain": row["secondary_domain"],
                **summary,
            }
            if error_text is not None and result is None:
                record["error"] = error_text

            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            try:
                os.fsync(sink.fileno())
            except OSError:
                pass

            records.append(record)
            total += 1
            refusals += int(record["refusals"] > 0)
            rounds = float(record["rounds"])
            avg_rounds.append(rounds)
            by_column.setdefault(row.get("source_column", "unknown"), {"num_samples": 0, "num_refusals": 0, "style_score_sum": 0.0})
            by_primary_domain.setdefault(row["primary_domain"], {"num_samples": 0, "num_refusals": 0, "style_score_sum": 0.0})
            by_column[row.get("source_column", "unknown")]["num_samples"] += 1
            by_column[row.get("source_column", "unknown")]["num_refusals"] += int(record["refusals"] > 0)
            by_column[row.get("source_column", "unknown")]["style_score_sum"] += rounds
            by_primary_domain[row["primary_domain"]]["num_samples"] += 1
            by_primary_domain[row["primary_domain"]]["num_refusals"] += int(record["refusals"] > 0)
            by_primary_domain[row["primary_domain"]]["style_score_sum"] += rounds
            processed_ids.add(row_id)

            checkpoint = {
                "output_path": str(output_path),
                "processed_samples": total,
                "summary": {
                    "num_samples": total,
                    "num_refusals": refusals,
                    "refusal_trigger_rate": round(refusals / total if total else 0.0, 4),
                    "avg_rounds": round(sum(avg_rounds) / total if total else 0.0, 4),
                    "by_source_column": {
                        key: {
                            "num_samples": int(stats["num_samples"]),
                            "num_refusals": int(stats["num_refusals"]),
                            "refusal_trigger_rate": round(stats["num_refusals"] / stats["num_samples"], 4) if stats["num_samples"] else 0.0,
                            "avg_rounds": round(stats["style_score_sum"] / stats["num_samples"], 4) if stats["num_samples"] else 0.0,
                        }
                        for key, stats in by_column.items()
                    },
                    "by_primary_domain": {
                        key: {
                            "num_samples": int(stats["num_samples"]),
                            "num_refusals": int(stats["num_refusals"]),
                            "refusal_trigger_rate": round(stats["num_refusals"] / stats["num_samples"], 4) if stats["num_samples"] else 0.0,
                            "avg_rounds": round(stats["style_score_sum"] / stats["num_samples"], 4) if stats["num_samples"] else 0.0,
                        }
                        for key, stats in by_primary_domain.items()
                    },
                },
                "updated_at": time.time(),
            }
            checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_path.write_text(json.dumps(checkpoint["summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "num_samples": total,
        "num_refusals": refusals,
        "refusal_trigger_rate": round(refusals / total if total else 0.0, 4),
        "avg_rounds": round(sum(avg_rounds) / total if total else 0.0, 4) if total else 0.0,
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "by_source_column": {
            key: {
                "num_samples": int(stats["num_samples"]),
                "num_refusals": int(stats["num_refusals"]),
                "refusal_trigger_rate": round(stats["num_refusals"] / stats["num_samples"], 4) if stats["num_samples"] else 0.0,
                "avg_rounds": round(stats["style_score_sum"] / stats["num_samples"], 4) if stats["num_samples"] else 0.0,
            }
            for key, stats in by_column.items()
        },
        "by_primary_domain": {
            key: {
                "num_samples": int(stats["num_samples"]),
                "num_refusals": int(stats["num_refusals"]),
                "refusal_trigger_rate": round(stats["num_refusals"] / stats["num_samples"], 4) if stats["num_samples"] else 0.0,
                "avg_rounds": round(stats["style_score_sum"] / stats["num_samples"], 4) if stats["num_samples"] else 0.0,
            }
            for key, stats in by_primary_domain.items()
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return pd.DataFrame(records), json.dumps(summary, ensure_ascii=False, indent=2)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="CC-BOS Red-Team Evaluation Agent") as demo:
        gr.Markdown(
            "# CC-BOS Red-Team Evaluation Agent\n"
            "A defensive multi-agent workflow for prompt robustness evaluation."
        )

        with gr.Tab("Single Sample"):
            seed = gr.Textbox(label="Seed Prompt", lines=6, value="请对以下内容进行安全评测。")
            with gr.Row():
                model_name = gr.Textbox(label="Model", value="qwen3-max")
                base_url = gr.Textbox(label="Base URL", value="")
            with gr.Row():
                api_key = gr.Textbox(label="API Key", type="password", value="")
                use_mock = gr.Checkbox(label="Mock Mode", value=True)
            with gr.Row():
                population_size = gr.Slider(2, 16, value=8, step=1, label="Population Size")
                max_rounds = gr.Slider(1, 10, value=4, step=1, label="Max Rounds")
            run_btn = gr.Button("Run")
            summary = gr.JSON(label="Summary")
            history = gr.Dataframe(label="History")
            raw = gr.Code(label="Raw Result", language="json")
            run_btn.click(
                run_single,
                inputs=[seed, model_name, base_url, api_key, population_size, max_rounds, use_mock],
                outputs=[summary, history, raw],
            )

        with gr.Tab("Batch Dataset"):
            dataset = gr.Textbox(label="Dataset Path (CSV/JSONL)", value=str(DEFAULT_DATASET))
            limit = gr.Slider(1, 200, value=10, step=1, label="Limit")
            resume = gr.Checkbox(label="Resume Existing Output", value=True)
            batch_run = gr.Button("Run Batch")
            batch_df = gr.Dataframe(label="Batch Results")
            batch_meta = gr.Code(label="Batch Meta", language="json")
            batch_run.click(
                run_batch,
                inputs=[dataset, limit, model_name, base_url, api_key, population_size, max_rounds, use_mock, resume],
                outputs=[batch_df, batch_meta],
            )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
