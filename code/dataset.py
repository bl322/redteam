from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


TEXT_COLUMN_CANDIDATES = [
    "goal",
    "query",
    "question_zh",
    "text",
    "prompt",
    "instruction",
    "original_instruction",
    "seed_text",
    "content",
]

PRIMARY_DOMAIN_CANDIDATES = [
    "primary_domain",
    "一级领域",
    "source_domain",
    "domain",
]

SECONDARY_DOMAIN_CANDIDATES = [
    "secondary_domain",
    "二级领域",
    "subcategory",
    "subdomain",
]


def _first_nonempty_value(row: Dict[str, object], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        value = row.get(candidate)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _get_record_id(row: Dict[str, object], fallback: int) -> str:
    for candidate in ("id", "ID", "_id", ""):
        value = row.get(candidate)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return str(fallback)


class DatasetLoader:
    def __init__(self, dataset_path: Path, candidate_columns: Optional[Sequence[str]] = None) -> None:
        self.dataset_path = dataset_path
        self.candidate_columns = list(candidate_columns or TEXT_COLUMN_CANDIDATES)

    def load_records(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        suffix = self.dataset_path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return self._load_jsonl_records(limit=limit)
        return self._load_csv_records(limit=limit)

    def _load_csv_records(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        with self.dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{self.dataset_path} has no header row.")
            text_column = self._detect_column(reader.fieldnames)
            records: List[Dict[str, str]] = []
            for idx, row in enumerate(reader):
                record = self._build_record(row, idx, text_column=text_column)
                if record is None:
                    continue
                records.append(record)
                if limit is not None and len(records) >= limit:
                    break
        return records

    def _load_jsonl_records(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        with self.dataset_path.open("r", encoding="utf-8-sig") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{self.dataset_path} line {idx + 1} is not a JSON object.")
                record = self._build_record(row, idx)
                if record is None:
                    continue
                records.append(record)
                if limit is not None and len(records) >= limit:
                    break
        return records

    def _detect_column(self, header: Sequence[str]) -> str:
        lowered = {name.strip().lower(): name for name in header}
        for candidate in self.candidate_columns:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        raise ValueError(
            f"Could not find a usable text column in {self.dataset_path}. "
            f"Tried: {', '.join(self.candidate_columns)}"
        )

    def _build_record(
        self,
        row: Dict[str, object],
        fallback_idx: int,
        text_column: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        if text_column is not None:
            raw_text = row.get(text_column)
            text = str(raw_text).strip() if raw_text is not None else ""
            source_column = text_column
        else:
            text = ""
            source_column = ""
            for candidate in self.candidate_columns:
                raw_text = row.get(candidate)
                if raw_text is None:
                    continue
                candidate_text = str(raw_text).strip()
                if candidate_text:
                    text = candidate_text
                    source_column = candidate
                    break
        if not text:
            return None

        return {
            "id": _get_record_id(row, fallback_idx),
            "source_column": source_column or "unknown",
            "goal": text,
            "primary_domain": _first_nonempty_value(row, PRIMARY_DOMAIN_CANDIDATES) or "unknown",
            "secondary_domain": _first_nonempty_value(row, SECONDARY_DOMAIN_CANDIDATES) or "unknown",
        }
