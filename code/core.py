from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset import DatasetLoader
from .engine import (
    ComplianceChecker,
    EchoMockLLM,
    OpenAICompatibleLLM,
    StyleOptimizer,
)


@dataclass
class AgentConfig:
    model_name: str = "qwen3-max"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    use_mock: bool = False
    population_size: int = 8
    max_rounds: int = 4
    temperature: float = 0.2
    max_tokens: int = 512
    seed: int = 42


class AdversarialSampleGenerator:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.optimizer = StyleOptimizer(
            population_size=config.population_size,
            max_iter=config.max_rounds,
            random_seed=config.seed,
        )

    def generate(self, seed_text: str) -> Dict[str, Any]:
        candidate = self.optimizer.optimize(seed_text)
        return {
            "prompt": candidate.rewritten_text,
            "strategy_labels": candidate.strategy_labels,
            "scores": candidate.scores,
        }


class TargetEnvironmentAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        if config.use_mock:
            self.client = EchoMockLLM()
        else:
            self.client = OpenAICompatibleLLM(
                model=config.model_name,
                api_key=config.api_key,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

    def call(self, prompt: str) -> str:
        return self.client.call_llm(prompt)


class JudgeAgent:
    def __init__(self) -> None:
        self.checker = ComplianceChecker()

    def evaluate(self, prompt: str, response: str) -> Dict[str, Any]:
        compliance = self.checker.check_compliance(response)
        refusal = bool(compliance["is_refusal_template"])
        return {
            "is_refusal_template": refusal,
            "matched_keywords": compliance["matched_keywords"],
            "refusal_score": compliance["score"],
            "response_length": len(response or ""),
            "status": "refusal" if refusal else "answered",
        }


def load_seed_prompts_from_csv(csv_path: str | Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    return DatasetLoader(Path(csv_path)).load_records(limit=limit)
