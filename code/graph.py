from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .core import AdversarialSampleGenerator, AgentConfig, JudgeAgent, TargetEnvironmentAgent


class RedTeamState(TypedDict, total=False):
    seed_text: str
    current_text: str
    round_index: int
    max_rounds: int
    prompt_candidate: Dict[str, Any]
    target_response: str
    judge: Dict[str, Any]
    trace: List[Dict[str, Any]]
    history: List[Dict[str, Any]]


def _append_trace(state: RedTeamState, node: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace = list(state.get("trace", []))
    trace.append({"node": node, **payload})
    return trace


def build_graph(config: AgentConfig):
    generator = AdversarialSampleGenerator(config)
    target = TargetEnvironmentAgent(config)
    judge = JudgeAgent()

    def generate_node(state: RedTeamState) -> Dict[str, Any]:
        seed = state.get("current_text") or state["seed_text"]
        candidate = generator.generate(seed)
        round_index = int(state.get("round_index", 0)) + 1
        trace = _append_trace(state, "generate", {"round": round_index, "prompt": candidate["prompt"][:200]})
        return {
            "current_text": candidate["prompt"],
            "prompt_candidate": candidate,
            "round_index": round_index,
            "trace": trace,
        }

    def interact_node(state: RedTeamState) -> Dict[str, Any]:
        prompt = state["prompt_candidate"]["prompt"]
        response = target.call(prompt)
        trace = _append_trace(state, "interact", {"round": state.get("round_index", 0), "response": response[:200]})
        return {"target_response": response, "trace": trace}

    def judge_node(state: RedTeamState) -> Dict[str, Any]:
        prompt = state["prompt_candidate"]["prompt"]
        response = state.get("target_response", "")
        result = judge.evaluate(prompt, response)
        history_entry = {
            "round": state.get("round_index", 0),
            "prompt": prompt,
            "response": response,
            "judge": result,
        }
        trace = _append_trace(state, "judge", history_entry)
        history = list(state.get("history", []))
        history.append(history_entry)
        return {"judge": result, "history": history, "trace": trace}

    def route(state: RedTeamState) -> str:
        if state.get("judge", {}).get("is_refusal_template"):
            return "stop"
        if int(state.get("round_index", 0)) >= int(state.get("max_rounds", 1)):
            return "stop"
        return "continue"

    graph = StateGraph(RedTeamState)
    graph.add_node("generate", generate_node)
    graph.add_node("interact", interact_node)
    graph.add_node("judge", judge_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "interact")
    graph.add_edge("interact", "judge")
    graph.add_conditional_edges("judge", route, {"continue": "generate", "stop": END})
    return graph.compile(checkpointer=MemorySaver())
