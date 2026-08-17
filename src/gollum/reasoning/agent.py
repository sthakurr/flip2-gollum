"""LLM-based re-ranking of acquisition-function candidates.

`select_acquisitions_with_llm` takes the top-k candidates proposed by a
BoTorch acquisition function, the history of already-acquired points, and the
dataset's wild-type sequence; formats them into the prompt template;
sends the prompt to an LLM (Claude, Gemini, or GPT, configured via
`llm_config`); and parses the returned JSON ranking to pick the batch that
actually gets acquired. The LLM's native reasoning/thinking trace is returned
alongside the chosen candidates so callers can log it.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from gollum.utils.config import instantiate_class

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "p0.txt")


def _mutations(sequence: str, wild_type: str) -> List[List[Any]]:
    """`sequence`'s mutations relative to `wild_type`, as [wt_aa, position,
    mut_aa] triples (1-indexed). Full sequences are mostly identical within a
    mutant library and burn far more tokens than the handful of substitutions
    that actually vary, so the LLM sees only the mutation list."""
    return [
        [wt_aa, i + 1, mut_aa]
        for i, (wt_aa, mut_aa) in enumerate(zip(wild_type, sequence))
        if wt_aa != mut_aa
    ]


def _format_history(history: List[Dict[str, Any]], wild_type: str) -> str:
    if not history:
        return "(none yet)"
    return "\n".join(
        f"{i}. {_mutations(point['sequence'], wild_type)} -> fitness: {point['fitness']}"
        for i, point in enumerate(history)
    )


def _format_candidates(candidates: List[Dict[str, Any]], wild_type: str) -> str:
    lines = []
    for i, candidate in enumerate(candidates):
        line = f"id {i}: {_mutations(candidate['sequence'], wild_type)}"
        if "gp_mean" in candidate and "gp_std" in candidate:
            line += f" | predicted fitness: {candidate['gp_mean']:.4f} +/- {candidate['gp_std']:.4f}"
        lines.append(line)
    return "\n".join(lines)


def format_ranking_prompt(
    candidates: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    wild_type: str,
    prompt_path: str = PROMPT_PATH,
) -> str:
    with open(prompt_path) as f:
        template = f.read()
    return template.format(
        wild_type=wild_type,
        history=_format_history(history, wild_type),
        candidates=_format_candidates(candidates, wild_type),
        n_candidates=len(candidates),
        max_id=len(candidates) - 1,
    )


def parse_ranking(response_text: str, n_candidates: int, batch_size: int) -> List[int]:
    """Extract the JSON ranking from the LLM's response, tolerating surrounding
    text or a markdown code fence.
    """
    start, end = response_text.find("["), response_text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response: {response_text!r}")
    ranking = json.loads(response_text[start : end + 1])
    if any(not (0 <= i < n_candidates) for i in ranking):
        raise ValueError(
            f"LLM ranking {ranking} has out-of-range ids "
            f"(expected ids in 0..{n_candidates - 1})"
        )
    seen = set()
    deduped = [i for i in ranking if not (i in seen or seen.add(i))]
    if len(deduped) < batch_size:
        raise ValueError(
            f"LLM ranking only has {len(deduped)} unique ids, fewer than batch_size={batch_size}"
        )
    return deduped


def _aggregate_rankings(rankings: List[List[int]], n_candidates: int) -> List[int]:
    """Combine multiple per-member rankings into one via Borda count: a
    member's top id scores n_candidates-1 points, the next n_candidates-2, ...
    down to 0, summed across members. An id a member omitted scores 0 from
    that member, same as if it had been ranked last."""
    scores = [0] * n_candidates
    for ranking in rankings:
        for rank_pos, cand_id in enumerate(ranking):
            scores[cand_id] += n_candidates - 1 - rank_pos
    return sorted(range(n_candidates), key=lambda i: -scores[i])


def select_acquisitions_with_llm(
    candidates: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    wild_type: str,
    llm_config: Dict[str, Any],
    batch_size: int,
    prompt_path: str = PROMPT_PATH,
    ensemble_size: int = 3,
) -> tuple:
    """Re-rank `candidates` with an LLM and return (chosen, reasoning): the top
    `batch_size` candidates by the LLM's ranking, and its native reasoning
    trace(s) for the call(s) (empty string if the model/provider exposes none).

    candidates: top-k candidates from the acquisition function, each a dict
        with at least a "sequence" key.
    history: previously acquired points, each a dict with "sequence" and
        "fitness" keys.
    wild_type: the dataset's wild-type sequence.
    llm_config: class_path/init_args config for a gollum.reasoning.llm.LLMClient,
        e.g. {"class_path": "gollum.reasoning.llm.AnthropicLLM", "init_args": {...}}.
    batch_size: number of candidates to acquire, taken from the top of the
        final ranking.
    ensemble_size: number of independent calls to `llm_config` to make (fired
        concurrently) and combine via Borda count, instead of trusting a
        single call. A member whose response is malformed is dropped from the
        vote rather than failing the whole re-rank; only raises if every
        member fails. 1 (default) is a single call, no aggregation. Costs
        `ensemble_size` LLM calls per invocation.
    """
    prompt = format_ranking_prompt(candidates, history, wild_type, prompt_path)
    llm = instantiate_class(llm_config)
    n_candidates = len(candidates)

    with ThreadPoolExecutor(max_workers=ensemble_size) as pool:
        responses = list(pool.map(lambda _: llm.complete(prompt), range(ensemble_size)))

    rankings, reasoning_traces = [], []
    for member, (text, member_reasoning) in enumerate(responses):
        try:
            rankings.append(parse_ranking(text, n_candidates, batch_size))
        except ValueError as e:
            print(f"[ensemble member {member}] discarded, invalid ranking: {e}")
            continue
        if member_reasoning:
            reasoning_traces.append(
                f"[member {member}]\n{member_reasoning}" if ensemble_size > 1 else member_reasoning
            )
    if not rankings:
        raise ValueError(f"All {ensemble_size} ensemble members produced invalid rankings")

    ranking = rankings[0] if ensemble_size == 1 else _aggregate_rankings(rankings, n_candidates)
    reasoning = "\n\n".join(reasoning_traces)
    return [candidates[i] for i in ranking[:batch_size]], reasoning
