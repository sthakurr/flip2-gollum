# Reasoning prompt versions

Each prompt formats the acquisition function's shortlist + acquisition history
into a ranking request for the LLM (see `format_ranking_prompt` in
`gollum/reasoning/agent.py`). All require `wild_type`/`history`/`candidates`/
`n_candidates`/`max_id`; some additionally need candidates to carry `gp_mean`/
`gp_std` (config: `reasoning.include_gp_stats: true`) or structural context
(config: `reasoning.include_structure: true`).

| Prompt | Needs | One-line description |
|---|---|---|
| p0 | -- | Plain baseline: mutation triples + biochemistry framing, no extra signal. |
| p1 | gp_stats | "Biochemist" framing, explicitly shown GP mean/std, soft guidance to weigh both. |
| p3 | -- | Rigid evidence-table procedure over history only; forbids structural/GP claims. |
| p4 | structure | p0-style + real ESM3 structural context per mutated position. |
| p5 | gp_stats + structure | p4 + explicit UCB-primary ranking rule; structure only breaks close ties. |
| p6 | gp_stats + structure | p5 + a second tie-break: prefer untested mutation/structure combinations. |

## p0 -- baseline

Minimal prompt: `[wt_aa, position, mut_aa]` mutation triples, wild type,
acquisition history, and a generic "use your biochemistry knowledge" ask. No
GP stats, no structure. The plainest version tried, and empirically a strong
baseline relative to the more elaborate prompts below.

## p1 -- biochemist + GP stats

Frames the LLM as a biochemist collaborating with the BO system, shown the
GP's predicted mean/std per candidate and asked to weigh statistical
confidence against biochemical judgment (conservative vs. disruptive
substitutions, structural/functional plausibility, epistasis hints in the
history). The guidance to use uncertainty is descriptive, not an enforced
rule. Measured close to neutral in the local fitness-delta diagnostic
(`amie_one`: mean delta +0.013 per iteration -- barely more exploitative than
the acquisition function itself), but `coverage_top5` still landed a bit below
the no-reasoning baseline.

## p3 -- rigid evidence-table procedure

Authored to eliminate ungrounded claims entirely. Forces a 3-step procedure:
(1) build an evidence table from the history only, keeping a mutation only if
it has >=5 carrier variants and their mean fitness clearly separates from
non-carriers; (2) score each candidate by beneficial-minus-harmful mutations
from that table, explicitly forbidding mutation-count reasoning, generic
substitution chemistry, any structural/conservation claims (no structure was
given), and re-deriving the GP's own mean/std; (3) emit a stable re-tiering of
the acquisition function's given order -- never inventing an order where there
is no evidence. Treats the acquisition order as a strong prior to correct, not
rebuild. No GP stats or structure fields used.

## p4 -- structural context, free-form

Same minimal style as p0, but each mutation triple gets a 4th element: real
ESM3-predicted structural context at that position (secondary structure,
solvent exposure, pLDDT confidence), computed once per run from a wild-type
fold. No GP stats. This was the first attempt to ground reasoning in real
structure -- and it measurably back-fired: the local fitness-delta diagnostic
showed p4 *always* overriding the acquisition function's own ranking (0% of
iterations deferred to it on `amie_one`/`rash_one`) with a strong positive
per-iteration fitness lift (+0.108 avg on `amie_one`) -- i.e. pure
exploitation, ignoring the acquisition function's exploration term entirely.
That showed up as *worse* `coverage_top5` than no reasoning at all, despite
locally "better" picks.

## p5 -- UCB-gated ranking rule

Same structural context as p4, plus GP mean/std, but with an explicit
2-step ranking procedure: (1) primary criterion is UCB = mean + 1.0*std,
reproducing the acquisition function's own tradeoff; (2) biochemical/
structural evidence may only reorder candidates whose UCB scores are close
(~1 std) -- a large UCB gap must dominate regardless of how promising a
lower-UCB candidate looks. This fixed the exploitation bias: the LLM's pick
matched the acquisition-only pick exactly in 72-92% of iterations tested (vs.
0% for p4), and `coverage_top5` improved (`amie_one` 0.243, `rash_one` 0.355)
without ever getting worse than p4. Still landed a bit below the no-reasoning
baseline on both single-mutant datasets tested -- safe, but close to a no-op.

## p6 -- + novelty-of-hypothesis tie-break

p5's UCB-gated procedure, but the tie-break step gets a second criterion
alongside biochemical/structural risk: among close-UCB candidates, prefer the
one testing a mutation-type/structural-context combination not well
represented in the acquisition history yet (e.g. a non-conservative
substitution at a buried helix position that hasn't been tried), since that's
informative to the campaign even at equal predicted fitness/uncertainty --
something the GP's mean/std can't express on its own. No change on
single-mutant landscapes (`amie_one` 0.234, `rash_one` 0.353 -- same as p5,
within noise): there's no real mutation-combination novelty to reason about
when every variant is a single substitution. On combinatorial landscapes
(actual multi-mutation data), the same prompt shows a clear win on `aa_far`
(0.555 vs. 0.536 no-reasoning baseline) and ties the no-reasoning baseline on
`aa_many`/`ired_many`/`aa_close`/`trpb_two` -- consistent with the criterion
only having real signal to act on where mutations actually combine.
