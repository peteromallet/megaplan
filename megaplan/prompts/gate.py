"""Gate-phase prompt builders and summaries."""

from __future__ import annotations

import textwrap
from pathlib import Path

from megaplan._core import (
    configured_robustness,
    current_iteration_artifact,
    intent_and_notes_block,
    json_dump,
    latest_plan_meta_path,
    latest_plan_path,
    load_flag_registry,
    read_json,
    unresolved_significant_flags,
)
from megaplan.types import FlagRegistry, PlanState

from ._shared import _gate_debt_block


def _gate_prompt(state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    gate_signals = read_json(
        current_iteration_artifact(plan_dir, "gate_signals", state["iteration"])
    )
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    open_flags = [
        {
            "id": flag["id"],
            "concern": flag["concern"],
            "evidence": flag.get("evidence", ""),
            "category": flag["category"],
            "severity": flag.get("severity", "unknown"),
            "status": flag["status"],
            "weight": flag.get("weight"),
        }
        for flag in unresolved
    ]
    robustness = configured_robustness(state)
    debt_block = _gate_debt_block(plan_dir, root)
    # Load critique checks for gate visibility — include ALL findings so the gate
    # can promote unflagged concerns to flags if it disagrees with the critique.
    critique_checks_block = ""
    critique_path = current_iteration_artifact(plan_dir, "critique", state["iteration"])
    if Path(critique_path).exists():
        critique_data = read_json(critique_path)
        checks = critique_data.get("checks", [])
        if checks:
            check_lines = []
            for check in checks:
                findings = check.get("findings", [])
                flagged = [f for f in findings if f.get("flagged")]
                unflagged = [f for f in findings if not f.get("flagged")]
                flagged_count = len(flagged)
                status = f"{flagged_count} flagged" if flagged_count else "clear"
                check_lines.append(f"- {check.get('id', '?')}: {status}")
                # Show unflagged findings as FYI — gate can promote if needed
                for f in unflagged:
                    detail = f.get("detail", "").strip()
                    if detail and len(detail) > 30:  # skip trivial "no issue" findings
                        check_lines.append(f"    FYI (not flagged): {detail[:200]}")
            critique_checks_block = (
                "Critique checks (flagged findings are flags; FYI findings are the critique's "
                "unflagged observations — review them and promote to a flag if any look risky):\n        "
                + "\n        ".join(check_lines)
            )
    return textwrap.dedent(
        f"""
        You are the gatekeeper for the megaplan workflow. Make the continuation decision directly.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Plan:
        {latest_plan}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Gate signals:
        {json_dump(gate_signals).strip()}

        {critique_checks_block}

        Unresolved significant flags:
        {json_dump(open_flags).strip()}

        {debt_block}

        Robustness level:
        {robustness}

        Requirements:
        - Decide exactly one of: PROCEED, ITERATE, ESCALATE.
        - Use the weighted score, flag details (including the `evidence` field — not just `concern`), plan delta, recurring critiques, loop summary, and preflight results as judgment context, not as a fixed decision table.
        - PROCEED when execution should move forward now.
        - ITERATE when revising the plan is the best next move.
        - ESCALATE when the loop is stuck, churn is recurring, or user intervention is needed.
        - `signals_assessment` should summarize the score trajectory, plan delta, recurring critiques, unresolved flag weight, and preflight posture in one compact paragraph.
        - Put any cautionary notes in `warnings`.
        - Populate `settled_decisions` with design choices that are now settled and should carry into review without being re-litigated. Return `[]` when there are no such decisions.

        Flag resolution — severity determines what you must do:
        - `likely-significant`: You MUST explicitly resolve each one in `flag_resolutions`. Either "iterate" (fix it), "disputed" (explain with evidence why the critique is wrong), or "accepted" (explain why it's safe to proceed despite the concern). You cannot PROCEED if any likely-significant flag is unresolved.
        - `likely-minor`: You MUST acknowledge each one in `flag_resolutions` — same options as above, but the bar for "accepted" is lower. A sentence is enough.
        - `uncertain`: No resolution required. Address if you choose.

        Populate `flag_resolutions` with one entry per resolved flag:
          - `flag_id`: the exact flag ID
          - `disposition`: one of "iterate", "accepted", "disputed"
          - `rationale`: why (required for likely-significant, brief for likely-minor)
        - Example output shape:
        ```json
        {{
          "recommendation": "PROCEED",
          "rationale": "All significant flags resolved. Minor convention concern accepted.",
          "signals_assessment": "Weighted score stable, plan delta clean, preflight passed.",
          "warnings": ["Double-check FLAG-005 while executing."],
          "flag_resolutions": [
            {{
              "flag_id": "correctness-1",
              "disposition": "disputed",
              "rationale": "The critique flagged missing null check, but the caller guarantees non-null per line 42."
            }},
            {{
              "flag_id": "conventions-1",
              "disposition": "accepted",
              "rationale": "Both API patterns are valid; the chosen one is simpler."
            }}
          ],
          "settled_decisions": [
            {{
              "id": "DECISION-001",
              "decision": "Use allow_migrate over allow_migrate_model — accepted at gate.",
              "rationale": "Convention flag resolved as accepted tradeoff."
            }}
          ]
        }}
        ```
        """
    ).strip()


def _collect_critique_summaries(
    plan_dir: Path, iteration: int
) -> list[dict[str, object]]:
    """Gather a compact list of all critique rounds for the finalize prompt."""
    summaries: list[dict[str, object]] = []
    for i in range(1, iteration + 1):
        path = plan_dir / f"critique_v{i}.json"
        if path.exists():
            data = read_json(path)
            summaries.append(
                {
                    "iteration": i,
                    "flag_count": len(data.get("flags", [])),
                    "verified": data.get("verified_flag_ids", []),
                }
            )
    return summaries


def _flag_summary(registry: FlagRegistry) -> list[dict[str, object]]:
    """Compact flag list for the finalize prompt."""
    return [
        {
            "id": f["id"],
            "concern": f["concern"],
            "evidence": f.get("evidence", ""),
            "status": f["status"],
            "severity": f.get("severity", "unknown"),
        }
        for f in registry["flags"]
    ]
