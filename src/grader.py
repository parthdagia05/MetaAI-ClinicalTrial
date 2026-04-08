from typing import Any
from .reward import CONCERN_PATTERNS, RewardCalculator


def grade_task(
    task_id: str,
    task_data: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    patient_state: dict[str, dict[str, Any]],
    protocol_checked: bool,
    steps_taken: int,
    reward_calc: RewardCalculator,
) -> dict[str, Any]:
    answer_key = task_data.get("answer_key", {})
    grading_weights = answer_key.get("grading", {})

    if task_id == "easy_diabetes_screening":
        return _grade_easy(answer_key, grading_weights, decisions, patient_state, reward_calc, steps_taken)
    elif task_id == "medium_cardiac_interactions":
        return _grade_medium(answer_key, grading_weights, decisions, patient_state, reward_calc, steps_taken)
    elif task_id == "hard_rare_disease_multi":
        return _grade_hard(answer_key, grading_weights, decisions, patient_state, reward_calc, steps_taken)

    return {"score": 0.0, "breakdown": {}, "reason": "unknown task"}


def _grade_easy(
    answer_key: dict,
    weights: dict,
    decisions: dict,
    patient_state: dict,
    reward_calc: RewardCalculator,
    steps_taken: int,
) -> dict[str, Any]:
    breakdown: dict[str, float] = {}

    total_concerns = reward_calc.get_total_concerns("default")
    matched = reward_calc.get_matched_concern_count("default")
    concern_ratio = matched / total_concerns if total_concerns > 0 else 0.0
    breakdown["found_all_concerns"] = concern_ratio * weights.get("found_all_concerns", 0.30)

    decision_data = decisions.get("default", {})
    correct = decision_data.get("decision") == answer_key.get("correct_decision")
    breakdown["correct_decision"] = weights.get("correct_decision", 0.30) if correct else 0.0

    reasoning = decision_data.get("reasoning", "") or ""
    reasoning_lower = reasoning.lower()
    evidence_terms = ["egfr", "55", "metformin", "dose", "renal", "kidney"]
    evidence_count = sum(1 for t in evidence_terms if t in reasoning_lower)
    reasoning_score = min(evidence_count / 2, 1.0)
    breakdown["correct_reasoning"] = reasoning_score * weights.get("correct_reasoning", 0.20)

    expected = answer_key.get("expected_steps", {})
    max_expected = expected.get("max", 7)
    if steps_taken <= max_expected:
        breakdown["efficiency"] = weights.get("efficiency", 0.10)
    elif steps_taken <= max_expected + 3:
        breakdown["efficiency"] = weights.get("efficiency", 0.10) * 0.5
    else:
        breakdown["efficiency"] = 0.0

    ps = patient_state.get("default", {})
    concerns = ps.get("concerns_found", [])
    false_count = len(concerns) - matched
    if false_count == 0:
        breakdown["no_false_concerns"] = weights.get("no_false_concerns", 0.10)
    elif false_count == 1:
        breakdown["no_false_concerns"] = weights.get("no_false_concerns", 0.10) * 0.5
    else:
        breakdown["no_false_concerns"] = 0.0

    total = sum(breakdown.values())
    total = max(0.0, min(1.0, total))

    return {"score": round(total, 4), "breakdown": breakdown, "reason": _build_reason(breakdown)}


def _grade_medium(
    answer_key: dict,
    weights: dict,
    decisions: dict,
    patient_state: dict,
    reward_calc: RewardCalculator,
    steps_taken: int,
) -> dict[str, Any]:
    breakdown: dict[str, float] = {}

    total_concerns = reward_calc.get_total_concerns("default")
    matched = reward_calc.get_matched_concern_count("default")
    concern_ratio = matched / total_concerns if total_concerns > 0 else 0.0
    breakdown["identified_concerns"] = concern_ratio * weights.get("identified_concerns", 0.25)

    legitimate_terms = ["age", "cha2ds2", "liver", "normal", "60 days", "371", "prior", "route", "injection"]
    reasoning = (decisions.get("default", {}).get("reasoning", "") or "").lower()
    legit_count = sum(1 for t in legitimate_terms if t in reasoning)
    legit_ratio = min(legit_count / 3, 1.0)
    breakdown["identified_legitimate"] = legit_ratio * weights.get("identified_legitimate", 0.15)

    decision_data = decisions.get("default", {})
    correct = decision_data.get("decision") == answer_key.get("correct_decision")
    breakdown["correct_decision"] = weights.get("correct_decision", 0.25) if correct else 0.0

    evidence_terms = ["st. john", "st john", "cyp3a4", "warfarin", "washout", "inducer"]
    evidence_count = sum(1 for t in evidence_terms if t in reasoning)
    reasoning_score = min(evidence_count / 2, 1.0)
    breakdown["correct_reasoning"] = reasoning_score * weights.get("correct_reasoning", 0.15)

    expected = answer_key.get("expected_steps", {})
    max_expected = expected.get("max", 11)
    if steps_taken <= max_expected:
        breakdown["efficiency"] = weights.get("efficiency", 0.10)
    elif steps_taken <= max_expected + 3:
        breakdown["efficiency"] = weights.get("efficiency", 0.10) * 0.5
    else:
        breakdown["efficiency"] = 0.0

    ps = patient_state.get("default", {})
    concerns = ps.get("concerns_found", [])
    false_count = len(concerns) - matched
    if false_count == 0:
        breakdown["no_false_concerns"] = weights.get("no_false_concerns", 0.10)
    elif false_count == 1:
        breakdown["no_false_concerns"] = weights.get("no_false_concerns", 0.10) * 0.5
    else:
        breakdown["no_false_concerns"] = 0.0

    total = sum(breakdown.values())
    total = max(0.0, min(1.0, total))

    return {"score": round(total, 4), "breakdown": breakdown, "reason": _build_reason(breakdown)}


def _grade_hard(
    answer_key: dict,
    weights: dict,
    decisions: dict,
    patient_state: dict,
    reward_calc: RewardCalculator,
    steps_taken: int,
) -> dict[str, Any]:
    breakdown: dict[str, float] = {}
    patients_key = answer_key.get("patients", {})

    pa_decision = decisions.get("patient_a", {}).get("decision")
    pa_correct = patients_key.get("patient_a", {}).get("correct_decision")
    pa_reasoning = (decisions.get("patient_a", {}).get("reasoning", "") or "").lower()

    pa_matched = reward_calc.get_matched_concern_count("patient_a")
    pa_evidence = any(t in pa_reasoning for t in ["mpo", "pr3", "anca", "subtype", "mercy", "egpa"])
    pa_score = 0.0
    if pa_decision == pa_correct and pa_matched >= 2 and pa_evidence:
        pa_score = 1.0
    elif pa_decision == pa_correct and pa_matched >= 1:
        pa_score = 0.7
    elif pa_decision == pa_correct:
        pa_score = 0.4
    breakdown["connected_patient_a_rejection_reasons"] = (
        pa_score * weights.get("connected_patient_a_rejection_reasons", 0.20)
    )

    pc_decision = decisions.get("patient_c", {}).get("decision")
    pc_correct = patients_key.get("patient_c", {}).get("correct_decision")
    pc_reasoning = (decisions.get("patient_c", {}).get("reasoning", "") or "").lower()

    pc_score = 0.0
    if pc_decision == pc_correct:
        pc_evidence = any(t in pc_reasoning for t in ["eligible", "criteria", "pr3", "rituximab", "qualif"])
        pc_score = 1.0 if pc_evidence else 0.7
        target_amount = patients_key.get("patient_c", {}).get("approved_amount")
        actual_amount = decisions.get("patient_c", {}).get("approved_amount")
        if target_amount and actual_amount:
            if abs(actual_amount - target_amount) / target_amount <= 0.15:
                pc_score = min(pc_score + 0.1, 1.0)
    breakdown["correctly_identified_patient_c_as_eligible"] = (
        pc_score * weights.get("correctly_identified_patient_c_as_eligible", 0.20)
    )

    all_correct = True
    for pid in ["patient_a", "patient_b", "patient_c"]:
        d = decisions.get(pid, {}).get("decision")
        c = patients_key.get(pid, {}).get("correct_decision")
        if d != c:
            all_correct = False
            break

    correct_count = sum(
        1
        for pid in ["patient_a", "patient_b", "patient_c"]
        if decisions.get(pid, {}).get("decision") == patients_key.get(pid, {}).get("correct_decision")
    )
    decision_ratio = correct_count / 3
    breakdown["correct_decisions_all_three"] = (
        decision_ratio * weights.get("correct_decisions_all_three", 0.25)
    )

    total_matched = sum(reward_calc.get_matched_concern_count(p) for p in ["patient_a", "patient_b", "patient_c"])
    if total_matched >= 5:
        concern_score = 1.0
    elif total_matched >= 4:
        concern_score = 0.8
    elif total_matched >= 3:
        concern_score = 0.6
    else:
        concern_score = total_matched / 5
    breakdown["found_4_plus_concerns"] = concern_score * weights.get("found_4_plus_concerns", 0.15)

    all_reasoning = " ".join(
        (decisions.get(pid, {}).get("reasoning", "") or "") for pid in ["patient_a", "patient_b", "patient_c"]
    ).lower()
    evidence_terms = ["mpo", "pr3", "vascuwatch", "interventional", "rituximab", "anca", "concurrent"]
    evidence_hits = sum(1 for t in evidence_terms if t in all_reasoning)
    reasoning_score = min(evidence_hits / 3, 1.0)
    breakdown["correct_reasoning_with_evidence"] = (
        reasoning_score * weights.get("correct_reasoning_with_evidence", 0.10)
    )

    expected = answer_key.get("expected_steps", {})
    max_expected = expected.get("max", 20)
    if steps_taken <= max_expected:
        breakdown["efficiency"] = weights.get("efficiency", 0.10)
    elif steps_taken <= max_expected + 5:
        breakdown["efficiency"] = weights.get("efficiency", 0.10) * 0.5
    else:
        breakdown["efficiency"] = 0.0

    total = sum(breakdown.values())
    total = max(0.0, min(1.0, total))

    return {"score": round(total, 4), "breakdown": breakdown, "reason": _build_reason(breakdown)}


def _build_reason(breakdown: dict[str, float]) -> str:
    parts = [f"{k}: {v:.3f}" for k, v in breakdown.items()]
    return " | ".join(parts)
