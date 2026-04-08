import json
from typing import Any, Optional

DOCUMENT_RELEVANCE: dict[str, dict[str, dict[str, float]]] = {
    "easy_diabetes_screening": {
        "default": {
            "medical_history": 0.05,
            "lab_results": 0.05,
            "medications": 0.05,
            "prior_trials": 0.02,
            "drug_interactions": 0.03,
            "genetic_markers": 0.02,
        }
    },
    "medium_cardiac_interactions": {
        "default": {
            "medical_history": 0.05,
            "lab_results": 0.05,
            "medications": 0.05,
            "prior_trials": 0.05,
            "drug_interactions": 0.08,
            "genetic_markers": 0.03,
        }
    },
    "hard_rare_disease_multi": {
        "patient_a": {
            "medical_history": 0.05,
            "lab_results": 0.05,
            "medications": 0.03,
            "prior_trials": 0.02,
            "drug_interactions": 0.03,
            "genetic_markers": 0.05,
        },
        "patient_b": {
            "medical_history": 0.05,
            "lab_results": 0.05,
            "medications": 0.03,
            "prior_trials": 0.08,
            "drug_interactions": 0.03,
            "genetic_markers": 0.03,
        },
        "patient_c": {
            "medical_history": 0.05,
            "lab_results": 0.05,
            "medications": 0.03,
            "prior_trials": 0.02,
            "drug_interactions": 0.03,
            "genetic_markers": 0.05,
        },
    },
}

CONCERN_PATTERNS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "easy_diabetes_screening": {
        "default": [
            {
                "description": "eGFR below threshold",
                "keywords": ["egfr", "55", "kidney", "renal", "glomerular", "filtration", "ckd", "gfr"],
            },
            {
                "description": "metformin dose change",
                "keywords": ["metformin", "dose", "changed", "increased", "stable", "timing", "recent", "1000"],
            },
        ]
    },
    "medium_cardiac_interactions": {
        "default": [
            {
                "description": "St. John's Wort CYP3A4 inducer",
                "keywords": ["st. john", "st john", "cyp3a4", "hypericum", "inducer", "herbal"],
            },
            {
                "description": "warfarin washout required",
                "keywords": ["warfarin", "washout", "discontinu", "14-day", "14 day", "anticoagul"],
            },
            {
                "description": "prior adverse event in similar trial",
                "keywords": ["afguard", "prior trial", "adverse", "previous", "reaction", "factor xa", "rash"],
            },
        ]
    },
    "hard_rare_disease_multi": {
        "patient_a": [
            {
                "description": "MPO-ANCA positive, not PR3-ANCA",
                "keywords": ["mpo", "wrong subtype", "not pr3", "pr3-anca negative", "pr3 negative", "mpo-anca positive", "42.8", "3.2"],
            },
            {
                "description": "conflicting records between hospitals",
                "keywords": ["conflicting", "mercy", "egpa", "different hospital", "inconsisten", "city general", "discrepan"],
            },
            {
                "description": "genetic markers inconsistent with GPA",
                "keywords": ["genetic", "hla", "dpb1", "marker", "negative"],
            },
        ],
        "patient_b": [
            {
                "description": "concurrent interventional trial (VASCUWATCH)",
                "keywords": ["vascuwatch", "concurrent", "interventional", "enrolled", "another trial", "current trial"],
            },
            {
                "description": "adherence/compliance concerns",
                "keywords": ["adherence", "compliance", "missed", "appointment", "non-compli"],
            },
        ],
        "patient_c": [
            {
                "description": "borderline PR3-ANCA titers",
                "keywords": ["borderline", "low titer", "15.2", "lower range", "trend", "low positive"],
            },
        ],
    },
}

REASONING_EVIDENCE: dict[str, dict[str, list[str]]] = {
    "easy_diabetes_screening": {
        "default": ["egfr", "55", "metformin", "dose", "renal", "kidney"],
    },
    "medium_cardiac_interactions": {
        "default": ["st. john", "st john", "cyp3a4", "warfarin", "washout", "inducer"],
    },
    "hard_rare_disease_multi": {
        "patient_a": ["mpo", "pr3", "anca", "subtype", "mercy", "egpa"],
        "patient_b": ["vascuwatch", "interventional", "concurrent", "enrollment"],
        "patient_c": ["pr3", "anca", "rituximab", "eligible", "criteria", "qualif"],
    },
}


class RewardCalculator:
    def __init__(self, task_id: str, task_data: dict[str, Any]):
        self.task_id = task_id
        self.task_data = task_data
        self.answer_key = task_data.get("answer_key", {})
        self._matched_concerns: dict[str, set[int]] = {}

    def document_reward(self, doc_key: str, patient_id: str) -> float:
        task_docs = DOCUMENT_RELEVANCE.get(self.task_id, {})
        patient_docs = task_docs.get(patient_id, {})
        return patient_docs.get(doc_key, 0.02)

    def check_concern(self, flag_reason: str, patient_id: str) -> tuple[bool, int]:
        if patient_id not in self._matched_concerns:
            self._matched_concerns[patient_id] = set()

        patterns = CONCERN_PATTERNS.get(self.task_id, {}).get(patient_id, [])
        reason_lower = flag_reason.lower()

        for i, pattern in enumerate(patterns):
            if i in self._matched_concerns[patient_id]:
                continue
            for keyword in pattern["keywords"]:
                if keyword in reason_lower:
                    self._matched_concerns[patient_id].add(i)
                    return True, i

        return False, -1

    def get_matched_concern_count(self, patient_id: str) -> int:
        return len(self._matched_concerns.get(patient_id, set()))

    def get_total_concerns(self, patient_id: str) -> int:
        return len(CONCERN_PATTERNS.get(self.task_id, {}).get(patient_id, []))

    def decision_reward(
        self,
        patient_id: str,
        decision: str,
        reasoning: str,
        approved_amount: Optional[float],
    ) -> float:
        correct = self._get_correct_decision(patient_id)
        correct_decision = correct.get("correct_decision", "")
        reward = 0.0

        if decision == correct_decision:
            if decision == "reject":
                reward += 0.30
            elif decision == "enroll":
                reward += 0.20
                target_amount = correct.get("approved_amount")
                if target_amount and approved_amount:
                    if abs(approved_amount - target_amount) / target_amount <= 0.15:
                        reward += 0.05
            elif decision == "waitlist":
                reward += 0.25
            elif decision == "refer_to_specialist":
                reward += 0.15
        else:
            if correct_decision == "reject" and decision == "enroll":
                reward -= 0.30
            elif correct_decision == "enroll" and decision == "reject":
                reward -= 0.40
            elif correct_decision == "waitlist" and decision == "enroll":
                reward -= 0.20
            elif correct_decision == "waitlist" and decision == "reject":
                reward -= 0.15
            elif correct_decision == "reject" and decision == "waitlist":
                reward -= 0.05
            else:
                reward -= 0.10

        if reasoning and self._reasoning_cites_evidence(patient_id, reasoning):
            reward += 0.10

        return reward

    def _get_correct_decision(self, patient_id: str) -> dict[str, Any]:
        if "patients" in self.answer_key:
            return self.answer_key["patients"].get(patient_id, {})
        return self.answer_key

    def _reasoning_cites_evidence(self, patient_id: str, reasoning: str) -> bool:
        evidence_terms = REASONING_EVIDENCE.get(self.task_id, {}).get(patient_id, [])
        reasoning_lower = reasoning.lower()
        matches = sum(1 for term in evidence_terms if term in reasoning_lower)
        return matches >= 2
