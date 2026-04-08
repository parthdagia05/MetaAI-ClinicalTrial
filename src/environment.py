import json
from typing import Any, Optional

from .models import Observation, Action
from .data_loader import load_task_data
from .reward import RewardCalculator

ACTION_TO_DOCUMENT = {
    "check_medical_history": "medical_history",
    "check_lab_results": "lab_results",
    "check_medications": "medications",
    "check_prior_trials": "prior_trials",
    "check_drug_interactions": "drug_interactions",
    "check_genetic_markers": "genetic_markers",
}

MAX_STEPS = {
    "easy_diabetes_screening": 10,
    "medium_cardiac_interactions": 15,
    "hard_rare_disease_multi": 25,
}

VALID_ACTIONS = [
    "view_patient",
    "check_trial_protocol",
    "check_medical_history",
    "check_lab_results",
    "check_medications",
    "check_drug_interactions",
    "check_genetic_markers",
    "check_prior_trials",
    "flag_concern",
    "request_additional_tests",
    "make_decision",
]

VALID_DECISIONS = ["enroll", "reject", "waitlist", "refer_to_specialist"]


class ClinicalTrialScreenerEnv:
    def __init__(self):
        self._task_id: Optional[str] = None
        self._task_data: Optional[dict[str, Any]] = None
        self._steps_taken: int = 0
        self._max_steps: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0

        self._current_patient: str = "default"
        self._patients: list[str] = []
        self._patient_state: dict[str, dict[str, Any]] = {}

        self._protocol_checked: bool = False
        self._screening_notes: list[str] = []
        self._action_history: list[str] = []
        self._decisions: dict[str, dict[str, Any]] = {}
        self._current_document_content: Optional[str] = None

        self._reward_calc: Optional[RewardCalculator] = None

    def reset(self, task_id: str) -> Observation:
        self._task_id = task_id
        self._task_data = load_task_data(task_id)
        self._steps_taken = 0
        self._max_steps = MAX_STEPS.get(task_id, 15)
        self._done = False
        self._cumulative_reward = 0.0
        self._protocol_checked = False
        self._screening_notes = []
        self._action_history = []
        self._decisions = {}
        self._current_document_content = None

        self._reward_calc = RewardCalculator(task_id, self._task_data)

        if "patients" in self._task_data:
            self._patients = sorted(self._task_data["patients"].keys())
            self._current_patient = self._patients[0]
        else:
            self._patients = ["default"]
            self._current_patient = "default"

        self._patient_state = {}
        for p in self._patients:
            self._patient_state[p] = {
                "viewed": False,
                "documents_reviewed": [],
                "concerns_found": [],
                "eligibility_findings": [],
            }

        self._screening_notes.append(f"screening session started for task: {task_id}")
        if len(self._patients) > 1:
            names = []
            for p in self._patients:
                pd = self._get_patient_data(p)
                names.append(pd.get("patient", {}).get("name", p))
            self._screening_notes.append(f"multiple patients to screen: {', '.join(names)}")

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._done:
            return self._build_observation(), 0.0, True, {"error": "episode already finished"}

        self._steps_taken += 1
        reward = 0.0
        info: dict[str, Any] = {}

        action_key = f"{action.action_type}:{action.target}"
        self._action_history.append(action_key)

        if self._detect_loop():
            self._done = True
            self._cumulative_reward = 0.0
            return self._build_observation(), -1.0, True, {"reason": "infinite loop detected"}

        is_repeat = (
            self._action_history.count(action_key) > 1
            and action.action_type not in ("flag_concern", "make_decision", "view_patient")
        )
        if is_repeat:
            reward -= 0.05
            self._screening_notes.append(f"[duplicate] already performed: {action.action_type}")

        if action.action_type == "view_patient":
            reward += self._handle_view_patient(action)
        elif action.action_type == "check_trial_protocol":
            reward += self._handle_check_protocol()
        elif action.action_type in ACTION_TO_DOCUMENT:
            reward += self._handle_check_document(action)
        elif action.action_type == "flag_concern":
            reward += self._handle_flag_concern(action)
        elif action.action_type == "request_additional_tests":
            reward += self._handle_request_tests(action)
        elif action.action_type == "make_decision":
            reward += self._handle_make_decision(action)
            info["decision"] = action.decision
        else:
            reward -= 0.02
            self._screening_notes.append(f"unknown action: {action.action_type}")

        if self._steps_taken >= self._max_steps and not self._done:
            self._done = True
            reward -= 0.15
            self._screening_notes.append("max steps reached — episode ended")

        if len(self._decisions) == len(self._patients) and not self._done:
            self._done = True

        self._cumulative_reward += reward
        return self._build_observation(), reward, self._done, info

    def state(self) -> dict[str, Any]:
        return {
            "task_id": self._task_id,
            "steps_taken": self._steps_taken,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "current_patient": self._current_patient,
            "patients": self._patients,
            "decisions": self._decisions,
            "protocol_checked": self._protocol_checked,
            "patient_state": {
                k: {
                    "viewed": v["viewed"],
                    "documents_reviewed": v["documents_reviewed"][:],
                    "concerns_found": v["concerns_found"][:],
                }
                for k, v in self._patient_state.items()
            },
            "screening_notes": self._screening_notes[:],
        }

    # -- action handlers --

    def _handle_view_patient(self, action: Action) -> float:
        if action.target and action.target in self._patients:
            self._current_patient = action.target

        p = self._current_patient
        ps = self._patient_state[p]
        patient_data = self._get_patient_data(p)
        patient_info = patient_data.get("patient", {})

        self._current_document_content = json.dumps(patient_info, indent=2)

        if not ps["viewed"]:
            ps["viewed"] = True
            name = patient_info.get("name", p)
            self._screening_notes.append(f"viewed patient: {name}")
            return 0.02

        return 0.0

    def _handle_check_protocol(self) -> float:
        protocol = self._task_data.get("trial_protocol", {})
        self._current_document_content = json.dumps(protocol, indent=2)

        if not self._protocol_checked:
            self._protocol_checked = True
            self._screening_notes.append(
                f"reviewed trial protocol: {protocol.get('trial_name', 'unknown')}"
            )
            return 0.03

        return 0.0

    def _handle_check_document(self, action: Action) -> float:
        doc_key = ACTION_TO_DOCUMENT[action.action_type]
        p = self._current_patient
        ps = self._patient_state[p]
        patient_data = self._get_patient_data(p)

        content = None
        if doc_key in patient_data.get("documents", {}):
            content = patient_data["documents"][doc_key]
        elif doc_key in patient_data.get("external_data", {}):
            content = patient_data["external_data"][doc_key]

        if content is None:
            self._current_document_content = f"no {doc_key} data available for this patient."
            self._screening_notes.append(f"checked {doc_key} — not available")
            return -0.01

        self._current_document_content = json.dumps(content, indent=2)

        if doc_key not in ps["documents_reviewed"]:
            ps["documents_reviewed"].append(doc_key)
            self._screening_notes.append(f"reviewed {doc_key} for {p}")
            return self._reward_calc.document_reward(doc_key, p)

        return 0.0

    def _handle_flag_concern(self, action: Action) -> float:
        if not action.flag_reason:
            self._screening_notes.append("attempted to flag concern without reason")
            return -0.02

        p = self._current_patient
        ps = self._patient_state[p]

        is_valid, _ = self._reward_calc.check_concern(action.flag_reason, p)
        ps["concerns_found"].append(action.flag_reason)

        if is_valid:
            self._screening_notes.append(f"[concern] valid concern for {p}: {action.flag_reason}")
            return 0.10
        else:
            self._screening_notes.append(
                f"[false concern] invalid concern for {p}: {action.flag_reason}"
            )
            return -0.05

    def _handle_request_tests(self, action: Action) -> float:
        if not action.target:
            return -0.02

        self._screening_notes.append(f"requested additional test: {action.target}")
        self._current_document_content = (
            f"additional test '{action.target}' requested. "
            "results would take 2-3 business days. proceeding with available data is recommended."
        )
        return -0.02

    def _handle_make_decision(self, action: Action) -> float:
        if not action.decision or action.decision not in VALID_DECISIONS:
            self._screening_notes.append(f"invalid decision: {action.decision}")
            return -0.02

        p = self._current_patient
        ps = self._patient_state[p]

        if p in self._decisions:
            self._screening_notes.append(f"decision already made for {p}")
            return -0.05

        penalty = 0.0
        if not self._protocol_checked:
            penalty -= 0.10
            self._screening_notes.append("[penalty] decision made without checking trial protocol")
        if len(ps["documents_reviewed"]) == 0:
            penalty -= 0.20
            self._screening_notes.append("[penalty] decision made without reviewing any documents")

        decision_reward = self._reward_calc.decision_reward(
            patient_id=p,
            decision=action.decision,
            reasoning=action.decision_reasoning or "",
            approved_amount=action.approved_amount,
        )

        self._decisions[p] = {
            "decision": action.decision,
            "reasoning": action.decision_reasoning,
            "approved_amount": action.approved_amount,
        }

        self._screening_notes.append(
            f"[decision] {p}: {action.decision.upper()} — "
            f"{action.decision_reasoning or 'no reasoning provided'}"
        )

        if len(self._patients) > 1:
            undecided = [pid for pid in self._patients if pid not in self._decisions]
            if undecided:
                self._current_patient = undecided[0]
                self._screening_notes.append(f"switching to next patient: {self._current_patient}")

        return decision_reward + penalty

    # -- helpers --

    def _get_patient_data(self, patient_id: str) -> dict[str, Any]:
        if "patients" in self._task_data:
            return self._task_data["patients"].get(patient_id, {})
        return self._task_data

    def _detect_loop(self) -> bool:
        if len(self._action_history) < 6:
            return False
        last_6 = self._action_history[-6:]
        return last_6[:3] == last_6[3:]

    def _build_observation(self) -> Observation:
        p = self._current_patient
        ps = self._patient_state[p]
        patient_data = self._get_patient_data(p)

        patient_info = patient_data.get("patient", {})
        protocol = self._task_data.get("trial_protocol", {})

        all_docs: list[str] = []
        for k in patient_data.get("documents", {}):
            all_docs.append(k)
        for k in patient_data.get("external_data", {}):
            if k != "insurance_coverage":
                all_docs.append(k)

        docs_available = [d for d in all_docs if d not in ps["documents_reviewed"]]

        summary = patient_info.get("summary", f"patient {p}")
        if len(self._patients) > 1:
            decided = [pid for pid in self._patients if pid in self._decisions]
            undecided = [pid for pid in self._patients if pid not in self._decisions]
            summary += (
                f"\n\n[multi-patient task: {len(decided)}/{len(self._patients)} decisions made. "
                f"current: {p}. undecided: {', '.join(undecided)}]"
            )

        all_concerns: list[str] = []
        for pid in self._patients:
            for c in self._patient_state[pid]["concerns_found"]:
                all_concerns.append(f"[{pid}] {c}")

        all_findings: list[str] = []
        for pid in self._patients:
            for f in self._patient_state[pid]["eligibility_findings"]:
                all_findings.append(f"[{pid}] {f}")

        return Observation(
            patient_id=patient_info.get("patient_id", p),
            trial_id=protocol.get("trial_id", ""),
            patient_summary=summary,
            trial_phase=protocol.get("phase", ""),
            claimed_costs=0.0,
            available_actions=list(VALID_ACTIONS),
            documents_available=docs_available,
            documents_reviewed=ps["documents_reviewed"][:],
            current_document_content=self._current_document_content,
            eligibility_findings=all_findings,
            concerns_found=all_concerns,
            steps_taken=self._steps_taken,
            max_steps=self._max_steps,
            screening_notes=self._screening_notes[:],
        )
