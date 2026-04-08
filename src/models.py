from typing import List, Optional
from pydantic import BaseModel


class Observation(BaseModel):
    patient_id: str
    trial_id: str
    patient_summary: str
    trial_phase: str
    claimed_costs: float
    available_actions: List[str]
    documents_available: List[str]
    documents_reviewed: List[str]
    current_document_content: Optional[str] = None
    eligibility_findings: List[str]
    concerns_found: List[str]
    steps_taken: int
    max_steps: int
    screening_notes: List[str]


class Action(BaseModel):
    action_type: str
    # one of: "view_patient", "check_trial_protocol", "check_medical_history",
    #   "check_lab_results", "check_medications", "check_drug_interactions",
    #   "check_genetic_markers", "check_prior_trials",
    #   "flag_concern", "request_additional_tests", "make_decision"

    target: Optional[str] = None
    # for check_drug_interactions: drug name
    # for request_additional_tests: test name

    flag_reason: Optional[str] = None
    # for flag_concern: description of the concern

    decision: Optional[str] = None
    # for make_decision: "enroll", "reject", "waitlist", "refer_to_specialist"

    decision_reasoning: Optional[str] = None
    # for make_decision: explanation

    approved_amount: Optional[float] = None
    # for make_decision with "enroll": estimated trial cost coverage


class Reward(BaseModel):
    score: float
    reason: str
    cumulative_score: float
