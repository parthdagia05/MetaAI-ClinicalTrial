from src.models import Observation, Action, Reward


def test_observation_creation():
    obs = Observation(
        patient_id="PT-001",
        trial_id="NCT-001",
        patient_summary="test patient",
        trial_phase="Phase III",
        claimed_costs=0.0,
        available_actions=["view_patient", "check_trial_protocol"],
        documents_available=["medical_history", "lab_results"],
        documents_reviewed=[],
        current_document_content=None,
        eligibility_findings=[],
        concerns_found=[],
        steps_taken=0,
        max_steps=10,
        screening_notes=[],
    )
    assert obs.patient_id == "PT-001"
    assert obs.steps_taken == 0
    assert len(obs.available_actions) == 2


def test_action_creation():
    action = Action(action_type="view_patient")
    assert action.action_type == "view_patient"
    assert action.target is None

    action = Action(
        action_type="make_decision",
        decision="reject",
        decision_reasoning="fails criteria",
    )
    assert action.decision == "reject"


def test_reward_creation():
    r = Reward(score=0.1, reason="checked document", cumulative_score=0.3)
    assert r.score == 0.1
    assert r.cumulative_score == 0.3


def test_action_with_all_fields():
    action = Action(
        action_type="make_decision",
        target=None,
        flag_reason=None,
        decision="enroll",
        decision_reasoning="meets all criteria",
        approved_amount=18000.0,
    )
    assert action.approved_amount == 18000.0
