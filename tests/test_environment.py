import pytest
from src.environment import ClinicalTrialScreenerEnv
from src.models import Action


@pytest.fixture
def env():
    return ClinicalTrialScreenerEnv()


def test_reset_easy(env):
    obs = env.reset("easy_diabetes_screening")
    assert obs.patient_id == "PT-2026-00847"
    assert obs.trial_id == "NCT-2026-GBL-0042"
    assert obs.steps_taken == 0
    assert obs.max_steps == 10
    assert "medical_history" in obs.documents_available
    assert "lab_results" in obs.documents_available
    assert len(obs.documents_reviewed) == 0
    assert len(obs.concerns_found) == 0


def test_reset_medium(env):
    obs = env.reset("medium_cardiac_interactions")
    assert obs.patient_id == "PT-2026-01234"
    assert obs.trial_id == "NCT-2026-HGD-0118"
    assert obs.max_steps == 15


def test_reset_hard(env):
    obs = env.reset("hard_rare_disease_multi")
    assert obs.trial_id == "NCT-2026-VRG-0007"
    assert obs.max_steps == 25


def test_view_patient(env):
    env.reset("easy_diabetes_screening")
    action = Action(action_type="view_patient")
    obs, reward, done, info = env.step(action)
    assert reward > 0
    assert obs.current_document_content is not None
    assert "Robert Nakamura" in obs.current_document_content
    assert obs.steps_taken == 1


def test_check_protocol(env):
    env.reset("easy_diabetes_screening")
    action = Action(action_type="check_trial_protocol")
    obs, reward, done, info = env.step(action)
    assert reward == 0.03
    assert "GLUCOBALANCE" in obs.current_document_content


def test_check_document(env):
    env.reset("easy_diabetes_screening")
    action = Action(action_type="check_lab_results")
    obs, reward, done, info = env.step(action)
    assert reward == 0.05
    assert "lab_results" in obs.documents_reviewed
    assert "lab_results" not in obs.documents_available


def test_repeated_action_penalty(env):
    env.reset("easy_diabetes_screening")
    action = Action(action_type="check_lab_results")
    env.step(action)
    obs, reward, done, info = env.step(action)
    assert reward <= 0


def test_flag_valid_concern(env):
    env.reset("easy_diabetes_screening")
    action = Action(
        action_type="flag_concern",
        flag_reason="eGFR of 55 is below the required threshold of 60",
    )
    obs, reward, done, info = env.step(action)
    assert reward == 0.10
    assert len(obs.concerns_found) == 1


def test_flag_invalid_concern(env):
    env.reset("easy_diabetes_screening")
    action = Action(
        action_type="flag_concern",
        flag_reason="patient has blue eyes which is suspicious",
    )
    obs, reward, done, info = env.step(action)
    assert reward == -0.05


def test_make_correct_decision_easy(env):
    env.reset("easy_diabetes_screening")
    env.step(Action(action_type="check_trial_protocol"))
    env.step(Action(action_type="check_lab_results"))

    action = Action(
        action_type="make_decision",
        decision="reject",
        decision_reasoning="eGFR of 55 is below threshold and metformin dose changed recently",
    )
    obs, reward, done, info = env.step(action)
    assert reward > 0
    assert done


def test_make_wrong_decision_easy(env):
    env.reset("easy_diabetes_screening")
    env.step(Action(action_type="check_trial_protocol"))
    env.step(Action(action_type="check_lab_results"))

    action = Action(
        action_type="make_decision",
        decision="enroll",
        decision_reasoning="looks good",
    )
    obs, reward, done, info = env.step(action)
    assert reward < 0


def test_decision_without_protocol_penalty(env):
    env.reset("easy_diabetes_screening")
    env.step(Action(action_type="check_lab_results"))
    action = Action(
        action_type="make_decision",
        decision="reject",
        decision_reasoning="fails criteria",
    )
    obs, reward, done, info = env.step(action)
    state = env.state()
    assert "[penalty] decision made without checking trial protocol" in state["screening_notes"]


def test_decision_without_documents_penalty(env):
    env.reset("easy_diabetes_screening")
    action = Action(
        action_type="make_decision",
        decision="reject",
        decision_reasoning="reject",
    )
    obs, reward, done, info = env.step(action)
    state = env.state()
    assert "[penalty] decision made without reviewing any documents" in state["screening_notes"]


def test_max_steps_ends_episode(env):
    env.reset("easy_diabetes_screening")
    for _ in range(10):
        if env._done:
            break
        env.step(Action(action_type="view_patient"))
    assert env._done


def test_state_returns_dict(env):
    env.reset("easy_diabetes_screening")
    state = env.state()
    assert state["task_id"] == "easy_diabetes_screening"
    assert state["steps_taken"] == 0
    assert state["done"] is False


def test_hard_multi_patient_switching(env):
    env.reset("hard_rare_disease_multi")
    obs = env.reset("hard_rare_disease_multi")
    assert "multi-patient" in obs.patient_summary.lower() or len(env._patients) == 3

    obs, _, _, _ = env.step(Action(action_type="view_patient", target="patient_b"))
    assert env._current_patient == "patient_b"

    obs, _, _, _ = env.step(Action(action_type="view_patient", target="patient_c"))
    assert env._current_patient == "patient_c"


def test_hard_all_decisions_ends_episode(env):
    env.reset("hard_rare_disease_multi")
    env.step(Action(action_type="check_trial_protocol"))
    env.step(Action(action_type="check_lab_results"))

    for pid in ["patient_a", "patient_b", "patient_c"]:
        env.step(Action(action_type="view_patient", target=pid))
        env.step(Action(action_type="check_lab_results"))
        env.step(
            Action(
                action_type="make_decision",
                decision="reject",
                decision_reasoning="test",
            )
        )

    assert env._done


def test_loop_detection(env):
    env.reset("easy_diabetes_screening")
    for _ in range(3):
        env.step(Action(action_type="check_trial_protocol"))
        env.step(Action(action_type="check_lab_results"))
        env.step(Action(action_type="check_medical_history"))
        if env._done:
            break
    assert env._done


def test_score_between_0_and_1(env):
    env.reset("easy_diabetes_screening")
    env.step(Action(action_type="check_trial_protocol"))
    env.step(Action(action_type="check_lab_results"))
    env.step(Action(
        action_type="flag_concern",
        flag_reason="eGFR 55 below threshold",
    ))
    env.step(Action(
        action_type="flag_concern",
        flag_reason="metformin dose recently increased",
    ))
    env.step(Action(
        action_type="make_decision",
        decision="reject",
        decision_reasoning="eGFR of 55 below 60 threshold and metformin dose change",
    ))

    from src.grader import grade_task
    state = env.state()
    result = grade_task(
        task_id="easy_diabetes_screening",
        task_data=env._task_data,
        decisions=state["decisions"],
        patient_state=state["patient_state"],
        protocol_checked=state["protocol_checked"],
        steps_taken=state["steps_taken"],
        reward_calc=env._reward_calc,
    )
    assert 0.0 <= result["score"] <= 1.0
