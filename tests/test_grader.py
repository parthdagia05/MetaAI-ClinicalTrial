import pytest
from src.environment import ClinicalTrialScreenerEnv
from src.grader import grade_task
from src.models import Action


def _run_perfect_easy():
    env = ClinicalTrialScreenerEnv()
    env.reset("easy_diabetes_screening")

    env.step(Action(action_type="view_patient"))
    env.step(Action(action_type="check_trial_protocol"))
    env.step(Action(action_type="check_lab_results"))
    env.step(Action(action_type="check_medications"))
    env.step(Action(
        action_type="flag_concern",
        flag_reason="eGFR of 55 is below the required 60 threshold",
    ))
    env.step(Action(
        action_type="flag_concern",
        flag_reason="metformin dose was recently increased, not stable for 3 months",
    ))
    env.step(Action(
        action_type="make_decision",
        decision="reject",
        decision_reasoning="patient fails two criteria: eGFR 55 below 60 threshold and metformin dose change within 3 months",
    ))

    state = env.state()
    return grade_task(
        task_id="easy_diabetes_screening",
        task_data=env._task_data,
        decisions=state["decisions"],
        patient_state=state["patient_state"],
        protocol_checked=state["protocol_checked"],
        steps_taken=state["steps_taken"],
        reward_calc=env._reward_calc,
    )


def test_perfect_easy_score():
    result = _run_perfect_easy()
    assert result["score"] >= 0.9
    assert result["score"] <= 1.0


def test_easy_wrong_decision():
    env = ClinicalTrialScreenerEnv()
    env.reset("easy_diabetes_screening")
    env.step(Action(action_type="check_trial_protocol"))
    env.step(Action(action_type="check_lab_results"))
    env.step(Action(
        action_type="make_decision",
        decision="enroll",
        decision_reasoning="patient looks fine",
    ))

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
    assert result["score"] < 0.3


def test_medium_correct_decision():
    env = ClinicalTrialScreenerEnv()
    env.reset("medium_cardiac_interactions")

    env.step(Action(action_type="view_patient"))
    env.step(Action(action_type="check_trial_protocol"))
    env.step(Action(action_type="check_medical_history"))
    env.step(Action(action_type="check_lab_results"))
    env.step(Action(action_type="check_medications"))
    env.step(Action(action_type="check_drug_interactions"))
    env.step(Action(action_type="check_prior_trials"))
    env.step(Action(
        action_type="flag_concern",
        flag_reason="St. John's Wort is a CYP3A4 inducer contraindicated with Vorixaban",
    ))
    env.step(Action(
        action_type="flag_concern",
        flag_reason="warfarin requires 14-day washout before trial drug",
    ))
    env.step(Action(
        action_type="flag_concern",
        flag_reason="prior adverse event in AFGUARD-1 trial with factor Xa inhibitor",
    ))
    env.step(Action(
        action_type="make_decision",
        decision="waitlist",
        decision_reasoning="patient meets core criteria but St. John's Wort CYP3A4 inducer must be discontinued, warfarin washout needed. Age and CHA2DS2-VASc qualify. Liver function normal.",
    ))

    state = env.state()
    result = grade_task(
        task_id="medium_cardiac_interactions",
        task_data=env._task_data,
        decisions=state["decisions"],
        patient_state=state["patient_state"],
        protocol_checked=state["protocol_checked"],
        steps_taken=state["steps_taken"],
        reward_calc=env._reward_calc,
    )
    assert result["score"] >= 0.8


def test_grader_deterministic():
    result1 = _run_perfect_easy()
    result2 = _run_perfect_easy()
    assert result1["score"] == result2["score"]
    assert result1["breakdown"] == result2["breakdown"]


def test_all_graders_produce_valid_scores():
    for task_id in ["easy_diabetes_screening", "medium_cardiac_interactions", "hard_rare_disease_multi"]:
        env = ClinicalTrialScreenerEnv()
        env.reset(task_id)
        env.step(Action(action_type="check_trial_protocol"))
        env.step(Action(action_type="check_lab_results"))

        if task_id == "hard_rare_disease_multi":
            for pid in ["patient_a", "patient_b", "patient_c"]:
                env.step(Action(action_type="view_patient", target=pid))
                env.step(Action(action_type="check_lab_results"))
                env.step(Action(
                    action_type="make_decision",
                    decision="reject",
                    decision_reasoning="test decision",
                ))
        else:
            env.step(Action(
                action_type="make_decision",
                decision="reject",
                decision_reasoning="test decision",
            ))

        state = env.state()
        result = grade_task(
            task_id=task_id,
            task_data=env._task_data,
            decisions=state["decisions"],
            patient_state=state["patient_state"],
            protocol_checked=state["protocol_checked"],
            steps_taken=state["steps_taken"],
            reward_calc=env._reward_calc,
        )
        assert 0.0 <= result["score"] <= 1.0, f"{task_id} score out of range: {result['score']}"
