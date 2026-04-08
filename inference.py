import json
import os
import sys
import time

from openai import OpenAI

from src.environment import ClinicalTrialScreenerEnv
from src.models import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", HF_TOKEN))

SYSTEM_PROMPT = """You are an expert clinical trial patient screener. Your job is to investigate whether a patient qualifies for a clinical trial by reviewing their records step-by-step.

Available actions:
- view_patient: View the current patient's info. For multi-patient tasks, use target to specify patient (e.g. "patient_a")
- check_trial_protocol: Review the trial's inclusion/exclusion criteria
- check_medical_history: Review the patient's medical history
- check_lab_results: Review laboratory results
- check_medications: Review current and past medications
- check_drug_interactions: Check for drug interactions with the trial drug
- check_genetic_markers: Review pharmacogenomic/genetic data
- check_prior_trials: Check if the patient has prior clinical trial participation
- flag_concern: Flag a concern (provide flag_reason)
- request_additional_tests: Request additional tests (provide target with test name)
- make_decision: Make final screening decision (provide decision: "enroll", "reject", "waitlist", or "refer_to_specialist", decision_reasoning, and optionally approved_amount)

Strategy:
1. First view the patient and check the trial protocol
2. Systematically review documents (medical history, labs, medications, prior trials)
3. Check external data (drug interactions, genetic markers)
4. Flag any concerns you find
5. Make a well-reasoned decision with evidence

For multi-patient tasks, investigate each patient thoroughly before making decisions.

Respond with ONLY valid JSON (no markdown, no explanation):
{
    "action_type": "...",
    "target": "..." or null,
    "flag_reason": "..." or null,
    "decision": "..." or null,
    "decision_reasoning": "..." or null,
    "approved_amount": ... or null
}"""


def observation_to_prompt(obs_dict: dict) -> str:
    parts = [
        f"Patient ID: {obs_dict['patient_id']}",
        f"Trial: {obs_dict['trial_id']} ({obs_dict['trial_phase']})",
        f"Summary: {obs_dict['patient_summary']}",
        f"Step: {obs_dict['steps_taken']}/{obs_dict['max_steps']}",
        f"Documents available: {', '.join(obs_dict['documents_available']) or 'none'}",
        f"Documents reviewed: {', '.join(obs_dict['documents_reviewed']) or 'none'}",
        f"Concerns found: {json.dumps(obs_dict['concerns_found']) if obs_dict['concerns_found'] else 'none'}",
    ]

    if obs_dict.get("current_document_content"):
        content = obs_dict["current_document_content"]
        if len(content) > 3000:
            content = content[:3000] + "... [truncated]"
        parts.append(f"\nLast document content:\n{content}")

    if obs_dict.get("screening_notes"):
        recent = obs_dict["screening_notes"][-5:]
        parts.append(f"\nRecent notes: {json.dumps(recent)}")

    return "\n".join(parts)


def parse_action(response_text: str) -> dict:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    return json.loads(text)


def run_task(task_id: str) -> dict:
    env = ClinicalTrialScreenerEnv()

    print(json.dumps({"type": "[START]", "task_id": task_id, "timestamp": time.time()}))
    sys.stdout.flush()

    obs = env.reset(task_id)
    obs_dict = obs.model_dump()
    total_reward = 0.0

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        prompt = observation_to_prompt(obs_dict)
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )
            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(json.dumps({"type": "[STEP]", "task_id": task_id, "error": str(e)}))
            sys.stdout.flush()
            break

        try:
            action_dict = parse_action(reply)
        except (json.JSONDecodeError, Exception):
            action_dict = {"action_type": "view_patient"}

        action = Action(
            action_type=action_dict.get("action_type", "view_patient"),
            target=action_dict.get("target"),
            flag_reason=action_dict.get("flag_reason"),
            decision=action_dict.get("decision"),
            decision_reasoning=action_dict.get("decision_reasoning"),
            approved_amount=action_dict.get("approved_amount"),
        )

        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        total_reward += reward

        step_log = {
            "type": "[STEP]",
            "task_id": task_id,
            "step": obs_dict["steps_taken"],
            "action": action_dict.get("action_type"),
            "reward": round(reward, 4),
            "cumulative_reward": round(total_reward, 4),
            "done": done,
        }
        print(json.dumps(step_log))
        sys.stdout.flush()

        if done:
            break

        if len(messages) > 30:
            messages = messages[:2] + messages[-10:]

    state = env.state()
    from src.grader import grade_task
    grade_result = grade_task(
        task_id=task_id,
        task_data=env._task_data,
        decisions=state["decisions"],
        patient_state=state["patient_state"],
        protocol_checked=state["protocol_checked"],
        steps_taken=state["steps_taken"],
        reward_calc=env._reward_calc,
    )

    end_log = {
        "type": "[END]",
        "task_id": task_id,
        "final_score": grade_result["score"],
        "cumulative_reward": round(total_reward, 4),
        "steps_taken": state["steps_taken"],
        "decisions": state["decisions"],
        "breakdown": grade_result["breakdown"],
        "timestamp": time.time(),
    }
    print(json.dumps(end_log))
    sys.stdout.flush()

    return grade_result


def main():
    tasks = [
        "easy_diabetes_screening",
        "medium_cardiac_interactions",
        "hard_rare_disease_multi",
    ]

    results = {}
    for task_id in tasks:
        print(f"\n{'='*60}")
        print(f"running task: {task_id}")
        print(f"{'='*60}\n")
        sys.stdout.flush()

        result = run_task(task_id)
        results[task_id] = result

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    scores = []
    for task_id, result in results.items():
        score = result["score"]
        scores.append(score)
        print(f"  {task_id}: {score:.4f}")
    avg = sum(scores) / len(scores) if scores else 0
    print(f"  average: {avg:.4f}")
    print(f"{'='*60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
