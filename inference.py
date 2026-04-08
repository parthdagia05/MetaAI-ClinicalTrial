"""
Clinical Trial Screener - Inference Script
Runs all 3 tasks using an LLM agent via OpenAI-compatible API.
"""

import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

from src.environment import ClinicalTrialScreenerEnv
from src.models import Action
from src.grader import grade_task

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")

client: Optional[OpenAI] = None

BENCHMARK = "clinical-trial-screener"

SYSTEM_PROMPT = """You are an expert clinical trial patient screener. Investigate whether a patient qualifies for a clinical trial by reviewing their records step-by-step.

Available actions:
- view_patient: View patient info. For multi-patient tasks, set target to patient id (e.g. "patient_a")
- check_trial_protocol: Review trial inclusion/exclusion criteria
- check_medical_history: Review medical history
- check_lab_results: Review lab results
- check_medications: Review medications
- check_drug_interactions: Check drug interactions with trial drug
- check_genetic_markers: Review genetic/pharmacogenomic data
- check_prior_trials: Check prior trial participation
- flag_concern: Flag a concern (set flag_reason)
- make_decision: Final decision (set decision to "enroll"/"reject"/"waitlist"/"refer_to_specialist", set decision_reasoning, optionally set approved_amount)

Strategy: view patient -> check protocol -> review docs -> check external data -> flag concerns -> decide.

Respond with ONLY valid JSON:
{"action_type": "...", "target": null, "flag_reason": null, "decision": null, "decision_reasoning": null, "approved_amount": null}"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


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


def run_task(task_id: str) -> float:
    env = ClinicalTrialScreenerEnv()

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset(task_id)
    obs_dict = obs.model_dump()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        max_steps = obs_dict["max_steps"]
        for step in range(1, max_steps + 1):
            prompt = observation_to_prompt(obs_dict)
            messages.append({"role": "user", "content": prompt})

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500,
                )
                reply = response.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                log_step(step=step, action="error", reward=0.0, done=False, error=str(e))
                rewards.append(0.0)
                steps_taken = step
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

            rewards.append(reward)
            steps_taken = step
            action_str = action_dict.get("action_type", "unknown")

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

            if len(messages) > 30:
                messages = messages[:2] + messages[-10:]

        state = env.state()
        grade_result = grade_task(
            task_id=task_id,
            task_data=env._task_data,
            decisions=state["decisions"],
            patient_state=state["patient_state"],
            protocol_checked=state["protocol_checked"],
            steps_taken=state["steps_taken"],
            reward_calc=env._reward_calc,
        )
        score = grade_result["score"]
        score = min(max(score, 0.0), 1.0)
        success = score > 0.1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    global client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [
        "easy_diabetes_screening",
        "medium_cardiac_interactions",
        "hard_rare_disease_multi",
    ]
    scores = []
    for task_id in tasks:
        score = run_task(task_id)
        scores.append(score)

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\naverage score: {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()
