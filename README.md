---
title: Clinical Trial Screener
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Clinical Trial Patient Screener - OpenEnv Environment

An OpenEnv-compliant reinforcement learning environment that simulates the job of a clinical trial patient screener. An AI agent receives patient referrals and must investigate them step-by-step - reviewing medical records, checking lab results, identifying drug interactions, and making enrollment decisions (enroll, reject, waitlist, or refer to specialist).

## Why This Matters

Clinical trials are the backbone of modern medicine, yet patient screening is a slow, error-prone bottleneck. Coordinators manually cross-reference eligibility criteria against medical records, lab values, medications, and drug interactions. Mistakes can endanger patients (enrolling someone with a contraindication) or delay trials (rejecting eligible candidates). AI agents are being built to assist this process, but **no standardized benchmark exists** to evaluate how good these agents are at clinical screening.

This environment fills that gap.

## Tasks

| Task | Difficulty | Scenario | Correct Decision | Expected Steps |
|------|-----------|----------|-----------------|----------------|
| `easy_diabetes_screening` | Easy | Screen a patient for a Phase III diabetes drug trial. Two clear disqualifiers hidden in labs and medication history. | Reject | 5-7 |
| `medium_cardiac_interactions` | Medium | Screen a cardiac patient with a hidden CYP3A4 drug interaction and warfarin washout requirement. | Waitlist | 8-11 |
| `hard_rare_disease_multi` | Hard | Screen three patients for a rare autoimmune disease (GPA) trial. One is fraudulently diagnosed, one has a hidden concurrent trial, one is legitimately eligible. | Reject / Reject / Enroll | 14-20 |

## Action Space

| Action | Description | Parameters |
|--------|-------------|------------|
| `view_patient` | View patient demographics and summary | `target`: patient ID (for multi-patient tasks) |
| `check_trial_protocol` | Review trial inclusion/exclusion criteria | - |
| `check_medical_history` | Review patient's medical history | - |
| `check_lab_results` | Review laboratory results | - |
| `check_medications` | Review current and past medications | - |
| `check_drug_interactions` | Check drug interactions with trial drug | - |
| `check_genetic_markers` | Review pharmacogenomic/genetic data | - |
| `check_prior_trials` | Check prior clinical trial participation | - |
| `flag_concern` | Flag a screening concern | `flag_reason`: description |
| `request_additional_tests` | Request additional diagnostic tests | `target`: test name |
| `make_decision` | Make final screening decision | `decision`, `decision_reasoning`, `approved_amount` |

## Observation Space

```python
class Observation(BaseModel):
    patient_id: str
    trial_id: str
    patient_summary: str
    trial_phase: str
    claimed_costs: float
    available_actions: List[str]
    documents_available: List[str]      # documents not yet reviewed
    documents_reviewed: List[str]       # documents already reviewed
    current_document_content: Optional[str]  # content of last viewed document
    eligibility_findings: List[str]
    concerns_found: List[str]           # concerns the agent has flagged
    steps_taken: int
    max_steps: int
    screening_notes: List[str]          # running log of actions taken
```

## Reward Function

### Per-Step Rewards

| Action | Condition | Reward |
|--------|-----------|--------|
| `view_patient` | First time | +0.02 |
| `check_trial_protocol` | First time | +0.03 |
| `check_document` | Document has relevant info | +0.02 to +0.08 |
| `check_document` | No useful info | -0.01 |
| `flag_concern` | Correct concern | +0.10 |
| `flag_concern` | False concern | -0.05 |
| `request_additional_tests` | - | -0.02 |
| Repeated action | Same action twice | -0.05 |

### Final Decision Rewards

| Decision | Condition | Reward |
|----------|-----------|--------|
| Reject | Correct | +0.30 |
| Reject | Wrong (was eligible) | -0.40 |
| Enroll | Correct | +0.20 |
| Enroll | Wrong (should reject) | -0.30 |
| Waitlist | Correct | +0.25 |
| Good reasoning | Cites evidence | +0.10 |

### Penalties

| Behavior | Penalty |
|----------|---------|
| Exceeding max steps | -0.15 |
| Decision without checking protocol | -0.10 |
| Decision without any documents | -0.20 |
| Infinite loop detected | Episode terminates, score = 0.0 |

## Setup

### Local Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # run tests
python -m uvicorn src.server:app    # start server
```

### Docker

```bash
docker build -t clinical-trial-screener .
docker run -p 7860:7860 clinical-trial-screener
```

### Run Baseline Agent

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="your-key"
python inference.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/reset` | Start a new task. Body: `{"task_id": "easy_diabetes_screening"}` |
| POST | `/step` | Take an action. Body: `{"action_type": "view_patient", ...}` |
| GET | `/state` | Get current environment state |
| GET | `/grade` | Get final grading for completed episode |

## Baseline Scores

Baseline agent using `llama-3.3-70b-versatile` via Groq:

| Task | Score | Decision | Notes |
|------|-------|----------|-------|
| `easy_diabetes_screening` | **0.80** | Reject (correct) | Found 1/2 concerns, correct reasoning |
| `medium_cardiac_interactions` | **0.43** | Reject (should be waitlist) | Identified CYP3A4 issue but was too aggressive |
| `hard_rare_disease_multi` | **0.13** | Incomplete | Multi-patient task challenges frontier models |
| **Average** | **0.45** | | |

The difficulty progression is clear: easy tasks are solvable, medium requires nuance, and hard genuinely challenges AI agents.

## Example Usage

```python
from src.environment import ClinicalTrialScreenerEnv
from src.models import Action

env = ClinicalTrialScreenerEnv()
obs = env.reset("easy_diabetes_screening")

obs, reward, done, info = env.step(Action(action_type="view_patient"))
obs, reward, done, info = env.step(Action(action_type="check_trial_protocol"))
obs, reward, done, info = env.step(Action(action_type="check_lab_results"))

obs, reward, done, info = env.step(Action(
    action_type="flag_concern",
    flag_reason="eGFR of 55 is below the required threshold of 60"
))

obs, reward, done, info = env.step(Action(
    action_type="make_decision",
    decision="reject",
    decision_reasoning="Patient fails eGFR threshold and metformin dose stability requirement"
))
# done = True, grading available
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API base URL (default: OpenAI) |
| `MODEL_NAME` | Model to use for inference (default: gpt-4o) |
| `HF_TOKEN` | Hugging Face token for deployment |

## Project Structure

```
├── openenv.yaml              # OpenEnv configuration
├── Dockerfile                # Docker deployment
├── requirements.txt
├── inference.py              # Baseline LLM agent
├── src/
│   ├── environment.py        # Core environment (reset/step/state)
│   ├── models.py             # Pydantic models (Observation, Action, Reward)
│   ├── reward.py             # Per-step reward calculation
│   ├── grader.py             # Task grading logic (0.0-1.0)
│   ├── data_loader.py        # JSON task data loader
│   └── server.py             # FastAPI server
├── tasks/
│   ├── easy_diabetes_screening/
│   ├── medium_cardiac_interactions/
│   └── hard_rare_disease_multi/
├── tests/                    # 28 unit tests
└── scripts/
    └── validate.sh           # Validation script
```
