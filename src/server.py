from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Optional

from .environment import ClinicalTrialScreenerEnv
from .models import Action, Observation
from .grader import grade_task

app = FastAPI(title="Clinical Trial Screener Environment", version="1.0.0")
env = ClinicalTrialScreenerEnv()

VALID_TASKS = ["easy_diabetes_screening", "medium_cardiac_interactions", "hard_rare_disease_multi"]


class StepRequest(BaseModel):
    action_type: str
    target: Optional[str] = None
    flag_reason: Optional[str] = None
    decision: Optional[str] = None
    decision_reasoning: Optional[str] = None
    approved_amount: Optional[float] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class GradeResponse(BaseModel):
    score: float
    breakdown: dict
    reason: str


@app.get("/")
def root():
    return {"status": "ok", "environment": "clinical-trial-screener", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
async def reset(request: Request, task_id: Optional[str] = Query(None)):
    tid = task_id
    if not tid:
        try:
            body = await request.json()
            tid = body.get("task_id") if isinstance(body, dict) else None
        except Exception:
            pass
    if not tid:
        tid = VALID_TASKS[0]
    if tid not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"invalid task_id. must be one of: {VALID_TASKS}")
    obs = env.reset(tid)
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    if env._task_id is None:
        raise HTTPException(status_code=400, detail="call /reset first")

    action = Action(
        action_type=req.action_type,
        target=req.target,
        flag_reason=req.flag_reason,
        decision=req.decision,
        decision_reasoning=req.decision_reasoning,
        approved_amount=req.approved_amount,
    )
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def get_state():
    if env._task_id is None:
        raise HTTPException(status_code=400, detail="call /reset first")
    return env.state()


@app.get("/grade", response_model=GradeResponse)
def grade():
    if env._task_id is None:
        raise HTTPException(status_code=400, detail="call /reset first")

    state = env.state()
    result = grade_task(
        task_id=state["task_id"],
        task_data=env._task_data,
        decisions=state["decisions"],
        patient_state=state["patient_state"],
        protocol_checked=state["protocol_checked"],
        steps_taken=state["steps_taken"],
        reward_calc=env._reward_calc,
    )
    return GradeResponse(
        score=result["score"],
        breakdown=result["breakdown"],
        reason=result["reason"],
    )


@app.get("/health")
def health():
    return {"status": "healthy"}
