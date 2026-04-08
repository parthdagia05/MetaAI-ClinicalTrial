#!/bin/bash
set -e

echo "=== Clinical Trial Screener - Validation ==="
echo ""

echo "[1/5] checking required files..."
for f in openenv.yaml Dockerfile requirements.txt inference.py src/environment.py src/models.py src/grader.py src/reward.py src/server.py src/data_loader.py; do
    if [ ! -f "$f" ]; then
        echo "FAIL: missing $f"
        exit 1
    fi
done
echo "  all required files present"

echo ""
echo "[2/5] checking task data..."
for task in easy_diabetes_screening medium_cardiac_interactions hard_rare_disease_multi; do
    if [ ! -d "tasks/$task" ]; then
        echo "FAIL: missing tasks/$task"
        exit 1
    fi
    if [ ! -f "tasks/$task/answer_key.json" ]; then
        echo "FAIL: missing tasks/$task/answer_key.json"
        exit 1
    fi
done
echo "  all task data present"

echo ""
echo "[3/5] running unit tests..."
python -m pytest tests/ -v --tb=short
echo "  tests passed"

echo ""
echo "[4/5] validating openenv.yaml structure..."
python -c "
import yaml
with open('openenv.yaml') as f:
    config = yaml.safe_load(f)
assert 'name' in config, 'missing name'
assert 'version' in config, 'missing version'
assert 'tasks' in config, 'missing tasks'
assert len(config['tasks']) >= 3, 'need at least 3 tasks'
for task in config['tasks']:
    assert 'id' in task, 'task missing id'
    assert 'difficulty' in task, 'task missing difficulty'
print('  openenv.yaml valid')
"

echo ""
echo "[5/5] testing environment reset for all tasks..."
python -c "
from src.environment import ClinicalTrialScreenerEnv
env = ClinicalTrialScreenerEnv()
for task_id in ['easy_diabetes_screening', 'medium_cardiac_interactions', 'hard_rare_disease_multi']:
    obs = env.reset(task_id)
    assert obs.patient_id, f'{task_id}: no patient_id'
    assert obs.trial_id, f'{task_id}: no trial_id'
    assert obs.max_steps > 0, f'{task_id}: invalid max_steps'
    assert len(obs.available_actions) > 0, f'{task_id}: no available actions'
    print(f'  {task_id}: OK (patient={obs.patient_id}, trial={obs.trial_id})')
print('  all tasks reset successfully')
"

echo ""
echo "=== VALIDATION PASSED ==="
