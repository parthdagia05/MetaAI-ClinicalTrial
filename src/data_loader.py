import json
from pathlib import Path
from typing import Any

TASKS_DIR = Path(__file__).parent.parent / "tasks"


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_task_data(task_id: str) -> dict[str, Any]:
    task_path = TASKS_DIR / task_id
    if not task_path.exists():
        raise ValueError(f"task '{task_id}' not found")

    data: dict[str, Any] = {"task_id": task_id}

    for json_file in task_path.glob("*.json"):
        data[json_file.stem] = load_json(json_file)

    docs_path = task_path / "documents"
    if docs_path.exists():
        data["documents"] = {
            f.stem: load_json(f) for f in docs_path.glob("*.json")
        }

    ext_path = task_path / "external_data"
    if ext_path.exists():
        data["external_data"] = {
            f.stem: load_json(f) for f in ext_path.glob("*.json")
        }

    # handle multi-patient tasks (hard)
    sub_patients = {}
    for sub in sorted(task_path.iterdir()):
        if sub.is_dir() and sub.name.startswith("patient_"):
            patient_data: dict[str, Any] = {}
            for json_file in sub.glob("*.json"):
                patient_data[json_file.stem] = load_json(json_file)

            pdocs = sub / "documents"
            if pdocs.exists():
                patient_data["documents"] = {
                    f.stem: load_json(f) for f in pdocs.glob("*.json")
                }

            pext = sub / "external_data"
            if pext.exists():
                patient_data["external_data"] = {
                    f.stem: load_json(f) for f in pext.glob("*.json")
                }

            sub_patients[sub.name] = patient_data

    if sub_patients:
        data["patients"] = sub_patients

    return data
