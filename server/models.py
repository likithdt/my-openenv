from pydantic import Field
from openenv.core.env_server import Observation, Action, State
from typing import List, Dict, Any, Optional

class DataObservation(Observation):
    summary: Dict[str, Any]
    sample_rows: List[Dict[str, Any]]
    column_names: List[str]
    health_score: float
    goal: str

class CleanAction(Action):
    command: str
    target_column: Optional[str] = None

class DataState(State):
    current_task_index: int = 0
    steps_taken: int = 0
