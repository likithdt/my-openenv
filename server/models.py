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
    # The command the agent sends (e.g., "drop_duplicates")
    command: str 
    target_column: Optional[str] = None

class DataState(State):
    # REQUIRED: OpenEnv State usually needs the 'data' field to render in the UI
    data: List[Dict[str, Any]] 
    integrity_score: float
    current_task_index: int = 0
    steps_taken: int = 0
    