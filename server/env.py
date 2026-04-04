import pandas as pd
import numpy as np
from openenv.core.env_server import Environment
from .models import DataObservation, CleanAction, DataState 

class DataCleaningEnv(Environment):
    def __init__(self):
        # Mandatory call to parent
        super().__init__()
        # Manually set the attribute the error is complaining about
        self.episode_id = "integrity-lab-v1"
        
        data = {
            'id': [1, 2, 3, 4, 1],
            'name': ['Likith', 'Arjun', 'Sneha', 'Rahul', 'Likith'],
            'age': [21, 22, 25, 23, 21],
            'salary': [50000, 60000, 55000, 70000, 50000]
        }
        self.original_df = pd.DataFrame(data)
        self.df = self.original_df.copy()
        self.current_task = 0
        self.step_count = 0

    def state(self) -> DataState:
        return DataState(current_task_index=self.current_task, steps_taken=self.step_count)

    def reset(self) -> DataObservation:
        self.df = self.original_df.copy()
        self.current_task = 0
        self.step_count = 0
        return self._get_observation("Task 1: Remove duplicates.")

    def step(self, action: CleanAction):
        try:
            self.step_count += 1
            reward = 0.0
            done = False
            
            # Logic for Task 0
            if getattr(self, 'current_task', 0) == 0 and action.command == "drop_duplicates":
                before = len(self.df)
                self.df = self.df.drop_duplicates()
                if len(self.df) < before:
                    reward = 1.0
                    self.current_task = 1
            
            # Default return if no specific task logic matches
            return {
                "observation": self._get_observation("Next Task: Fill missing values in Age"),
                "reward": reward,
                "done": done,
                "info": {"task": getattr(self, 'current_task', 0)}
            }
        except Exception as e:
            # This helps you see the error in the logs more clearly
            print(f"Error in step: {e}")
            raise e

    def _get_observation(self, goal: str) -> DataObservation:
        # Crucial: Convert numpy types to native Python types for JSON
        null_counts = self.df.isnull().sum().to_dict()
        clean_summary = {str(k): int(v) for k, v in null_counts.items()}
        
        return DataObservation(
            summary=clean_summary,
            sample_rows=self.df.head().to_dict('records'),
            column_names=list(self.df.columns),
            goal=goal
        )
