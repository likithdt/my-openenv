import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from openenv.core.env_server import Environment as Env 
from server.models import CleanAction, DataObservation, DataState

class DataCleaningEnv(Env):
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initializes the environment. 
        Supports dynamic uploads via the 'df' parameter.
        """
        super().__init__() 
        self.initial_df = df
        self.df = pd.DataFrame()
        self.step_count = 0
        self.history = []
        self.current_task = "easy"

    def calculate_integrity(self) -> float:
        """
        THE GRADER METRIC: 
        Calculates a score between 0.0 and 1.0 based on Uniqueness and Completeness.
        """
        if self.df.empty:
            return 0.0
        
        # 1. Uniqueness (Penalty for duplicates)
        u = 1.0 - (self.df.duplicated().sum() / len(self.df))
        
        # 2. Completeness (Penalty for NaN values)
        total_cells = self.df.size
        c = 1.0 - (self.df.isnull().sum().sum() / total_cells) if total_cells > 0 else 1.0
        
        # Final Score is a product of both dimensions
        return round(u * c, 4)

    # --- MANDATORY METHOD: reset ---
    def reset(self, task_id: str = "easy") -> DataObservation:
        """
        REQUIREMENT: 3 Tasks (Graders).
        Resets the environment to a specific difficulty level.
        """
        self.current_task = task_id
        self.step_count = 0
        self.history = []

        if task_id == "easy":
            # TASK 1: Basic Redundancy (Only Duplicates)
            self.df = pd.DataFrame({
                'id': [1, 1, 2, 3, 3],
                'value': [10, 10, 20, 30, 30]
            })
        elif task_id == "medium":
            # TASK 2: Duplicates + Missing Values
            self.df = pd.DataFrame({
                'id': [101, 101, 102, 103],
                'value': [np.nan, 5.0, 5.0, 10.0],
                'category': [None, 'A', 'A', 'B']
            })
        elif task_id == "hard":
            # TASK 3: Real-world / User-Uploaded Data
            if self.initial_df is not None:
                self.df = self.initial_df.copy()
            else:
                # Fallback to a complex generated set if no upload exists
                self.df = pd.DataFrame({
                    'id': np.random.randint(0, 10, 10),
                    'val': np.random.choice([1.0, np.nan], 10),
                    'tag': np.random.choice(['X', None], 10)
                })

        return self._get_observation(f"Task {task_id.upper()} started. Objective: Reach score 1.0")

    # --- MANDATORY METHOD: state ---
    def state(self) -> DataState:
        """Returns the current raw state of the dataset."""
        return DataState(
            data=self.df.to_dict(orient='records'),
            integrity_score=self.calculate_integrity()
        )

    def step(self, action: CleanAction) -> Dict[str, Any]:
        """
        Executes an action, calculates the Reward, and returns the next Observation.
        """
        self.step_count += 1
        old_score = self.calculate_integrity()
        
        # Logic for Action Space
        if action.command == "drop_duplicates":
            self.df = self.df.drop_duplicates()
        elif action.command == "fill_median":
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        elif action.command == "drop_nulls":
            self.df = self.df.dropna()

        new_score = self.calculate_integrity()
        
        # Reward is the partial progress signal (improvement in score)
        reward = round((new_score - old_score) * 100, 2)
        
        # Log to History for Audit Trail
        self.history.append({
            "step": self.step_count,
            "action": action.command,
            "reward": reward,
            "score": new_score
        })
        
        return {
            "observation": self._get_observation(f"Executed {action.command}"),
            "reward": reward,
            "done": new_score >= 0.999 or self.step_count >= 10,
            "history": self.history
        }

    def _get_observation(self, goal_text: str) -> DataObservation:
        """Helper to format the internal state for the AI Agent."""
        return DataObservation(
            summary=self.df.describe(include='all').fillna(0).to_dict(),
            sample_rows=self.df.head(5).to_dict(orient='records'),
            column_names=list(self.df.columns),
            health_score=self.calculate_integrity(),
            goal=goal_text
        )
        