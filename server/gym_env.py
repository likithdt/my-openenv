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
            integrity_score=float(self.calculate_integrity()),
            current_task_index=0, # You can update this based on your task logic
            steps_taken=self.step_count
        )

    def step(self, action: CleanAction) -> Dict[str, Any]:
        """
        Executes an action and returns a strictly JSON-serializable dictionary.
        """
        try:
            self.step_count += 1
            # old_score MUST be a standard float
            old_score = float(self.calculate_integrity())
            
            # 1. Standardize the command input
            cmd = str(action.command).lower()
            
            # 2. Execute Action Logic
            if cmd == "drop_duplicates":
                self.df = self.df.drop_duplicates()
            elif cmd == "fill_median":
                num_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
            elif cmd == "drop_nulls":
                self.df = self.df.dropna()

            # 3. Calculate New Metrics
            new_score = float(self.calculate_integrity())
            reward = float(round((new_score - old_score) * 100, 2))
            
            # 4. Update History (Ensure everything is a standard Python type)
            self.history.append({
                "step": int(self.step_count),
                "action": str(cmd),
                "reward": float(reward),
                "score": float(new_score)
            })
            
            # 5. Return the Response (Force all types for FastAPI validation)
            return {
                "observation": self._get_observation(f"Executed {cmd}"),
                "reward": float(reward),
                "done": bool(new_score >= 0.999 or self.step_count >= 10),
                "history": self.history
            }
            
        except Exception as e:
            # This prints the REAL error in your Hugging Face Logs
            print(f"--- STEP EXECUTION CRASH ---")
            print(f"Error: {str(e)}")
            raise e

    def _get_observation(self, goal_text: str) -> DataObservation:
        """
        Force-converts all NumPy/Pandas types to standard Python types 
        to prevent FastAPI validation 500 errors.
        """
        return DataObservation(
            # Convert DataFrame summary to a clean dict
            summary=self.df.describe(include='all').fillna(0).to_dict(),
            # Convert sample rows to a list of dicts
            sample_rows=self.df.head(5).to_dict(orient='records'),
            # Ensure column names are a list of strings
            column_names=[str(c) for c in self.df.columns],
            # CRITICAL: Force health_score to a standard Python float
            health_score=float(self.calculate_integrity()),
            # Ensure goal is a string
            goal=str(goal_text)
        )

