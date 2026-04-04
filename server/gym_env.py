import pandas as pd
import numpy as np
from typing import Any, Dict, List
from openenv.core.env_server import Environment as Env 
from server.models import CleanAction, DataObservation, DataState

class DataCleaningEnv(Env):
    def __init__(self, df=None):
        """
        Initializes the environment. 
        If df is None, it uses a default messy dataset for the initial boot.
        """
        super().__init__() 
        
        if df is None:
            # Default messy data for startup
            self.initial_df = pd.DataFrame({
                'id': [101, 102, 102, 104, 105, 105],
                'value': [10.5, 20.0, 20.0, np.nan, 30.0, 30.0],
                'category': ['A', 'B', 'B', None, 'C', 'C']
            })
        else:
            self.initial_df = df.copy()
            
        self.df = self.initial_df.copy()
        self.step_count = 0
        self.history = []  # NEW: The Audit Trail / Provenance Log

    def calculate_integrity(self) -> float:
        """
        The Mathematical Auditor: 
        Calculates a score between 0 and 1 based on Uniqueness and Completeness.
        """
        if self.df.empty:
            return 0.0
        
        # 1. Uniqueness (1.0 if no duplicates)
        u = 1.0 - (self.df.duplicated().sum() / len(self.df))
        
        # 2. Completeness (1.0 if no null values)
        total_cells = self.df.size
        c = 1.0 - (self.df.isnull().sum().sum() / total_cells) if total_cells > 0 else 1.0
        
        return round(u * c, 4)

    # --- MANDATORY METHOD: reset ---
    def reset(self) -> DataObservation:
        """Resets the environment and clears the audit history."""
        self.df = self.initial_df.copy()
        self.step_count = 0
        self.history = [] # NEW: Clear history on reset
        return self._get_observation("Goal: Reach Integrity Index 1.0 by cleaning the data.")

    # --- MANDATORY METHOD: state ---
    def state(self) -> DataState:
        """Returns the current raw state of the dataset and its health score."""
        return DataState(
            data=self.df.to_dict(orient='records'),
            integrity_score=self.calculate_integrity()
        )

    def step(self, action: CleanAction) -> Dict[str, Any]:
        """
        Executes a cleaning action, calculates the reward, and logs it to the history.
        """
        self.step_count += 1
        old_idx = self.calculate_integrity()
        
        # Action Logic
        if action.command == "drop_duplicates":
            self.df = self.df.drop_duplicates()
        elif action.command == "fill_median":
            # Target numeric columns for median imputation
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        elif action.command == "drop_nulls":
            self.df = self.df.dropna()

        new_idx = self.calculate_integrity()
        
        # Reward is the Delta of the Integrity Index
        reward = round((new_idx - old_idx) * 100, 2)
        
        # NEW: Log to Audit Trail for the judge's review
        self.history.append({
            "step": self.step_count,
            "action": action.command,
            "reward": reward,
            "new_integrity": new_idx
        })
        
        return {
            "observation": self._get_observation(f"Action '{action.command}' executed."),
            "reward": reward,
            "done": new_idx >= 0.999,
            "history": self.history # Returns the log in the response
        }

    def _get_observation(self, goal_text: str) -> DataObservation:
        """Helper to format the data for the AI agent / API."""
        return DataObservation(
            summary=self.df.describe(include='all').fillna(0).to_dict(),
            sample_rows=self.df.head(5).to_dict(orient='records'),
            column_names=list(self.df.columns),
            health_score=self.calculate_integrity(),
            goal=goal_text
        )
        