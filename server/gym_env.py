import pandas as pd
import numpy as np
from typing import Any
from openenv.core.env_server import Environment as Env 
from server.models import CleanAction, DataObservation, DataState

class DataCleaningEnv(Env):
    def __init__(self, df=None):
        super().__init__() 
        if df is None:
            self.initial_df = pd.DataFrame({
                'id': [101, 102, 102, 104],
                'value': [10.5, 20.0, 20.0, np.nan],
                'category': ['A', 'B', 'B', None]
            })
        else:
            self.initial_df = df.copy()
            
        self.df = self.initial_df.copy()
        self.step_count = 0

    def calculate_integrity(self) -> float:
        """Calculates a global quality score (Uniqueness * Completeness)."""
        if self.df.empty: return 0.0
        u = 1.0 - (self.df.duplicated().sum() / len(self.df))
        c = 1.0 - (self.df.isnull().sum().sum() / self.df.size) if self.df.size > 0 else 1.0
        return round(u * c, 4)

    # MANDATORY METHOD 1: reset
    def reset(self) -> DataObservation:
        self.df = self.initial_df.copy()
        self.step_count = 0
        return self._get_observation("Environment Reset: Ready for dynamic cleaning.")

    # MANDATORY METHOD 2: state
    def state(self) -> DataState:
        """Returns the current raw state of the environment."""
        return DataState(
            data=self.df.to_dict(orient='records'),
            integrity_score=self.calculate_integrity()
        )

    def step(self, action: CleanAction) -> dict:
        old_idx = self.calculate_integrity()
        
        if action.command == "drop_duplicates":
            self.df = self.df.drop_duplicates()
        elif action.command == "fill_median":
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())
        elif action.command == "drop_nulls":
            self.df = self.df.dropna()

        new_idx = self.calculate_integrity()
        reward = (new_idx - old_idx) * 100 
        
        return {
            "observation": self._get_observation(f"Current Integrity: {new_idx}"),
            "reward": reward,
            "done": new_idx >= 0.99,
            "info": {"improvement": reward}
        }

    def _get_observation(self, goal_text: str) -> DataObservation:
        """Helper to format the observation for the API."""
        return DataObservation(
            summary=self.df.describe().to_dict(),
            sample_rows=self.df.head(5).to_dict(orient='records'),
            column_names=list(self.df.columns),
            health_score=self.calculate_integrity(),
            goal=goal_text
        )
        