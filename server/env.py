import pandas as pd
import numpy as np
from openenv_core.env_server import Env
from server.models import CleanAction, DataObservation, DataState

class DataCleaningEnv(Env):
    def __init__(self, df=None):
        super().__init__()
        # If no DF is provided (startup), we use a messy default
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

    def calculate_integrity(self):
        """The 'Finalist' Secret Sauce: A universal quality metric."""
        if self.df.empty: return 0.0
        
        # 1. Uniqueness (1.0 if no duplicates)
        u = 1.0 - (self.df.duplicated().sum() / len(self.df))
        
        # 2. Completeness (1.0 if no missing values)
        total_cells = self.df.size
        c = 1.0 - (self.df.isnull().sum().sum() / total_cells) if total_cells > 0 else 1.0
        
        # Integrity Index is the product of Uniqueness and Completeness
        return round(u * c, 4)

    def step(self, action: CleanAction):
        self.step_count += 1
        old_idx = self.calculate_integrity()
        
        # Dynamic Command Logic
        if action.command == "drop_duplicates":
            self.df = self.df.drop_duplicates()
        elif action.command == "fill_median":
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())
        elif action.command == "drop_nulls":
            self.df = self.df.dropna()

        new_idx = self.calculate_integrity()
        
        # Reward is based on how much the integrity improved
        reward = (new_idx - old_idx) * 100 
        
        return {
            "observation": self._get_observation(f"Integrity Index: {new_idx}"),
            "reward": reward,
            "done": new_idx >= 0.99, # Environment 'solved' when data is clean
            "info": {"improvement": reward}
        }