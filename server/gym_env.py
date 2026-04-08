import pandas as pd
import numpy as np
import os
import sys
import io
from typing import List, Dict, Any, Optional

try:
    from .models import DataObservation, CleanAction, DataState
except (ImportError, ValueError):
    try:
        from models import DataObservation, CleanAction, DataState
    except ImportError:
        from server.models import DataObservation, CleanAction, DataState

class DataCleaningEnv:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df if df is not None else pd.DataFrame()
        self.initial_df = self.df.copy() if not self.df.empty else None
        self.current_task = "User Input Mode"
        self.step_count = 0
        self.history = []

    def load_user_data(self, csv_content: str):
        self.df = pd.read_csv(io.StringIO(csv_content))
        self.initial_df = self.df.copy()
        self.step_count = 0
        self.history = []
        return self._get_observation("Dataset loaded")

    def calculate_integrity(self) -> float:
        if self.df is None or self.df.empty:
            return 0.30
        
        total_elements = self.df.size + 1e-9
        total_rows = len(self.df) + 1e-9
        
        null_penalty = self.df.isnull().sum().sum() / total_elements
        dup_penalty = self.df.duplicated().sum() / total_rows
        
        score = 1.0 - (null_penalty + dup_penalty)
        return float(max(0.15, min(0.90, score)))

    def _get_observation(self, goal_text: str) -> DataObservation:
        null_count = int(self.df.isnull().sum().sum())
        dup_count = int(self.df.duplicated().sum())
        status_hint = f"{goal_text}. Current: {null_count} nulls, {dup_count} dups."
        summary_dict = self.df.describe(include='all').fillna(0).to_dict() if not self.df.empty else {}
        
        return DataObservation(
            summary=summary_dict,
            sample_rows=self.df.head(5).to_dict(orient='records'),
            column_names=[str(c) for c in self.df.columns],
            health_score=float(self.calculate_integrity()),
            goal=status_hint
        )

    def reset(self, task_id: str = "user_current") -> DataObservation:
        self.step_count = 0
        self.history = []
        if self.initial_df is not None:
            self.df = self.initial_df.copy()
        return self._get_observation("Environment reset")

    def step(self, action_input: Any) -> Dict[str, Any]:
        self.step_count += 1
        old_score = self.calculate_integrity()
        
        cmd = str(action_input.get("command", "") if isinstance(action_input, dict) else action_input.command).lower()
        
        if "duplicate" in cmd:
            self.df = self.df.drop_duplicates()
        elif "median" in cmd:
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                self.df[col] = self.df[col].fillna(float(self.df[col].median()))
        elif "null" in cmd:
            self.df = self.df.dropna()

        new_score = self.calculate_integrity()
        reward = float(round((new_score - old_score) * 100, 2)) - 1.001

        self.history.append({
            "step": int(self.step_count),
            "action": cmd,
            "reward": reward,
            "score": float(new_score)
        })

        return {
            "observation": self._get_observation("Action processed"),
            "reward": reward,
            "done": bool(new_score >= 0.98 or self.step_count >= 15),
            "history": self.history
        }

    def state(self) -> DataState:
        return DataState(
            data=self.df.to_dict(orient='records'),
            integrity_score=float(self.calculate_integrity()),
            current_task_index=0,
            steps_taken=self.step_count
        )