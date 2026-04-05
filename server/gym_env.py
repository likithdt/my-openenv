import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from server.models import DataObservation, CleanAction, DataState

class DataCleaningEnv:
    def __init__(self):
        self.df = pd.DataFrame()
        self.initial_df = None
        self.current_task = "easy"
        self.step_count = 0
        self.history = []

    def calculate_integrity(self) -> float:
        """
        Deterministic Grader:
        1.0 = No nulls AND no duplicates.
        """
        if self.df.empty:
            return 0.0
        
        null_penalty = self.df.isnull().sum().sum() / (self.df.size + 1e-9)
        dup_penalty = self.df.duplicated().sum() / (len(self.df) + 1e-9)
        
        score = 1.0 - (null_penalty + dup_penalty)
        return float(max(0.0, min(1.0, score)))

    def _get_observation(self, goal_text: str) -> DataObservation:
        """
        Creates a rich observation. 
        Includes 'Hints' in the goal field to help LLMs avoid loops.
        """
        null_count = int(self.df.isnull().sum().sum())
        dup_count = int(self.df.duplicated().sum())
        
        status_hint = f"{goal_text}. Found {null_count} nulls and {dup_count} duplicates."
        
        return DataObservation(
            summary=self.df.describe(include='all').fillna(0).to_dict(),
            sample_rows=self.df.head(5).to_dict(orient='records'),
            column_names=[str(c) for c in self.df.columns],
            health_score=float(self.calculate_integrity()),
            goal=status_hint
        )

    def reset(self, task_id: str = "easy") -> DataObservation:
        self.current_task = task_id
        self.step_count = 0
        self.history = []
        
        if task_id == "easy":
            self.df = pd.DataFrame({
                'id': [1, 2, 2, 3, 3],
                'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'Charlie'],
                'age': [25, 30, 30, 35, 35]
            })
        elif task_id == "medium":
            self.df = pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'val': [10, np.nan, 30, np.nan, 50],
                'cat': ['A', 'B', None, 'D', 'E']
            })
        elif task_id == "hard":
            np.random.seed(42)
            n_rows = 100
            self.df = pd.DataFrame({
                'transaction_id': range(n_rows),
                'user_id': np.random.choice(range(50), n_rows), 
                'amount': [float(x) if np.random.random() > 0.15 else np.nan for x in np.random.uniform(10, 500, n_rows)],
                'status': np.random.choice(['Success', 'Pending', None, 'Error'], n_rows)
            })
            
            extra_dups = self.df.iloc[:10]
            self.df = pd.concat([self.df, extra_dups]).sample(frac=1).reset_index(drop=True)
            
        self.initial_df = self.df.copy()
        return self._get_observation(f"Clean the {task_id} dataset to reach score 1.0")

    def step(self, action: CleanAction) -> Dict[str, Any]:
        """
        Executes action, calculates reward, and provides feedback.
        """
        self.step_count += 1
        old_score = self.calculate_integrity()
        
        cmd = str(action.command).lower()
        
        if "duplicate" in cmd:
            self.df = self.df.drop_duplicates()
        elif "median" in cmd:
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                self.df[col] = self.df[col].fillna(float(self.df[col].median()))
        elif "null" in cmd:
            self.df = self.df.dropna()

        new_score = self.calculate_integrity()
        progress = new_score - old_score
        step_penalty = -1.0

        if progress > 0:
            reward = float(round(progress * 100, 2))+step_penalty
            feedback = "Success! Integrity improved."
        else:
            reward = -2.0+step_penalty
            feedback = "No change. Try a different cleaning command."

        self.history.append({
            "step": int(self.step_count),
            "action": cmd,
            "reward": reward,
            "score": float(new_score)
        })

        done = bool(new_score >= 0.999 or self.step_count >= 10)

        return {
            "observation": self._get_observation(feedback),
            "reward": reward,
            "done": done,
            "history": self.history
        }

    def state(self) -> DataState:
        return DataState(
            data=self.df.to_dict(orient='records'),
            integrity_score=float(self.calculate_integrity()),
            current_task_index=0,
            steps_taken=self.step_count
        )
