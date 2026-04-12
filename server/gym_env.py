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


# ---------------------------------------------------------------------------
# Built-in task datasets — each task starts with intentionally dirty data
# ---------------------------------------------------------------------------

def _make_easy_df() -> pd.DataFrame:
    """Easy: dataset with ~30% duplicate rows, no nulls."""
    base = pd.DataFrame({
        "id":    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name":  ["Alice", "Bob", "Carol", "Dave", "Eve",
                  "Frank", "Grace", "Hank", "Ivy", "Jack"],
        "score": [82, 91, 74, 65, 88, 77, 95, 61, 83, 70],
        "grade": ["B", "A", "C", "D", "B", "C", "A", "D", "B", "C"],
    })
    # Add ~30% duplicates
    dups = base.iloc[[0, 1, 2]].copy()
    return pd.concat([base, dups], ignore_index=True)


def _make_medium_df() -> pd.DataFrame:
    """Medium: dataset with ~35% null values, no duplicates."""
    rng = np.random.default_rng(42)
    n = 20
    ages   = rng.integers(20, 60, n).astype(float)
    salaries = rng.integers(30000, 90000, n).astype(float)
    scores = rng.uniform(0.4, 0.9, n)

    # Inject nulls at fixed indices
    null_idx_ages    = [2, 5, 9, 14, 17]
    null_idx_salary  = [1, 4, 7, 11, 16, 19]
    null_idx_scores  = [0, 6, 12]

    for i in null_idx_ages:
        ages[i] = np.nan
    for i in null_idx_salary:
        salaries[i] = np.nan
    for i in null_idx_scores:
        scores[i] = np.nan

    return pd.DataFrame({
        "emp_id":  range(1, n + 1),
        "age":     ages,
        "salary":  salaries,
        "perf":    scores,
        "dept":    (["HR", "Eng", "Sales", "Fin"] * 5),
    })


def _make_hard_df() -> pd.DataFrame:
    """Hard: dataset with both duplicates AND null values."""
    base = pd.DataFrame({
        "product": ["Widget", "Gadget", "Doohickey", "Thingamajig", "Whatsit",
                    "Gizmo",  "Doodad",  "Contraption", "Knick-knack", "Tchotchke"],
        "price":   [9.99, 24.99, 4.99, 14.99, 7.49,
                    19.99, 3.49, 29.99, 11.99, 5.99],
        "stock":   [100, 200, 150, 80, 300, 120, 250, 60, 175, 220],
        "rating":  [4.2, 3.8, 4.5, 3.1, 4.7, 3.9, 4.1, 3.5, 4.3, 3.6],
    })
    # Add duplicates
    dups = base.iloc[[0, 2, 5]].copy()
    df = pd.concat([base, dups], ignore_index=True)
    # Inject nulls
    df.loc[1,  "price"]  = np.nan
    df.loc[4,  "stock"]  = np.nan
    df.loc[7,  "rating"] = np.nan
    df.loc[10, "price"]  = np.nan
    df.loc[12, "stock"]  = np.nan
    return df


TASK_FACTORIES = {
    "easy":   _make_easy_df,
    "medium": _make_medium_df,
    "hard":   _make_hard_df,
}

TASK_GOALS = {
    "easy":   "Remove all duplicate rows from the dataset.",
    "medium": "Fill missing numerical values using column medians.",
    "hard":   "Remove duplicates AND fill missing values to fully clean the dataset.",
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DataCleaningEnv:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df: pd.DataFrame = df if df is not None else pd.DataFrame()
        self.initial_df: Optional[pd.DataFrame] = self.df.copy() if not self.df.empty else None
        self.current_task: str = "user_input"
        self.step_count: int = 0
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_user_data(self, csv_content: str) -> DataObservation:
        self.df = pd.read_csv(io.StringIO(csv_content))
        self.initial_df = self.df.copy()
        self.current_task = "user_input"
        self.step_count = 0
        self.history = []
        return self._get_observation("Dataset loaded successfully.")

    def calculate_integrity(self) -> float:
        """
        Returns a data-quality score strictly in the open interval (0, 1).
        Score = 1 - (null_fraction + duplicate_fraction), clamped to (0.01, 0.99).
        """
        if self.df is None or self.df.empty:
            return 0.35  # non-zero, non-one sentinel for empty state

        total_elements = float(self.df.size) or 1.0
        total_rows     = float(len(self.df)) or 1.0

        null_fraction = self.df.isnull().sum().sum() / total_elements
        dup_fraction  = self.df.duplicated().sum()  / total_rows

        raw = 1.0 - (null_fraction + dup_fraction)

        # Strictly exclude boundary values 0.0 and 1.0
        return float(max(0.01, min(0.99, raw)))

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy") -> DataObservation:
        """Load a built-in task dataset (or restore user data) and reset counters."""
        self.step_count = 0
        self.history = []
        self.current_task = task_id

        if task_id in TASK_FACTORIES:
            self.df = TASK_FACTORIES[task_id]()
            self.initial_df = self.df.copy()
            goal = TASK_GOALS[task_id]
        elif task_id in ("user_current", "user_input") and self.initial_df is not None:
            self.df = self.initial_df.copy()
            goal = "Clean the user-provided dataset."
        else:
            # Fallback: use easy task
            self.df = _make_easy_df()
            self.initial_df = self.df.copy()
            self.current_task = "easy"
            goal = TASK_GOALS["easy"]

        return self._get_observation(f"Task '{self.current_task}' started. {goal}")

    def step(self, action_input: Any) -> Dict[str, Any]:
        """Apply a cleaning action and return the transition tuple."""
        self.step_count += 1
        old_score = self.calculate_integrity()

        # Parse command
        if isinstance(action_input, dict):
            cmd = str(action_input.get("command", "")).strip().lower()
            col = action_input.get("target_column", None)
        else:
            cmd = str(getattr(action_input, "command", action_input)).strip().lower()
            col = getattr(action_input, "target_column", None)

        action_taken = cmd or "none"

        if not self.df.empty:
            if "duplicate" in cmd:
                self.df = self.df.drop_duplicates().reset_index(drop=True)

            elif "median" in cmd or "fill" in cmd:
                if col and col in self.df.columns:
                    med = self.df[col].median()
                    self.df[col] = self.df[col].fillna(float(med))
                else:
                    # Fill all numeric columns
                    num_cols = self.df.select_dtypes(include=[np.number]).columns
                    for c in num_cols:
                        med = self.df[c].median()
                        if not np.isnan(med):
                            self.df[c] = self.df[c].fillna(float(med))

            elif "null" in cmd or "drop_null" in cmd or "dropna" in cmd:
                self.df = self.df.dropna().reset_index(drop=True)

        new_score = self.calculate_integrity()

        # Reward: improvement in score (percentage points) minus small step cost
        improvement = (new_score - old_score) * 100.0
        reward = float(round(improvement - 1.0, 4))

        entry: Dict[str, Any] = {
            "step":   int(self.step_count),
            "action": action_taken,
            "reward": reward,
            "score":  float(new_score),   # always in (0.01, 0.99)
        }
        self.history.append(entry)

        # Task is done when significantly clean or max steps reached
        done = bool(new_score >= 0.93 or self.step_count >= 15)

        return {
            "observation": self._get_observation("Action processed."),
            "reward":      reward,
            "done":        done,
            "score":       float(new_score),   # top-level score for grader
            "history":     self.history,
        }

    def state(self) -> DataState:
        return DataState(
            data=self.df.to_dict(orient="records") if not self.df.empty else [],
            integrity_score=float(self.calculate_integrity()),
            current_task_index=list(TASK_FACTORIES.keys()).index(self.current_task)
                               if self.current_task in TASK_FACTORIES else 0,
            steps_taken=int(self.step_count),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_observation(self, goal_text: str) -> DataObservation:
        null_count = int(self.df.isnull().sum().sum()) if not self.df.empty else 0
        dup_count  = int(self.df.duplicated().sum())   if not self.df.empty else 0
        status     = f"{goal_text} | nulls={null_count}, duplicates={dup_count}"

        summary_dict: Dict[str, Any] = {}
        if not self.df.empty:
            try:
                summary_dict = self.df.describe(include="all").fillna(0).to_dict()
            except Exception:
                summary_dict = {}

        health = self.calculate_integrity()  # always in (0.01, 0.99)

        return DataObservation(
            summary=summary_dict,
            sample_rows=self.df.head(5).to_dict(orient="records") if not self.df.empty else [],
            column_names=[str(c) for c in self.df.columns],
            health_score=float(health),
            goal=status,
        )