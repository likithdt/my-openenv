"""
Inference script for my-env (Data Integrity Lab).

Mandatory env vars:
    API_BASE_URL   LLM API endpoint
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace / API key
    BASE_URL       Running env URL (HF Space)

stdout format (required by validator):
    [START] task=<task> env=<env> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

score is the health_score of the final observation, always strictly in (0, 1).
"""

import os
import requests
from typing import List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

_raw_base    = os.getenv("BASE_URL", "https://likithdt-my-env.hf.space")
BASE_URL     = _raw_base.strip("[]() ").rstrip("/")

ENV_NAME     = "my-env"
MAX_STEPS    = 8

# ---------------------------------------------------------------------------
# Required log helpers — exact format mandated by the spec
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    """
    IMPORTANT: 'score' must appear in the [END] line and be strictly in (0, 1).
    We use health_score from the final environment observation, which is
    always clamped to [0.01, 0.99] by the environment.
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

def get_llm_action(client: OpenAI, model_name: str, obs: dict) -> str:
    """Ask the LLM which cleaning action to take given the current observation."""
    health = obs.get("health_score", 0.35)
    goal   = obs.get("goal", "Clean the dataset.")

    prompt = (
        f"You are a Data Quality Agent.\n"
        f"Current data quality score: {health:.3f} (higher is better, max ~0.99).\n"
        f"Goal: {goal}\n\n"
        f"Choose exactly ONE action from the list below that will most improve "
        f"data quality, and respond with ONLY that single word:\n"
        f"  drop_duplicates  — removes duplicate rows\n"
        f"  fill_median      — fills missing numeric values with column medians\n"
        f"  drop_nulls       — removes rows that have any missing values\n"
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
            stream=False,
        )
        reply = (completion.choices[0].message.content or "").strip().lower()
        for valid in ["drop_duplicates", "fill_median", "drop_nulls"]:
            if valid in reply:
                return valid
        # Fallback: pick based on score heuristic
        return "drop_duplicates" if health < 0.6 else "fill_median"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "drop_duplicates"


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def _safe_score(obs: dict, fallback: float = 0.35) -> float:
    """
    Extract health_score from an observation dict.
    Result is always strictly within (0.01, 0.99) to satisfy the validator.
    """
    raw = obs.get("health_score", fallback)
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        raw = fallback
    return max(0.01, min(0.99, raw))


def run_benchmark() -> None:
    if not API_KEY:
        print("[DEBUG] CRITICAL: No API_KEY or HF_TOKEN found.", flush=True)
        # Still emit [END] lines so the validator can parse something
        for task_id in ["easy", "medium", "hard"]:
            log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.35, rewards=[])
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks  = ["easy", "medium", "hard"]

    for task_id in tasks:
        log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

        rewards:     List[float] = []
        steps_taken: int         = 0
        # Default score — non-boundary sentinel, safe even if env call fails
        score:       float       = 0.35
        success:     bool        = False
        obs:         dict        = {}

        try:
            # ── Reset ──────────────────────────────────────────────────────
            reset_res = requests.post(
                f"{BASE_URL}/reset",
                json={"task_id": task_id},
                timeout=30,
            )
            reset_res.raise_for_status()
            reset_data = reset_res.json()

            # Unwrap nested observation if present
            if isinstance(reset_data, dict):
                obs = reset_data.get("observation", reset_data)
            else:
                obs = {}

            score = _safe_score(obs)  # initial score from reset

            # ── Steps ──────────────────────────────────────────────────────
            for step_n in range(1, MAX_STEPS + 1):
                action_cmd = get_llm_action(client, MODEL_NAME, obs)

                step_res = requests.post(
                    f"{BASE_URL}/step",
                    json={"command": action_cmd},
                    timeout=30,
                )
                step_res.raise_for_status()
                result = step_res.json()

                reward  = float(result.get("reward", 0.0))
                done    = bool(result.get("done", False))
                raw_obs = result.get("observation", {})
                error   = result.get("error", None)

                # Unwrap nested observation
                if isinstance(raw_obs, dict):
                    obs = raw_obs
                else:
                    obs = {}

                rewards.append(reward)
                steps_taken = step_n

                # Score is the current health_score — always strictly in (0,1)
                score = _safe_score(obs, fallback=score)

                log_step(
                    step=step_n,
                    action=action_cmd,
                    reward=reward,
                    done=done,
                    error=str(error) if error else None,
                )

                if done:
                    break

            # Final score: health_score of last observation, strictly in (0.01, 0.99)
            score   = _safe_score(obs, fallback=score)
            success = score > 0.82

        except Exception as exc:
            print(f"[DEBUG] Error on task '{task_id}': {exc}", flush=True)
            # score stays at its last known good value (or 0.35 sentinel)

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


if __name__ == "__main__":
    run_benchmark()