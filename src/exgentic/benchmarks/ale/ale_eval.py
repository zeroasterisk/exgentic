# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""ALE evaluator, session, and scoring classes.

These classes import ``ale_run`` at method level so the heavy ALE dependencies
are only loaded inside the runner subprocess, never in the host process.

The adapter bridges ALE's VM-based task execution into Exgentic's
Session/Evaluator contract:

- **Task discovery**: reads ``task_card.json`` files from the ALE repo tree
  or from a ``selected_tasks/*.txt`` file list.
- **Session**: presents the ALE task prompt to the agent with ``bash`` and
  ``finish`` actions.  The agent operates inside the task sandbox via shell
  commands; when it calls ``finish``, scoring is triggered.
- **Scoring**: invokes the per-task ``scripts/score_outputs.py`` (or
  ``scripts/verify_outputs.py``) and normalizes the result to a 0-1 score.

Source benchmark: https://github.com/rdi-berkeley/agents-last-exam
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from ...core.actions import ActionsHandler, extract_argument
from ...core.evaluator import Evaluator
from ...core.session import Session
from ...core.types import (
    Action,
    ActionType,
    BenchmarkResults,
    EmptyObservation,
    Observation,
    SessionIndex,
    SessionScore,
    SingleAction,
)
from .ale_benchmark import BashAction, FinishTaskAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task card helpers (no ALE import needed — task_card.json is plain JSON)
# ---------------------------------------------------------------------------

_TASK_SUBSETS: dict[str, str] = {
    "hello": "hello_both.txt",
    "cpu_unlicensed": "cpu_unlicensed.txt",
    "unlicensed": "unlicensed.txt",
    "full": "full.txt",
}


def _load_task_card(ale_repo: Path, task_path: str) -> dict[str, Any]:
    card_file = ale_repo / "tasks" / task_path / "task_card.json"
    if not card_file.exists():
        raise FileNotFoundError(f"task_card.json not found at {card_file}")
    return json.loads(card_file.read_text(encoding="utf-8"))


def _discover_tasks_from_list(ale_repo: Path, list_file: str) -> list[str]:
    path = ale_repo / list_file
    if not path.exists():
        path = ale_repo / "selected_tasks" / list_file
    if not path.exists():
        raise FileNotFoundError(
            f"Task list file not found: {list_file} "
            f"(tried {ale_repo / list_file} and {ale_repo / 'selected_tasks' / list_file})"
        )
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    return tasks


def _discover_tasks_from_repo(ale_repo: Path, subset: str) -> list[str]:
    list_file = _TASK_SUBSETS.get(subset)
    if list_file:
        return _discover_tasks_from_list(ale_repo, list_file)
    tasks_dir = ale_repo / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"tasks/ directory not found in {ale_repo}")
    tasks = []
    for card in sorted(tasks_dir.rglob("task_card.json")):
        task_path = card.parent.relative_to(tasks_dir)
        tasks.append(str(task_path))
    return tasks


def _find_scoring_script(ale_repo: Path, task_path: str) -> Path | None:
    scripts_dir = ale_repo / "tasks" / task_path / "scripts"
    if not scripts_dir.is_dir():
        return None
    for name in ("score_outputs.py", "verify_outputs.py", "evaluate.py"):
        script = scripts_dir / name
        if script.exists():
            return script
    return None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class ALESession(Session):
    """Session for a single ALE task.

    The session presents the task prompt from ``task_card.json`` to the
    agent.  The agent interacts via ``bash`` (shell execution inside the
    task's working directory) and ``finish`` (signal completion).

    In the current adapter, ``bash`` commands run on the host in a
    subprocess, not inside a remote VM.  This works for Docker-based and
    local task execution.  For full GCP/QEMU VM execution, the ALE
    orchestrator (``python -m ale_run run``) should be used directly;
    this adapter focuses on making ALE tasks discoverable and scorable
    through Exgentic's evaluation pipeline.
    """

    def __init__(
        self,
        task_id: str,
        subset: str,
        ale_repo_path: str = "",
        session_id: str | None = None,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id

        self._task_id = task_id
        self._subset = subset
        self._ale_repo = Path(ale_repo_path) if ale_repo_path else None
        self._done = False
        self._step_count = 0
        self._final_summary: str | None = None

        self._task_card: dict[str, Any] = {}
        self._work_dir: Path | None = None

        self._registry = ActionsHandler(
            logger=self.logger,
            warn_on_validation_error=False,
            warn_on_unknown_action=True,
        )
        self._registry.add_action(
            name="bash",
            description="Execute a shell command in the task working directory.",
            action_cls=BashAction,
            handler=self._handle_bash,
        )
        self._registry.add_action(
            name="finish",
            description=(
                "Signal that the task is complete. Provide a brief summary "
                "of what was done and where outputs were written."
            ),
            action_cls=FinishTaskAction,
            handler=self._handle_finish,
            is_finish=True,
        )

        if self._ale_repo and (self._ale_repo / "tasks" / task_id / "task_card.json").exists():
            self._task_card = _load_task_card(self._ale_repo, task_id)

        super().__init__()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def task(self) -> str:
        prompt = self._task_card.get("taskPrompt", "")
        if not prompt:
            prompt = (
                f"Complete the ALE task: {self._task_id}\n\n"
                f"Title: {self._task_card.get('title', 'Unknown')}\n"
                f"Summary: {self._task_card.get('summary', 'No summary available.')}\n"
            )
        return prompt

    @property
    def context(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        if self._task_card.get("agentMustDo"):
            ctx["requirements"] = self._task_card["agentMustDo"]
        if self._task_card.get("software"):
            ctx["software"] = self._task_card["software"]
        if self._task_card.get("evaluation"):
            ctx["evaluation_criteria"] = self._task_card["evaluation"]
        return ctx

    @property
    def actions(self) -> list[ActionType]:
        return self._registry.actions

    def start(self) -> Observation | None:
        return EmptyObservation()

    def step(self, action: Action) -> Observation | None:
        if self._done:
            return None
        self._step_count += 1
        return self._registry.execute(action)

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        if self._ale_repo is None:
            self.logger.warning("No ALE repo path; scoring as 0.")
            sc = SessionScore(score=0.0, success=False, is_finished=self._done)
            self.save_standard_results(sc)
            return sc

        scoring_script = _find_scoring_script(self._ale_repo, self._task_id)
        if scoring_script is None:
            self.logger.warning(
                "No scoring script found for %s; scoring as 0.",
                self._task_id,
            )
            sc = SessionScore(score=0.0, success=False, is_finished=self._done)
            self.save_standard_results(sc)
            return sc

        task_dir = self._ale_repo / "tasks" / self._task_id
        try:
            result = subprocess.run(
                ["python3", str(scoring_script)],
                cwd=str(task_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.logger.info(
                "Scoring script exit=%d stdout=%s stderr=%s",
                result.returncode,
                result.stdout[:500],
                result.stderr[:500],
            )
            raw_score = self._parse_score(result.stdout)
        except subprocess.TimeoutExpired:
            self.logger.error("Scoring script timed out for %s", self._task_id)
            raw_score = 0.0
        except Exception as e:
            self.logger.exception("Scoring failed for %s: %s", self._task_id, e)
            raw_score = 0.0

        score = max(0.0, min(1.0, raw_score))
        sc = SessionScore(
            score=score,
            success=score > 0.0,
            is_finished=self._done,
        )
        self.save_standard_results(sc)
        return sc

    def close(self):
        super().close()

    def _handle_bash(self, action: SingleAction) -> Any:
        command = extract_argument(action.arguments, "command", "")
        self.logger.info("BASH | step=%d | %s", self._step_count, command[:200])
        cwd = str(self._work_dir) if self._work_dir else None
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=cwd,
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            if len(output) > 50000:
                output = output[:25000] + "\n\n[...truncated...]\n\n" + output[-25000:]
            return {"output": output, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": "Command timed out after 300 seconds.", "returncode": 124}
        except Exception as e:
            return {"output": f"Error: {e}", "returncode": 1}

    def _handle_finish(self, action: SingleAction) -> None:
        self._final_summary = extract_argument(action.arguments, "summary", "")
        self._done = True

    @staticmethod
    def _parse_score(stdout: str) -> float:
        stdout = stdout.strip()
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    for key in ("score", "normalized_score", "reward"):
                        if key in data:
                            return float(data[key])
                if isinstance(data, (int, float)):
                    return float(data)
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                return float(line)
            except ValueError:
                continue
        return 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ALEEvaluator(Evaluator):
    """Task discovery and result aggregation for ALE."""

    def __init__(
        self,
        subset: str = "cpu_unlicensed",
        ale_repo_path: str = "",
        task_list_file: str | None = None,
    ) -> None:
        self._subset = subset
        self._ale_repo = Path(ale_repo_path) if ale_repo_path else None
        self._task_list_file = task_list_file

    def list_tasks(self) -> list[str]:
        if self._ale_repo is None:
            raise RuntimeError(
                "ale_repo_path is required for task discovery. "
                "Clone rdi-berkeley/agents-last-exam and set ale_repo_path."
            )
        if self._task_list_file:
            return _discover_tasks_from_list(self._ale_repo, self._task_list_file)
        return _discover_tasks_from_repo(self._ale_repo, self._subset)

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        scores: list[float] = []
        for paths in self.get_sessions_paths(sessions):
            fp = paths.benchmark_results
            if not fp.exists():
                raise FileNotFoundError(f"Missing benchmark result for session '{paths.session_id}' at {fp}")
            with open(fp, encoding="utf-8") as f:
                payload = json.load(f)
            scores.append(float(payload.get("score", 0.0)))

        avg = sum(scores) / len(scores) if scores else 0.0
        return BenchmarkResults(
            benchmark_name=f"ale-{self._subset}",
            total_tasks=len(sessions),
            score=avg,
            metrics={
                "avg_score": avg,
                "total_tasks": len(sessions),
                "tasks_scored": len(scores),
                "tasks_passing": sum(1 for s in scores if s > 0.0),
            },
        )
