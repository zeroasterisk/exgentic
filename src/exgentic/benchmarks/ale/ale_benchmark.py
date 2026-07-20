# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Agents' Last Exam (ALE) benchmark adapter — light benchmark class only.

ALE (UC Berkeley RDI) evaluates agents on long-horizon, economically valuable
tasks inside sandboxed VMs.  Each task has a ``task_card.json`` that defines the
prompt, input/reference files, VM snapshot, and a per-task scoring script.

Evaluator, session, and scoring helpers live in ``ale_eval.py`` and are loaded
inside the runner subprocess via ``_get_evaluator_class()`` /
``_get_session_class()``.  This file must remain importable without the
``agent-last-exam`` package or its heavy dependencies installed.

Source benchmark: https://github.com/rdi-berkeley/agents-last-exam
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ...core import Benchmark
from ...core.types import FinishAction, SingleAction


class BashArgs(BaseModel):
    command: str = Field(description="Shell command to execute inside the task VM")


class BashAction(SingleAction):
    name: str = "bash"
    arguments: BashArgs


class FinishTaskArgs(BaseModel):
    summary: str = Field(description="Brief summary of what was done and where outputs were written")


class FinishTaskAction(FinishAction):
    name: str = "finish"
    arguments: FinishTaskArgs


class ALEBenchmark(Benchmark, BaseModel):
    """Benchmark configuration for Agents' Last Exam evaluation."""

    display_name: ClassVar[str] = "Agents' Last Exam"
    slug_name: ClassVar[str] = "ale"
    available_subsets: ClassVar[list[str]] = [
        "hello",
        "cpu_unlicensed",
        "unlicensed",
        "full",
    ]
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @classmethod
    def _get_evaluator_class(cls):
        return "exgentic.benchmarks.ale.ale_eval:ALEEvaluator"

    @classmethod
    def _get_session_class(cls):
        return "exgentic.benchmarks.ale.ale_eval:ALESession"

    subset: str = "cpu_unlicensed"
    ale_repo_path: str = Field(
        default="",
        description="Path to a local clone of rdi-berkeley/agents-last-exam. "
        "If empty, the evaluator downloads task cards from HuggingFace.",
    )
    task_list_file: str | None = Field(
        default=None,
        description="Path to a selected_tasks/*.txt file. Overrides subset-based task discovery.",
    )

    def list_subsets(self) -> list[str]:
        return list(self.available_subsets)

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "ale_repo_path": self.ale_repo_path,
            "task_list_file": self.task_list_file,
        }

    def _get_session_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "ale_repo_path": self.ale_repo_path,
        }
