"""Example: Run ALE benchmark tasks through Exgentic.

Prerequisites:
  1. Clone ALE: git clone https://github.com/rdi-berkeley/agents-last-exam.git
  2. Set the path below to your local clone.

Usage:
  uv run python examples/run_ale.py
"""

from exgentic.benchmarks.ale.ale_benchmark import ALEBenchmark

# Point to your local ALE repo clone
ALE_REPO = "path/to/agents-last-exam"

benchmark = ALEBenchmark(
    subset="hello",
    ale_repo_path=ALE_REPO,
)

evaluator = benchmark.get_evaluator()
tasks = evaluator.list_tasks()
print(f"Discovered {len(tasks)} tasks: {tasks}")

for task_id in tasks[:1]:
    session = benchmark.get_session(task_id)
    print(f"\nTask: {task_id}")
    print(f"Prompt: {session.task[:200]}...")
    print(f"Actions: {[a.name for a in session.actions]}")
