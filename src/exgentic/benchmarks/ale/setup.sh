#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Setup script for the Agents' Last Exam (ALE) benchmark adapter.
#
# ALE tasks require a local clone of the benchmark repository.
# This script clones it if ALE_REPO_PATH is not already set.

set -euo pipefail

ALE_REPO_URL="https://github.com/rdi-berkeley/agents-last-exam.git"
DEFAULT_CLONE_DIR="${HOME}/.exgentic/benchmarks/ale/agents-last-exam"

if [ -n "${ALE_REPO_PATH:-}" ] && [ -d "${ALE_REPO_PATH}/tasks" ]; then
    echo "ALE repo already available at ${ALE_REPO_PATH}"
    exit 0
fi

if [ -d "${DEFAULT_CLONE_DIR}/tasks" ]; then
    echo "ALE repo found at ${DEFAULT_CLONE_DIR}"
    echo "Set ALE_REPO_PATH=${DEFAULT_CLONE_DIR} in your experiment config."
    exit 0
fi

echo "Cloning ALE benchmark repository..."
mkdir -p "$(dirname "${DEFAULT_CLONE_DIR}")"
git clone --depth 1 "${ALE_REPO_URL}" "${DEFAULT_CLONE_DIR}"
echo "ALE repo cloned to ${DEFAULT_CLONE_DIR}"
echo "Set ale_repo_path in your benchmark config to: ${DEFAULT_CLONE_DIR}"
