#!/usr/bin/env python3
"""
Shroud Reconstruction Task Orchestrator
========================================
Reads tasks from tasks.json, sends each to MiniMax M2.7 for code generation,
executes the output in a sandboxed subprocess, validates results, and
auto-commits passing work.

Usage:
    python shroud_runner.py                  # Run all pending tasks
    python shroud_runner.py --dry-run        # Show what would run without executing
    python shroud_runner.py --task 0         # Run only task at index 0
    python shroud_runner.py --retry-failed   # Re-run tasks marked "failed"

Task queue format (tasks.json):
[
  {
    "task": "Description of what to do",
    "type": "compute|write|publish",
    "status": "pending|running|done|failed",
    "output_files": [],
    "error": null
  }
]
"""

import json
import os
import sys
import subprocess
import time
import hashlib
import argparse
import re
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
TASKS_FILE = PROJECT_ROOT / "tasks.json"
MARATHON_LOG = PROJECT_ROOT / "marathon_log.md"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"
SRC_DIR = PROJECT_ROOT / "src"
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"

load_dotenv(PROJECT_ROOT / ".env")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_ENDPOINT = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-M2.7"

# System prompt that gives MiniMax context about the project
SYSTEM_PROMPT = """You are a code generation assistant for the Shroud of Turin AI Forensic Reconstruction project.

Project structure:
- src/ — Python analysis scripts
- docs/ — GitHub Pages site (HTML pages)
- docs/images/ — visualization outputs shown on the site
- docs/css/style.css — shared stylesheet
- output/ — raw script outputs (analysis/, 3d_print/, full_body/, etc.)
- data/processed/ — depth maps (.npy files)
- data/source/ — original photographs

Key data files:
- data/processed/depth_map_smooth_15.npy — Enrie 1931 depth map (load with numpy, resize to 150x150)
- data/source/shroud_full_negatives.jpg — full body image
- venv/ — Python 3.11 virtual environment with numpy, opencv, matplotlib, scipy, pywt, torch, etc.

Rules:
- Use matplotlib Agg backend (import matplotlib; matplotlib.use('Agg'))
- Dark figure backgrounds (#1a1a1a), gold accent (#c4a35a), white text
- Save figures to both output/analysis/ and docs/images/
- When writing HTML, use <link rel="stylesheet" href="css/style.css">
- Navigation must include all 14 pages (see nav template below)
- Print all numeric findings to stdout so they can be captured

When asked to write Python: output ONLY the complete Python script, no markdown fences.
When asked to write HTML: output ONLY the complete HTML file, no markdown fences.
When asked for any other file: output ONLY the file content.

Always include error handling and print clear status messages."""

# The canonical nav block for all HTML pages
NAV_TEMPLATE = """<nav>
  <div class="nav-inner">
    <a class="nav-brand" href="index.html">Shroud Reconstruction</a>
    <ul class="nav-links">
      <li><a href="index.html">Home</a></li>
      <li><a href="methodology.html">Methodology</a></li>
      <li><a href="findings.html">Findings</a></li>
      <li><a href="reconstruction.html">Reconstruction</a></li>
      <li><a href="study2.html">Study 2</a></li>
      <li><a href="neural-depth.html">Neural Depth</a></li>
      <li><a href="full-body.html">Full Body</a></li>
      <li><a href="formation-analysis.html">Formation</a></li>
      <li><a href="distance-function.html">Distance</a></li>
      <li><a href="scourge-analysis.html">Scourge</a></li>
      <li><a href="neave-comparison.html">Neave</a></li>
      <li><a href="viewer.html">3D Viewer</a></li>
      <li><a href="summary.html">Summary</a></li>
      <li><a href="about.html">About</a></li>
    </ul>
  </div>
</nav>"""


# ---------------------------------------------------------------------------
# MiniMax API
# ---------------------------------------------------------------------------
def call_minimax(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 8192) -> str:
    """Send a prompt to MiniMax M2.7 and return the response text."""
    if not MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY not set. Check .env file.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        MINIMAX_ENDPOINT,
        headers={
            "Authorization": f"Bearer {MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MINIMAX_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    # Check for API-level errors
    base_resp = data.get("base_resp", {})
    if base_resp.get("status_code", 0) != 0:
        raise RuntimeError(
            f"MiniMax API error {base_resp.get('status_code')}: "
            f"{base_resp.get('status_msg', 'unknown')}"
        )

    # Extract the assistant's reply
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"MiniMax returned no choices: {json.dumps(data)[:300]}")
    return choices[0]["message"]["content"]


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------
def extract_code(text: str, lang: str = "python") -> str:
    """Extract code from MiniMax response. Strips markdown fences if present."""
    # Try to find fenced code block
    pattern = rf"```{lang}?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fences, assume the whole response is code
    return text.strip()


def validate_output_files(files: list[str]) -> tuple[bool, list[str]]:
    """Check that output files exist and aren't empty. Returns (all_ok, errors)."""
    errors = []
    for f in files:
        p = Path(f) if os.path.isabs(f) else PROJECT_ROOT / f
        if not p.exists():
            errors.append(f"Missing: {f}")
        elif p.stat().st_size == 0:
            errors.append(f"Empty: {f}")
        elif p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            # Basic image validation: check file header
            with open(p, "rb") as fh:
                header = fh.read(8)
            if p.suffix.lower() == ".png" and header[:4] != b"\x89PNG":
                errors.append(f"Invalid PNG: {f}")
            elif p.suffix.lower() in (".jpg", ".jpeg") and header[:2] != b"\xff\xd8":
                errors.append(f"Invalid JPEG: {f}")
    return len(errors) == 0, errors


def run_python_script(script_path: str, timeout: int = 300) -> tuple[int, str, str]:
    """Execute a Python script in the project venv. Returns (returncode, stdout, stderr)."""
    result = subprocess.run(
        [str(VENV_PYTHON), str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return result.returncode, result.stdout, result.stderr


def git_commit_and_push(message: str, files: list[str]) -> bool:
    """Stage specific files, commit, and push."""
    try:
        # Stage files
        for f in files:
            subprocess.run(
                ["git", "add", str(f)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                check=True,
            )
        # Commit
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=True,
        )
        # Push
        subprocess.run(
            ["git", "push"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        log(f"Git error: {e.stderr if hasattr(e, 'stderr') else e}")
        return False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(message: str):
    """Print and append to marathon log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    with open(MARATHON_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_task_result(task: dict, success: bool, stdout: str = "", stderr: str = ""):
    """Log a detailed task result to the marathon log."""
    with open(MARATHON_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n### Task: {task['task'][:80]}\n")
        f.write(f"- **Type:** {task.get('type', 'unknown')}\n")
        f.write(f"- **Status:** {'PASS' if success else 'FAIL'}\n")
        f.write(f"- **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if task.get("output_files"):
            f.write(f"- **Output:** {', '.join(task['output_files'])}\n")
        if stdout.strip():
            # Truncate long output
            short = stdout.strip()[:500]
            f.write(f"- **Stdout:**\n```\n{short}\n```\n")
        if stderr.strip() and not success:
            short = stderr.strip()[:300]
            f.write(f"- **Stderr:**\n```\n{short}\n```\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# Task handlers by type
# ---------------------------------------------------------------------------
def handle_compute_task(task: dict) -> bool:
    """Generate a Python script via MiniMax, execute it, validate outputs."""
    prompt = (
        f"Write a complete Python script for the following task.\n\n"
        f"TASK: {task['task']}\n\n"
        f"The script will be saved to src/ and run with the project venv.\n"
        f"Print all results to stdout. Save figures to both output/analysis/ "
        f"and docs/images/.\n"
        f"Use matplotlib.use('Agg'). Dark backgrounds (#1a1a1a)."
    )

    log(f"Sending compute task to MiniMax: {task['task'][:60]}...")
    response = call_minimax(prompt)
    code = extract_code(response)

    # Determine script filename from task description
    slug = re.sub(r"[^a-z0-9]+", "_", task["task"][:40].lower()).strip("_")
    script_path = SRC_DIR / f"mm_{slug}.py"
    script_path.write_text(code, encoding="utf-8")
    log(f"Script written to {script_path.name}")

    # Execute
    log("Executing script...")
    returncode, stdout, stderr = run_python_script(script_path)
    log(f"Exit code: {returncode}")
    if stdout.strip():
        log(f"Output: {stdout.strip()[:200]}")

    success = returncode == 0
    if success and task.get("output_files"):
        ok, errors = validate_output_files(task["output_files"])
        if not ok:
            log(f"Validation failed: {errors}")
            success = False

    task["status"] = "done" if success else "failed"
    if not success:
        task["error"] = (stderr or stdout)[:500]

    log_task_result(task, success, stdout, stderr)

    # Auto-commit if passed
    if success:
        files_to_commit = [str(script_path)]
        if task.get("output_files"):
            files_to_commit.extend(task["output_files"])
        git_commit_and_push(
            f"[auto] {task['task'][:60]}",
            files_to_commit,
        )

    return success


def handle_write_task(task: dict) -> bool:
    """Generate a file (HTML, markdown, etc.) via MiniMax and save it."""
    # Determine output path from task description
    # Look for explicit file paths in the task
    path_match = re.search(r"(docs/\S+\.html|docs/\S+\.md|src/\S+\.py)", task["task"])
    if path_match:
        output_path = PROJECT_ROOT / path_match.group(1)
    else:
        slug = re.sub(r"[^a-z0-9]+", "_", task["task"][:40].lower()).strip("_")
        output_path = DOCS_DIR / f"mm_{slug}.html"

    prompt = (
        f"Write the complete file content for the following task.\n\n"
        f"TASK: {task['task']}\n\n"
        f"Output path: {output_path.relative_to(PROJECT_ROOT)}\n\n"
        f"NAVIGATION TEMPLATE (use this exact nav in every HTML page):\n"
        f"{NAV_TEMPLATE}\n\n"
        f"Mark the appropriate page as class=\"active\" in the nav."
    )

    log(f"Sending write task to MiniMax: {task['task'][:60]}...")
    response = call_minimax(prompt, max_tokens=16384)

    # Strip any markdown fences
    content = response.strip()
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n", "", content)
        content = re.sub(r"\n```$", "", content)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    log(f"File written to {output_path.relative_to(PROJECT_ROOT)}")

    task["status"] = "done"
    task["output_files"] = [str(output_path.relative_to(PROJECT_ROOT))]

    log_task_result(task, True)

    # Auto-commit
    git_commit_and_push(
        f"[auto] {task['task'][:60]}",
        [str(output_path)],
    )
    return True


def handle_publish_task(task: dict) -> bool:
    """Commit all pending changes and push."""
    log("Running publish task: staging all docs/ changes...")
    try:
        subprocess.run(
            ["git", "add", "docs/"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=True,
        )
        # Check if there's anything to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
        )
        if result.returncode == 0:
            log("Nothing to commit.")
            task["status"] = "done"
            return True

        subprocess.run(
            ["git", "commit", "-m", f"[auto] {task['task'][:60]}"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "push"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=True,
        )
        task["status"] = "done"
        log("Published successfully.")
        log_task_result(task, True)
        return True
    except subprocess.CalledProcessError as e:
        task["status"] = "failed"
        task["error"] = str(e)[:300]
        log(f"Publish failed: {e}")
        log_task_result(task, False, stderr=str(e))
        return False


HANDLERS = {
    "compute": handle_compute_task,
    "write": handle_write_task,
    "publish": handle_publish_task,
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def load_tasks() -> list[dict]:
    """Load task queue from JSON file."""
    if not TASKS_FILE.exists():
        log(f"No {TASKS_FILE.name} found. Creating empty queue.")
        TASKS_FILE.write_text("[]", encoding="utf-8")
        return []
    return json.loads(TASKS_FILE.read_text(encoding="utf-8"))


def save_tasks(tasks: list[dict]):
    """Save task queue back to JSON."""
    TASKS_FILE.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_tasks(tasks: list[dict], dry_run: bool = False, task_index: int = None):
    """Execute pending tasks from the queue."""
    if task_index is not None:
        # Run a single task
        if task_index >= len(tasks):
            log(f"Task index {task_index} out of range (have {len(tasks)} tasks)")
            return
        indices = [task_index]
    else:
        indices = range(len(tasks))

    total = len(tasks)
    done = 0
    failed = 0

    for i in indices:
        task = tasks[i]
        if task.get("status") in ("done", "running"):
            continue

        task_type = task.get("type", "compute")
        handler = HANDLERS.get(task_type)
        if not handler:
            log(f"Unknown task type: {task_type}")
            task["status"] = "failed"
            task["error"] = f"Unknown type: {task_type}"
            continue

        log(f"\n{'='*60}")
        log(f"Task {i+1}/{total}: [{task_type}] {task['task'][:70]}")
        log(f"{'='*60}")

        if dry_run:
            log("(dry run - skipping execution)")
            continue

        task["status"] = "running"
        save_tasks(tasks)

        try:
            success = handler(task)
            if success:
                done += 1
            else:
                failed += 1
        except Exception as e:
            log(f"Exception: {e}")
            task["status"] = "failed"
            task["error"] = str(e)[:500]
            failed += 1

        save_tasks(tasks)

    log(f"\nComplete. Done: {done}, Failed: {failed}, "
        f"Remaining: {sum(1 for t in tasks if t.get('status') == 'pending')}")


def main():
    parser = argparse.ArgumentParser(description="Shroud Reconstruction Task Runner")
    parser.add_argument("--dry-run", action="store_true", help="Show tasks without executing")
    parser.add_argument("--task", type=int, default=None, help="Run a specific task by index")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run failed tasks")
    args = parser.parse_args()

    log("\n" + "=" * 60)
    log("SHROUD RUNNER — MiniMax M2.7 Orchestrator")
    log("=" * 60)

    tasks = load_tasks()
    if not tasks:
        log("Task queue is empty. Add tasks to tasks.json and re-run.")
        return

    # Reset failed tasks if retrying
    if args.retry_failed:
        for t in tasks:
            if t.get("status") == "failed":
                t["status"] = "pending"
                t.pop("error", None)
        save_tasks(tasks)

    log(f"Loaded {len(tasks)} tasks. "
        f"Pending: {sum(1 for t in tasks if t.get('status', 'pending') == 'pending')}, "
        f"Done: {sum(1 for t in tasks if t.get('status') == 'done')}, "
        f"Failed: {sum(1 for t in tasks if t.get('status') == 'failed')}")

    run_tasks(tasks, dry_run=args.dry_run, task_index=args.task)


if __name__ == "__main__":
    main()
