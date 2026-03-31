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
    "type": "compute|write",
    "status": "pending|running|done|failed",
    "output_files": [],
    "results_json": null,        // compute tasks: path to structured JSON output
    "data_source": null,         // write tasks: path to JSON from a compute task
    "error": null
  }
]

Safeguards:
- Compute tasks MUST produce a JSON results file in output/task_results/.
- Write tasks MUST reference a data_source JSON — no JSON, no page generated.
- No auto-commit, no auto-push. Publishing is manual.
- The "publish" task type has been removed.
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
RESULTS_DIR = OUTPUT_DIR / "task_results"
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

COMPUTE TASKS — JSON output is mandatory:
- At the end of every Python script, collect ALL numeric findings into a dict.
- Write the dict as JSON to the path specified in the prompt.
- Include an 'image_files' key listing every image path saved (relative to project root).

WRITE TASKS — data integrity rules:
- You will receive verified JSON data in the prompt. Use ONLY those numbers.
- Do NOT fabricate quotes from researchers, papers, or any other source.
- Do NOT make authenticity claims about the Shroud of Turin.
- Do NOT invent data that is not in the provided JSON.

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
        timeout=300,
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
    """Extract code from MiniMax response.

    Strips <think> blocks, markdown fences, and any prose/reasoning that
    appears before the first real code line.  MiniMax sometimes emits
    reasoning as plain text (no <think> tags), so we can't rely on tag
    matching alone — we scan for the first line that looks like actual
    code and discard everything above it.
    """
    # 1. Strip explicit <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # 2. Try to find fenced code block (```python ... ``` or ``` ... ```)
    pattern = rf"```{lang}?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Find the first line that is unambiguously code, not prose.
    #    Covers Python (import/from/def/class/#!) and HTML (<!DOCTYPE/<html).
    _CODE_START = re.compile(
        r"^("
        r"import\s|from\s|def\s|class\s"
        r"|#!|#!/"
        r"|<!DOCTYPE|<html"
        r"|'\"'\"'\"'|\"\"\"|\'\'\'"  # docstrings
        r"|#\s"                        # Python comments
        r")",
    )
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if _CODE_START.match(line.strip()):
            start = i
            break

    if start is None:
        # Fallback: return everything (caller will see the error at runtime)
        start = 0

    code = "\n".join(lines[start:])

    # 4. Strip any trailing markdown fences
    code = re.sub(r"\n```\s*$", "", code)
    return code.strip()


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
    """Generate a Python script via MiniMax, execute it, validate outputs.

    The generated script MUST write a structured JSON results file to
    output/task_results/<slug>_results.json containing all numeric
    findings.  Write-type tasks consume this JSON downstream — if the
    JSON is missing or malformed, the pipeline stops here.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", task["task"][:40].lower()).strip("_")
    results_json = RESULTS_DIR / f"{slug}_results.json"
    results_json_rel = results_json.relative_to(PROJECT_ROOT)

    prompt = (
        f"CRITICAL: Output ONLY a Python script. No explanations, no numbered lists, "
        f"no prose before or after. Start with imports. The response must be valid "
        f"Python that can be saved to a .py file and executed directly.\n\n"
        f"Write a complete Python script for the following task:\n\n"
        f"{task['task']}\n\n"
        f"Requirements:\n"
        f"- Start with: import matplotlib; matplotlib.use('Agg')\n"
        f"- The script will be saved to src/ and run with the project venv.\n"
        f"- Save figures to both output/analysis/ and docs/images/.\n"
        f"- Use os.makedirs(exist_ok=True) for output directories.\n"
        f"- Dark figure backgrounds (#1a1a1a), gold accent (#c4a35a), white text.\n"
        f"- MANDATORY: At the end of the script, collect ALL numeric findings "
        f"into a Python dict and write it as JSON to:\n"
        f"    {results_json_rel}\n"
        f"  The JSON must be a flat or shallow dict of result_name → value.\n"
        f"  Also include an 'image_files' key listing every image path the "
        f"  script saved (relative to project root).\n"
        f"  Use: json.dump(results, open(r'{results_json}', 'w'), indent=2)\n"
        f"- Also print the JSON to stdout so it is captured in logs.\n"
        f"- NO markdown fences. NO explanations. ONLY Python code."
    )

    log(f"Sending compute task to MiniMax: {task['task'][:60]}...")
    response = call_minimax(prompt)
    code = extract_code(response)

    script_path = SRC_DIR / f"mm_{slug}.py"
    script_path.write_text(code, encoding="utf-8")
    log(f"Script written to {script_path.name}")

    # Ensure results dir exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Execute
    log("Executing script...")
    returncode, stdout, stderr = run_python_script(script_path)
    log(f"Exit code: {returncode}")
    if stdout.strip():
        log(f"Output: {stdout.strip()[:200]}")

    success = returncode == 0

    # Validate output files
    if success and task.get("output_files"):
        ok, errors = validate_output_files(task["output_files"])
        if not ok:
            log(f"Validation failed: {errors}")
            success = False

    # Validate the mandatory JSON results file
    if success:
        if not results_json.exists():
            log(f"FAIL: Compute task did not produce results JSON at {results_json_rel}")
            success = False
        else:
            try:
                results_data = json.loads(results_json.read_text(encoding="utf-8"))
                if not isinstance(results_data, dict):
                    raise ValueError("Top-level value must be a JSON object")
                log(f"Results JSON validated: {len(results_data)} keys")
            except (json.JSONDecodeError, ValueError) as e:
                log(f"FAIL: Results JSON is malformed: {e}")
                success = False

    task["status"] = "done" if success else "failed"
    task["results_json"] = str(results_json_rel) if success else None
    if not success:
        task["error"] = (stderr or stdout)[:500]

    log_task_result(task, success, stdout, stderr)
    return success


def handle_write_task(task: dict) -> bool:
    """Generate a file (HTML, markdown, etc.) via MiniMax, using ONLY
    verified JSON data from a prior compute task.

    The task dict MUST include a ``data_source`` key pointing to a JSON
    file produced by a compute task (e.g. ``output/task_results/wound_mapping_results.json``).
    If the file is missing, empty, or malformed, the write task FAILS
    immediately — MiniMax never sees a prompt and no page is generated.
    This prevents fabricated data from reaching published pages.
    """
    # ------------------------------------------------------------------
    # 1. Validate the data source
    # ------------------------------------------------------------------
    data_source = task.get("data_source")
    if not data_source:
        log("FAIL: Write task has no 'data_source' key. "
            "Every write task must reference a compute-produced JSON file.")
        task["status"] = "failed"
        task["error"] = "Missing data_source field"
        log_task_result(task, False)
        return False

    json_path = Path(data_source) if os.path.isabs(data_source) else PROJECT_ROOT / data_source
    if not json_path.exists():
        log(f"FAIL: Data source not found: {data_source}. "
            "Run the corresponding compute task first.")
        task["status"] = "failed"
        task["error"] = f"data_source not found: {data_source}"
        log_task_result(task, False)
        return False

    try:
        results_data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(results_data, dict):
            raise ValueError("Top-level value must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        log(f"FAIL: Data source is malformed JSON: {e}")
        task["status"] = "failed"
        task["error"] = f"Malformed data_source: {e}"
        log_task_result(task, False)
        return False

    log(f"Data source validated: {data_source} ({len(results_data)} keys)")

    # Verify that image files listed in the results actually exist
    image_files = results_data.get("image_files", [])
    missing_images = []
    for img in image_files:
        img_path = Path(img) if os.path.isabs(img) else PROJECT_ROOT / img
        if not img_path.exists():
            missing_images.append(img)
    if missing_images:
        log(f"WARNING: {len(missing_images)} images referenced in JSON not found: "
            f"{missing_images[:5]}")

    # ------------------------------------------------------------------
    # 2. Determine output path
    # ------------------------------------------------------------------
    path_match = re.search(r"(docs/\S+\.html|docs/\S+\.md|src/\S+\.py)", task["task"])
    if path_match:
        output_path = PROJECT_ROOT / path_match.group(1)
    else:
        slug = re.sub(r"[^a-z0-9]+", "_", task["task"][:40].lower()).strip("_")
        output_path = DOCS_DIR / f"mm_{slug}.html"

    # ------------------------------------------------------------------
    # 3. Send to MiniMax with the verified data inlined
    # ------------------------------------------------------------------
    # Truncate very large JSON to stay within token limits
    data_str = json.dumps(results_data, indent=2)
    if len(data_str) > 6000:
        data_str = data_str[:6000] + "\n... (truncated)"

    prompt = (
        f"Write the complete file content for the following task.\n\n"
        f"TASK: {task['task']}\n\n"
        f"Output path: {output_path.relative_to(PROJECT_ROOT)}\n\n"
        f"VERIFIED DATA (use ONLY these numbers and image paths — do not "
        f"invent additional data, quotes, or claims):\n"
        f"```json\n{data_str}\n```\n\n"
        f"STRICT RULES:\n"
        f"- Use ONLY the numeric values from the JSON above.\n"
        f"- Do NOT fabricate quotes from researchers or papers.\n"
        f"- Do NOT make authenticity claims about the Shroud.\n"
        f"- Do NOT add data that is not in the JSON.\n"
        f"- Image paths: use ONLY images listed in the JSON 'image_files' key.\n\n"
        f"NAVIGATION TEMPLATE (use this exact nav in every HTML page):\n"
        f"{NAV_TEMPLATE}\n\n"
        f"Mark the appropriate page as class=\"active\" in the nav."
    )

    log(f"Sending write task to MiniMax: {task['task'][:60]}...")
    response = call_minimax(prompt, max_tokens=16384)

    # Strip think blocks / prose preamble / markdown fences
    content = extract_code(response, lang="html")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    log(f"File written to {output_path.relative_to(PROJECT_ROOT)}")

    task["status"] = "done"
    task["output_files"] = [str(output_path.relative_to(PROJECT_ROOT))]

    log_task_result(task, True)
    # No auto-push — publishing is manual
    return True


HANDLERS = {
    "compute": handle_compute_task,
    "write": handle_write_task,
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
