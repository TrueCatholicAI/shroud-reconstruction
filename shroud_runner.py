#!/usr/bin/env python3
"""
Shroud Reconstruction Task Orchestrator
========================================
Fully autonomous MiniMax M2.7 pipeline. Reads a task JSON file, calls the
MiniMax API for code/content generation, executes the output, validates
results, auto-commits, and optionally pushes.  Designed to run unattended
from a terminal — no Claude Code required.

Usage:
    python shroud_runner.py                       # Run all pending tasks, commit+push
    python shroud_runner.py --dry-run             # Run all tasks but skip git push
    python shroud_runner.py --task 0              # Run only task at index 0
    python shroud_runner.py --retry-failed        # Re-run tasks marked "failed"
    python shroud_runner.py --tasks wave7.json    # Use a different task file

Task queue format:
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
- Every task is wrapped in try/except — one failure never crashes the run.
- Missing pip packages are auto-installed before script execution.
- Git push failures are caught and logged, never fatal.
- Full run summary saved to output/runner_summary.json.

Filename convention (so compute and write tasks agree on JSON paths):
- Compute task descriptions should end with:  OUTPUT: my_results.json
  → saves to output/task_results/my_results.json
- Write task descriptions should start with:  READ: my_results.json
  → reads from output/task_results/my_results.json
- If these tags are absent, compute falls back to slug-based naming
  and write falls back to the "data_source" field in the task JSON.
"""

import json
import os
import sys
import subprocess
import time
import traceback
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
MARATHON_LOG = PROJECT_ROOT / "marathon_log.md"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = OUTPUT_DIR / "task_results"
ERROR_LOG = OUTPUT_DIR / "runner_errors.log"
SUMMARY_JSON = OUTPUT_DIR / "runner_summary.json"
DOCS_DIR = PROJECT_ROOT / "docs"
SRC_DIR = PROJECT_ROOT / "src"
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
VENV_PIP = PROJECT_ROOT / "venv" / "Scripts" / "pip.exe"

load_dotenv(PROJECT_ROOT / ".env")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_ENDPOINT = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-M2.7"

# Runtime counters
_tokens_used = {"prompt": 0, "completion": 0}
_files_created: list[str] = []
_errors: list[dict] = []

# System prompt
SYSTEM_PROMPT = """You are a code generation assistant for the Shroud of Turin AI Forensic Reconstruction project.

Project structure:
- src/ — Python analysis scripts
- docs/ — GitHub Pages site (HTML pages)
- docs/images/ — visualization outputs shown on the site
- docs/css/style.css — shared stylesheet
- output/ — raw script outputs (analysis/, 3d_print/, full_body/, etc.)
- data/processed/ — depth maps (.npy files)
- data/final/ — final analysis-ready depth maps
- data/source/ — original photographs
- venv/ — Python 3.11 virtual environment

Rules:
- Use matplotlib Agg backend (import matplotlib; matplotlib.use('Agg'))
- Dark figure backgrounds (#1a1a1a), gold accent (#c4a35a), white text
- Save figures to both output/analysis/ and docs/images/
- When writing HTML, use <link rel="stylesheet" href="css/style.css">

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

# Nav template — loaded from fix_nav.py if available, otherwise hardcoded
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
      <li><a href="wound-mapping.html">Wound Mapping</a></li>
      <li><a href="wavelet-analysis.html">Wavelet</a></li>
      <li><a href="bilateral-analysis.html">Bilateral</a></li>
      <li><a href="seed-sweep.html">Seed Sweep</a></li>
      <li><a href="curvature-analysis.html">Curvature</a></li>
      <li><a href="sudarium.html">Sudarium</a></li>
      <li><a href="art-historical.html">Art History</a></li>
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
    global _tokens_used
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

    # Track token usage
    usage = data.get("usage", {})
    _tokens_used["prompt"] += usage.get("prompt_tokens", 0)
    _tokens_used["completion"] += usage.get("completion_tokens", 0)

    base_resp = data.get("base_resp", {})
    if base_resp.get("status_code", 0) != 0:
        raise RuntimeError(
            f"MiniMax API error {base_resp.get('status_code')}: "
            f"{base_resp.get('status_msg', 'unknown')}"
        )

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"MiniMax returned no choices: {json.dumps(data)[:300]}")
    return choices[0]["message"]["content"]


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------
_CODE_START_RE = re.compile(
    r"^("
    r"import\s|from\s|def\s|class\s"
    r"|#!|#!/"
    r"|<!DOCTYPE|<html"
    r"|\"\"\"|\'\'\'"
    r"|#\s"
    r")",
)


def extract_code(text: str, lang: str = "python") -> str:
    """Extract code from MiniMax response, stripping think blocks and prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    pattern = rf"```{lang}?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if _CODE_START_RE.match(line.strip()):
            start = i
            break
    if start is None:
        start = 0

    code = "\n".join(lines[start:])
    code = re.sub(r"\n```\s*$", "", code)
    return code.strip()


# ---------------------------------------------------------------------------
# Dependency scanning and auto-install
# ---------------------------------------------------------------------------
# Packages that are part of stdlib or known aliases — skip these
_STDLIB_AND_ALIASES = {
    "os", "sys", "re", "json", "math", "time", "datetime", "pathlib",
    "subprocess", "hashlib", "io", "collections", "itertools", "functools",
    "shutil", "glob", "copy", "traceback", "typing", "abc", "enum",
    "dataclasses", "argparse", "logging", "csv", "struct", "base64",
    "urllib", "http", "string", "textwrap", "warnings", "contextlib",
    "tempfile", "threading", "multiprocessing", "socket", "pickle",
    # Common pip-name != import-name mappings handled below
    "cv2", "PIL", "skimage", "sklearn", "yaml", "mpl_toolkits",
}

_IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "pywt": "PyWavelets",
    "dotenv": "python-dotenv",
}


def scan_and_install_deps(code: str) -> list[str]:
    """Scan code for imports and pip-install anything missing."""
    installed = []
    import_re = re.compile(r"^\s*(?:import|from)\s+(\w+)", re.MULTILINE)
    modules = set(import_re.findall(code))

    for mod in modules:
        if mod in _STDLIB_AND_ALIASES:
            continue
        # Check if importable
        check = subprocess.run(
            [str(VENV_PYTHON), "-c", f"import {mod}"],
            capture_output=True, text=True, timeout=15,
        )
        if check.returncode == 0:
            continue

        pip_name = _IMPORT_TO_PIP.get(mod, mod)
        log(f"  Auto-installing missing package: {pip_name}")
        result = subprocess.run(
            [str(VENV_PIP), "install", pip_name],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            installed.append(pip_name)
            log(f"  Installed {pip_name}")
        else:
            log(f"  WARNING: Failed to install {pip_name}: {result.stderr[:200]}")

    return installed


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------
def validate_output_files(files: list[str]) -> tuple[bool, list[str]]:
    """Check that output files exist and aren't empty."""
    errors = []
    for f in files:
        p = Path(f) if os.path.isabs(f) else PROJECT_ROOT / f
        if not p.exists():
            errors.append(f"Missing: {f}")
        elif p.stat().st_size == 0:
            errors.append(f"Empty: {f}")
        elif p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            with open(p, "rb") as fh:
                header = fh.read(8)
            if p.suffix.lower() == ".png" and header[:4] != b"\x89PNG":
                errors.append(f"Invalid PNG: {f}")
            elif p.suffix.lower() in (".jpg", ".jpeg") and header[:2] != b"\xff\xd8":
                errors.append(f"Invalid JPEG: {f}")
    return len(errors) == 0, errors


def run_python_script(script_path: str, timeout: int = 300) -> tuple[int, str, str]:
    """Execute a Python script in the project venv."""
    result = subprocess.run(
        [str(VENV_PYTHON), str(script_path)],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------
def git_add_commit(message: str, files: list[str]) -> bool:
    """Stage files and commit. Returns True on success."""
    try:
        for f in files:
            subprocess.run(
                ["git", "add", str(f)],
                cwd=str(PROJECT_ROOT), capture_output=True, check=True,
            )
        # Check if there's anything staged
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(PROJECT_ROOT), capture_output=True,
        )
        if diff.returncode == 0:
            log("  Nothing to commit (no changes)")
            return True
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(PROJECT_ROOT), capture_output=True, check=True,
        )
        log(f"  Committed: {message[:60]}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"  Git commit failed: {e}")
        return False


def git_push() -> bool:
    """Push to origin. Returns True on success."""
    try:
        subprocess.run(
            ["git", "push"],
            cwd=str(PROJECT_ROOT), capture_output=True, check=True, timeout=60,
        )
        log("  Pushed to origin")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log(f"  PUSH FAILED: {e}")
        _errors.append({
            "task": "git push",
            "error": str(e)[:300],
            "timestamp": datetime.now().isoformat(),
        })
        return False


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------
def log_error(task_desc: str, exc: Exception):
    """Write full traceback to error log file."""
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Task: {task_desc[:120]}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n")
        traceback.print_exc(file=f)
        f.write("\n")


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
            f.write(f"- **Stdout:**\n```\n{stdout.strip()[:500]}\n```\n")
        if stderr.strip() and not success:
            f.write(f"- **Stderr:**\n```\n{stderr.strip()[:300]}\n```\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------
def _extract_output_filename(task_desc: str) -> str | None:
    """Extract explicit output filename from 'OUTPUT: filename.json' at end of task."""
    match = re.search(r"OUTPUT:\s*(\S+\.json)\s*$", task_desc)
    return match.group(1) if match else None


def _extract_read_filename(task_desc: str) -> str | None:
    """Extract explicit data source from 'READ: filename.json' at start of task."""
    match = re.match(r"READ:\s*(\S+\.json)", task_desc)
    return match.group(1) if match else None


def handle_compute_task(task: dict, dry_run: bool = False) -> bool:
    """Generate a Python script via MiniMax, execute it, validate outputs."""
    # Determine JSON output path: explicit OUTPUT: tag or slug fallback
    slug = re.sub(r"[^a-z0-9]+", "_", task["task"][:40].lower()).strip("_")
    explicit_name = _extract_output_filename(task["task"])
    if explicit_name:
        results_json = RESULTS_DIR / explicit_name
    else:
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
        f"  The JSON must be a flat or shallow dict of result_name -> value.\n"
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
    _files_created.append(str(script_path.relative_to(PROJECT_ROOT)))
    log(f"Script written to {script_path.name}")

    if dry_run:
        log("  (dry-run: skipping execution)")
        task["status"] = "done"
        return True

    # Auto-install missing deps
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scan_and_install_deps(code)

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

    # Auto-commit compute artifacts
    if success:
        commit_files = [str(script_path)]
        if task.get("output_files"):
            commit_files.extend(task["output_files"])
        git_add_commit(
            f"[runner] compute: {task['task'][:50]}",
            commit_files,
        )

    return success


def handle_write_task(task: dict, dry_run: bool = False) -> bool:
    """Generate a file via MiniMax using ONLY verified JSON data."""
    # Resolve data source: READ: tag in task desc > data_source field
    read_name = _extract_read_filename(task["task"])
    data_source = (
        str(RESULTS_DIR / read_name) if read_name
        else task.get("data_source")
    )
    if not data_source:
        log("FAIL: Write task has no 'data_source' field and no READ: tag in description.")
        task["status"] = "failed"
        task["error"] = "Missing data_source / READ: tag"
        log_task_result(task, False)
        return False

    json_path = Path(data_source) if os.path.isabs(data_source) else PROJECT_ROOT / data_source
    if not json_path.exists():
        log(f"FAIL: Data source not found: {data_source}")
        task["status"] = "failed"
        task["error"] = f"data_source not found: {data_source}"
        log_task_result(task, False)
        return False

    try:
        results_data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(results_data, dict):
            raise ValueError("Top-level value must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        log(f"FAIL: Data source is malformed: {e}")
        task["status"] = "failed"
        task["error"] = f"Malformed data_source: {e}"
        log_task_result(task, False)
        return False

    log(f"Data source validated: {data_source} ({len(results_data)} keys)")

    image_files = results_data.get("image_files", [])
    missing = [p for p in image_files
               if not (Path(p) if os.path.isabs(p) else PROJECT_ROOT / p).exists()]
    if missing:
        log(f"WARNING: {len(missing)} images missing: {missing[:5]}")

    path_match = re.search(r"(docs/\S+\.html|docs/\S+\.md|src/\S+\.py)", task["task"])
    if path_match:
        output_path = PROJECT_ROOT / path_match.group(1)
    else:
        slug = re.sub(r"[^a-z0-9]+", "_", task["task"][:40].lower()).strip("_")
        output_path = DOCS_DIR / f"mm_{slug}.html"

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
    content = extract_code(response, lang="html")

    if dry_run:
        log(f"  (dry-run: would write to {output_path.relative_to(PROJECT_ROOT)})")
        task["status"] = "done"
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    rel = str(output_path.relative_to(PROJECT_ROOT))
    log(f"File written to {rel}")
    _files_created.append(rel)

    task["status"] = "done"
    task["output_files"] = [rel]
    log_task_result(task, True)

    # Auto-commit
    git_add_commit(
        f"[runner] write: {task['task'][:50]}",
        [str(output_path)],
    )

    return True


HANDLERS = {
    "compute": handle_compute_task,
    "write": handle_write_task,
}


# ---------------------------------------------------------------------------
# Task file I/O
# ---------------------------------------------------------------------------
def load_tasks(tasks_file: Path) -> list[dict]:
    """Load task queue from JSON file."""
    if not tasks_file.exists():
        log(f"No {tasks_file.name} found. Creating empty queue.")
        tasks_file.write_text("[]", encoding="utf-8")
        return []
    return json.loads(tasks_file.read_text(encoding="utf-8"))


def save_tasks(tasks: list[dict], tasks_file: Path):
    """Save task queue back to JSON."""
    tasks_file.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_tasks(tasks: list[dict], tasks_file: Path, dry_run: bool = False,
              task_index: int = None) -> dict:
    """Execute pending tasks. Returns summary dict."""
    run_start = time.time()

    if task_index is not None:
        if task_index >= len(tasks):
            log(f"Task index {task_index} out of range (have {len(tasks)} tasks)")
            return {"attempted": 0, "passed": 0, "failed": 0}
        indices = [task_index]
    else:
        indices = list(range(len(tasks)))

    total = len(tasks)
    attempted = 0
    passed = 0
    failed = 0
    push_ok = True

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
            failed += 1
            save_tasks(tasks, tasks_file)
            continue

        log(f"\n{'='*60}")
        log(f"Task {i+1}/{total}: [{task_type}] {task['task'][:70]}")
        log(f"{'='*60}")

        task["status"] = "running"
        save_tasks(tasks, tasks_file)
        attempted += 1

        try:
            success = handler(task, dry_run=dry_run)
            if success:
                passed += 1
            else:
                failed += 1
                _errors.append({
                    "task": task["task"][:120],
                    "error": task.get("error", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                })
        except Exception as e:
            log(f"EXCEPTION: {e}")
            log_error(task["task"], e)
            task["status"] = "failed"
            task["error"] = str(e)[:500]
            failed += 1
            _errors.append({
                "task": task["task"][:120],
                "error": str(e)[:300],
                "traceback": traceback.format_exc()[:500],
                "timestamp": datetime.now().isoformat(),
            })

        save_tasks(tasks, tasks_file)

    # Git push (unless dry-run)
    if not dry_run and attempted > 0:
        log("\nPushing to origin...")
        push_ok = git_push()
    elif dry_run:
        log("\n(dry-run: skipping git push)")
        push_ok = True

    elapsed = time.time() - run_start

    summary = {
        "timestamp": datetime.now().isoformat(),
        "tasks_file": str(tasks_file),
        "dry_run": dry_run,
        "attempted": attempted,
        "passed": passed,
        "failed": failed,
        "remaining": sum(1 for t in tasks if t.get("status") == "pending"),
        "files_created": _files_created.copy(),
        "files_created_count": len(_files_created),
        "tokens_prompt": _tokens_used["prompt"],
        "tokens_completion": _tokens_used["completion"],
        "tokens_total": _tokens_used["prompt"] + _tokens_used["completion"],
        "runtime_seconds": round(elapsed, 1),
        "push_succeeded": push_ok,
        "errors": _errors.copy(),
    }

    # Save summary
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print summary
    log(f"\n{'='*60}")
    log("RUN SUMMARY")
    log(f"{'='*60}")
    log(f"  Attempted:  {attempted}")
    log(f"  Passed:     {passed}")
    log(f"  Failed:     {failed}")
    log(f"  Files:      {len(_files_created)}")
    log(f"  Tokens:     {summary['tokens_total']} "
        f"(prompt: {_tokens_used['prompt']}, completion: {_tokens_used['completion']})")
    log(f"  Runtime:    {elapsed:.1f}s")
    if not push_ok:
        log("  *** PUSH FAILED — run 'git push' manually ***")
    if _errors:
        log(f"  Errors:     {len(_errors)}")
        for err in _errors:
            log(f"    - {err['task'][:60]}: {err['error'][:80]}")
    log(f"\nSummary saved to {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Shroud Reconstruction Task Runner — autonomous MiniMax pipeline",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run tasks but skip git push (review before going live)")
    parser.add_argument("--task", type=int, default=None,
                        help="Run a specific task by index")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-run tasks marked 'failed'")
    parser.add_argument("--tasks", type=str, default="tasks.json",
                        help="Path to task file (default: tasks.json)")
    args = parser.parse_args()

    tasks_file = Path(args.tasks)
    if not tasks_file.is_absolute():
        tasks_file = PROJECT_ROOT / tasks_file

    log("\n" + "=" * 60)
    log("SHROUD RUNNER — MiniMax M2.7 Autonomous Orchestrator")
    log(f"Tasks file: {tasks_file}")
    log(f"Dry run: {args.dry_run}")
    log("=" * 60)

    tasks = load_tasks(tasks_file)
    if not tasks:
        log("Task queue is empty.")
        return

    if args.retry_failed:
        reset = 0
        for t in tasks:
            if t.get("status") == "failed":
                t["status"] = "pending"
                t.pop("error", None)
                reset += 1
        save_tasks(tasks, tasks_file)
        log(f"Reset {reset} failed tasks to pending")

    pending = sum(1 for t in tasks if t.get("status", "pending") == "pending")
    done = sum(1 for t in tasks if t.get("status") == "done")
    log(f"Loaded {len(tasks)} tasks. Pending: {pending}, Done: {done}")

    if pending == 0:
        log("No pending tasks. Nothing to do.")
        return

    run_tasks(tasks, tasks_file, dry_run=args.dry_run, task_index=args.task)


if __name__ == "__main__":
    main()
