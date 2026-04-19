# tools/record_tools.py
import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mcp.server.fastmcp import FastMCP

try:
    __version__ = version("mcp-action-recorder")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"


# --- Global state ---
action_log: list[dict] = []
action_counter: int = 0
work_dir: Path = Path(".")
log_dir: Path = Path("./log")

mcp = FastMCP("action-recorder")


def next_id() -> int:
    global action_counter
    action_counter += 1
    return action_counter


def record_action(tool: str, arguments: dict, result: dict, duration_ms: int) -> None:
    action_log.append({
        "id": next_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "arguments": arguments,
        "result": result,
        "duration_ms": duration_ms,
    })


def export_log() -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / "actions.json"
    with open(log_path, "w") as f:
        json.dump(action_log, f, indent=2, ensure_ascii=False)
    return str(log_path)


def validate_path(path_str: str) -> Path:
    """Resolve a path relative to work_dir and ensure it stays within it."""
    requested = Path(path_str)
    if not requested.is_absolute():
        resolved = (work_dir / requested).resolve()
    else:
        resolved = requested.resolve()
    work_dir_resolved = work_dir.resolve()
    if not (resolved == work_dir_resolved or str(resolved).startswith(str(work_dir_resolved) + os.sep)):
        raise ValueError(f"Path '{path_str}' is outside the working directory '{work_dir_resolved}'")
    return resolved


@mcp.tool(name="read_file", description="Read a file from the workspace. Returns file content as text.")
def read_file(path: str, start_offset: int | None = None, end_offset: int | None = None) -> str:
    """Read a file from the workspace.

    Args:
        path: Path to the file (absolute or relative to working directory).
        start_offset: Optional 1-indexed starting line number (inclusive).
        end_offset: Optional 1-indexed ending line number (inclusive).
    """
    start = time.time()
    arguments = {"path": path, "start_offset": start_offset, "end_offset": end_offset}

    try:
        resolved = validate_path(path)
        with open(resolved, "r") as f:
            lines = f.readlines()

        # Apply line offsets (1-indexed, inclusive)
        start_idx = (start_offset - 1) if start_offset is not None else 0
        end_idx = end_offset if end_offset is not None else len(lines)
        selected = lines[start_idx:end_idx]
        content = "".join(selected)

        result = {"content": content, "success": True}
        duration_ms = int((time.time() - start) * 1000)
        record_action("Read", arguments, result, duration_ms)
        return content

    except Exception as e:
        result = {"content": "", "success": False, "error": str(e)}
        duration_ms = int((time.time() - start) * 1000)
        record_action("Read", arguments, result, duration_ms)
        return f"Error: {e}"


@mcp.tool(name="write_file", description="Write content to a file in the workspace. Creates parent directories if needed.")
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace.

    Args:
        path: Path to the file (absolute or relative to working directory).
        content: The content to write to the file.
    """
    start = time.time()
    arguments = {"path": path, "content": content}

    try:
        resolved = validate_path(path)
        os.makedirs(resolved.parent, exist_ok=True)
        with open(resolved, "w") as f:
            f.write(content)

        result = {"success": True}
        duration_ms = int((time.time() - start) * 1000)
        record_action("Write", arguments, result, duration_ms)
        return f"Successfully wrote to {path}"

    except Exception as e:
        result = {"success": False, "error": str(e)}
        duration_ms = int((time.time() - start) * 1000)
        record_action("Write", arguments, result, duration_ms)
        return f"Error: {e}"


@mcp.tool(name="execute_cmd", description="Execute a bash command. Runs in the workspace directory.")
def execute_cmd(cmd: str) -> str:
    """Execute a bash command.

    Args:
        cmd: The bash command to execute.
    """
    start = time.time()
    arguments = {"cmd": cmd}

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            cwd=str(work_dir),
        )

        result = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "success": proc.returncode == 0,
        }
        duration_ms = int((time.time() - start) * 1000)
        record_action("Execute", arguments, result, duration_ms)

        log_parts = []
        if proc.stdout:
            log_parts.append(proc.stdout)
        if proc.stderr:
            log_parts.append(f"[stderr]\n{proc.stderr}")
        log_parts.append(f"[exit code: {proc.returncode}]")
        return "\n".join(log_parts)

    except Exception as e:
        result = {"stdout": "", "stderr": str(e), "exit_code": -1, "success": False}
        duration_ms = int((time.time() - start) * 1000)
        record_action("Execute", arguments, result, duration_ms)
        return f"Error: {e}"


@mcp.tool(name="export_log", description="Export the action log to actions.json in the log directory.")
def export_record() -> str:
    """Export the recorded action log to the configured log directory."""
    start = time.time()
    arguments = {}

    try:
        log_path = export_log()
        result = {
            "log_path": log_path,
            "actions_count": len(action_log),
            "success": True,
        }
        duration_ms = int((time.time() - start) * 1000)
        record_action("ExportRecord", arguments, result, duration_ms)
        return f"Exported {len(action_log)} actions to {log_path}"

    except Exception as e:
        result = {"log_path": "", "actions_count": len(action_log), "success": False, "error": str(e)}
        duration_ms = int((time.time() - start) * 1000)
        record_action("ExportRecord", arguments, result, duration_ms)
        return f"Error: {e}"


def main():
    global work_dir, log_dir

    parser = argparse.ArgumentParser(description="MCP Action Recorder Server")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--work-dir", required=True, help="Agent working directory. Read/Write restricted to this path.")
    parser.add_argument("--log-dir", default=None, help="log directory for actions.json. Defaults to <work-dir>/log/")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    if not work_dir.is_dir():
        print(f"Error: --work-dir '{args.work_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        log_dir = work_dir / "log"

    atexit.register(export_log)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
