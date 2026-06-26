# MCP Action Recorder

An MCP server that provides agents with real file and shell tools (`read_file`, `write_file`, `execute_cmd`) while recording every action into a chronological JSON log. Designed for evaluating agent behavior during ML paper reproduction.

## Quick Start

### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Run the server

```bash
uvx --from tools mcp-action-recorder --work-dir /path/to/workspace
```

Or install and run manually:

```bash
cd tools && pip install .
mcp-action-recorder --work-dir /path/to/workspace
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--work-dir` | yes | — | Agent working directory. Read/Write operations are restricted to this path. |
| `--log-dir` | no | `<work-dir>/log/` | Directory where `actions.json` will be saved. |

## Tools

### read_file

Read a file from the workspace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `str` | yes | Path to the file (absolute or relative to work-dir) |
| `start_offset` | `int` | no | 1-indexed starting line number (inclusive) |
| `end_offset` | `int` | no | 1-indexed ending line number (inclusive) |

### write_file

Write content to a file in the workspace. Creates parent directories if needed.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `str` | yes | Path to the file (absolute or relative to work-dir) |
| `content` | `str` | yes | The content to write |

### execute_cmd

Execute a bash command. Runs with `cwd` set to the working directory.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cmd` | `str` | yes | The bash command to execute |

### export_log

Export the action log to `actions.json` in the log directory. Takes no parameters.

## Usage Example

### 1. Configure as an MCP server

Add the server to your MCP client configuration. For example, in Claude Code's `mcp_config.json`:

```json
{
  "mcpServers": {
    "action-recorder": {
      "command": "uvx",
      "args": ["mcp-action-recorder", "--work-dir", "/path/to/workspace"]
    }
  }
}
```

### 2. Agent uses tools during a reproduction run

The agent interacts with the workspace through the MCP tools:

```
Agent → read_file(path="paper.md")                          # reads the paper
Agent → write_file(path="model.py", content="import ...")   # writes implementation
Agent → execute_cmd(cmd="python train.py --epochs 10")      # runs experiment
Agent → read_file(path="results/metrics.json")              # checks results
Agent → export_log()                                        # exports the log
```

### 3. Review the action log

After the run, `actions.json` is saved to the output directory (auto-exported on shutdown, or manually via `export_log`):

```json
[
  {
    "id": 1,
    "timestamp": "2026-04-14T15:30:00.123456+00:00",
    "tool": "Read",
    "arguments": {
      "path": "paper.md",
      "start_offset": null,
      "end_offset": null
    },
    "result": {
      "content": "# My Paper\n\n## Abstract\n...",
      "success": true
    },
    "duration_ms": 5
  },
  {
    "id": 2,
    "timestamp": "2026-04-14T15:30:10.456789+00:00",
    "tool": "Write",
    "arguments": {
      "path": "model.py",
      "content": "import torch\nimport torch.nn as nn\n..."
    },
    "result": {
      "success": true
    },
    "duration_ms": 2
  },
  {
    "id": 3,
    "timestamp": "2026-04-14T15:30:15.000000+00:00",
    "tool": "Execute",
    "arguments": {
      "cmd": "python train.py --epochs 10"
    },
    "result": {
      "stdout": "Epoch 1/10 loss=0.45\nEpoch 2/10 loss=0.32\n...",
      "stderr": "",
      "exit_code": 0,
      "success": true
    },
    "duration_ms": 120000
  },
  {
    "id": 4,
    "timestamp": "2026-04-14T15:32:15.000000+00:00",
    "tool": "Read",
    "arguments": {
      "path": "results/metrics.json",
      "start_offset": null,
      "end_offset": null
    },
    "result": {
      "content": "{\"accuracy\": 0.87, \"f1\": 0.85}",
      "success": true
    },
    "duration_ms": 3
  }
]
```

### Action Record Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Sequential counter, starting at 1 |
| `timestamp` | `str` | ISO 8601 UTC timestamp |
| `tool` | `str` | `"Read"`, `"Write"`, `"Execute"`, or `"ExportRecord"` |
| `arguments` | `object` | Exact parameters passed to the tool |
| `result` | `object` | Full output (no truncation) |
| `duration_ms` | `int` | Wall-clock execution time in milliseconds |

## Path Restriction

Read and Write operations are restricted to the `--work-dir` directory. Attempts to access paths outside (e.g., `../../etc/passwd` or `/etc/secret`) return an error. Execute commands are not path-restricted but run with `cwd` set to `--work-dir`.

## Lifecycle

1. **Startup** — server parses args, initializes empty action list, listens on stdio
2. **Recording** — each tool call performs the real operation and appends a record
3. **Export** — call `export_log` anytime, or let the server auto-export on shutdown via `atexit`
