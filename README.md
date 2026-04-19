# NLPCC 2026 Shared Task 11

## Agent-Based Experiment Reproduction from Scientific Papers

### Task Introduction

As LLM-based AI systems become increasingly integral to scientific workflows, they provide substantial support across multiple stages of research. However, experiment reproduction remains a significant challenge for current AI agents. This task evaluates agents' ability to reconstruct experimental pipelines from research papers by interpreting text, planning implementations, generating code, and executing experiments within a controlled environment. Unlike prior work that focuses only on final outputs, this task emphasizes the full reproduction process through fine-grained action logs and predefined rubrics covering key stages such as paper understanding, planning, coding, and execution.

### Data Description & Rules

For each paper in the dataset, both PDF and Markdown formats are provided. In this task, you need to build your agent system to read the given paper and reproduce the codebase for experiment reproduction. During this process, the system must use the tools provided by the [MCP Action Recorder](#mcp-action-recorder) to record all actions. Finally, you need to submit the action log generated during this process along with the corresponding repository. We will evaluate your reproduction results using a series of rubrics, each consisting of criteria and their assigned scores.

#### Training & Validation Data

In the training and validation data, we also release the ground-truth `rubrics.json` paired with each paper so you can examine your agent system's results and adjust accordingly.

Each `rubrics.json` is a JSON array of rubric entries. Each entry is structured as follows:

```json
{
  "criteria": "A description of what the agent is expected to accomplish.",
  "score": 3,
  "type": "Code Implementation",
  "comment": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `criteria` | `string` | A human-readable description of the specific requirement to be evaluated. |
| `score` | `integer` | The score weight assigned to this criterion (higher values indicate greater importance). |
| `type` | `string` | The evaluation category this criterion belongs to (see below). |
| `comment` | `string \| null` | Optional context, such as the relevant section or table in the paper. |

The `type` field categorizes each criterion into one of the following evaluation stages:

| Type | Description |
|------|-------------|
| `Paper Observation` | The agent has correctly read and understood the relevant sections of the paper. |
| `Plan Writing` | The agent has formulated a plan that covers the necessary components for reproduction. |
| `Code Implementation` | The agent has written code that implements the required functionality. |
| `Command Execution` | The agent has successfully executed the implemented code without errors. |
| `Result Matching` | The agent's execution outcomes are consistent with the results reported in the paper. |

#### Test Data

In the test data, the rubrics for each paper will not be released. You can still obtain the original papers. Please submit your reproduction repository and action log for evaluation.

*To be released.*

### Submission

Your submission should contain the reproduced repositories for each paper in the test data along with their action logs. The name for each repository must exactly match the subdirectory name provided in the `data/` directory, and the action log file must be saved to `log/actions.json` under that repository.

For instance, if the paper to reproduce is saved under `data/test/paper_name`, its reproduced repository should also be saved in a directory named `paper_name` in your submission, and the action log should be saved to `paper_name/log/actions.json`.

The complete submission should be packaged as a single zip file named with your team ID (`your_team_id.zip`) containing all reproduced repositories, and sent to [hanhua.hong@postgrad.manchester.ac.uk](mailto:hanhua.hong@postgrad.manchester.ac.uk).

#### Submission Format

```
your_team_id.zip
|
|- paper_1/
|  |- log/
|  |  |- actions.json
|  |- <reproduced source files>
|
|- paper_2/
|  |- log/
|  |  |- actions.json
|  |- <reproduced source files>
|
|- ...
```

---

## MCP Action Recorder

An MCP server that provides agents with real file and shell tools (`read_file`, `write_file`, `execute_cmd`) while recording every action into a chronological JSON log. Designed for evaluating agent behavior during ML paper reproduction.

### Quick Start

#### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

#### Run the Server

```bash
uvx --from tools mcp-action-recorder --work-dir /path/to/workspace
```

Or install and run manually:

```bash
pip install mcp
python tools/record_tools.py --work-dir /path/to/workspace
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--work-dir` | Yes | — | Agent working directory. Read/Write operations are restricted to this path. |
| `--log-dir` | No | `<work-dir>/log/` | Directory where `actions.json` will be saved. |

### Tools

#### `read_file`

Read a file from the workspace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `str` | Yes | Path to the file (absolute or relative to work-dir). |
| `start_offset` | `int` | No | 1-indexed starting line number (inclusive). |
| `end_offset` | `int` | No | 1-indexed ending line number (inclusive). |

#### `write_file`

Write content to a file in the workspace. Creates parent directories if needed.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `str` | Yes | Path to the file (absolute or relative to work-dir). |
| `content` | `str` | Yes | The content to write. |

#### `execute_cmd`

Execute a bash command. Runs with `cwd` set to the working directory.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cmd` | `str` | Yes | The bash command to execute. |

#### `export_log`

Export the action log to `actions.json` in the log directory. Takes no parameters.

### Usage

#### 1. Configure as an MCP Server

Add the server to your MCP client configuration. Below are examples for [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

**Option A: Using `uvx` (recommended)**

Add the following to your `.mcp.json` in the workspace root:

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

**Option B: Running the script directly**

If you prefer not to use `uvx`, you can point directly to the `record_tools.py` script:

```json
{
  "mcpServers": {
    "action-recorder": {
      "command": "python",
      "args": ["tools/record_tools.py", "--work-dir", "/path/to/workspace"]
    }
  }
}
```

To ensure that all agent actions are routed through the MCP Action Recorder (and thus properly logged), you should restrict built-in tools and only allow the MCP tools. For example, you can create a `.claude/settings.local.json` file in your workspace with the following content for Claude Code:

```json
{
  "permissions": {
    "allow": [
      "mcp__action-recorder"
    ],
    "deny": [
      "Read",
      "Edit",
      "Write",
      "Bash"
    ]
  },
  "enabledMcpjsonServers": [
    "action-recorder"
  ]
}
```

#### 2. Agent Uses Tools During a Reproduction Run

The agent interacts with the workspace through the MCP tools:

```
Agent -> read_file(path="paper.md")                          # reads the paper
Agent -> write_file(path="model.py", content="import ...")   # writes implementation
Agent -> execute_cmd(cmd="python train.py --epochs 10")      # runs experiment
Agent -> read_file(path="results/metrics.json")              # checks results
Agent -> export_log()                                        # exports the log
```

#### 3. Review the Action Log

After the run, `actions.json` is saved to the log directory (auto-exported on shutdown, or manually via `export_log`):

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

#### Action Record Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Sequential counter, starting at 1. |
| `timestamp` | `str` | ISO 8601 UTC timestamp. |
| `tool` | `str` | `"Read"`, `"Write"`, `"Execute"`, or `"ExportRecord"`. |
| `arguments` | `object` | Exact parameters passed to the tool. |
| `result` | `object` | Full output (no truncation). |
| `duration_ms` | `int` | Wall-clock execution time in milliseconds. |

### Example Repository

The [`example_repo/`](./example_repo/) directory provides a complete, minimal reference for setting up and running the MCP Action Recorder with Claude Code. It includes:

- **`.mcp.json`** — MCP server configuration pointing to the Action Recorder.
- **`.claude/settings.local.json`** — Claude Code permission settings that restrict built-in tools and route all actions through the MCP server.
- **`log/actions.json`** — A sample action log demonstrating the expected output format, including `Write`, `Read`, `Execute`, and `ExportRecord` entries.
- **`hello_world.py`** — A trivial script produced during the example run.

You can use this directory as a template when setting up your own reproduction workspace.

### Path Restriction

Read and Write operations are restricted to the `--work-dir` directory. Attempts to access paths outside (e.g., `../../etc/passwd` or `/etc/secret`) return an error. Execute commands are not path-restricted but run with `cwd` set to `--work-dir`.