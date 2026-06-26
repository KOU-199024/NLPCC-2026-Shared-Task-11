# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLPCC 2026 Shared Task 11: Agent-Based Experiment Reproduction from Scientific Papers. The goal is to build an agent system that reads a research paper and reproduces its experimental codebase, with all actions recorded via an MCP Action Recorder.

## Repository Structure

- `data/train_valid/<domain>/<paper_name>/` — Training/validation papers, each containing `paper.pdf`, `paper.md`, `images/`, and `rubrics.json` (ground-truth evaluation criteria with scores and types like "Paper Observation", "Plan Writing", etc.)
- `data/test/<domain>/<paper_name>/` — Test papers (no rubrics provided)
- `tools/record_tools.py` — MCP Action Recorder server (the core tool for this task)
- `example_repo/` — Reference example showing correct MCP setup (`.mcp.json`, `.claude/settings.local.json`) and expected log format (`log/actions.json`)

## MCP Action Recorder

The recorder provides 4 tools: `read_file`, `write_file`, `execute_cmd`, `export_log`. It records every agent action into `log/actions.json`.

### Running the MCP server

```bash
# With uvx (recommended)
uvx --from tools mcp-action-recorder --work-dir /path/to/workspace

# Or directly
python tools/record_tools.py --work-dir /path/to/workspace
```

Requires `mcp` Python package. The `--work-dir` argument is required; `--log-dir` defaults to `<work-dir>/log/`.

### MCP configuration for Claude Code

Place `.mcp.json` in the reproduction workspace:

```json
{
    "mcpServers": {
        "action-recorder": {
            "command": "uvx",
            "args": ["mcp-action-recorder", "--work-dir", "./"]
        }
    }
}
```

When running reproduction, Claude Code should use `.claude/settings.local.json` to restrict tools to only MCP tools (deny `Read`, `Edit`, `Write`, `Bash`) so all actions are recorded.

## Submission Format

The submission should follow the two-layer `domain/paper_name` structure from `data/test/` and must include `log/actions.json` under each paper directory:

```
your_team_id.zip
|- domain/
   |- paper_name/
      |- log/
         |- actions.json
      |- <reproduced code files>
```

## Evaluation

Rubrics (`rubrics.json`) evaluate the reproduction across stages: paper understanding, planning, coding, and execution. Each criterion has a score weight and a type category.
