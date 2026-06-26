"""
Test script for the MCP Action Recorder server.

Spawns the server as a subprocess, sends JSON-RPC requests over stdio,
and verifies that tools are listed and functional.

Usage:
    python tools/test_mcp_server.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import shutil

# --- Config ---
#_HERE = os.path.dirname(os.path.abspath(__file__))
#SERVER_CMD = [sys.executable, os.path.join(_HERE, "record_tools.py")]
SERVER_CMD = ["uvx", "mcp-action-recorder", "--work-dir", "./"]
TIMEOUT = 10  # seconds per request
PROTOCOL_VERSION = "2024-11-05"


def send_request(proc, request: dict) -> dict:
    """Send a JSON-RPC request and read the response."""
    line = json.dumps(request) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()

    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        resp_line = proc.stdout.readline()
        if resp_line.strip():
            return json.loads(resp_line)
        time.sleep(0.05)
    raise TimeoutError(f"No response for request id={request.get('id')}")


def test_server():
    work_dir = tempfile.mkdtemp(prefix="mcp_test_")
    errors = []

    try:
        proc = subprocess.Popen(
            SERVER_CMD + ["--work-dir", work_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print("FAIL: Could not start server. Is 'uvx' installed and mcp-action-recorder available?")
        shutil.rmtree(work_dir, ignore_errors=True) 
        return False

    req_id = 0

    def next_id():
        nonlocal req_id
        req_id += 1
        return req_id

    try:
        # --------------------------------------------------
        # 1. Initialize
        # --------------------------------------------------
        print("1. Testing initialize...")
        resp = send_request(proc, {
            "jsonrpc": "2.0",
            "id": next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1"},
            },
        })
        assert "result" in resp, f"initialize failed: {resp}"
        server_info = resp["result"].get("serverInfo", {})
        print(f"   Server: {server_info.get('name')} v{server_info.get('version')}")

        # Send initialized notification (required by protocol)
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()

        # --------------------------------------------------
        # 2. List tools
        # --------------------------------------------------
        print("2. Testing tools/list...")
        resp = send_request(proc, {
            "jsonrpc": "2.0",
            "id": next_id(),
            "method": "tools/list",
            "params": {},
        })
        assert "result" in resp, f"tools/list failed: {resp}"
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        print(f"   Found {len(tools)} tools: {tool_names}")

        expected_tools = {"read_file", "write_file", "execute_cmd", "export_log"}
        actual_tools = set(tool_names)
        if expected_tools != actual_tools:
            msg = f"Tool name mismatch: expected {expected_tools}, got {actual_tools}"
            print(f"   FAIL: {msg}")
            errors.append(msg)
        else:
            print("   OK: Tool names match expected")

        # --------------------------------------------------
        # 3. Test Write tool
        # --------------------------------------------------
        write_tool = None
        for t in tools:
            if "write" in t["name"].lower():
                write_tool = t["name"]
                break

        if write_tool:
            print(f"3. Testing {write_tool}...")
            resp = send_request(proc, {
                "jsonrpc": "2.0",
                "id": next_id(),
                "method": "tools/call",
                "params": {
                    "name": write_tool,
                    "arguments": {"path": "hello.txt", "content": "hello world\nsecond line\n"},
                },
            })
            assert "result" in resp, f"Write failed: {resp}"
            written_path = os.path.join(work_dir, "hello.txt")
            if os.path.exists(written_path):
                print(f"   OK: File created at {written_path}")
            else:
                msg = "Write tool returned success but file not found on disk"
                print(f"   FAIL: {msg}")
                errors.append(msg)
        else:
            msg = "No Write-like tool found to test"
            print(f"3. SKIP: {msg}")
            errors.append(msg)

        # --------------------------------------------------
        # 4. Test Read tool
        # --------------------------------------------------
        read_tool = None
        for t in tools:
            if "read" in t["name"].lower():
                read_tool = t["name"]
                break

        if read_tool:
            print(f"4. Testing {read_tool}...")
            resp = send_request(proc, {
                "jsonrpc": "2.0",
                "id": next_id(),
                "method": "tools/call",
                "params": {
                    "name": read_tool,
                    "arguments": {"path": "hello.txt"},
                },
            })
            assert "result" in resp, f"Read failed: {resp}"
            content_parts = resp["result"].get("content", [])
            text = "".join(p.get("text", "") for p in content_parts if p.get("type") == "text")
            if "hello world" in text:
                print(f"   OK: Read back correct content")
            else:
                msg = f"Read content unexpected: {text!r}"
                print(f"   FAIL: {msg}")
                errors.append(msg)

            # Test with offsets
            print(f"   Testing {read_tool} with start_offset/end_offset...")
            resp = send_request(proc, {
                "jsonrpc": "2.0",
                "id": next_id(),
                "method": "tools/call",
                "params": {
                    "name": read_tool,
                    "arguments": {"path": "hello.txt", "start_offset": 2, "end_offset": 2},
                },
            })
            assert "result" in resp, f"Read with offsets failed: {resp}"
            content_parts = resp["result"].get("content", [])
            text = "".join(p.get("text", "") for p in content_parts if p.get("type") == "text")
            if "second line" in text:
                print(f"   OK: Line offset read correct")
            else:
                msg = f"Read offset content unexpected: {text!r}"
                print(f"   FAIL: {msg}")
                errors.append(msg)
        else:
            msg = "No Read-like tool found to test"
            print(f"4. SKIP: {msg}")
            errors.append(msg)

        # --------------------------------------------------
        # 5. Test Execute tool
        # --------------------------------------------------
        exec_tool = None
        for t in tools:
            if "exec" in t["name"].lower():
                exec_tool = t["name"]
                break

        if exec_tool:
            print(f"5. Testing {exec_tool}...")
            resp = send_request(proc, {
                "jsonrpc": "2.0",
                "id": next_id(),
                "method": "tools/call",
                "params": {
                    "name": exec_tool,
                    "arguments": {"cmd": "echo ok && pwd"},
                },
            })
            assert "result" in resp, f"Execute failed: {resp}"
            content_parts = resp["result"].get("content", [])
            text = "".join(p.get("text", "") for p in content_parts if p.get("type") == "text")
            if "ok" in text and work_dir in text:
                print(f"   OK: Execute ran correctly in work_dir")
            else:
                msg = f"Execute output unexpected: {text!r}"
                print(f"   FAIL: {msg}")
                errors.append(msg)
        else:
            msg = "No Execute-like tool found to test"
            print(f"5. SKIP: {msg}")
            errors.append(msg)

        # --------------------------------------------------
        # 6. Test path restriction (Read outside work_dir)
        # --------------------------------------------------
        if read_tool:
            print(f"6. Testing path restriction...")
            resp = send_request(proc, {
                "jsonrpc": "2.0",
                "id": next_id(),
                "method": "tools/call",
                "params": {
                    "name": read_tool,
                    "arguments": {"path": "/etc/passwd"},
                },
            })
            assert "result" in resp, f"Path restriction test failed: {resp}"
            content_parts = resp["result"].get("content", [])
            text = "".join(p.get("text", "") for p in content_parts if p.get("type") == "text")
            if "outside" in text.lower() or "error" in text.lower():
                print(f"   OK: Path outside work_dir was rejected")
            else:
                msg = f"Path restriction not enforced: {text!r}"
                print(f"   FAIL: {msg}")
                errors.append(msg)

        # --------------------------------------------------
        # 7. Test ExportRecord tool
        # --------------------------------------------------
        export_tool = None
        for t in tools:
            if "export" in t["name"].lower() or "log" in t["name"].lower():
                export_tool = t["name"]
                break

        if export_tool:
            print(f"7. Testing {export_tool}...")
            resp = send_request(proc, {
                "jsonrpc": "2.0",
                "id": next_id(),
                "method": "tools/call",
                "params": {
                    "name": export_tool,
                    "arguments": {},
                },
            })
            assert "result" in resp, f"ExportRecord failed: {resp}"
            log_path = os.path.join(work_dir, "log", "actions.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    actions = json.load(f)
                print(f"   OK: Exported {len(actions)} actions to {log_path}")
                # Verify action log has entries from previous tool calls
                tool_names_logged = [a["tool"] for a in actions]
                print(f"   Logged tools: {tool_names_logged}")
            else:
                msg = f"ExportRecord returned success but {log_path} not found"
                print(f"   FAIL: {msg}")
                errors.append(msg)
        else:
            msg = "No ExportRecord-like tool found to test"
            print(f"7. SKIP: {msg}")
            errors.append(msg)

    except Exception as e:
        errors.append(f"Unexpected error: {e}")
        print(f"ERROR: {e}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(work_dir, ignore_errors=True) 

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED ({len(errors)} issue(s)):")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("ALL TESTS PASSED")
        return True


if __name__ == "__main__":
    ok = test_server()
    sys.exit(0 if ok else 1)
