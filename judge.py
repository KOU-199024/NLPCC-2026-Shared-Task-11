import os
import re
import json
import argparse
import logging
from datetime import datetime

from openai import OpenAI
from pydantic import BaseModel


RUBRIC_TYPE_TOOLS = {
    "Paper Observation": ["Read"],
    "Plan Writing": ["Write"],
    "Code Implementation": ["Write"],
    "Command Execution": ["Execute"],
    "Result Matching": ["Execute", "Read"],
}
DEFAULT_TOOLS = ["Read", "Write", "Execute"]

system_prompt = """You are a serious judge of experiment reproduction.
You are given a research paper, a single rubric criterion, and the relevant action log entries recorded by a candidate reproduction agent.
Your job is to decide whether the reproduction satisfies the criterion based on what the agent actually did.

Read the paper carefully as the ground truth. Then inspect the provided action log and decide whether the criterion has been met.
Respond in three parts:

# Expectations
Identify the parts of the paper relevant to the criterion. Describe what a correct reproduction should look like and what specific actions the agent should have taken.

# Reality
Inspect the provided action log entries and describe what the agent actually did that is relevant to the criterion. Be explicit about which logged actions you cite.

# Score
Give a binary score of either 0 (fail) or 1 (pass), with a short justification.
Be strict and thorough, but only judge what the criterion asks for.
If the relevant actions are missing or insufficient, treat that as a failure.
IMPORTANT: If the rubric criterion is of type "Paper Observation", you will be provided only with the agent’s "Read" logs. In this case, judge only whether the agents have read the relevant parts of the paper. Do not consider any other actions, outputs, or reproduction results.
"""

content_prompt = """Here is the paper:
<paper>
{paper}
</paper>

Here is the rubric criterion you are grading (type: {rubric_type}, importance score: {rubric_score}):
<criterion>
{rubric}
</criterion>

Here are the relevant action log entries from the candidate reproduction (filtered for tools: {tools}):
<action_log>
{action_log}
</action_log>

Now produce your Expectations / Reality / Score response. End with a final line of the form `FINAL: 0` or `FINAL: 1`.
"""


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JudgeResponse(BaseModel):
    valid_score: bool
    score: int
    explanation: str


def normalize_rubric(rubric):
    """Normalize a rubric dict so downstream code can rely on `criteria`, `score` (int),
    `type`, and `comment` regardless of which generation pipeline produced it."""
    normalized = dict(rubric)
    normalized.setdefault("criteria", "")
    normalized.setdefault("type", "Unknown")
    normalized.setdefault("comment", None)
    raw_score = normalized.get("score", 0)
    try:
        normalized["score"] = int(raw_score)
    except (TypeError, ValueError):
        normalized["score"] = 0
    return normalized


def format_action_entry(action, max_chars=100000):
    """Format a single action log entry as a readable text block."""
    tool = action.get("tool", "")
    args = action.get("arguments", {})
    result = action.get("result", {})

    if tool == "Read":
        path = args.get("path", "unknown")
        content = ""
        if isinstance(result, dict):
            content = result.get("content", "")
        text = f"[Read] path={path}\n{content}"
    elif tool == "Write":
        path = args.get("path", "unknown")
        content = args.get("content", "")
        text = f"[Write] path={path}\n{content}"
    elif tool == "Execute":
        cmd = args.get("cmd", "")
        stdout = ""
        if isinstance(result, dict):
            stdout = result.get("stdout", result.get("output", ""))
            stderr = "Empty" if len(result.get("stderr", "")) == 0 else result.get("stderr", "Empty")
        text = f"[Execute] cmd={cmd}\nOutput: {stdout}\nError: {stderr}"
    else:
        return None

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text


def load_and_filter_actions(action_log_path, rubric_type, max_total_chars=150000, per_entry_chars=100000):
    """Load actions.json and return filtered + formatted content for the given rubric type."""
    with open(action_log_path, "r") as f:
        actions = json.load(f)

    relevant_tools = RUBRIC_TYPE_TOOLS.get(rubric_type, DEFAULT_TOOLS)
    filtered = [a for a in actions if a.get("tool") in relevant_tools]

    blocks = []
    total = 0
    for action in filtered:
        block = format_action_entry(action, max_chars=per_entry_chars)
        if block is None:
            continue
        if total + len(block) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 200:
                blocks.append(block[:remaining] + "\n... [truncated]")
            break
        blocks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(blocks), relevant_tools


def parse_judge_response(raw_response):
    """Extract binary score from the FINAL: 0/1 marker the model is prompted to emit."""
    match = re.search(r'FINAL:\s*([01])\b', raw_response)
    if match:
        score = int(match.group(1))
        pre = raw_response[:match.start()].rstrip()
        paragraphs = [p.strip() for p in pre.split('\n\n') if p.strip()]
        explanation = paragraphs[-1] if paragraphs else ""
        if len(explanation) > 300:
            explanation = explanation[-300:]
        return JudgeResponse(valid_score=True, score=score, explanation=explanation)
    return JudgeResponse(valid_score=False, score=0, explanation="no FINAL score found in response")


def grade_rubric(client, args, paper, rubric, action_log_path):
    action_log_content, relevant_tools = load_and_filter_actions(action_log_path, rubric["type"])

    if not action_log_content:
        action_log_content = "(no relevant action log entries found)"

    user_msg = content_prompt.format(
        paper=paper,
        rubric=rubric["criteria"],
        rubric_type=rubric["type"],
        rubric_score=rubric["score"],
        tools=", ".join(relevant_tools),
        action_log=action_log_content,
    )
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content or ""
    parsed = parse_judge_response(raw)
    return {
        "relevant_tools": relevant_tools,
        "judge_response": raw,
        "valid_score": parsed.valid_score,
        "passed": bool(parsed.score),
        "explanation": parsed.explanation,
    }


def judge_paper(paper_name, domain, action_log_path, paper_dir, output_dir, args, client):
    paper_file = os.path.join(paper_dir, f"paper.{args.paper_format}")
    if not os.path.exists(paper_file):
        logger.warning(f"Paper file not found: {paper_file}, skipping")
        return
    with open(paper_file, "r") as f:
        paper_content = f.read()

    rubrics_file = os.path.join(paper_dir, "rubrics.json")
    if not os.path.exists(rubrics_file):
        logger.warning(f"Rubrics file not found: {rubrics_file}, skipping")
        return
    with open(rubrics_file, "r") as f:
        rubrics = json.load(f)
    rubrics = [normalize_rubric(r) for r in rubrics]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "judge.json")
    if os.path.exists(output_path) and not args.overwrite:
        logger.info(f"Judge results already exist for paper: {paper_name}")
        return

    logger.info(f"Paper {paper_name} ({domain}): {len(rubrics)} rubrics")

    results = []
    total_score = 0
    earned_score = 0
    for i, rubric in enumerate(rubrics):
        logger.info(f"  [{i + 1}/{len(rubrics)}] grading [{rubric['type']}]: {rubric['criteria'][:80]}...")
        weight = rubric["score"]
        total_score += weight
        try:
            outcome = grade_rubric(client, args, paper_content, rubric, action_log_path)
        except Exception as e:
            logger.exception(f"Error grading rubric {i}: {e}")
            outcome = {
                "relevant_tools": [],
                "judge_response": "",
                "valid_score": False,
                "passed": False,
                "explanation": f"error: {e}",
            }
        if outcome["passed"]:
            earned_score += weight
        results.append({**rubric, **outcome})

    final_score = earned_score / total_score if total_score > 0 else 0.0
    summary = {
        "paper_id": paper_name,
        "domain": domain,
        "model": args.model,
        "system_name": args.system_name,
        "total_rubrics": len(rubrics),
        "total_score": total_score,
        "earned_score": earned_score,
        "final_score": final_score,
        "timestamp": datetime.now().isoformat(),
        "rubrics": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Paper {paper_name}: final score = {earned_score}/{total_score} = {final_score:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_base_dir", type=str, default="outputs")
    parser.add_argument("--papers_base_dir", type=str, default="papers")
    parser.add_argument("--system_name", type=str, required=True)
    parser.add_argument("--results_base_dir", type=str, default="results")
    parser.add_argument("--paper_format", type=str, default="md")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def aggregate_avg_scores(results_root, judge_model):
    """Collect each paper's final_score from judge.json files and write avg_score.json."""
    detailed_scores = {}
    for domain in sorted(os.listdir(results_root)):
        domain_dir = os.path.join(results_root, domain)
        if not os.path.isdir(domain_dir):
            continue
        for paper_name in sorted(os.listdir(domain_dir)):
            paper_dir = os.path.join(domain_dir, paper_name)
            if not os.path.isdir(paper_dir):
                continue
            judge_file = os.path.join(paper_dir, "judge.json")
            if not os.path.exists(judge_file):
                continue
            with open(judge_file, "r") as f:
                summary = json.load(f)
            final_score = summary.get("final_score")
            if final_score is None:
                continue
            detailed_scores[f"{domain}/{paper_name}"] = final_score

    avg_score = sum(detailed_scores.values()) / len(detailed_scores) if detailed_scores else 0.0
    aggregate = {
        judge_model: {
            "avg_score": avg_score,
            "detailed_scores": detailed_scores,
        }
    }
    output_path = os.path.join(results_root, "avg_score.json")
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    logger.info(f"Aggregated avg_score written to {output_path}: avg={avg_score:.4f} over {len(detailed_scores)} papers")


def main():
    args = parse_args()

    system_output_root = os.path.join(args.outputs_base_dir, args.system_name)
    results_root = os.path.join(args.results_base_dir, args.system_name)
    os.makedirs(results_root, exist_ok=True)

    logger.info(f"System output root: {system_output_root}")
    logger.info(f"Papers base dir: {args.papers_base_dir}")
    logger.info(f"Results dir: {results_root}")
    logger.info(f"Judge model: {args.model}")

    client = OpenAI()

    for domain in sorted(os.listdir(system_output_root)):
        domain_dir = os.path.join(system_output_root, domain)
        if not os.path.isdir(domain_dir):
            continue
        for paper_name in sorted(os.listdir(domain_dir)):
            paper_output_dir = os.path.join(domain_dir, paper_name)
            if not os.path.isdir(paper_output_dir):
                continue
            action_log_path = os.path.join(paper_output_dir, "log", "actions.json")
            if not os.path.exists(action_log_path):
                logger.warning(f"No actions.json found for {domain}/{paper_name}, skipping")
                continue

            paper_dir = os.path.join(args.papers_base_dir, domain, paper_name)
            result_dir = os.path.join(results_root, domain, paper_name)

            logger.info(f"Processing paper: {domain}/{paper_name}")
            judge_paper(paper_name, domain, action_log_path, paper_dir, result_dir, args, client)

    aggregate_avg_scores(results_root, args.model)


if __name__ == "__main__":
    main()
