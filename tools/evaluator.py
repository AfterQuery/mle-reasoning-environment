"""Evaluator for MLE tasks using rubric-based scoring."""

import json
import os
import re
from typing import Any, Dict, List

from litellm import acompletion


class RubricEvaluator:
    def __init__(self, judge_model: str = "openai/gpt-4o-mini"):
        self.judge_model = judge_model

    def parse_rubric_md(self, rubric_text: str) -> List[Dict]:
        """Parse pipe-delimited rubric.md format: +5|TYPE|criterion|justification"""
        criteria = []
        for line in rubric_text.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith(("+", "-")):
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                points = (
                    int(re.search(r"[+-]?\d+", parts[0]).group())
                    if re.search(r"[+-]?\d+", parts[0])
                    else 0
                )
                criteria.append(
                    {
                        "points": points,
                        "type": parts[1].strip(),
                        "criterion": parts[2].strip(),
                        "justification": parts[3].strip() if len(parts) > 3 else "",
                    }
                )
        return criteria

    async def evaluate_criterion(
        self, criterion: Dict, task_dir: str, agent_answer: str
    ) -> Dict:
        """Evaluate a single criterion."""
        crit_type = criterion["type"]
        crit_text = criterion["criterion"]

        # Handle NO_FILE checks directly
        if crit_type == "`NO_FILE`" or "exists" in crit_text.lower():
            # Extract path from criterion - look for path after "at" or in backticks containing /
            # First try to find a path with slashes (more specific)
            path_match = re.search(r"`([^`]*[/\\][^`]*)`", crit_text)
            if not path_match:
                # Fall back to "at <path>" pattern
                path_match = re.search(r"at\s+`?([^\s`]+)`?", crit_text)
            if path_match:
                rel_path = path_match.group(1).strip()
                # Remove common prefixes from rubric paths
                for prefix in ["../tasking/", "../grading/", "./"]:
                    if rel_path.startswith(prefix):
                        rel_path = rel_path[len(prefix) :]
                        break
                full_path = os.path.join(task_dir, rel_path)
                exists = os.path.exists(full_path)
                return {
                    "criterion": crit_text,
                    "points": criterion["points"],
                    "passed": exists,
                    "reasoning": f"File {'exists' if exists else 'does not exist'} at {full_path}",
                }

        # Use LLM judge for other criteria
        prompt = f"""Evaluate if the agent's work meets this criterion.

CRITERION: {crit_text}

AGENT'S ANSWER/OUTPUT:
{agent_answer[:8000]}

Respond with JSON only:
{{"passed": true/false, "reasoning": "brief explanation"}}"""

        try:
            response = await acompletion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            # Clean markdown
            if text.startswith("```"):
                text = re.sub(r"```\w*\n?", "", text).strip()
            result = json.loads(text)
            return {
                "criterion": crit_text,
                "points": criterion["points"],
                "passed": result.get("passed", False),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            return {
                "criterion": crit_text,
                "points": criterion["points"],
                "passed": False,
                "reasoning": f"Evaluation error: {e}",
            }

    async def evaluate(
        self, rubric_text: str, task_dir: str, agent_answer: str
    ) -> Dict[str, Any]:
        """Evaluate agent against all rubric criteria."""
        criteria = self.parse_rubric_md(rubric_text)
        results = []
        total_possible = sum(max(0, c["points"]) for c in criteria)
        earned = 0

        for crit in criteria:
            result = await self.evaluate_criterion(crit, task_dir, agent_answer)
            results.append(result)
            if result["passed"] and result["points"] > 0:
                earned += result["points"]
            elif not result["passed"] and result["points"] < 0:
                earned += result["points"]  # penalties

        return {
            "score": earned,
            "total_possible": total_possible,
            "percentage": (earned / total_possible * 100) if total_possible > 0 else 0,
            "results": results,
        }
