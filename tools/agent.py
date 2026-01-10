"""Agent loop for MLE tasks."""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from llm import LLM
from tools import Tool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert ML engineer debugging ML pipelines.
You have access to tools to read files, write files, list directories, run Python scripts, and execute bash commands.

When you have completed the task and have your final answer, respond with:
FINAL ANSWER: <your answer>

Your answer should also include all files changed in full, with explanations of the changes made. e.g.
"{
  'file1.py': '<full contents of file1.py>',
  'file2.py': '<full contents of file2.py>'
}"

Be systematic: read the code, understand the issue, write fixes, and verify they work."""


class Agent:
    def __init__(
        self,
        tools: Dict[str, Tool],
        llm: LLM,
        max_turns: int = 50,
        log_context: Optional[Dict[str, Any]] = None,
    ):
        self.tools = tools
        self.llm = llm
        self.max_turns = max_turns
        self.log_context = dict(log_context or {})
        self.log_events: List[Dict[str, Any]] = []

    def _get_tool_defs(self):
        return [
            t.get_tool_json(provider=self.llm.provider) for t in self.tools.values()
        ]

    def _ensure_serializable(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: self._ensure_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._ensure_serializable(v) for v in value]
        return repr(value)

    def _snapshot_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        try:
            return json.loads(json.dumps(messages))
        except (TypeError, ValueError):
            return [self._ensure_serializable(m) for m in messages]

    def _log_event(self, event: str, **payload: Any) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            **{k: self._ensure_serializable(v) for k, v in payload.items()},
        }
        self.log_events.append(entry)

    def _finalize_run(
        self,
        run_id: str,
        status: str,
        metadata: Dict[str, Any],
        messages: List[Dict[str, Any]],
        answer: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self._log_event(
            "conversation_snapshot",
            run_id=run_id,
            messages=self._snapshot_messages(messages),
        )
        payload: Dict[str, Any] = {
            "run_id": run_id,
            "status": status,
            "metadata": metadata,
        }
        if answer is not None:
            payload["answer"] = answer
        if error is not None:
            payload["error"] = error
        self._log_event("run_end", **payload)

    async def run(
        self, task_prompt: str
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]
        metadata = {"turns": 0, "tool_calls": 0, "errors": []}
        self.log_events = []
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        context = dict(self.log_context)
        context.setdefault("run_id", run_id)

        self._log_event(
            "run_start",
            run_id=run_id,
            context=context,
            system_prompt=SYSTEM_PROMPT,
            task_prompt=task_prompt,
        )

        for turn in range(self.max_turns):
            metadata["turns"] = turn + 1
            logger.info(f"[Turn {turn + 1}]")

            self._log_event(
                "model_prompt",
                run_id=run_id,
                turn=turn + 1,
                messages=self._snapshot_messages(messages),
            )

            try:
                response = await self.llm.chat(messages, tools=self._get_tool_defs())
            except Exception as e:
                metadata["errors"].append(str(e))
                logger.error(f"LLM error: {e}")
                error_message = f"Error: {e}"
                self._log_event(
                    "error",
                    run_id=run_id,
                    turn=turn + 1,
                    error=str(e),
                )
                self._finalize_run(
                    run_id,
                    status="error",
                    metadata=metadata,
                    messages=messages,
                    error=error_message,
                )
                return error_message, metadata, self.log_events

            tool_calls = self.llm.get_tool_calls(response)
            text = self.llm.get_text(response)

            self.llm.append_assistant(messages, response)

            self._log_event(
                "model_response",
                run_id=run_id,
                turn=turn + 1,
                text=text,
                tool_calls=tool_calls,
            )

            if tool_calls:
                for tc in tool_calls:
                    metadata["tool_calls"] += 1
                    tool_name, args, tc_id = tc["name"], tc["arguments"], tc["id"]
                    logger.info(f"  Tool: {tool_name}({args})")

                    if tool_name not in self.tools:
                        result = f"Unknown tool: {tool_name}"
                    else:
                        result_data = await self.tools[tool_name](args)
                        result = result_data["result"]
                        if len(result) > 10000:
                            result = result[:10000] + "\n... [truncated]"

                    self._log_event(
                        "tool_result",
                        run_id=run_id,
                        turn=turn + 1,
                        tool_name=tool_name,
                        arguments=args,
                        result=result,
                    )
                    self.llm.append_tool_result(messages, tc_id, result)
            elif text:
                logger.info(f"  Response: {text[:200]}...")
                if re.search(r"FINAL ANSWER:", text, re.IGNORECASE):
                    match = re.search(
                        r"FINAL ANSWER:\s*(.*)", text, re.DOTALL | re.IGNORECASE
                    )
                    answer = match.group(1).strip() if match else text
                    self._finalize_run(
                        run_id,
                        status="completed",
                        metadata=metadata,
                        messages=messages,
                        answer=answer,
                    )
                    return answer, metadata, self.log_events

        final_message = "Max turns reached without final answer."
        self._finalize_run(
            run_id,
            status="max_turns",
            metadata=metadata,
            messages=messages,
            answer=final_message,
        )
        return final_message, metadata, self.log_events
