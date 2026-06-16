"""Agentic RAG loop — the LLM decides which tools to call and in what order.

Architecture:

  User query
       |
       v
  Agent loop (this file):
    1. Send query + tool definitions to LLM
    2. LLM returns either:
       a. tool_calls -> execute tool, append result, go to step 1
       b. message    -> final answer, stop
    3. Emit a trace event for every step
    4. Stop if max_steps reached (safety limit)

The trace is the key observable: every tool call, every result,
every decision the LLM made is logged with timing. This is the
"context mid-pipeline" observability story from Phase 4's spec.

Tool-calling uses Ollama's native tools API (llama3.2 supports it).
The LLM sees all previous tool results in its context window on each
iteration — this is how it "carries state between steps without
losing the original question."

Streaming: ask_agent_stream() yields TraceEvent dicts as they happen,
so the UI can show the agent's reasoning in real time.
"""

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.timing import timed_stage
from app.tools import search
from app.tools import list_notes
from app.tools import summarize

logger = logging.getLogger(__name__)

MAX_STEPS = 8   # prevent infinite loops
AGENT_TIMEOUT_S = 60.0

# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = {
    "search_notes": search,
    "list_notes": list_notes,
    "summarize_note": summarize,
}

TOOL_DEFINITIONS = [t.TOOL_DEFINITION for t in TOOLS.values()]

SYSTEM_PROMPT = """You are a helpful assistant with access to the user's personal notes. Use the provided tools to find information before answering.

Guidelines:
- Always search or list notes before answering questions about their content
- Use list_notes to discover what notes exist, then search_notes or summarize_note for details
- Use summarize_note when you need a complete picture of one note
- Use search_notes for targeted lookups across all notes
- After gathering enough information from tools, synthesize a clear, direct answer
- If tools return no relevant results, say so clearly rather than guessing
- Be specific — cite which notes your answer comes from"""


# ── Trace events ─────────────────────────────────────────────────────────────

@dataclass
class TraceEvent:
    """One observable step in the agent's reasoning chain.

    type values:
      "thinking"    — LLM is deciding what to do next (emitted before each LLM call)
      "tool_call"   — LLM decided to call a tool (name + args visible immediately)
      "tool_result" — tool executed, result available
      "answer"      — final answer from LLM
      "error"       — something went wrong
    """
    type: str
    step: int
    data: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "step": self.step,
            "duration_ms": round(self.duration_ms, 1),
            "timestamp": self.timestamp,
            **self.data,
        }


# ── Ollama tool-calling client ────────────────────────────────────────────────

def _call_ollama_with_tools(messages: list[dict]) -> dict:
    """One call to Ollama's /api/chat with tools enabled.

    Returns the raw response dict. The caller checks response["message"]
    for either a "tool_calls" list or a "content" string.

    Using /api/chat (not /api/generate) because tool-calling is only
    supported via the chat endpoint in Ollama's API.
    """
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "stream": False,
        },
        timeout=AGENT_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def _execute_tool(name: str, arguments: dict) -> tuple[str, dict]:
    """Execute a named tool with given arguments.

    Returns (formatted_result_for_llm, raw_result_dict).
    Catches all exceptions so a bad tool call doesn't kill the agent loop.
    """
    tool = TOOLS.get(name)
    if tool is None:
        error = f"Unknown tool: {name!r}. Available: {list(TOOLS.keys())}"
        return error, {"error": error}

    # Coerce known integer parameters — LLMs sometimes send "4" instead of 4
    int_params = {"top_k"}
    for param in int_params:
        if param in arguments:
            try:
                arguments[param] = int(arguments[param])
            except (TypeError, ValueError):
                arguments.pop(param)  # drop bad value, let tool use its default

    try:
        raw = tool.run(**arguments)
        formatted = tool.format_for_llm(raw)
        return formatted, raw
    except TypeError as e:
        # Wrong arguments passed by LLM
        error = f"Tool {name!r} called with invalid arguments {arguments}: {e}"
        logger.warning(error)
        return error, {"error": error}
    except Exception as e:  # noqa: BLE001
        error = f"Tool {name!r} failed: {e}"
        logger.error(error)
        return error, {"error": error}


# ── Agent loop ────────────────────────────────────────────────────────────────

def ask_agent_stream(query: str) -> Iterator[TraceEvent]:
    """Run the agent loop, yielding TraceEvents as they happen.

    This is the streaming interface used by /agent in main.py.
    Each event is small enough to send immediately via SSE, so the
    UI shows the agent's reasoning in real time rather than waiting
    for the full answer.

    Message format follows Ollama's /api/chat protocol:
      messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "tool_calls": [...]},   <- added per step
        {"role": "tool", "content": "..."},            <- tool result
        ...
      ]
    The full message history is sent on every iteration so the LLM
    has complete context of what's been tried and what was returned.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for step in range(1, MAX_STEPS + 1):
        # ── LLM call ────────────────────────────────────────────────────────
        yield TraceEvent(type="thinking", step=step, data={"message": f"Step {step}: deciding what to do..."})

        t0 = time.perf_counter()
        try:
            with timed_stage(f"agent_step_{step}"):
                response = _call_ollama_with_tools(messages)
        except Exception as e:
            yield TraceEvent(
                type="error", step=step,
                data={"message": f"LLM call failed: {e}"},
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
            return

        llm_ms = (time.perf_counter() - t0) * 1000
        message = response.get("message", {})

        # ── Tool calls ───────────────────────────────────────────────────────
        tool_calls = message.get("tool_calls") or []

        if tool_calls:
            # Add the assistant's tool-call decision to message history
            messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                arguments = fn.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                # Emit tool_call event immediately (before executing)
                yield TraceEvent(
                    type="tool_call", step=step,
                    data={"tool": tool_name, "arguments": arguments},
                    duration_ms=llm_ms,
                )

                # Execute the tool
                t1 = time.perf_counter()
                formatted, raw = _execute_tool(tool_name, arguments)
                tool_ms = (time.perf_counter() - t1) * 1000

                # Emit tool_result event
                yield TraceEvent(
                    type="tool_result", step=step,
                    data={
                        "tool": tool_name,
                        "result_preview": formatted[:300] + ("..." if len(formatted) > 300 else ""),
                        "result_full": formatted,
                        "raw": raw,
                    },
                    duration_ms=tool_ms,
                )

                # Add tool result to message history for next LLM call
                messages.append({
                    "role": "tool",
                    "content": formatted,
                })

            # Continue loop — LLM will see tool results and decide next step
            continue

        # ── Final answer ─────────────────────────────────────────────────────
        content = message.get("content", "").strip()

        if content:
            # Guard: if the model is explaining a tool failure and planning
            # to retry (common with smaller models like llama3.2:3B), it
            # sometimes returns a text explanation instead of a tool_call.
            # Detect this by checking if the content looks like a retry
            # description rather than a real answer — if so, inject a hint
            # and continue rather than surfacing the model's internal monologue.
            retry_signals = [
                "i was unable", "i will try again", "invalid argument",
                "error in the parameter", "let me try", "i'll try",
                "corrected version",
            ]
            is_retry = any(sig in content.lower() for sig in retry_signals)

            if is_retry and step < MAX_STEPS:
                # Append a gentle correction to the message history and
                # continue — the model will get another chance to call
                # the tool correctly.
                messages.append({
                    "role": "assistant",
                    "content": content,
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Please retry the tool call with corrected arguments. "
                        "Remember: top_k must be an integer (e.g. 4, not '4')."
                    ),
                })
                yield TraceEvent(
                    type="thinking", step=step,
                    data={"message": f"Step {step}: model retrying after tool error..."},
                    duration_ms=llm_ms,
                )
                continue

            yield TraceEvent(
                type="answer", step=step,
                data={
                    "answer": content,
                    "total_steps": step,
                    "tools_called": sum(
                        1 for m in messages if m.get("role") == "tool"
                    ),
                },
                duration_ms=llm_ms,
            )
            return

        # LLM returned neither tool_calls nor content — unexpected
        yield TraceEvent(
            type="error", step=step,
            data={"message": "LLM returned empty response (no tool calls, no answer)"},
        )
        return

    # Hit max steps without a final answer
    yield TraceEvent(
        type="error", step=MAX_STEPS,
        data={
            "message": f"Agent exceeded {MAX_STEPS} steps without a final answer. "
                       "Try a more specific question.",
        },
    )
