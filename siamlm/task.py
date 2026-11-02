"""The Task API: what a world needs from an environment (a task family), so that the shared-context world (scheduling,
the context, editing, the cap, logging) does not depend on any one benchmark.

Terms (Harbor's in brackets, see docs/harbor.md):
- an **item** [task]: one unit of work with a statement, given to one agent at a time; each agent has its own
  **stream** of items, worked in order;
- a **work call**: a tool call on the current item. A *try* call (`try_name`, e.g. LCB's run_tests) gets feedback and
  the item stays open; the *final* call (`final_name`, e.g. submit) is graded and closes the item. After
  `world.max_work_calls` work calls, or `world.max_steps_per_problem` steps, the item is closed with the latest attempt
  (`forced`). A task without work calls (`open`) has messages only;
- **grading** [verifier]: `grade` returns an Outcome; `success` is the item's pass/fail and `log["reward"]`, when
  given, its numeric reward (otherwise the reward is 1.0 / 0.0 from success); both go to results.jsonl;
- **per-item files**: `files(item)` are put in /work/env/<slug>/ of the agent's sandbox while the item is open
  (state_chain); `sandbox(item)` instead gives the item a container of its own, where that agent's bash commands run
  while the item is open (Harbor tasks, siamlm/tasks/harbor.py).

The world adds everything about the context (sharing, the limit, editing, the other agents). Everything the model
reads about the task comes from the methods and words below, so a new environment is one subclass in siamlm/tasks/
(the methods marked "words" only phrase the instructions; copy LcbTask's or StateChainTask's and adapt). The
existing families: lcb (LiveCodeBench), codeeval_pro, open (the task dial), state_chain (generated shell tasks) and
harbor (Harbor-format task directories). siamlm/run.py's `build_task` maps `task.name` to one of them.

For tasks with one action per step and an environment observation after each (text games), `attempt` is the move
and its observation, `terminal` ends the item, and `final_name` may be None (no separate final call).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Outcome:
    """The answer to one work call: the text added to the context, whether the item is now closed, whether it was
    solved (None while open), and fields for the logs (results.jsonl / actions.jsonl; `reward`: a number)."""
    text: str
    terminal: bool = False
    success: bool | None = None
    log: dict = field(default_factory=dict)


class Task:
    name = "task"
    try_name: str | None = None          # a work call with feedback (item stays open)
    final_name: str | None = None        # the graded call that closes the item
    arg_name = "code"                    # the attempt's argument in a work call (the call is logged as {arg_name: payload})
    needs_shell = False                  # True: the task needs ctx.edit=bash (its items are worked with shell commands)

    # words used in the instructions and the status line (see LcbTask for what each one reads like)
    items = "problems"                   # "It holds <items>, each starting with ..." (pool setup)
    tool_list = ""                       # "Call one tool: <tool_list> or edit_context."
    work_or = ""                         # "(at most 3 calls before each <work_or>)"
    work_and = ""                        # "<work_and> are not carried out until it fits again"
    left_words = ""                      # "<left_words> left for it: 3."
    runs_word = "runs"                   # "you have used your 8 <runs_word> on ..."
    last_try_note = ""                   # added to the result of the last allowed try call

    @property
    def work_names(self) -> tuple:
        return tuple(n for n in (self.try_name, self.final_name) if n)

    # --- items --------------------------------------------------------------------------------------------------
    def statement(self, item: dict) -> str:
        """The item's statement, as the agent reads it."""
        raise NotImplementedError

    def meta(self, item: dict) -> dict:
        """Item fields for results.jsonl."""
        return {}

    # --- instructions (words) -----------------------------------------------------------------------------------
    def preamble(self) -> str:
        """The first lines of every agent's instructions."""
        raise NotImplementedError

    def series(self, n: int) -> str:
        """'You solve a series of n ...'."""
        raise NotImplementedError

    def record_words(self, view: str) -> str:
        """What is added to the context, for the instructions; view = shared | alone | chat."""
        raise NotImplementedError

    def tool_lines(self, setup: str) -> list[str]:
        """One line per work tool, for the instructions' tool list."""
        raise NotImplementedError

    def limit_line(self, T: int, pool: bool) -> str:
        """The instructions' line about how many work calls an item allows ('' for none)."""
        raise NotImplementedError

    def work_tools(self, pool: bool) -> list[dict]:
        """The work tools' schemas (OpenAI function tools)."""
        raise NotImplementedError

    # --- work calls ---------------------------------------------------------------------------------------------
    def payload(self, arguments: dict) -> str:
        """The attempt in a work call (LCB: the program)."""
        raise NotImplementedError

    def shown(self, words: str, name: str, payload: str) -> str:
        """How a work call is written into the context."""
        raise NotImplementedError

    def attempt(self, item: dict, payload: str) -> Outcome:
        """A try call."""
        raise NotImplementedError

    def grade(self, item: dict, payload: str) -> Outcome:
        """The final grading of an attempt (a final call, or a forced close). It closes the item."""
        raise NotImplementedError

    def verdict(self, success: bool, title: str | None = None) -> str:
        """What the agent is told after grading (world.feedback_on_submit=bit)."""
        raise NotImplementedError

    def forced_notice(self, position: int, how: str) -> str:
        """The notice when the harness closes an item (work or step limit)."""
        raise NotImplementedError

    # --- per-item environment -----------------------------------------------------------------------------------
    def files(self, item: dict) -> dict:
        """Files the item needs in its agent's sandbox ({relative path: text}; bash editing only). The world puts
        them in /work/env/<slug>/ while the item is open and removes them when it closes. Default: none."""
        return {}

    def sandbox(self, item: dict):
        """The container this item's bash commands run in (a siamlm.context.sandbox.Sandbox with /ctx mounted, so the
        context file is there too), or None for the agent's own sandbox (ctx.bash). The task starts it on first use
        and stops it when the item is graded. Default: None."""
        return None

    def close(self) -> None:
        pass
