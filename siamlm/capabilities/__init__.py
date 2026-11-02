"""Capabilities: tools an agent may have beyond the world's own (editing the context, the task's work calls).

Three layers (docs/capabilities.md). The KERNEL (grading, limits, holds and ends, logging, the sandbox, controls) is
never changed by an agent at run time. CAPABILITIES are tools, each opt-in and off by default (the `capabilities:`
block of siamlm/configs/siamlm.yaml), each logged, and each may change only (a) a context, its own or others', or
(b) activations, its own or others'. There is no capability that changes the harness.

A capability is an object with `name` (its log file: <name>.jsonl), `names` (the tool names it answers), `tools()`
(OpenAI function tools, added after the world's), `lines()` (lines added to the tool list of the agent's instructions;
all wording in siamlm/capabilities/words.py) and `call(name, arguments) -> (result text, log fields)`. The world runs
a call to it like an editing call (it does not count as a work call), writes the call and its result into the
context and logs it (siamlm/world.py capability_call). An agent without capabilities (the default) is sent exactly
what it was sent before capabilities existed.

The capabilities: pin / unpin (pin.py), fork (fork.py) and steer (steer.py, EXPERIMENTAL; docs/steering.md): things
an agent cannot do for itself (a cost rule in the counting, a model call on an uncommitted context, activations).
The hooks the world calls are the module functions below; every one of them does nothing for an agent without that
capability.
"""
from __future__ import annotations


class Capability:
    name = ""
    names: tuple = ()
    log_name = ""                 # <log_name>.jsonl (default: <name>.jsonl)

    def __init__(self, world, agent):
        self.world, self.agent = world, agent
        if not self.names:
            self.names = (self.name,)
        self.log_name = self.log_name or self.name

    def tools(self) -> list[dict]:
        return []

    def lines(self) -> list[str]:
        return []

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        raise NotImplementedError


def settings(world) -> dict:
    caps = world.cfg.get("capabilities") if world is not None else None
    if not caps:
        return {}
    from omegaconf import OmegaConf
    return OmegaConf.to_container(caps, resolve=True) if not isinstance(caps, dict) else dict(caps)


def make_capabilities(pcfg, backend, world=None, agent=None) -> list:
    """The capabilities an agent has: the world's `capabilities:` block, plus the older per-agent switch
    provider.steer.enabled (= capabilities.steer.self for that agent)."""
    out = []
    caps = settings(world)
    steer = dict(caps.get("steer") or {})
    own = dict(pcfg.get("steer") or {})
    if world is not None and agent is not None:
        if caps.get("pin"):
            from .pin import Pin
            out.append(Pin(world, agent))
        if caps.get("fork"):
            from .fork import Fork
            out.append(Fork(world, agent))
    if steer.get("self") or steer.get("others") or own.get("enabled", False):
        from .steer import make_steer
        cap = make_steer(pcfg, backend, world, agent, steer)
        if cap is not None:
            out.append(cap)
    return out


# --- the hooks the world calls (siamlm/world.py) --------------------------------------------------------------
def find(agent, name: str):
    return next((c for c in agent.capabilities if name in c.names), None)


def tool_schemas(agent) -> list[dict]:
    return [t for c in agent.capabilities for t in c.tools()]


def tool_lines(agent) -> list[str]:
    return [l for c in agent.capabilities for l in c.lines()]


def status_lines(world, agent) -> list[str]:
    """Private lines for the agent's next status (steer.notice=private); [] otherwise."""
    cap = find(agent, "steer")
    return cap.notices() if cap is not None else []


def pins_removed(world, removed: list, remover) -> None:
    """A turn with a pin is gone from the file: logged with its pinner and its remover (None: the harness)."""
    if not getattr(world, "pin_cost", None):
        return
    from .pin import log_removed
    log_removed(world, removed, remover)


def pin_extra_chars(world, turns) -> int:
    """The extra size pinned turns count: (pin_cost - 1) x their text, in characters."""
    cost = getattr(world, "pin_cost", None)
    if not cost:
        return 0
    return int(sum(len(t.text) * (cost - 1) for t in turns if t.pinned is not None))
