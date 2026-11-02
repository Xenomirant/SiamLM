"""Optional capabilities: tools an agent can have beyond the world's own (editing the context, the task's work calls).

A capability has a `name` (its tool's name), `tool()` (an OpenAI function tool, added to the agent's tools after
the world's), `line()` (one line added to the tool list in the agent's instructions) and `call(arguments) ->
(result text, log fields)`. The world runs a call to it like an editing call (it does not count as a work call),
writes the call and its result into the context, and logs it in <name>.jsonl (siamlm/world.py capability_call).
An agent without capabilities (the default) is sent exactly what it was sent before capabilities existed.

The one capability so far is `steer` (EXPERIMENTAL; siamlm/steer.py, docs/steering.md), configured per agent under
provider.steer and available only with provider=hf.
"""
from __future__ import annotations


def make_capabilities(pcfg, backend) -> list:
    """The capabilities an agent's provider config turns on, for its backend."""
    out = []
    steer = pcfg.get("steer") or {}
    if steer.get("enabled", False):
        steering = getattr(backend, "steering", None)
        if steering is None:
            raise ValueError("the steer capability needs a local model: provider=hf with provider.steer.enabled=true "
                             f"(got provider {pcfg.get('name')})")
        from siamlm.steer import SteerCapability
        out.append(SteerCapability(steering))
    return out
