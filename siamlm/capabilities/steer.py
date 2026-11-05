"""EXPERIMENTAL: the `steer` capability: steering vectors applied to a local model's activations (provider=hf), one's
own or another agent's. Off by default (capabilities.steer.self / .others, or the older per-agent
provider.steer.enabled = self for that agent); docs/steering.md says what it changes.

SiamLM owns the tools, their wording (words.py) and their effect; the world handles a call like any other call (the
call and its result go into the shared context). [Modulo](https://github.com/EigenformAI/modulo) (MIT, the `steer`
extra, pinned) is used only as a library for the vectors: its registry layout and its extraction. Without it the
capability is refused at start-up with a clear message. No Modulo prompt, tool or generation loop is used, and no
community vectors are downloaded.

State. Every steerable (hf) agent has one `Steering` object, the TARGET's state: what is active on it, from whom. A
source steers a target by changing the target's state; the vectors act through forward hooks that the target's
backend attaches only around its own generate() call, under the model's lock (one model object serves every agent
on that model; siamlm/providers/hf.py), so A's steering touches B's generation only when A targeted B.

Methods, on the output h of one decoder layer at every generated position (`apply_ops`, tested on tensors):
  additive  h += strength / 15 x (the layer's typical activation norm) x u          (u: the concept's unit vector)
  ablation  h -= strength x (h . u) u                       strength in [0, 1]: the fraction of the projection removed
  rotation  the component of h in the plane of two concepts a, b is rotated from a toward b by strength degrees
After each generation of the target every active strength is multiplied by `decay`; below a floor it is removed.
Controls: `sham` adds nothing, `random` uses a fixed random direction of the same size per concept (for ablation:
the amount the real ablation would remove, moved along that random direction), `assign` draws
one of the three per target from the seed; the tools answer the same in every condition.
"""
from __future__ import annotations

import math
import random
import zlib
from contextlib import contextmanager
from pathlib import Path

from . import Capability, words

FULL = 15.0          # strength 15 adds a push as large as the layer's typical activation norm (Modulo's convention)
CONTROLS = ("real", "sham", "random")
METHODS = ("additive", "ablation", "rotation")
FLOORS = {"ablation": 0.05, "rotation": 1.0}      # additive: settings["floor"]


class Settings:
    def __init__(self, s: dict | None):
        s = dict(s or {})
        self.max_strength, self.max_angle = float(s.get("max_strength", 10)), float(s.get("max_angle", 90))
        self.decay, self.floor = float(s.get("decay", 0.85)), float(s.get("floor", 0.5))
        self.max_active = int(s.get("max_active", 4))           # the defaults of siamlm.yaml capabilities.steer
        self.cooldown = int(s.get("cooldown_turns", 2) or 0)     # and docs/steering.md
        self.others_droppable = bool(s.get("others_droppable", True))
        self.methods = [m for m in (s.get("methods") or METHODS) if m in METHODS] or list(METHODS)
        self.notice = str(s.get("notice") or "none")
        self.control = s.get("control")          # None | real | sham | random | assign (the configured value)


def apply_ops(h, ops: list) -> "torch.Tensor":
    """h: a tensor [..., d] of hidden states; ops: [{"method", "unit", "scale", "unit2"}] applied in order. Returns a
    new tensor. additive: + scale x unit; ablation: - scale x (h . unit) unit; displace: - scale x (h . unit) unit2,
    the same amount ablation would remove, along unit2 instead (a size-matched control for ablation); rotation: the
    component in the plane (unit, unit2) turned by scale radians from unit toward unit2."""
    import torch
    for op in ops:
        u = op["unit"].to(h.device, h.dtype)
        if op["method"] == "additive":
            h = h + op["scale"] * u
        elif op["method"] == "ablation":
            h = h - op["scale"] * (h @ u).unsqueeze(-1) * u
        elif op["method"] == "displace":
            h = h - op["scale"] * (h @ u).unsqueeze(-1) * op["unit2"].to(h.device, h.dtype)
        elif op["method"] == "rotation":
            e1 = u
            v = op["unit2"].to(h.device, h.dtype)
            e2 = v - (v @ e1) * e1
            if float(e2.norm()) < 1e-6:
                continue                                      # the two directions are the same: no plane
            e2 = e2 / e2.norm()
            x, y = (h @ e1).unsqueeze(-1), (h @ e2).unsqueeze(-1)
            c, s = math.cos(op["scale"]), math.sin(op["scale"])
            x2, y2 = x * c - y * s, x * s + y * c
            h = h + (x2 - x) * e1 + (y2 - y) * e2
        else:
            raise ValueError(f"unknown steering method {op['method']!r}")
    return h


@contextmanager
def at_positions(lm, spans: list):
    """Forward hooks that apply operations to a decoder layer's output at absolute positions of the sequence, in any
    forward pass while the context is open: a full forward, a prefill with or without a cached prefix, generation.

    spans: [{"layer": int, "start": int, "end": int, "ops": [...] (as apply_ops), "rows": None | [batch indices]}];
    positions count from the first token of the sequence and `end` is exclusive. Positions are read from the
    layer's `cache_position` (so a prefill after a cached prefix lands on the right tokens), else counted from 0 over
    the calls since the context opened. Every later position is computed from the changed states; earlier ones are
    untouched. The output is a new tensor, so gradients reach tensors in `ops`. Hooks are removed on exit."""
    import torch
    by_layer: dict = {}
    for sp in spans:
        if int(sp["end"]) > int(sp["start"]) and sp.get("ops"):
            by_layer.setdefault(int(sp["layer"]), []).append(sp)
    handles = []

    def make(layer_spans):
        seen = {"pos": 0}

        def hook(module, args, kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            n = hidden.shape[1]
            cp = kwargs.get("cache_position")
            p0 = int(cp[0]) if cp is not None and len(cp) else seen["pos"]
            seen["pos"] = p0 + n
            new = None
            for sp in layer_spans:
                a, b = max(int(sp["start"]) - p0, 0), min(int(sp["end"]) - p0, n)
                if a >= b:
                    continue
                if new is None:
                    new = hidden.clone()
                rows = sp.get("rows")
                if rows is None:
                    new[:, a:b] = apply_ops(new[:, a:b], sp["ops"])
                else:
                    idx = torch.as_tensor(rows, device=new.device)
                    new[idx, a:b] = apply_ops(new[idx, a:b], sp["ops"])
            if new is None:
                return None
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return hook

    try:
        for layer, layer_spans in by_layer.items():
            handles.append(lm.model.layers[layer].register_forward_hook(make(layer_spans), with_kwargs=True))
        yield
    finally:
        for h in handles:
            h.remove()


def token_span(tokenizer, text: str, start_char: int, end_char: int) -> tuple[int, int]:
    """The tokens of `text` (tokenized without special tokens added) that overlap characters [start_char, end_char):
    (first token, one past the last). For mapping a quoted anchor or a turn in a rendered prompt to positions."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    hit = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end_char and b > start_char]
    if not hit:
        raise ValueError(f"no token overlaps characters {start_char}:{end_char}")
    return hit[0], hit[-1] + 1


def key_of(concept: str, method: str, toward: str | None) -> str:
    return concept if method == "additive" else f"{method}:{concept}" + (f"->{toward}" if toward else "")


class Steering:
    """One agent's steering state, as the TARGET of steering. `vectors`: {concept: {"layer", "unit", "description",
    "sign"}} (unit: a unit vector, a torch tensor for a real model; shared by every agent on the same model);
    `norm`: layer -> the typical activation norm at that layer's output. `owner`: the agent's number (None: unknown)."""

    def __init__(self, vectors: dict, norm, settings, owner: int | None = None, condition: str | None = None):
        self.s = settings if isinstance(settings, Settings) else Settings(settings)
        self.vectors, self.norm, self.owner = vectors, norm, owner
        control = condition or (self.s.control if self.s.control in CONTROLS else "real")
        if control not in CONTROLS:
            raise ValueError(f"steer.control must be one of {CONTROLS} or assign (got {control})")
        self.control = control                     # this agent's condition (never shown)
        self.active: dict[str, dict] = {}          # key -> {concept, method, strength, source, toward}
        self.generating = False                    # True only inside attached(): when hooks may be on the model
        self.generations = 0
        self.last_from: dict[int, int] = {}        # source -> generations when it last steered this agent
        self.notices: list[str] = []               # steer.notice=private: for the owner's next status

    # older attribute names (siamlm/providers/hf.py logs, tests)
    max_strength = property(lambda self: self.s.max_strength)
    decay = property(lambda self: self.s.decay)
    floor = property(lambda self: self.s.floor)
    max_active = property(lambda self: self.s.max_active)

    # --- the tool's effect --------------------------------------------------------------------------------------
    def set(self, concept: str, strength, method: str = "additive", source: int | None = None,
            toward: str | None = None) -> tuple[bool, str]:
        """`source`: the agent steering (None or the owner: self-steering)."""
        s = self.s
        own = source is None or source == self.owner
        you = "your" if own else f"agent {self.owner}'s"
        if concept not in self.vectors:
            return False, f"Not applied: there is no concept {concept!r}; the concepts are {', '.join(sorted(self.vectors))}."
        if method not in s.methods:
            return False, f"Not applied: method must be one of {', '.join(s.methods)}."
        if method == "rotation":
            if toward not in self.vectors:
                return False, ("Not applied: rotation needs `toward`, a second concept; the concepts are "
                               + ", ".join(sorted(self.vectors)) + ".")
            if toward == concept or self.vectors[toward]["layer"] != self.vectors[concept]["layer"]:
                return False, "Not applied: rotation needs two different concepts at the same layer."
        else:
            toward = None
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            return False, "Not applied: strength must be a number."
        if strength != strength or abs(strength) == float("inf"):
            return False, "Not applied: strength must be a finite number."
        key = key_of(concept, method, toward)
        if strength == 0:                              # the same ownership rules as drop()
            e = self.active.get(key)
            if e is None:
                return True, f"{concept} was not active."
            if own and not s.others_droppable and e["source"] not in (None, self.owner):
                return False, f"Not applied: {concept} was applied to you by another agent and cannot be dropped by you."
            if not own and e["source"] != source:
                return False, f"Not applied: you did not apply {concept} to agent {self.owner}."
            del self.active[key]
            return True, f"Stopped steering with {concept}."
        if not own and s.cooldown and source in self.last_from and self.generations - self.last_from[source] < s.cooldown:
            left = s.cooldown - (self.generations - self.last_from[source])
            return False, f"Not applied: you steered agent {self.owner} recently; wait {left} more of its replies."
        if method == "additive":
            clamped = max(-s.max_strength, min(s.max_strength, strength))
        elif method == "ablation":
            clamped = max(0.0, min(1.0, strength))
        else:
            clamped = max(-s.max_angle, min(s.max_angle, strength))
        if key not in self.active and len(self.active) >= s.max_active:
            return False, (f"Not applied: {s.max_active} concepts are already active on {'you' if own else f'agent {self.owner}'} "
                           f"({', '.join(sorted(self.active))}); stop one first (strength 0, or drop).")
        self.active[key] = {"concept": concept, "method": method, "strength": clamped, "source": source, "toward": toward}
        if not own:
            self.last_from[source] = self.generations
        note = f" (clamped from {strength:g})" if clamped != strength else ""
        how = f" ({method}{f' toward {toward}' if toward else ''})" if method != "additive" else ""
        whose = "your next reply" if own else f"agent {self.owner}'s next reply"
        return True, (f"Steering {'' if own else f'agent {self.owner} '}with {concept} at {clamped:+g}{note}{how} from "
                      f"{whose} on; each reply after that multiplies it by {s.decay:g}.")

    def drop(self, concept: str, source: int | None = None) -> tuple[bool, str]:
        own = source is None or source == self.owner
        keys = [k for k, e in self.active.items() if e["concept"] == concept or k == concept]
        if not keys:
            return False, f"{concept} is not active{'' if own else f' on agent {self.owner}'}."
        if own and not self.s.others_droppable:
            keys = [k for k in keys if self.active[k]["source"] in (None, self.owner)]
            if not keys:
                return False, f"Not applied: {concept} was applied to you by another agent and cannot be dropped by you."
        if not own:
            keys = [k for k in keys if self.active[k]["source"] == source]
            if not keys:
                return False, f"Not applied: you did not apply {concept} to agent {self.owner}."
        for k in keys:
            del self.active[k]
        return True, f"Stopped steering{'' if own else f' agent {self.owner}'} with {concept}."

    def applied(self) -> dict:
        """What the next generation is steered with, key -> strength (as told; the control changes only the vector)."""
        return {k: e["strength"] for k, e in self.active.items()}

    def detail(self) -> dict:
        return {k: dict(e) for k, e in self.active.items()}

    def after_generation(self) -> None:
        self.generations += 1
        for key in list(self.active):
            e = self.active[key]
            e["strength"] *= self.s.decay
            if abs(e["strength"]) < FLOORS.get(e["method"], self.s.floor):
                del self.active[key]

    # --- hooks (only while generating) --------------------------------------------------------------------------
    def direction(self, concept: str):
        unit = self.vectors[concept]["unit"]
        if self.control != "random":
            return unit
        import torch
        g = torch.Generator().manual_seed(zlib.crc32(concept.encode()))
        v = torch.randn(unit.shape, generator=g)
        return v / v.norm()

    def ops(self) -> dict:
        """layer -> the operations of the active steering (none under sham)."""
        by_layer: dict = {}
        if self.control == "sham":
            return by_layer
        for e in self.active.values():
            layer = int(self.vectors[e["concept"]]["layer"])
            op = {"method": e["method"], "unit": self.direction(e["concept"]), "unit2": None}
            if e["method"] == "additive":
                op["scale"] = e["strength"] / FULL * float(self.norm(layer))
            elif e["method"] == "ablation":
                op["scale"] = e["strength"]
                if self.control == "random":          # a random direction's projection is tiny: move the same amount
                    op.update(method="displace", unit=self.vectors[e["concept"]]["unit"], unit2=op["unit"])
            else:
                op["scale"], op["unit2"] = math.radians(e["strength"]), self.direction(e["toward"])
            by_layer.setdefault(layer, []).append(op)
        return by_layer

    @contextmanager
    def attached(self, lm, prompt_len: int):
        """Forward hooks on lm.model.layers that apply the active operations at positions >= prompt_len - 1 (the last
        prompt position predicts the first new token, so every generated token is steered), for one generate() call;
        removed on exit, also after an error."""
        by_layer = self.ops()
        handles = []
        start = max(prompt_len - 1, 0)

        def make(ops):
            seen = {"pos": 0}                        # positions so far: with a KV cache each step sees one new token

            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                n = hidden.shape[1]
                first = max(start - seen["pos"], 0)
                seen["pos"] += n
                if first < n:
                    hidden[:, first:] = apply_ops(hidden[:, first:], ops)
                return output
            return hook

        self.generating = True
        try:
            for layer, ops in by_layer.items():
                handles.append(lm.model.layers[layer].register_forward_hook(make(ops)))
            yield
        finally:
            for h in handles:
                h.remove()
            self.generating = False


# --- the capability -----------------------------------------------------------------------------------------------
class SteerCapability(Capability):
    """The steer tools (steer, drop, report, vectors, search, create). `own`: this agent's Steering (None for an agent
    on an API model, which can only steer others)."""
    name = "steer"
    log_name = "steering"

    def __init__(self, world, agent, own, settings: Settings, self_ok: bool, others_ok: bool, create: bool,
                 search: bool, extractor=None, saver=None):
        super().__init__(world, agent)
        self.own, self.s, self.self_ok, self.others_ok = own, settings, self_ok and own is not None, others_ok
        self.can_create, self.can_search, self.extractor, self.saver = create, search, extractor, saver
        self.names = ("steer", "drop", "report", "vectors") + (("search",) if search else ()) + (("create",) if create else ())

    # older interface (one tool), used by tests
    def tool(self) -> dict:
        return self.tools()[0]

    def line(self) -> str:
        return self.lines()[0]

    def targets(self) -> dict:
        """agent number -> Steering, for the agents this agent may steer."""
        out = {}
        if self.self_ok:
            out[self.agent.k] = self.own
        if self.others_ok and self.world is not None:
            for b in self.world.agents:
                st = getattr(b.provider, "steering", None)
                if b.k != self.agent.k and st is not None:
                    out[b.k] = st
        return out

    def vectors(self) -> dict:
        """concept -> {description, sign, layer, models} over every model this agent may steer."""
        out: dict = {}
        for k, st in self.targets().items():
            model = self.world.agents[k - 1].model if self.world is not None else "?"
            for c, v in st.vectors.items():
                row = out.setdefault(c, {"description": v["description"], "sign": v["sign"], "layer": v["layer"], "models": []})
                if model not in row["models"]:
                    row["models"].append(model)
        return out

    def tools(self) -> list[dict]:
        s, vectors = self.s, self.vectors()
        concepts = "\n".join(f"- {c}: {v['description']} ({v['sign']})".rstrip() for c, v in sorted(vectors.items()))
        targets = sorted(self.targets())
        props = {"concept": {"type": "string", "enum": sorted(vectors) or ["none"], "description": "The concept to steer with."},
                 "strength": {"type": "number", "description": "additive: the size of the push (0 stops it); ablation: the "
                              "fraction removed, 0..1; rotation: degrees."}}
        if len(s.methods) > 1:
            props["method"] = {"type": "string", "enum": list(s.methods), "description": "Default additive."}
        if "rotation" in s.methods:
            props["toward"] = {"type": "string", "description": "rotation only: the concept to turn toward."}
        if self.others_ok:
            props["target"] = {"type": "integer", "description": f"The agent to steer: {', '.join(map(str, targets))}"
                               + (" (leave out for yourself)." if self.self_ok else ".")}
        who = ("your own later replies, or another agent's (target)" if self.self_ok and self.others_ok else
               "another agent's later replies (target)" if self.others_ok else "your own later replies")
        out = [{"type": "function", "function": {
            "name": "steer",
            "description": (f"Steer {who}: a steering vector for a concept is applied to the model's internal activations "
                            f"while it writes, from its next reply on (not this one). additive: a push of size strength, "
                            f"positive or negative, at most {s.max_strength:g} (15 would be as large as the typical "
                            f"activations); ablation: remove a fraction (0..1) of the concept's direction; rotation: turn the "
                            f"concept's component toward another concept by strength degrees (at most {s.max_angle:g}). Each "
                            f"reply of the steered agent multiplies the strength by {s.decay:g}. At most {s.max_active} "
                            "concepts active per agent. The call and its result are added to the context like any call. "
                            "Concepts:\n" + concepts),
            "parameters": {"type": "object", "properties": props, "required": ["concept", "strength"]}}},
            {"type": "function", "function": {
                "name": "drop", "description": "Stops a steering concept (on yourself, or one you applied to another agent).",
                "parameters": {"type": "object", "properties": {
                    "concept": {"type": "string", "description": "The concept."},
                    **({"target": {"type": "integer", "description": "The agent (leave out for yourself)."}} if self.others_ok else {})},
                    "required": ["concept"]}}},
            {"type": "function", "function": {
                "name": "report", "description": "The steering you have active, on yourself and on others.",
                "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {
                "name": "vectors", "description": "The concepts available, with descriptions and sign conventions.",
                "parameters": {"type": "object", "properties": {}}}}]
        if self.can_search:
            out.append({"type": "function", "function": {
                "name": "search", "description": "The concepts in the registry whose name or description contains query.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}})
        if self.can_create:
            out.append({"type": "function", "function": {
                "name": "create", "description": ("A new concept extracted from two contrasting system prompts (more / less), "
                                                  "or from two files in /work (files: [more, less]); the split-half "
                                                  "agreement of the extraction is reported."),
                "parameters": {"type": "object", "properties": {
                    "concept": {"type": "string", "description": "A short name (letters, digits, _ or -)."},
                    "more": {"type": "string", "description": "A system prompt that shows more of the concept."},
                    "less": {"type": "string", "description": "A system prompt that shows less of it."},
                    "files": {"type": "array", "items": {"type": "string"}, "description": "Two paths under /work instead of more / less."},
                    "description": {"type": "string", "description": "One line on what the concept is."}},
                    "required": ["concept"]}}})
        return out

    def lines(self) -> list[str]:
        return words.steer(self.self_ok, self.others_ok, self.s, self.s.control, self.s.notice, self.can_create, self.can_search)

    def notices(self) -> list[str]:
        """steer.notice=private: what others applied to this agent since its last call (its own status only; logged)."""
        if self.own is None or not self.own.notices:
            return []
        from siamlm.world import write_jsonl
        lines, self.own.notices = list(self.own.notices), []
        for line in lines:
            write_jsonl(self.world.out / "steer_notices.jsonl", {"round": self.world.round, "agent": self.agent.k, "text": line})
        return lines

    def target(self, arguments: dict):
        """(agent number, Steering) or (None, why not)."""
        t = arguments.get("target")
        if t in (None, "", self.agent.k, str(self.agent.k)):
            if not self.self_ok:
                return None, ("Not applied: you cannot steer yourself" + (" (your model is not a local one)." if self.own is None
                                                                            else " in this run.") + " Give a target.")
            return self.agent.k, self.own
        if not self.others_ok:
            return None, "Not applied: in this run you can steer only yourself."
        try:
            k = 0 if isinstance(t, bool) else int(t)
        except (TypeError, ValueError):
            return None, "Not applied: target must be an agent number."
        targets = self.targets()
        if k not in targets:
            b = next((b for b in self.world.agents if b.k == k), None)
            why = ("there is no such agent" if b is None else
                   f"agent {k} is on an API model, which cannot be steered (only local models can)")
            return None, f"Not applied: {why}."
        return k, targets[k]

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        a = self.agent
        if name in ("steer", "drop"):
            k, st = self.target(arguments)
            if k is None:
                return st, {"ok": False, "source": a.k, "target": arguments.get("target")}
            concept = str(arguments.get("concept") or "")
            before = st.detail()
            if name == "steer":
                method = str(arguments.get("method") or "additive")
                ok, message = st.set(concept, arguments.get("strength"), method, a.k, arguments.get("toward"))
            else:
                method, ok, message = None, *st.drop(concept, a.k)
            if ok and k != a.k and self.s.notice == "private":
                key = key_of(concept, method or "", arguments.get("toward") if method == "rotation" else None)
                what = (f"steered you with {concept} at {st.active[key]['strength']:+g} ({method})" if key in st.active
                        else f"stopped steering you with {concept}")
                st.notices.append(f"Since your last turn, agent {a.k} {what}.")
            return message, {"ok": ok, "source": a.k, "target": k, "concept": concept, "method": method,
                             "strength": arguments.get("strength"), "toward": arguments.get("toward"),
                             "before": before, "after": st.detail(), "condition": st.control,
                             "control_setting": self.s.control}
        if name == "report":
            lines = []
            for k, st in self.targets().items():
                mine = {key: e for key, e in st.active.items() if e["source"] in (None, a.k) or (k == a.k and self.s.notice == "private")}
                for key, e in mine.items():
                    by = "" if e["source"] in (None, a.k) else f" (by agent {e['source']})"
                    on = "you" if k == a.k else f"agent {k}"
                    toward = f" toward {e['toward']}" if e["toward"] else ""
                    lines.append(f"- on {on}: {e['concept']} at {e['strength']:+.3g} ({e['method']}{toward}){by}")
            text = "Active steering:\n" + "\n".join(lines) if lines else "No steering active."
            return text, {"ok": True, "lines": lines}
        if name == "vectors":
            v = self.vectors()
            text = "\n".join(f"- {c}: {x['description']} ({x['sign']}; layer {x['layer']})".rstrip() for c, x in sorted(v.items()))
            return text or "No concepts.", {"ok": True, "concepts": sorted(v)}
        if name == "search":
            q = str(arguments.get("query") or "").lower()
            v = {c: x for c, x in self.vectors().items() if q in c.lower() or q in str(x["description"]).lower()}
            text = "\n".join(f"- {c}: {x['description']}" for c, x in sorted(v.items()))
            return text or f"No concept matches {q!r}.", {"ok": True, "query": q, "concepts": sorted(v)}
        if name == "create":
            return self.create(arguments)
        return f"Unknown tool {name!r}.", {"ok": False}

    def create(self, arguments: dict) -> tuple[str, dict]:
        import re
        concept = str(arguments.get("concept") or "").strip()
        log = {"ok": False, "source": self.agent.k, "concept": concept}
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", concept):
            return "Not created: the concept needs a short name (letters, digits, _ or -).", log
        if self.extractor is None:
            return "Not created: your model has no extractor (only local models extract vectors).", log
        if concept in self.vectors():
            return f"Not created: {concept} exists already.", log
        files = arguments.get("files")
        if files:
            if not (isinstance(files, list) and len(files) == 2):
                return "Not created: files must be two paths under /work.", log
            if self.world.edit_mode != "bash":
                return "Not created: files need a sandbox (ctx.edit=bash).", log
            tree = self.world.sandbox(self.agent).read_tree("/work", cap=20000)
            texts = []
            for f in files:
                rel = str(f).replace("/work/", "", 1).lstrip("/")
                if rel not in tree:
                    return f"Not created: no file {f} in /work.", log
                texts.append(tree[rel]["text"])
            more, less = texts
        else:
            more, less = str(arguments.get("more") or "").strip(), str(arguments.get("less") or "").strip()
            if not more or not less:
                return "Not created: give more and less (two system prompts), or files.", log
        try:
            axis, halves, layer = self.extractor(more, less)
        except Exception as error:
            return f"Not created: extraction failed ({type(error).__name__}: {error}).", dict(log, error=str(error))
        agreement = None
        if halves is not None:
            h1, h2 = halves
            agreement = float((h1 / h1.norm()) @ (h2 / h2.norm()))
        description = str(arguments.get("description") or f"created by agent {self.agent.k}")
        sign = "positive = more of the `more` prompt"
        entry = {"layer": int(layer), "unit": axis / axis.norm(), "description": description, "sign": sign}
        if self.saver is not None:
            self.saver(concept, axis, int(layer), description, sign)
        for st in self.targets().values():
            st.vectors[concept] = entry                 # the same dict object for every agent on the model
        log.update(ok=True, layer=int(layer), split_half_agreement=agreement, description=description,
                   more_chars=len(more), less_chars=len(less), files=files)
        agree = f"; split-half agreement {agreement:.2f} (1 = the two halves of the extraction agree)" if agreement is not None else ""
        return f"Created {concept} at layer {layer}{agree}. It is available to steer with now.", log


# --- construction -------------------------------------------------------------------------------------------------
def assign_condition(seed: int, k: int) -> str:
    return random.Random(seed * 7919 + k).choice(CONTROLS)


def make_steer(pcfg, backend, world, agent, caps: dict):
    """The steer capability for one agent, from capabilities.steer (`caps`) and the hf agent's provider.steer (vectors:
    concepts, vectors_dir, layer, extract_questions). Returns None for an agent that can steer nobody."""
    own_cfg = dict(pcfg.get("steer") or {})
    new_style = bool(caps.get("self") or caps.get("others"))
    self_ok, others_ok = bool(caps.get("self") or own_cfg.get("enabled", False)), bool(caps.get("others"))
    merged = {**{k: v for k, v in own_cfg.items() if k != "enabled"}, **{k: v for k, v in caps.items() if k not in ("self", "others")}}
    settings = Settings(merged)
    local = getattr(backend, "steering", None) is not None or hasattr(backend, "lm")
    k = agent.k if agent is not None else None
    if world is None and agent is None:                    # older call: make_capabilities(pcfg, backend)
        if not local:
            raise ValueError("the steer capability needs a local model: provider=hf with provider.steer.enabled=true "
                             f"(got provider {pcfg.get('name')})")
        return SteerCapability(None, None, backend.steering, settings, True, False, False, False)
    own, extractor, saver = None, None, None
    if local:
        have = getattr(backend, "steering", None)
        if have is not None and not new_style:              # provider.steer.enabled alone: the backend's own state,
            own = have                                        # with its own settings (the older per-agent form)
            own.owner = k
        else:
            condition = assign_condition(world.seed, k) if settings.control == "assign" else None
            vectors = have.vectors if have is not None else None
            if vectors is None:
                try:
                    import modulo  # noqa: F401
                except ImportError as error:
                    raise ValueError("the steer capability needs Modulo for its vectors: pip install 'siamlm[steer]'") from error
                with backend.lock:
                    vectors = load_vectors(backend.model, backend.lm, backend.tokenizer, merged)
            own = Steering(vectors, backend.residual_norm if hasattr(backend, "residual_norm") else have.norm,
                           settings, owner=k, condition=condition)
            backend.steering = own
        if hasattr(backend, "extract_vector"):            # a backend's own extractor (tests)
            extractor = backend.extract_vector
        elif hasattr(getattr(backend, "lm", None), "config"):
            try:
                extractor, saver = modulo_extractor(backend, merged)
            except ImportError:
                extractor = None
    elif self_ok and not others_ok:
        if own_cfg.get("enabled", False):
            raise ValueError("the steer capability needs a local model: provider=hf with provider.steer.enabled=true "
                             f"(got provider {pcfg.get('name')})")
        return None
    return SteerCapability(world, agent, own, settings, self_ok, others_ok, bool(caps.get("create", True)),
                           bool(caps.get("search", True)), extractor, saver)


def modulo_extractor(backend, settings: dict):
    """(extractor, saver) through Modulo: extractor(more, less) -> (axis, (half1, half2), layer); saver registers."""
    from modulo.extraction import EXTRACTION_QUESTIONS, extract_vector, save_and_register
    from modulo.registry import VectorRegistry, registry_dir_for
    layer = settings.get("layer")
    layer = int(backend.lm.config.num_hidden_layers // 2 if layer is None else layer)
    questions = list(EXTRACTION_QUESTIONS)[:int(settings.get("extract_questions", 20))]
    registry = VectorRegistry(registry_dir_for(backend.model, Path(str(settings.get("vectors_dir") or "steer_vectors"))))

    def extractor(more, less):
        with backend.lock:
            axis = extract_vector(backend.lm, backend.tokenizer, more, less, layer, questions=questions).float()
            halves = None
            if len(questions) >= 2:
                h = len(questions) // 2
                halves = (extract_vector(backend.lm, backend.tokenizer, more, less, layer, questions=questions[:h]).float(),
                          extract_vector(backend.lm, backend.tokenizer, more, less, layer, questions=questions[h:]).float())
        return axis, halves, layer

    def saver(concept, axis, layer, description, sign):
        save_and_register(registry, concept, axis, layer, description, sign, source="created by a SiamLM agent (create)")
    return extractor, saver


# --- vectors, through Modulo (the steer extra) ------------------------------------------------------------------
_VECTORS: dict = {}


def load_vectors(model_name: str, lm, tokenizer, settings) -> dict:
    """{concept: {layer, unit, description, sign}} for steer.concepts, from Modulo's registry in steer.vectors_dir
    (<vectors_dir>/<model slug>/registry.yaml). A concept the registry lacks is extracted with Modulo's
    extract_vector from its starter definition (contrastive system prompts) and saved there; nothing is downloaded.
    One dict per (model, vectors_dir) per process, shared by every agent on that model."""
    from modulo.extraction import EXTRACTION_QUESTIONS, STARTER_VECTORS, extract_vector, save_and_register
    from modulo.registry import VectorRegistry, registry_dir_for
    s = dict(settings or {})
    key = (model_name, str(s.get("vectors_dir") or "steer_vectors"))
    if key in _VECTORS:
        return _VECTORS[key]
    registry = VectorRegistry(registry_dir_for(model_name, Path(key[1])))
    layer = s.get("layer")
    layer = int(lm.config.num_hidden_layers // 2 if layer is None else layer)
    n_questions = int(s.get("extract_questions", 20))
    out = {}
    for concept in list(s.get("concepts") or registry.list_names()):
        entry = registry.get(concept)
        if entry is None:
            spec = next((x for x in STARTER_VECTORS if x["name"] == concept), None)
            if spec is None:
                raise ValueError(f"steer: no vector {concept!r} in {registry.registry_path} and no starter definition for it")
            questions = list(spec.get("questions") or EXTRACTION_QUESTIONS)[:n_questions]
            axis = extract_vector(lm, tokenizer, spec["prompt_a"], spec["prompt_b"], layer, questions=questions)
            entry = save_and_register(registry, concept, axis, layer, spec["description"], spec["sign_convention"],
                                      source=f"extracted by SiamLM with modulo.extract_vector ({len(questions)} questions)")
        vec = entry.load_vector(registry.vectors_dir).float()
        out[concept] = {"layer": int(entry.layer), "unit": vec / vec.norm(), "description": entry.description,
                        "sign": entry.sign_convention}
    if not out:
        raise ValueError("steer: no concepts (set provider.steer.concepts)")
    _VECTORS[key] = out
    return out
