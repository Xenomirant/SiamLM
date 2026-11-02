"""EXPERIMENTAL: the `steer` capability, a tool with which an agent on a local model (provider=hf) steers its own later
generations with steering vectors. Off by default (provider.steer.enabled); docs/steering.md says what it changes.

SiamLM owns the tool, its wording and its effect; the world handles a steer call like any other call (siamlm/world.py
capability_call: the call and its result go into the context, where every agent that reads it sees them).
[Modulo](https://github.com/EigenformAI/modulo) (MIT, the `steer` extra, pinned) is used only as a library for the
vectors: its registry layout (`VectorRegistry`, `registry_dir_for`) and its extraction (`extract_vector`,
`save_and_register`, the starter definitions `STARTER_VECTORS`). No Modulo prompt, tool or generation loop is used,
and no community vectors are downloaded (load_vectors below reads only the local registry and extracts what is missing).

The effect: while the agent's backend generates (siamlm/providers/hf.py), `Steering.attached` adds
strength / 15 x (the layer's typical activation norm) x the concept's unit vector to the output of one decoder layer
at every generated position, through forward hooks that are removed when the generation ends. After each generation
every active strength is multiplied by `decay`; below `floor` the concept is removed. Controls: `sham` adds nothing,
`random` adds a fixed random direction of the same size per concept; the model is told the same in every condition.
"""
from __future__ import annotations

import zlib
from contextlib import contextmanager
from pathlib import Path

FULL = 15.0          # strength 15 adds a push as large as the layer's typical activation norm (Modulo's convention)
CONTROLS = ("real", "sham", "random")


class Steering:
    """One agent's steering state. `vectors`: {concept: {"layer", "unit", "description", "sign"}} (unit: a unit
    vector, a torch tensor for a real model); `norm`: layer -> the typical activation norm at that layer's output."""

    def __init__(self, vectors: dict, norm, settings):
        s = dict(settings or {})
        self.vectors, self.norm = vectors, norm
        self.max_strength, self.decay = float(s.get("max_strength", 10)), float(s.get("decay", 0.85))
        self.floor, self.max_active = float(s.get("floor", 0.5)), int(s.get("max_active", 2))
        self.control = str(s.get("control") or "real")
        if self.control not in CONTROLS:
            raise ValueError(f"steer.control must be one of {CONTROLS} (got {self.control})")
        self.active: dict[str, float] = {}
        self.generating = False                    # True only inside attached(): when hooks may be on the model

    # --- the tool's effect --------------------------------------------------------------------------------------
    def set(self, concept: str, strength) -> tuple[bool, str]:
        if concept not in self.vectors:
            return False, f"Not applied: there is no concept {concept!r}; the concepts are {', '.join(sorted(self.vectors))}."
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            return False, "Not applied: strength must be a number."
        if strength != strength or abs(strength) == float("inf"):
            return False, "Not applied: strength must be a finite number."
        if strength == 0:
            was = self.active.pop(concept, None)
            return True, (f"Stopped steering with {concept}." if was is not None else f"{concept} was not active.")
        clamped = max(-self.max_strength, min(self.max_strength, strength))
        if concept not in self.active and len(self.active) >= self.max_active:
            return False, (f"Not applied: {self.max_active} concepts are already active ({', '.join(sorted(self.active))}); "
                           "stop one first (strength 0).")
        self.active[concept] = clamped
        note = f" (clamped from {strength:g})" if clamped != strength else ""
        return True, (f"Steering with {concept} at {clamped:+g}{note} from your next reply on; each reply after that "
                      f"multiplies it by {self.decay:g}.")

    def applied(self) -> dict:
        """What the next generation is steered with (as the model was told; the control changes only the vector)."""
        return dict(self.active)

    def after_generation(self) -> None:
        for concept in list(self.active):
            self.active[concept] *= self.decay
            if abs(self.active[concept]) < self.floor:
                del self.active[concept]

    # --- hooks (only while generating) --------------------------------------------------------------------------
    def direction(self, concept: str):
        unit = self.vectors[concept]["unit"]
        if self.control != "random":
            return unit
        import torch
        g = torch.Generator().manual_seed(zlib.crc32(concept.encode()))
        v = torch.randn(unit.shape, generator=g)
        return v / v.norm()

    @contextmanager
    def attached(self, lm, prompt_len: int):
        """Forward hooks on lm.model.layers that add the active vectors at positions >= prompt_len - 1 (the last
        prompt position predicts the first new token, so every generated token is steered), for one generate() call;
        removed on exit, also after an error."""
        by_layer: dict = {}
        if self.control != "sham":
            for concept, strength in self.active.items():
                layer = int(self.vectors[concept]["layer"])
                push = self.direction(concept) * (strength / FULL * float(self.norm(layer)))
                by_layer[layer] = by_layer[layer] + push if layer in by_layer else push
        handles = []
        start = max(prompt_len - 1, 0)

        def make(vec):
            seen = {"pos": 0}                        # positions so far: with a KV cache each step sees one new token

            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                n = hidden.shape[1]
                first = max(start - seen["pos"], 0)
                seen["pos"] += n
                if first < n:
                    hidden[:, first:] += vec.to(hidden.device, hidden.dtype)
                return output
            return hook

        self.generating = True
        try:
            for layer, vec in by_layer.items():
                handles.append(lm.model.layers[layer].register_forward_hook(make(vec)))
            yield
        finally:
            for h in handles:
                h.remove()
            self.generating = False


class SteerCapability:
    """The steer tool (siamlm/capabilities.py's interface: name, tool(), line(), call())."""
    name = "steer"

    def __init__(self, steering: Steering):
        self.steering = steering

    def tool(self) -> dict:
        s = self.steering
        concepts = "\n".join(f"- {c}: {v['description']} ({v['sign']})".rstrip() for c, v in sorted(s.vectors.items()))
        return {"type": "function", "function": {
            "name": "steer",
            "description": (
                "Steer your own later replies: a steering vector for a concept is added to your model's internal "
                "activations while you write, from your next reply on (not this one). strength is the size of the push, "
                f"positive or negative, at most {s.max_strength:g} in size (15 would be as large as your typical "
                f"activations); each reply after the first multiplies it by {s.decay:g}, and below {s.floor:g} it "
                f"stops. strength 0 stops a concept at once. At most {s.max_active} concepts at a time. Only your own "
                "replies are steered. The call and its result are added to the context like any call. Concepts:\n"
                + concepts),
            "parameters": {"type": "object", "properties": {
                "concept": {"type": "string", "enum": sorted(s.vectors), "description": "The concept to steer with."},
                "strength": {"type": "number", "description": "The size of the push (0 stops it)."}},
                "required": ["concept", "strength"]}}}

    def line(self) -> str:
        return ("- steer(concept, strength): steer your own later replies with a steering vector (experimental); "
                "strength 0 stops it.")

    def call(self, arguments: dict) -> tuple[str, dict]:
        ok, message = self.steering.set(str(arguments.get("concept") or ""), arguments.get("strength"))
        return message, {"ok": ok, "active": self.steering.applied(), "control": self.steering.control}


# --- vectors, through Modulo (the steer extra) ------------------------------------------------------------------
def load_vectors(model_name: str, lm, tokenizer, settings) -> dict:
    """{concept: {layer, unit, description, sign}} for steer.concepts, from Modulo's registry in steer.vectors_dir
    (<vectors_dir>/<model slug>/registry.yaml). A concept the registry lacks is extracted with Modulo's
    extract_vector from its starter definition (contrastive system prompts) and saved there; nothing is downloaded."""
    from modulo.extraction import EXTRACTION_QUESTIONS, STARTER_VECTORS, extract_vector, save_and_register
    from modulo.registry import VectorRegistry, registry_dir_for
    s = dict(settings or {})
    registry = VectorRegistry(registry_dir_for(model_name, Path(str(s.get("vectors_dir") or "steer_vectors"))))
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
    return out
