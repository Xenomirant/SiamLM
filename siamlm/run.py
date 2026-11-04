"""Run siamese worlds: N agents sharing one capped context, each working on its own stream of task items.

    python -m siamlm.run task.name=state_chain ctx.edit=bash world.setup=external ...     # mock provider (default)
    OPENROUTER_API_KEY=... python -m siamlm.run provider=openrouter \
        'world.agent_providers=[{model: deepseek/deepseek-v4.1-flash}, {model: anthropic/claude-haiku-5.5}]' \
        world.seeds=[1,2,3] output.batch=pilot

Any key of siamlm/configs/siamlm.yaml can be overridden (Hydra syntax). One folder per world:
<output.directory>/<output.name>/<output.batch>/<cell>/world_s<seed>/ (see siamlm/world.py). A world whose world.json
says complete is skipped, so a cut-off batch can be run again.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / "configs"
REASONING = {"low": {"effort": "low"}, "off": {"enabled": False}}


def compose(overrides=()):
    from hydra import compose as hydra_compose, initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        return hydra_compose("siamlm", overrides=list(overrides))


def with_reasoning(cfg):
    """reasoning=low|off -> provider.extra_body.reasoning for every agent (an agent override with extra_body wins)."""
    from omegaconf import OmegaConf
    mode = str(cfg.get("reasoning", "default"))
    if mode == "default":
        return cfg
    if mode not in REASONING:
        raise ValueError(f"reasoning must be default, low or off (got {mode})")
    OmegaConf.set_struct(cfg, False)
    cfg.provider.extra_body = OmegaConf.merge(cfg.provider.get("extra_body") or {}, {"reasoning": REASONING[mode]})
    return cfg


def models(cfg) -> list[str]:
    over = list(cfg.world.get("agent_providers") or [])
    return [str((over[(k - 1) % len(over)] if over else {}).get("model", cfg.provider.model)) for k in range(1, int(cfg.world.n_agents) + 1)]


def cell_name(cfg) -> str:
    if cfg.output.get("cell"):
        return str(cfg.output.cell)
    short = lambda m: m.split("/")[-1].replace("deepseek-", "ds-").replace("claude-", "").replace(".", "")
    n = int(cfg.world.n_agents)
    sharing = "single" if n == 1 else str(cfg.world.sharing)
    cap = f"b{int(cfg.ctx.budget_tokens)}" if cfg.ctx.get("cap", True) else "uncapped"
    rel = "" if n == 1 else f"_{cfg.world.relation}"
    tc = cfg.get("task") or {}
    cep = f"_cep-{tc.arrangement}-g{int(tc.gap)}" if str(tc.get("name", "lcb")) == "codeeval_pro" else ""
    return f"{sharing}{rel}{cep}_{'+'.join(short(m) for m in models(cfg))}_{cap}_r{cfg.get('reasoning', 'default')}"


def build_task(cfg, seed: int, out: Path, pool=(), grader_factory=None):
    """(task, streams) for one world: the environment named by task.name (siamlm/task.py's Task API) and each agent's
    stream of items. The generated families (codeeval_pro, state_chain) write their ground truth to manifest.json."""
    from siamlm.problems import streams
    tc = cfg.get("task") or {}
    name, n = str(tc.get("name", "lcb")), int(cfg.world.n_agents)
    if name == "open":                       # task dial: no problems
        from siamlm.tasks.open import OpenTask
        task = OpenTask(str(tc.get("level", "weak")))
        return task, task.streams(n)
    if name == "state_chain":                # binding by state: generated shell tasks, no grader
        from siamlm.tasks import state_chain
        st, manifest = state_chain.build(n, int(seed), int(tc.length), gate=bool(tc.get("gate", False)))
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return state_chain.StateChainTask(bool(tc.get("hint", False))), st
    if name == "harbor":                     # Harbor-format task directories, each item in its own container
        from omegaconf import OmegaConf
        from siamlm.tasks import harbor
        settings = {**OmegaConf.to_container(cfg.ctx.bash, resolve=True), **OmegaConf.to_container(cfg.harbor, resolve=True)}
        st = harbor.streams(harbor.load(cfg.harbor.path), n, int(cfg.harbor.get("per_agent") or 0),
                            str(cfg.world.relation), int(seed))
        return harbor.HarborTask(settings, out), st
    if name == "codeeval_pro":               # CodeEval-Pro: streams with known dependencies; manifest = ground truth
        from siamlm.graders.cep import CepGrader
        from siamlm.tasks import cep_streams, codeeval_pro
        st, manifest = cep_streams.build(codeeval_pro.load(tc.data), n, int(seed), str(tc.arrangement), int(tc.length),
                                         int(tc.gap), rotate=int(tc.get("rotate") or 0), gate=bool(tc.get("gate", False)))
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        grader = (grader_factory or (lambda: CepGrader(cfg.cep_grader)))()
        return codeeval_pro.CodeEvalProTask(grader), st
    if name != "lcb":
        raise ValueError(f"task.name must be lcb, codeeval_pro, open, state_chain or harbor (got {name})")
    from siamlm.graders.lcb import Grader
    from siamlm.tasks.lcb import LcbTask
    st = streams(list(pool), n, int(cfg.problems.per_agent), str(cfg.world.relation), int(seed))
    return LcbTask((grader_factory or (lambda: Grader(cfg.grader)))()), st


def run(cfg, make_provider=None, grader_factory=None):
    from siamlm.problems import load, select
    from siamlm.world import World
    from siamlm.context.sandbox import stop_all
    if make_provider is None:
        from siamlm.providers import make_provider
    tc = cfg.get("task") or {}
    cfg = with_reasoning(cfg)
    pc = cfg.problems
    pool = (select(load(pc.file), list(pc.difficulty or []), list(pc.platform or []), pc.start_date, pc.end_date)
            if str(tc.get("name", "lcb")) == "lcb" else [])
    root = Path(cfg.output.directory) / str(cfg.output.name) / str(cfg.output.batch) / cell_name(cfg)
    root.mkdir(parents=True, exist_ok=True)

    def one(seed):
        out = root / f"world_s{seed}"
        done = out / "world.json"
        if done.exists() and json.loads(done.read_text()).get("status") in ("complete", "round_limit"):
            return {"seed": seed, "status": "skipped (finished)"}      # complete, or ended by the round limit: final
        if out.exists():                      # a cut-off world: keep it aside, start again
            out.rename(out.with_name(out.name + f"_cut{len(list(root.glob(out.name + '_cut*')))}"))
        task, st = build_task(cfg, int(seed), out, pool, grader_factory)
        try:
            return World(cfg, int(seed), st, out, task, make_provider, label=f"{cell_name(cfg)}/s{seed}").run()
        finally:
            task.close()

    try:
        seeds = list(cfg.world.seeds)
        with ThreadPoolExecutor(max_workers=max(1, int(cfg.get("parallel", 1)))) as ex:
            results = list(ex.map(one, seeds))
    finally:
        stop_all()
    (root / "cell.json").write_text(json.dumps({"cell": cell_name(cfg), "pool": len(pool), "results": results}, indent=2), encoding="utf-8")
    return root, results


def main(argv=None):
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # a stopped run (kill, timeout) still removes its containers
    root, results = run(compose(list(sys.argv[1:] if argv is None else argv)))
    for r in results:
        print(json.dumps(r))
    print(f"-> {root}")


if __name__ == "__main__":
    main()
