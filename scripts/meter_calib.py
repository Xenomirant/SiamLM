"""Meter calibration for agent-edited weights (SiamLM idea notes, section 3, constraint 2).
For a set of weight deltas on Qwen2.5-0.5B-Instruct, record: relative Frobenius change of the edited tensor, the
tensor's relative output change on a calibration batch, mean per-token KL(base || edited) on a probe set (teacher-forced
on the base model's own greedy continuations), exact-match score on generated arithmetic word problems, and the loop
rate (generations that hit the token limit). CPU only. Usage: python meter_calib.py <out.json> [n_items] [max_new]"""
import json, math, random, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(16)
OUT = sys.argv[1]; N_ITEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 60; MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
rng = random.Random(7)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
L = model.config.num_hidden_layers; MID = L // 2

# --- items --------------------------------------------------------------------------------------------------------
NAMES = ["Tom", "Ana", "Lee", "Mia", "Raj", "Zoe"]; THINGS = ["apples", "coins", "books", "cards", "stamps", "marbles"]
def item():
    n, t = rng.choice(NAMES), rng.choice(THINGS); a, b, c = rng.randint(12, 90), rng.randint(5, 40), rng.randint(2, 9)
    kind = rng.randint(0, 2)
    if kind == 0: q, ans = f"{n} has {a} {t} and buys {b} more, then gives away {c}. How many {t} does {n} have now?", a + b - c
    elif kind == 1: q, ans = f"{n} has {a} {t} and packs them into boxes of {c}. How many full boxes are there?", a // c
    else: q, ans = f"{n} has {a} {t}. {n} gets {c} times as many more. How many {t} does {n} have now?", a + c * a
    return q + " Answer with the number only.", ans
ITEMS = [item() for _ in range(N_ITEMS)]
PROBE = [q for q, _ in ITEMS[:40]] + ["Write one sentence about the sea.", "What is the capital of France?",
         "List three colours.", "Explain what a prime number is in one sentence.", "Say hello in Spanish.",
         "Give a synonym for 'quick'.", "What comes after Tuesday?", "Name a mammal that can fly.",
         "Translate 'thank you' to French.", "What is 15 plus 27?"] * 2
CALIB = PROBE[:32]

def chat(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate(qs):
    outs = []
    for q in qs:
        ids = tok(chat(q), return_tensors="pt")
        g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
        new = g[0, ids["input_ids"].shape[1]:]
        outs.append((tok.decode(new, skip_special_tokens=True), new.shape[0] >= MAX_NEW, new.tolist()))
    return outs

def score(outs):
    ok = 0
    for (text, _, _), (_, ans) in zip(outs, ITEMS):
        nums = re.findall(r"-?\d+", text.replace(",", ""))
        ok += bool(nums) and int(nums[-1]) == ans
    return ok / len(ITEMS), sum(1 for _, hit, _ in outs if hit) / len(outs)

@torch.no_grad()
def logprobs_on(conts):
    """Teacher-forced log-probs of the base continuations under the current weights: list of [T, V] log-softmax."""
    res = []
    for q, cont in zip(PROBE, conts):
        ids = tok(chat(q), return_tensors="pt")["input_ids"]
        full = torch.cat([ids, torch.tensor([cont])], dim=1)
        logits = model(full).logits[0, ids.shape[1] - 1:-1]
        res.append(torch.log_softmax(logits.float(), -1))
    return res

def kl(base_lp, cur_lp):
    tot, n = 0.0, 0
    for b, c in zip(base_lp, cur_lp):
        tot += (b.exp() * (b - c)).sum(-1).sum().item(); n += b.shape[0]
    return tot / max(n, 1)

# --- capture inputs to the tensors we edit (for the output-change meter) --------------------------------------
TARGETS = {"down": model.model.layers[MID].mlp.down_proj, "o": model.model.layers[MID].self_attn.o_proj,
           "down_late": model.model.layers[L - 3].mlp.down_proj}
captured = {k: [] for k in TARGETS}
hooks = [m.register_forward_hook(lambda mod, inp, out, k=k: captured[k].append(inp[0][0].detach().clone())) for k, m in TARGETS.items()]
with torch.no_grad():
    for q in CALIB:
        model(tok(chat(q), return_tensors="pt")["input_ids"])
for h in hooks: h.remove()
X = {k: torch.cat(v, 0) for k, v in captured.items()}          # [positions, in_dim]

def out_change(k, delta):
    W = TARGETS[k].weight.data; base = X[k] @ W.T; d = X[k] @ delta.T
    return (d.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-6)).mean().item()

# --- deltas -----------------------------------------------------------------------------------------------------
def random_sparse(k, n, scale):
    W = TARGETS[k].weight.data; delta = torch.zeros_like(W); idx = torch.randperm(W.numel(), generator=torch.Generator().manual_seed(n + int(scale * 10)))[:n]
    delta.view(-1)[idx] = scale * W.std() * torch.randn(n, generator=torch.Generator().manual_seed(1))
    return delta
def delete_direction(k, cols=None):
    """Project out of the tensor's output the top principal direction of its outputs on the calibration batch (all
    output rows, or only `cols` of them as in the post's 64-column edit)."""
    W = TARGETS[k].weight.data; Y = X[k] @ W.T; Y = Y - Y.mean(0)
    u = torch.linalg.svd(Y, full_matrices=False)[2][0]           # top right-singular vector: a direction in output space
    full = -torch.outer(u, u) @ W                                   # (I - uu^T) W - W
    if cols is None: return full
    delta = torch.zeros_like(W); top = u.abs().topk(cols).indices; delta[top] = full[top]; return delta
def scale_rows(k, factor, rows):
    W = TARGETS[k].weight.data; delta = torch.zeros_like(W); r = torch.arange(rows); delta[r] = (factor - 1) * W[r]; return delta
def zero_top_magnitude(k, n):
    W = TARGETS[k].weight.data; delta = torch.zeros_like(W); idx = W.abs().view(-1).topk(n).indices; delta.view(-1)[idx] = -W.view(-1)[idx]; return delta

CONDS = [("base", None, None)]
for n in (1_000, 10_000, 100_000, 1_000_000):
    for sc in (0.5, 2.0):
        CONDS.append((f"random down n={n} x{sc}", "down", lambda n=n, sc=sc: random_sparse("down", n, sc)))
CONDS += [("random o n=100000 x2", "o", lambda: random_sparse("o", 100_000, 2.0)),
          ("delete top direction (down, all rows)", "down", lambda: delete_direction("down")),
          ("delete top direction (down, 64 rows)", "down", lambda: delete_direction("down", 64)),
          ("delete top direction (o, all rows)", "o", lambda: delete_direction("o")),
          ("delete top direction (down, layer L-3, all rows)", "down_late", lambda: delete_direction("down_late")),
          ("delete top direction (down, layer L-3, 64 rows)", "down_late", lambda: delete_direction("down_late", 64)),
          ("scale 64 rows of down x0.5", "down", lambda: scale_rows("down", 0.5, 64)),
          ("scale 64 rows of down x0", "down", lambda: scale_rows("down", 0.0, 64)),
          ("zero top-10000 |w| of down", "down", lambda: zero_top_magnitude("down", 10_000)),
          ("zero top-100000 |w| of down", "down", lambda: zero_top_magnitude("down", 100_000))]

rows = []; t0 = time.time()
base_outs = generate([q for q, _ in ITEMS]); base_conts = [generate([q])[0][2] for q in PROBE]
base_lp = logprobs_on(base_conts); acc, loop = score(base_outs)
rows.append(dict(cond="base", tensor="-", changed=0, rel_frob=0.0, out_change=0.0, kl=0.0, acc=acc, loop=loop))
print(f"base acc {acc:.2f} loop {loop:.2f}  ({time.time()-t0:.0f}s)", flush=True)
for name, k, make in CONDS[1:]:
    W = TARGETS[k].weight; delta = make(); saved = W.data.clone()
    rel = (delta.norm() / saved.norm()).item(); oc = out_change(k, delta)
    W.data.add_(delta)
    acc, loop = score(generate([q for q, _ in ITEMS])); k_ = kl(base_lp, logprobs_on(base_conts))
    W.data.copy_(saved)
    rows.append(dict(cond=name, tensor=k, changed=int((delta != 0).sum()), rel_frob=rel, out_change=oc, kl=k_, acc=acc, loop=loop))
    print(f"{name:40s} changed {rows[-1]['changed']:8d} relF {rel:.4f} outΔ {oc:.3f} KL {k_:.3f} acc {acc:.2f} loop {loop:.2f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
