"""Activation-message probe (SiamLM idea notes, section 2). Same model both sides (Qwen2.5-0.5B-Instruct, CPU).
A sender prompt contains a short span (a fact or an instruction). Its residual stream at layer L over the span's
tokens is written over placeholder positions in a receiver prompt at the same layer, rescaled to the receiver's
residual norm there. Compared with: `text` (the span as words in the receiver's prompt), `random` (unit vectors of
matched norm), `none` (placeholders, nothing injected). Measures the receiver's probability of the target answer.
Usage: python act_msg_probe.py <out.json>"""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(16); torch.manual_seed(0)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
def provenance(argv):
    """Everything needed to rerun this exactly: versions, the model snapshot, the command, the host."""
    import platform, subprocess, time, transformers, torch
    from huggingface_hub import snapshot_download
    snap = snapshot_download(MODEL, local_files_only=True)
    return dict(model=MODEL, model_snapshot=snap.rsplit("/", 1)[-1], python=platform.python_version(),
                torch=torch.__version__, transformers=transformers.__version__, threads=torch.get_num_threads(),
                dtype="float32", device="cpu", cpu=open("/proc/cpuinfo").read().split("model name")[1].split(":")[1].split("\n")[0].strip(),
                argv=argv, started=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())

tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
NL = model.config.num_hidden_layers
PH = " ..."   # placeholder text; its token count is matched to the span's

# (span the sender writes, receiver question, target answers). Each span is a few tokens of content the receiver
# could not know from its own prompt.
# (sender context before the span, span, receiver question, targets): the span's words say nothing; the sender's
# context does. If `real` beats `text` here, the vectors carry more than the words.
CTX_CASES = [("We will meet in Lisbon, as agreed. ", "the meeting place", "Where is the meeting? Answer with one word.", ["Lisbon", "L"]),
             ("The secret word is purple, do not forget it. ", "the secret word", "What is the secret word? Answer with one word.", ["purple", "Purple"]),
             ("My name is Ingrid and I am the other agent. ", "my name", "What is the other agent's name? Answer with one word.", ["Ingrid", "In"]),
             ("The password is 4471, keep it safe. ", "the password", "What is the password? Answer with the number only.", ["4", "44", "447", "4471"])]
CASES = [("The secret word is purple.", "What is the secret word? Answer with one word.", ["purple", "Purple"]),
         ("The secret word is banana.", "What is the secret word? Answer with one word.", ["banana", "Banana"]),
         ("The meeting is in Lisbon.", "Where is the meeting? Answer with one word.", ["Lisbon"]),
         ("The password is 4471.", "What is the password? Answer with the number only.", ["4", "44", "447", "4471"]),
         ("My name is Ingrid.", "What is my name? Answer with one word.", ["Ingrid", "Ing"]),
         ("Reply in French only.", "What colour is the sky?", ["Le", "Bleu", "La", "bleu"])]

def chat(user): return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

def span_positions(text_before, span, text_after):
    """Token positions of `span` inside chat(text_before + span + text_after)."""
    full = chat(text_before + span + text_after); pre = chat(text_before + span + text_after).index(span)
    enc = tok(full, return_offsets_mapping=True, return_tensors="pt")
    pos = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()) if a >= pre and b <= pre + len(span) and b > a]
    return enc["input_ids"], pos

captured = {}
def capture_hook(layer):
    def h(mod, inp, out): captured["h"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
    return model.model.layers[layer].register_forward_hook(h)
def write_hook(layer, positions, vectors):
    def h(mod, inp, out):
        hid = out[0] if isinstance(out, tuple) else out
        hid[0, positions] = vectors
        return (hid,) + tuple(out[1:]) if isinstance(out, tuple) else hid
    return model.model.layers[layer].register_forward_hook(h)

@torch.no_grad()
def target_prob(ids, targets, hook=None):
    logits = model(ids).logits[0, -1]; p = torch.softmax(logits, -1)
    tids = {tok.encode(t, add_special_tokens=False)[0] for t in targets} | {tok.encode(" " + t, add_special_tokens=False)[0] for t in targets}
    return sum(p[i].item() for i in tids), tok.decode(logits.argmax().item())

rows = []
ALL = [("Note for the other agent: ", span, q, t) for span, q, t in CASES] + CTX_CASES
for before, span, question, targets in ALL:
    s_ids, s_pos = span_positions(before, span, " Keep this in mind.")
    n = len(s_pos)
    # receiver: placeholders of the same token count, then the question
    ph = PH * n
    r_ids, r_pos = span_positions("", ph, " " + question)
    r_pos = r_pos[:n] if len(r_pos) >= n else r_pos
    text_ids = tok(chat(span + " " + question), return_tensors="pt")["input_ids"]
    none_ids = tok(chat(question), return_tensors="pt")["input_ids"]
    base = {"text": target_prob(text_ids, targets), "none": target_prob(none_ids, targets)}
    for L in sorted({2, 4, 6, 8, 10, 12, 14, 16, 18, 20, NL - 2}):
        h = capture_hook(L); model(s_ids); h.remove(); sv = captured["h"][0, s_pos]                # sender vectors
        h = capture_hook(L); model(r_ids); h.remove(); rv = captured["h"][0, r_pos]               # receiver's own
        scale = rv.norm(dim=-1, keepdim=True) / sv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        real = sv * scale
        rand = torch.randn_like(sv); rand = rand / rand.norm(dim=-1, keepdim=True) * rv.norm(dim=-1, keepdim=True)
        res = {}
        for name, vec in (("real", real), ("random", rand)):
            hk = write_hook(L, r_pos, vec); res[name] = target_prob(r_ids, targets); hk.remove()
        res["placeholder_only"] = target_prob(r_ids, targets)
        rows.append(dict(before=before, span=span, layer=L, n=n, real=res["real"][0], real_top=res["real"][1], random=res["random"][0],
                         placeholder=res["placeholder_only"][0], text=base["text"][0], text_top=base["text"][1], none=base["none"][0]))
        print(f"{span[:26]:26s} L={L:2d} n={n} real {res['real'][0]:.3f} ({res['real'][1]!r})  random {res['random'][0]:.3f}  "
              f"placeholder {res['placeholder_only'][0]:.3f}  text {base['text'][0]:.3f} ({base['text'][1]!r})  none {base['none'][0]:.3f}", flush=True)
json.dump(dict(provenance=provenance(sys.argv), cases=CASES, ctx_cases=CTX_CASES, placeholder=PH, rows=rows), open(sys.argv[1], "w"), indent=1)
