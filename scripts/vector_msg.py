"""Vector messages: can one model send another a message as k optimised input vectors, and can the receiver use it?
After Kuratov et al. 2025 ("Cramming 1568 Tokens into a Single Vector", ACL 2025): k trainable input vectors are
prepended to a frozen LM and optimised (AdamW) so the LM reproduces the message. New here: the receiver (the same
frozen model) is asked questions with only the vectors as prefix, with no decoding step, and compared with the
message as text and with decode-then-answer. CPU, Qwen2.5-0.5B-Instruct. Usage:
    python vector_msg.py <out.json> [max_steps=300] [lengths=16,32,64,128] [ks=1,4]"""
import json, math, random, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(16); torch.manual_seed(0)
OUT = sys.argv[1]; MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LENGTHS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "16,32,64,128").split(",")]
KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,4").split(",")]
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
for p in model.parameters(): p.requires_grad_(False)
E = model.get_input_embeddings(); d = E.weight.shape[1]; emb_std = E.weight.std().item()

# --- messages: agent-style notes made of facts (so questions have answers), and random words --------------------
rng = random.Random(11)
NAMES = ["Ingrid", "Tomas", "Aiko", "Mateo", "Zara", "Pavel", "Lina", "Omar"]; CITIES = ["Lisbon", "Oslo", "Lima", "Hanoi", "Cairo", "Perth"]
COLOURS = ["purple", "amber", "teal", "crimson", "olive", "ivory"]; THINGS = ["ledger", "key", "map", "parcel", "ticket", "badge"]
FACT_MAKERS = [
    lambda: (f"The password is {rng.randint(1000, 9999)}.", "What is the password? Answer with the number only.", None),
    lambda: (f"The meeting is in {rng.choice(CITIES)}.", "Where is the meeting? Answer with one word.", None),
    lambda: (f"The secret colour is {rng.choice(COLOURS)}.", "What is the secret colour? Answer with one word.", None),
    lambda: (f"The {rng.choice(THINGS)} belongs to {rng.choice(NAMES)}.", None, None),
    lambda: (f"{rng.choice(NAMES)} arrives at {rng.randint(1, 12)} o'clock.", None, None),
    lambda: (f"The code word for the door is {rng.choice(COLOURS)}-{rng.randint(10, 99)}.", "What is the code word for the door? Answer with the code only.", None),
]
WORDS = open("/usr/share/dict/words").read().split() if __import__("os").path.exists("/usr/share/dict/words") else None
if WORDS is None:
    WORDS = [w for w in (open("/srv/shared/worktrees/slm/README.md").read().split()) if w.isalpha()]
WORDS = sorted({w.lower() for w in WORDS if w.isalpha() and 3 <= len(w) <= 8})

def answer_of(fact):
    return fact.rstrip(".").split(" is ")[-1] if " is " in fact else None

def make_message(length, kind):
    """A text of about `length` tokens: facts (with up to 3 questions) or random words."""
    if kind == "facts":
        facts, qas, used = [], [], set()
        makers = list(range(len(FACT_MAKERS))); rng.shuffle(makers)
        for i in makers * 4:                                  # each fact type at most once per 6 facts, no repeats of a sentence
            f, q, _ = FACT_MAKERS[i]()
            if f in used: continue
            cand = " ".join(facts + [f])
            if len(tok.encode(cand)) > length: break
            facts.append(f); used.add(f)
            if q and len(qas) < 3 and not any(q == x[0] for x in qas): qas.append((q, answer_of(f)))
        return " ".join(facts), qas
    ws = []
    while len(tok.encode(" ".join(ws + [rng.choice(WORDS)]))) <= length: ws.append(rng.choice(WORDS))
    return " ".join(ws), []

def chat(user): return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

def compress(text, k, max_steps):
    """Optimise k input vectors so the frozen model reproduces `text` after them. Returns vectors, steps, final acc."""
    ids = tok(text, return_tensors="pt")["input_ids"]; tgt = ids[0]
    mem = torch.nn.Parameter(torch.randn(1, k, d) * emb_std)
    opt = torch.optim.AdamW([mem], lr=0.01, betas=(0.9, 0.9), weight_decay=0.01)
    x = E(ids).detach(); acc = 0.0
    for step in range(1, max_steps + 1):
        logits = model(inputs_embeds=torch.cat([mem, x], 1)).logits[0, k - 1:-1]
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        acc = (logits.argmax(-1) == tgt).float().mean().item()
        if acc >= 1.0: break                                   # stop before the update, so the returned vectors are the ones measured
        opt.zero_grad(); loss.backward(); opt.step()
    return mem.detach(), step, acc, loss.item()

@torch.no_grad()
def greedy(prefix_embeds, max_new):
    """Greedy decoding from an embedding prefix with a full forward pass per token (no KV cache: HF generate with a
    short inputs_embeds prefix produced degenerate output in testing). Returns the new token ids."""
    x = prefix_embeds; out = []
    for _ in range(max_new):
        nxt = model(inputs_embeds=x).logits[0, -1].argmax().item()
        if nxt == tok.eos_token_id or nxt == tok.convert_tokens_to_ids("<|im_end|>"): break
        out.append(nxt); x = torch.cat([x, E(torch.tensor([[nxt]]))], 1)
    return out

@torch.no_grad()
def reconstruct(mem, text):
    ids = tok(text, return_tensors="pt")["input_ids"]; tgt = ids[0]; k = mem.shape[1]
    logits = model(inputs_embeds=torch.cat([mem, E(ids)], 1)).logits[0, k - 1:-1]
    tf_acc = (logits.argmax(-1) == tgt).float().mean().item()
    dec = tok.decode(greedy(mem, len(tgt) + 4), skip_special_tokens=True)
    return tf_acc, dec, dec.strip().startswith(text.strip())

@torch.no_grad()
def answer(prefix_embeds, question, max_new=12):
    ids = tok(chat(question), return_tensors="pt")["input_ids"]
    x = torch.cat([prefix_embeds, E(ids)], 1) if prefix_embeds is not None else E(ids)
    return tok.decode(greedy(x, max_new), skip_special_tokens=True).strip()

def correct(reply, target):
    return target.lower().strip(".") in reply.lower()

def qa_scores(prefix_embeds, qas):
    return [correct(answer(prefix_embeds, q), a) for q, a in qas]

rows = []; t0 = time.time()
for kind in ("facts", "random"):
    for length in LENGTHS:
        text, qas = make_message(length, kind); n_tok = len(tok.encode(text))
        for k in KS:
            ts = time.time(); mem, steps, acc, loss = compress(text, k, MAX_STEPS); opt_s = time.time() - ts
            tf_acc, dec, exact = reconstruct(mem, text)
            row = dict(kind=kind, length=length, tokens=n_tok, k=k, steps=steps, train_acc=acc, loss=loss, opt_seconds=round(opt_s),
                       tf_acc=tf_acc, exact=exact, decoded=dec[:300], text=text, qas=qas)
            if qas:
                row["qa_vectors"] = qa_scores(mem, qas)                                  # answer from the vectors alone
                row["qa_text"] = qa_scores(E(tok(text, return_tensors="pt")["input_ids"]), qas)  # the message as text, same position
                row["qa_decoded"] = qa_scores(E(tok(dec, return_tensors="pt")["input_ids"]), qas)  # decode first, then answer
                row["qa_none"] = qa_scores(None, qas)
                row["replies_vectors"] = [answer(mem, q)[:60] for q, _ in qas]
            rows.append(row)
            qa = "" if not qas else (f"  QA vectors {sum(row['qa_vectors'])}/{len(qas)} text {sum(row['qa_text'])}/{len(qas)} "
                                    f"decoded {sum(row['qa_decoded'])}/{len(qas)} none {sum(row['qa_none'])}/{len(qas)}")
            print(f"{kind:6s} {n_tok:4d} tok k={k} steps {steps:3d} ({opt_s:4.0f}s) tf_acc {tf_acc:.2f} exact {int(exact)}{qa}  [{time.time()-t0:.0f}s]", flush=True)
            json.dump(dict(provenance=provenance(sys.argv), max_steps=MAX_STEPS, rows=rows), open(OUT, "w"), indent=1)
