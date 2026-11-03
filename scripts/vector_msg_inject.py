"""Vector messages injected into the residual stream: can a receiver read an optimised message in place?
k vectors are written over k placeholder positions of the receiver's prompt at layer L (the embedding output, or
after block L) by a forward hook, and optimised (AdamW, frozen model) under one of three objectives:
  reconstruct: cross-entropy of the message text after "The note says:" (Kuratov et al.'s loss, moved to layer L);
  use:         cross-entropy of the correct answers to questions about the message (2 training questions);
  both:        the sum.
Measured: teacher-forced reconstruction accuracy; answers from the vectors alone (greedy, no decoding step) on the
training questions and on one held-out question; the same with the message as text. CPU, Qwen2.5-0.5B-Instruct.
Usage: python vector_msg_inject.py <out.json> [steps=300] [k=4] [layers=embed,3,7,11] [n_messages=3]"""
import json, random, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(16); torch.manual_seed(0)
OUT = sys.argv[1]; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300; K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
LAYERS = (sys.argv[4] if len(sys.argv) > 4 else "embed,3,7,11").split(","); N_MSG = int(sys.argv[5]) if len(sys.argv) > 5 else 3
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"; PH = " ..."

def provenance(argv):
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
E = model.get_input_embeddings(); IM_END = tok.convert_tokens_to_ids("<|im_end|>")

# --- messages with three questions each --------------------------------------------------------------------------
rng = random.Random(23)
CITIES = ["Lisbon", "Oslo", "Lima", "Hanoi", "Cairo", "Perth"]; COLOURS = ["purple", "amber", "teal", "crimson", "olive", "ivory"]
def make_message():
    pw, city, col, code = rng.randint(1000, 9999), rng.choice(CITIES), rng.choice(COLOURS), f"{rng.choice(COLOURS)}-{rng.randint(10, 99)}"
    text = f"The password is {pw}. The meeting is in {city}. The secret colour is {col}. The code word for the door is {code}."
    qas = [("What is the password? Answer with the number only.", str(pw)),
           ("Where is the meeting? Answer with one word.", city),
           ("What is the secret colour? Answer with one word.", col),
           ("What is the code word for the door? Answer with the code only.", code)]
    rng.shuffle(qas)
    return text, qas[:2], qas[2:]            # train questions, held-out questions

def chat(user): return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

def placeholder_prompt(after):
    """chat(placeholders + after); returns ids and the placeholder positions."""
    full = chat(PH * K + after); pre = full.index(PH)
    enc = tok(full, return_offsets_mapping=True, return_tensors="pt")
    pos = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()) if a >= pre and b <= pre + len(PH) * K and b > a][:K]
    assert len(pos) == K, (len(pos), K)
    return enc["input_ids"], pos

class Injector:
    """Writes `vectors` over `positions` at the chosen layer during forward passes while active."""
    def __init__(self, layer):
        self.module = model.model.embed_tokens if layer == "embed" else model.model.layers[int(layer)]
        self.vectors = None; self.positions = None; self.handle = self.module.register_forward_hook(self.hook)
    def hook(self, mod, inp, out):
        if self.vectors is None: return
        hid = out[0] if isinstance(out, tuple) else out
        hid = hid.clone(); hid[0, self.positions] = self.vectors
        return (hid,) + tuple(out[1:]) if isinstance(out, tuple) else hid
    def own_residual(self, ids, positions):
        """The receiver's own residual at those positions and layer (initialisation and scale)."""
        cap = {}
        h = self.module.register_forward_hook(lambda m, i, o: cap.setdefault("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
        with torch.no_grad(): model(ids)
        h.remove(); return cap["h"][0, positions]

def loss_on(ids, start, inj, vectors, positions):
    """Cross-entropy of ids[start:] given the prefix, with vectors injected."""
    inj.vectors, inj.positions = vectors, positions
    logits = model(ids).logits[0, start - 1:-1]; inj.vectors = None
    tgt = ids[0, start:]
    return torch.nn.functional.cross_entropy(logits, tgt), (logits.argmax(-1) == tgt).float().mean().item()

@torch.no_grad()
def greedy(ids, inj, vectors, positions, max_new=12):
    inj.vectors, inj.positions = vectors, positions; out = []
    for _ in range(max_new):
        nxt = model(ids).logits[0, -1].argmax().item()
        if nxt in (tok.eos_token_id, IM_END): break
        out.append(nxt); ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
    inj.vectors = None
    return tok.decode(out, skip_special_tokens=True).strip()

def qa_prompt(question, answer=None):
    ids, pos = placeholder_prompt(" " + question)
    if answer is None: return ids, pos, None
    ans = tok(answer, return_tensors="pt")["input_ids"]
    return torch.cat([ids, ans], 1), pos, ids.shape[1]

def recon_prompt(text):
    ids, pos = placeholder_prompt(" The note says:")
    body = tok(" " + text, return_tensors="pt")["input_ids"]
    return torch.cat([ids, body], 1), pos, ids.shape[1]

def correct(reply, target): return target.lower() in reply.lower()

rows = []; t0 = time.time()
for m in range(N_MSG):
    text, train_q, held_q = make_message()
    text_scores = {}
    for q, a in train_q + held_q:                                   # the message as words, same prompt shape
        ids = tok(chat(text + " " + q), return_tensors="pt")["input_ids"]
        text_scores[q] = correct(greedy(ids, Injector("embed"), None, []), a)
    for layer in LAYERS:
        inj = Injector(layer)
        r_ids, r_pos, r_start = recon_prompt(text)
        init = inj.own_residual(r_ids, r_pos); scale = init.std().item()
        for objective in ("reconstruct", "use", "both"):
            delta = torch.nn.Parameter(torch.zeros(K, init.shape[1]))
            opt = torch.optim.AdamW([delta], lr=0.02, betas=(0.9, 0.9), weight_decay=0.0)
            ts = time.time(); last = {}
            for step in range(1, STEPS + 1):
                vec = init + scale * delta; loss = 0.0
                if objective in ("reconstruct", "both"):
                    l, acc = loss_on(r_ids, r_start, inj, vec, r_pos); loss = loss + l; last["recon_acc"] = acc
                if objective in ("use", "both"):
                    for q, a in train_q:
                        ids, pos, start = qa_prompt(q, a); l, acc = loss_on(ids, start, inj, vec, pos); loss = loss + l
                opt.zero_grad(); loss.backward(); opt.step()
            vec = (init + scale * delta).detach()
            with torch.no_grad(): _, recon_acc = loss_on(r_ids, r_start, inj, vec, r_pos)
            replies = {}
            for q, _ in train_q + held_q:
                ids, pos, _ = qa_prompt(q); replies[q] = greedy(ids, inj, vec, pos)
            row = dict(message=m, layer=layer, objective=objective, k=K, steps=STEPS, seconds=round(time.time() - ts),
                       final_loss=float(loss.detach()), recon_acc=recon_acc,
                       qa_train=[correct(replies[q], a) for q, a in train_q], qa_held=[correct(replies[q], a) for q, a in held_q],
                       qa_text_train=[text_scores[q] for q, _ in train_q], qa_text_held=[text_scores[q] for q, _ in held_q],
                       replies=replies, text=text, train_q=train_q, held_q=held_q, vector_norm=vec.norm(dim=-1).mean().item(), init_norm=init.norm(dim=-1).mean().item())
            rows.append(row)
            print(f"msg {m} layer {layer:5s} {objective:11s} recon {recon_acc:.2f}  QA train {sum(row['qa_train'])}/2 held {sum(row['qa_held'])}/2  "
                  f"(text {sum(row['qa_text_train'])}/2, {sum(row['qa_text_held'])}/2)  loss {float(loss.detach()):.3f}  [{time.time()-t0:.0f}s]", flush=True)
            json.dump(dict(provenance=provenance(sys.argv), steps=STEPS, k=K, layers=LAYERS, rows=rows), open(OUT, "w"), indent=1)
        inj.handle.remove()
