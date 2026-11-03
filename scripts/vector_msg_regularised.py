"""Vector messages, three remedies for the lookup-table result of vector_msg_inject.py, on the same frozen 0.5B model:
  curve:    the `use` objective with the held-out accuracy logged every 10 steps (is there ever a point to stop at?);
  many:     the `use` objective over many paraphrased questions covering every fact, held-out paraphrases and
            held-out facts measured separately;
  anchored: vectors initialised from the note's own residuals at the layer (mean-pooled chunks) and kept near them by
            an L2 penalty, with the `use` loss over the training questions;
  anchored+many: both remedies.
Injection as in vector_msg_inject.py (k placeholders at the start of the user turn, a hook at the layer). CPU.
Usage: python vector_msg_regularised.py <out.json> [steps=300] [k=4] [layer=7] [n_messages=3] [lam=1.0]"""
import json, random, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(16); torch.manual_seed(0)
OUT = sys.argv[1]; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300; K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
LAYER = sys.argv[4] if len(sys.argv) > 4 else "7"; N_MSG = int(sys.argv[5]) if len(sys.argv) > 5 else 3
LAM = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
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
IM_END = tok.convert_tokens_to_ids("<|im_end|>")

# --- messages, with paraphrased questions per fact -----------------------------------------------------------------
rng = random.Random(23)
CITIES = ["Lisbon", "Oslo", "Lima", "Hanoi", "Cairo", "Perth"]; COLOURS = ["purple", "amber", "teal", "crimson", "olive", "ivory"]
PARA = {  # fact -> question paraphrases; the answer format is fixed per fact
    "password": ["What is the password? Answer with the number only.", "Tell me the password, number only.",
                 "Which number is the password? Just the number.", "Password? Reply with the number and nothing else.",
                 "I need the password. Give only the digits."],
    "city": ["Where is the meeting? Answer with one word.", "In which city is the meeting? One word.",
             "Name the meeting's city, one word only.", "The meeting takes place where? One word.",
             "Which city hosts the meeting? Reply with the city name only."],
    "colour": ["What is the secret colour? Answer with one word.", "Name the secret colour, one word.",
               "Which colour is the secret one? One word only.", "Secret colour? Reply with one word.",
               "Tell me the secret colour in a single word."],
    "code": ["What is the code word for the door? Answer with the code only.", "Door code word? Give the code only.",
             "Which code opens the door? Code only.", "Tell me the door's code word and nothing else.",
             "What code word is used for the door? Reply with just the code."]}

def make_message():
    pw, city, col, code = rng.randint(1000, 9999), rng.choice(CITIES), rng.choice(COLOURS), f"{rng.choice(COLOURS)}-{rng.randint(10, 99)}"
    text = f"The password is {pw}. The meeting is in {city}. The secret colour is {col}. The code word for the door is {code}."
    answers = {"password": str(pw), "city": city, "colour": col, "code": code}
    facts = list(answers); rng.shuffle(facts)
    held_fact = facts[-1]                                       # one whole fact never asked about in training
    train = [(q, answers[f]) for f in facts[:-1] for q in PARA[f][:3]]        # 3 paraphrases x 3 facts = 9 questions
    held_para = [(q, answers[f]) for f in facts[:-1] for q in PARA[f][3:]]    # 2 unseen paraphrases x 3 facts = 6
    held_fact_q = [(q, answers[held_fact]) for q in PARA[held_fact]]          # 5 questions on the unseen fact
    two = [(PARA[f][0], answers[f]) for f in facts[:2]]                       # the original two-question regime
    return text, train, held_para, held_fact_q, two

def chat(user): return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

def placeholder_prompt(after):
    full = chat(PH * K + after); pre = full.index(PH)
    enc = tok(full, return_offsets_mapping=True, return_tensors="pt")
    pos = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()) if a >= pre and b <= pre + len(PH) * K and b > a][:K]
    assert len(pos) == K
    return enc["input_ids"], pos

class Injector:
    def __init__(self, layer):
        self.module = model.model.embed_tokens if layer == "embed" else model.model.layers[int(layer)]
        self.vectors = None; self.positions = None; self.handle = self.module.register_forward_hook(self.hook)
    def hook(self, mod, inp, out):
        if self.vectors is None: return
        hid = out[0] if isinstance(out, tuple) else out
        hid = hid.clone(); hid[0, self.positions] = self.vectors
        return (hid,) + tuple(out[1:]) if isinstance(out, tuple) else hid
    def residual(self, ids, positions=None):
        cap = {}
        h = self.module.register_forward_hook(lambda m, i, o: cap.setdefault("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
        with torch.no_grad(): model(ids)
        h.remove(); return cap["h"][0] if positions is None else cap["h"][0, positions]

def qa_ids(question, answer=None):
    ids, pos = placeholder_prompt(" " + question)
    if answer is None: return ids, pos, None
    return torch.cat([ids, tok(answer, return_tensors="pt")["input_ids"]], 1), pos, ids.shape[1]

def loss_on(ids, start, inj, vectors, positions):
    inj.vectors, inj.positions = vectors, positions
    logits = model(ids).logits[0, start - 1:-1]; inj.vectors = None
    return torch.nn.functional.cross_entropy(logits, ids[0, start:])

@torch.no_grad()
def greedy(ids, inj, vectors, positions, max_new=12):
    inj.vectors, inj.positions = vectors, positions; out = []
    for _ in range(max_new):
        nxt = model(ids).logits[0, -1].argmax().item()
        if nxt in (tok.eos_token_id, IM_END): break
        out.append(nxt); ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
    inj.vectors = None
    return tok.decode(out, skip_special_tokens=True).strip()

def correct(reply, target): return target.lower() in reply.lower()
def score(qas, inj, vec):
    out = []
    for q, a in qas:
        ids, pos, _ = qa_ids(q); out.append(correct(greedy(ids, inj, vec, pos), a))
    return out

def note_residual_chunks(inj, text):
    """The note's own residuals at the layer, inside the same prompt shape, mean-pooled into K chunks."""
    full = chat(text + " The note says:"); pre = full.index(text)
    enc = tok(full, return_offsets_mapping=True, return_tensors="pt")
    pos = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()) if a >= pre and b <= pre + len(text) and b > a]
    h = inj.residual(enc["input_ids"], pos)
    return torch.stack([c.mean(0) for c in torch.tensor_split(h, K)])

rows = []; t0 = time.time(); inj = Injector(LAYER)
for m in range(N_MSG):
    text, train, held_para, held_fact_q, two = make_message()
    q_ids, q_pos = placeholder_prompt(" What is the password? Answer with the number only.")
    own = inj.residual(q_ids, q_pos); scale = own.std().item()
    anchor = note_residual_chunks(inj, text)
    for variant in ("curve", "many", "anchored", "anchored+many"):
        train_set = train if "many" in variant else two
        init = anchor if "anchored" in variant else own
        delta = torch.nn.Parameter(torch.zeros(K, init.shape[1])); opt = torch.optim.AdamW([delta], lr=0.02, betas=(0.9, 0.9), weight_decay=0.0)
        curve = []; ts = time.time()
        for step in range(1, STEPS + 1):
            vec = init + scale * delta; loss = 0.0
            for q, a in train_set:
                ids, pos, start = qa_ids(q, a); loss = loss + loss_on(ids, start, inj, vec, pos)
            loss = loss / len(train_set)
            if "anchored" in variant: loss = loss + LAM * ((vec - anchor) ** 2).sum(-1).mean() / (anchor.norm(dim=-1) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if variant == "curve" and (step % 10 == 0 or step == 1):
                v = (init + scale * delta).detach()
                curve.append(dict(step=step, loss=float(loss.detach()), train=sum(score(two, inj, v)), held_para=sum(score(held_para[:3], inj, v)),
                                  held_fact=sum(score(held_fact_q[:2], inj, v))))
                print(f"  curve step {step:3d} loss {float(loss.detach()):.3f} train {curve[-1]['train']}/2 held_para {curve[-1]['held_para']}/3 held_fact {curve[-1]['held_fact']}/2", flush=True)
        vec = (init + scale * delta).detach()
        # the zero-step reading for the anchored start: are the note's own pooled residuals readable at all?
        base_read = score(train + held_para + held_fact_q, inj, anchor) if variant == "anchored" else None
        row = dict(message=m, variant=variant, layer=LAYER, k=K, steps=STEPS, seconds=round(time.time() - ts), lam=LAM if "anchored" in variant else 0,
                   train=score(train_set, inj, vec), held_para=score(held_para, inj, vec), held_fact=score(held_fact_q, inj, vec),
                   anchor_read=base_read, curve=curve, text=text, dist_to_anchor=((vec - anchor).norm(dim=-1) / anchor.norm(dim=-1)).mean().item())
        rows.append(row)
        extra = f"  anchor alone reads {sum(base_read)}/{len(base_read)}" if base_read else ""
        print(f"msg {m} {variant:14s} train {sum(row['train'])}/{len(row['train'])}  held_para {sum(row['held_para'])}/{len(held_para)}  "
              f"held_fact {sum(row['held_fact'])}/{len(held_fact_q)}  dist {row['dist_to_anchor']:.2f}{extra}  [{time.time()-t0:.0f}s]", flush=True)
        json.dump(dict(provenance=provenance(sys.argv), steps=STEPS, k=K, layer=LAYER, lam=LAM, rows=rows), open(OUT, "w"), indent=1)
