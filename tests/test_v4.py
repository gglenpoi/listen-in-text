"""Smoke test for listen_in_text_v4.ipynb on a tiny random LMModel. Downloads (weights, Mimi, LibriSpeech/AMI,
MMS/Whisper, GPT-2) are replaced by synthetic stand-ins; every analysis cell runs unmodified."""
import os, sys, types, math, random, time
os.environ["NO_TORCH_COMPILE"] = "1"
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, "/root/personaplex/moshi")
import nbformat, numpy as np, torch
from safetensors.torch import save_file

SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp"); os.makedirs(SCR, exist_ok=True)
nb = nbformat.read(sys.argv[1], as_version=4)
src = [c.source for c in nb.cells if c.cell_type == "code"]
def cell(marker):
    hits = [s for s in src if marker in s]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]

# ---- fake transformers (GPT-2) ----
tf = types.ModuleType("transformers")
class FakeTok:
    bos_token_id = 0
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def __call__(self, text, return_offsets_mapping=True, return_tensors="pt"):
        offs, pos = [], 0
        for w in text.split(" "):
            offs.append((pos, pos + len(w))); pos += len(w) + 1
        return types.SimpleNamespace(input_ids=torch.randint(1, 100, (1, len(offs))), offset_mapping=torch.tensor([offs]))
class FakeLM(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def forward(self, ids): return types.SimpleNamespace(logits=torch.randn(1, ids.shape[1], 100))
tf.AutoTokenizer, tf.AutoModelForCausalLM = FakeTok, FakeLM
sys.modules["transformers"] = tf

g = {}
src_cfg = cell("CFG = dict(").replace('CFG["out_dir"] = f"{BASE}/{CFG[\'model\']}_v4"', f'CFG["out_dir"] = "{SCR}/v4_test"')
exec(src_cfg, g)
g["CFG"].update(n_turns=6, tl_layers=[0, 2, 3], tl_steps=30, tl_batch=64, n_boot=50, tl_rank=8)
g["BASE"] = SCR
exec(cell("import json, math, random, re, tarfile"), g)

# tiny model
LMModel, loaders = g["LMModel"], g["loaders"]
kw = dict(loaders._lm_kwargs, dim=64, num_heads=4, num_layers=3, depformer_dim=32,
          depformer_dim_feedforward=64, depformer_num_layers=1, depformer_num_heads=2, dep_q=16)
ref = LMModel(device="cpu", dtype=torch.float32, **kw).eval()
for p in ref.parameters(): torch.nn.init.normal_(p, std=0.05)
path = f"{SCR}/tiny.safetensors"
save_file({k: v.contiguous().to(torch.bfloat16) for k, v in ref.state_dict().items()}, path)
model_src = cell("def load_lm_for_lens")
exec(model_src.split("DTYPE = pick_dtype()")[0], g)
g["DTYPE"] = torch.float32
g["lm"], g["LAYER_DEVS"] = g["load_lm_for_lens"](path, kw, torch.float32)
g["N_LAYERS"] = len(g["lm"].transformer.layers) + 1
class FakeMimi(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1)); self.sample_rate = 24000; self.frame_rate = 12.5
    def encode(self, x): return torch.randint(0, 2048, (1, 8, x.shape[-1] // 1920))
g["mimi"] = FakeMimi(); g["FRAME"], g["FPS"] = 1920, 12.5

VOCAB = ["<epad>", "<bos>", "<eos>", "<pad>", "hello", "world", "speech", "model", "difficult", "hard",
         "whole", "entire", "knowledge", "table", "chair", "river", "water", "green", "listen", "talking"]
class SP:
    def encode(self, s): return [VOCAB.index(w.lower()) if w.lower() in VOCAB else 19 for w in s.split()]
    def id_to_piece(self, i): return "▁" + VOCAB[i] if i < len(VOCAB) else f"<{i}>"
g["sp"] = SP()

# synthetic data: two datasets, 6 turns each, words every 0.8 s
rng = random.Random(0)
DATA = {}
for name in ("librispeech", "ami"):
    turns = []
    for i in range(6):
        dur = 12.0
        words = [dict(word=rng.choice(VOCAB[4:18]), start=0.6 + 0.8 * j, end=1.0 + 0.8 * j) for j in range(13)]
        turns.append(dict(id=f"{name}-{i}", spk=f"{name}-spk{i}", wav=np.random.randn(int(dur * 24000)).astype(np.float32) * 0.01,
                          segments=[(0, dur, " ".join(w["word"] for w in words))], words=words))
    DATA[name] = turns
g["DATA"], g["ALIGNER_USED"] = DATA, "test"
g["CFG"]["aligner_used"] = "test"

exec(cell("def text_track"), g)
tt = g["text_track"]([dict(word="hello", start=0.8, end=1.0), dict(word="world", start=0.85, end=1.2)], 30)
assert tt[10] == VOCAB.index("hello") and tt[11] == VOCAB.index("world") and tt[9] == 0, tt[:14]
exec(cell("def lens_forward"), g)
RUNS = g["RUNS"]
assert set(RUNS) == {(n, c) for n in DATA for c in ("listen", "reversed", "talking")}
# talking condition really feeds model-own text
c_talk, _ = g["build_codes"](DATA["librispeech"][0]["codes_user"], [], 2, "sine", DATA["librispeech"][1]["codes_user"],
                             g["text_track"](DATA["librispeech"][1]["words"], DATA["librispeech"][0]["codes_user"].shape[1]))
assert (c_talk[0] != 3).sum() > 5
# plant: layer 2 reads the word in listen (+ half in talking), nothing in reversed
for (name, cond), turns in RUNS.items():
    if cond == "reversed": continue
    for t in turns:
        for wi, w in enumerate(t["words"]):
            if cond == "talking" and wi % 2: continue
            if wi % 3 == 0: continue
            a, b = g["word_window"](t, w, 0)
            for p in range(a + 1, b + 1):
                t["top"][2, p, 0] = VOCAB.index(w["word"])
exec(cell("def word_targets"), g)
C = g["CURVES"]; sel = C[(C.k == 5) & (C.subset == "content") & (C.layer == 2)].set_index(["dataset", "cond"]).excess
print(sel.round(3).to_dict())
assert sel[("librispeech", "listen")] > 0.5 and abs(sel[("librispeech", "reversed")]) < 0.1 and 0.1 < sel[("librispeech", "talking")] < sel[("librispeech", "listen")]
exec(cell("fig, axes = plt.subplots(1, len(DATA)"), g)
exec(cell("OFFS = np.arange"), g)
assert g["BEST"]["librispeech"] == 2
# make stored hidden states carry word identity so the lens has something learnable
lm_ = g["lm"]; E = torch.randn(len(VOCAB), 64)
for name, tf_ in g["TL_FRAMES"].items():
    words = [RUNS[(name, "listen")][ti]["words"][wi]["word"] for ti, wi, _ in tf_["meta"]]
    base = torch.stack([E[VOCAB.index(w)] for w in words])
    for L in tf_["h"]:
        tf_["h"][L] = (base * (L + 1) + 0.1 * torch.randn_like(base)).half()
with torch.no_grad():   # in a 64-d toy model most of 32k random rows can never reach top-5; make the vocab words reachable
    lm_.text_linear.weight[4:len(VOCAB)] *= 20
g["CFG"].update(tl_steps=300, tl_lr=1e-2, tl_rank=32)
exec(cell("class LowRankLens"), g)
assert (g["TL"].tuned_seen > g["TL"].logit_seen + 0.3).all(), "tuned lens failed to learn a learnable mapping"
print(g["TL"].round(3).to_string())
exec(cell("from wordfreq import zipf_frequency"), g)
exec(cell("np.savez_compressed(f\"{d}/top5.npz\""), g)
print(sorted(os.listdir(f"{SCR}/v4_test")))
assert os.path.exists(f"{SCR}/v4_test/librispeech_talking/top5.npz")
print("ALL V4 CHECKS PASSED")
