"""Smoke test for listen_in_text_v2.ipynb: runs every analysis cell on a tiny random LMModel with
planted hits, stubbing only downloads (weights, LibriSpeech, Whisper, Mimi, nltk, MiniLM)."""
import os, sys, types, random, math, re
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

# ---- stubs for nltk and sentence-transformers ----
VOCAB = ["<epad>", "<bos>", "<eos>", "<pad>", "hello", "world", "speech", "model", "difficult", "hard",
         "whole", "entire", "knowledge", "know", "table", "chair", "river", "water", "green", "a"]
nltk = types.ModuleType("nltk"); nltk.download = lambda *a, **k: True
corpus = types.ModuleType("nltk.corpus")
class _CMU:
    @staticmethod
    def dict():
        rng = random.Random(1)
        return {w: [[rng.choice("AA AE B D EH F G HH IY K L M N OW P R S T UW V Z".split()) + "1" for _ in range(len(w) // 2 + 1)]] for w in VOCAB[4:]}
corpus.cmudict = _CMU; nltk.corpus = corpus
sys.modules["nltk"] = nltk; sys.modules["nltk.corpus"] = corpus
st = types.ModuleType("sentence_transformers")
class SentenceTransformer:
    def __init__(self, *a, **k): pass
    def encode(self, words, **k):
        out = []
        for w in words:
            v = np.random.default_rng(abs(hash(w)) % 2**32).normal(size=16)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)
st.SentenceTransformer = SentenceTransformer; sys.modules["sentence_transformers"] = st

g = {}
exec(cell("CFG = dict("), g)
os.makedirs(f"{SCR}/out2", exist_ok=True)
g["CFG"].update(out_dir=f"{SCR}/out2", topk=10, n_control=3, save_layers=[0, 1, 2, 3], sem_layers=[0, 1, 2, 3], dtype="fp16")
assert g["CFG"]["model_repo"] == "nvidia/personaplex-7b-v1" and g["CFG"]["prefix_user"] == "sine"
g["BASE"] = SCR
exec("import json, math, random, re, gc\nfrom pathlib import Path\nimport numpy as np, pandas as pd, torch\n"
     "from safetensors import safe_open\nfrom moshi.models import loaders\n"
     "from moshi.models.lm import LMModel, SILENCE_TOKENS, SINE_TOKENS, _delay_sequence\n", g)

kw = dict(g["loaders"]._lm_kwargs, dim=64, num_heads=4, num_layers=3, depformer_dim=32,
          depformer_dim_feedforward=64, depformer_num_layers=1, depformer_num_heads=2, dep_q=16)
ref = g["LMModel"](device="cpu", dtype=torch.float32, **kw).eval()
for p in ref.parameters():
    torch.nn.init.normal_(p, std=0.05)
path = f"{SCR}/tiny.safetensors"
save_file({k: v.contiguous().to(torch.bfloat16) for k, v in ref.state_dict().items()}, path)
exec(cell("def load_lm_for_lens").split("DTYPE = pick_dtype()")[0], g)
lm, devs = g["load_lm_for_lens"](path, kw, torch.float32)
g.update(lm=lm, LAYER_DEVS=devs)

class SP:
    def encode(self, s): return [VOCAB.index(w.lower()) if w.lower() in VOCAB else 19 for w in s.split()]
    def id_to_piece(self, i): return "▁" + VOCAB[i] if i < len(VOCAB) else f"<{i}>"
    def piece_to_id(self, p): return VOCAB.index(p[1:]) if p[1:] in VOCAB else 0
    def get_piece_size(self): return len(VOCAB)
g["sp"] = SP()

seq_src = cell("def build_codes")
exec("def wrap_system(text):\n    return f'<system> {text.strip()} <system>'\n"
     + seq_src[seq_src.index("def build_codes"):seq_src.index("prompt_ids =")], g)
g["FPS"] = 12.5
# moshi prefix uses silence on the user stream
c_moshi, _ = g["build_codes"](torch.zeros(8, 3, dtype=torch.long), [], 2, "silence")
assert torch.equal(c_moshi[9:, 0], torch.as_tensor(g["SILENCE_TOKENS"]))

rng = random.Random(0)
WORDS = VOCAB[4:-1]
turns = []
for i in range(6):
    codes, start = g["build_codes"](torch.randint(0, 2048, (8, 150)), [5, 6, 7], 6, "sine")
    ws = [rng.choice(WORDS) for _ in range(10)]
    words = [dict(word=w, start=0.8 + 1.0 * j, end=1.3 + 1.0 * j) for j, w in enumerate(ws)]
    turns.append(dict(id=f"{100+i}-{i}", codes=codes, user_start=start, words=words, text=" ".join(ws)))

exec(cell("def lens_forward").split("import time")[0], g)
for t in turns:
    t["top"], t["hid"] = g["lens_forward"](lm, devs, t["codes"], 10, [0, 1, 2, 3])
    # plant: at layer 2, the word appears at rank 0 from 2 frames after onset to word end;
    # a semantic neighbour ("hard" for "difficult") sits at rank 1
    for w in t["words"]:
        f0 = int(math.floor(w["start"] * 12.5)); f1 = int(math.ceil(w["end"] * 12.5))
        for p in range(t["user_start"] + f0 + 3, t["user_start"] + f1 + 1):
            t["top"][2, p, 0] = VOCAB.index(w["word"])
            if w["word"] == "difficult":
                t["top"][2, p, 1] = VOCAB.index("hard")
g["turns"] = turns

exec(cell("def word_targets"), g)
exec(cell("def hit_curves"), g)
c = g["curves"]
assert c[(c.k == 5) & (c.subset == "all") & (c.layer == 2)].real.item() > 0.9
exec(cell("import matplotlib.pyplot as plt"), g)
exec(cell("def final_argmax"), g)
exec(cell("L = int(best.sort_values"), g)
exec(cell("feats = {L: []"), g)
exec(cell("from sklearn.linear_model import LogisticRegression"), g)

exec(cell("OFFS = np.arange"), g)
real, ctrl, n = g["timing"]["start"]
OFFS = g["OFFS"]
i0 = list(OFFS).index(0)
assert real[2, i0] < 0.1 and real[2, i0 + 2] > 0.9, (real[2, i0], real[2, i0 + 2])   # planted from d=+2
exec(cell("best_layer = int("), g)
assert g["best_layer"] == 2

exec(cell("import nltk"), g)
exec(cell("CAND = {}"), g)
s = g["summary"]
print(s.round(3).to_string())
exec(cell("fig, ax = plt.subplots(figsize=(8, 4))"), g)
exec(cell("np.savez_compressed(f\"{CFG['out_dir']}/top5.npz\""), g)
z = np.load(f"{SCR}/out2/top5.npz")
assert z[turns[0]["id"]].shape == (4, turns[0]["top"].shape[1], 5)
print(sorted(os.listdir(f"{SCR}/out2")))
print("ALL V2 CHECKS PASSED")
