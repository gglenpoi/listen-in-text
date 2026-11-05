"""Smoke test: run the notebook's model/sequence/lens/metric cells on a tiny random LMModel (CPU)."""
import os, sys, random, math, re
os.environ["NO_TORCH_COMPILE"] = "1"
sys.path.insert(0, "/root/personaplex/moshi")
import nbformat, numpy as np, torch
from safetensors.torch import save_file

nb = nbformat.read(sys.argv[1], as_version=4)
src = [c.source for c in nb.cells if c.cell_type == "code"]
def cell(marker):
    hits = [s for s in src if marker in s]
    assert len(hits) == 1, marker
    return hits[0]

g = {}
exec(cell("CFG = dict("), g)
os.makedirs("out", exist_ok=True); g["CFG"].update(out_dir="out", topk=10, n_control=3, save_layers=[0, 1, 2], dtype="fp16")
exec("import json, math, random, re\nfrom pathlib import Path\nimport numpy as np, pandas as pd, torch\n"
     "from safetensors import safe_open\nfrom moshi.models import loaders\n"
     "from moshi.models.lm import LMModel, SILENCE_TOKENS, SINE_TOKENS, _delay_sequence\n", g)

# tiny model with the real stream layout (17 streams, card 2048, text 32000)
kw = dict(g["loaders"]._lm_kwargs, dim=64, num_heads=4, num_layers=3, depformer_dim=32,
          depformer_dim_feedforward=64, depformer_num_layers=1, depformer_num_heads=2, dep_q=16)
LMModel, _delay_sequence = g["LMModel"], g["_delay_sequence"]
ref = LMModel(device="cpu", dtype=torch.float32, **kw).eval()
for p in ref.parameters():
    torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad(): p.copy_(p.to(torch.bfloat16).float())  # bf16-exact
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_tiny.safetensors")
save_file({k: v.contiguous().to(torch.bfloat16) for k, v in ref.state_dict().items()}, path)

# loader cell, with model creation lines swapped for the tiny config
loader_src = cell("def load_lm_for_lens").split("DTYPE = pick_dtype()")[0]
exec(loader_src, g)
g["DTYPE"] = torch.float32   # CPU test in fp32 for exact comparison
lm, devs = g["load_lm_for_lens"](path, kw, torch.float32)
g["lm"], g["LAYER_DEVS"] = lm, devs

# fake tokenizer: deterministic ids per piece
class SP:
    def encode(self, s): return [100 + (sum(map(ord, w)) % 30000) for w in s.split()]
    def id_to_piece(self, i): return f"<{i}>"
    def piece_to_id(self, p): return int(p[1:-1]) if p.startswith("<") else 0
g["sp"] = SP()

# sequence builder from the Mimi cell (without loading Mimi)
seq_src = cell("def build_codes")
exec(seq_src.split("mimi = loaders.get_mimi")[0] + "def wrap_system(text):\n    return f'<system> {text.strip()} <system>'\n"
     + seq_src[seq_src.index("def build_codes"):seq_src.index("prompt_ids =")], g)
g["FPS"] = 12.5

turns = []
for i in range(4):
    user = torch.randint(0, 2048, (8, 120))
    codes, start = g["build_codes"](user, [5, 6, 7], 6)
    assert codes.shape == (17, 120 + 6 + 3 + 6) and start == 15
    words = [dict(word=w, start=0.5 + 0.6 * j, end=0.9 + 0.6 * j) for j, w in enumerate(["hello", "world", "speech", "model", "a"])]
    turns.append(dict(id=str(i), codes=codes, user_start=start, words=words, text="hello world speech model a" if i < 2 else "other text"))

lens_src = cell("def lens_forward").split("import time")[0]
exec(lens_src, g)
top, hid = g["lens_forward"](lm, devs, turns[0]["codes"], 10, [0, 1, 2])
assert top.shape == (4, 135, 10), top.shape
assert set(hid) == {0, 1, 2}

# lens at the last layer must equal the model's own forward_codes text logits (special tokens masked)
initial = lm._get_initial_token()
seq = torch.cat([initial, g["_delay_sequence"](lm.delays, turns[0]["codes"].unsqueeze(0), initial)], 2)[:, :, :-1]
_, text_logits = lm.forward_codes(seq)
ref_logits = text_logits[0, 0].float().clone()
ref_logits[:, [0, 1, 2, 3]] = -float("inf")
assert torch.equal(ref_logits.topk(10, -1).indices.int(), top[-1]), "final-layer lens != model text head"

# same weights as the original model (bf16 rounding aside)?
_, ref_text = ref.forward_codes(seq)
d = (ref_text.float() - text_logits.float()).abs().max().item(); print("max |ref - loaded| logits:", d)
assert d < 1e-3, "loaded weights differ from reference"

for t in turns:
    t["top"], t["hid"] = g["lens_forward"](lm, devs, t["codes"], 10, [0, 1, 2])
g["turns"] = turns
exec(cell("def word_targets"), g)
rows = g["rows"]
assert rows and rows[0]["real"].shape == (4,) and rows[0]["ctrl"].shape[1] == 4
# a planted hit: put the word's target id into the window at layer 2 and check rank 0 there
t0 = turns[0]; w = t0["words"][0]
a, b = g["window"](t0, w, 4)
tid = next(iter(g["word_targets"](w["word"])))
t0["top"][2, a, 0] = tid
r = g["best_rank"](t0["top"], a, b, {tid})
assert r[2] == 0
exec(cell("def hit_curves"), g)
print(g["curves"].head())
print("ALL CHECKS PASSED")
