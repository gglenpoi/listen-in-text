import os, sys, random, string
os.environ["NO_TORCH_COMPILE"] = "1"
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, "/root/personaplex/moshi")
import nbformat, numpy as np, torch
from safetensors.torch import save_file
SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp"); os.makedirs(SCR, exist_ok=True)
nb = nbformat.read(sys.argv[1], as_version=4)
src = [c.source for c in nb.cells if c.cell_type == "code"]
def cell(m):
    h = [s for s in src if m in s]; assert len(h) == 1, m; return h[0]
g = {}
exec(cell("CFG = dict(").replace('CFG["out_dir"] = f"{BASE}/{CFG[\'model\']}_v9"', f'CFG["out_dir"] = "{SCR}/v9_test"'), g)
g["CFG"].update(n_boot=100, n_turns=8, n_nat=3, n_beh=4, n_patch=2, nat_turn_sec=4.0, trig_sec=2.0, patch_blocks=[[1, 1], [2, 3], [1, 3]])
exec(cell("import json, math, random, re, tarfile"), g)
kw = dict(g["loaders"]._lm_kwargs, dim=64, num_heads=4, num_layers=3, depformer_dim=32, depformer_dim_feedforward=64,
          depformer_num_layers=1, depformer_num_heads=2, dep_q=16)
ref = g["LMModel"](device="cpu", dtype=torch.float32, **kw).eval()
for p in ref.parameters(): torch.nn.init.normal_(p, std=0.05)
sd = {k: v.contiguous().to(torch.bfloat16) for k, v in ref.state_dict().items()}
# emulate an 8-step depformer checkpoint: drop modules 8..15 and halve per-step self-attn weights
for k in list(sd):
    if any(f"{r}.{i}." in k for r in ("linears", "depformer_in", "depformer_emb") for i in range(8, 16)): del sd[k]
save_file(sd, f"{SCR}/tiny9.safetensors")
cl = cell("def load_lm_full")
exec(cl.split("DTYPE = pick_dtype()")[0], g)
g["DTYPE"] = torch.float32
g["lm"], g["LAYER_DEVS"] = g["load_lm_full"](f"{SCR}/tiny9.safetensors", kw, torch.float32)
g["N_LAYERS"] = 4
lm = g["lm"]
assert torch.equal(lm.linears[9].weight, lm.linears[1].weight)
class FakeMimi(torch.nn.Module):
    def __init__(s): super().__init__(); s.p = torch.nn.Parameter(torch.zeros(1)); s.sample_rate = 24000; s.frame_rate = 12.5
    def encode(s, x): return torch.randint(0, 2048, (1, 8, x.shape[-1] // 1920))
    def decode(s, c): return torch.randn(1, 1, c.shape[-1] * 1920) * 0.1
g["mimi"] = FakeMimi(); g["FRAME"], g["FPS"] = 1920, 12.5
rng = random.Random(0); WORDS = sorted({"".join(rng.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(300)})
VOCAB = ["<epad>", "<bos>", "<eos>", "<pad>"] + WORDS; IDX = {w: i for i, w in enumerate(VOCAB)}
class SP:
    def encode(s, x): return [IDX.get(w.lower(), 5) for w in x.split()]
    def id_to_piece(s, i): return "▁" + VOCAB[i]
    def decode(s, ids): return " ".join(VOCAB[i] if i < len(VOCAB) else "?" for i in ids)
g["sp"] = SP()
DATA = {}
for name in ("librispeech", "ami"):
    DATA[name] = []
    for i in range(8):
        words = [dict(word=rng.choice(WORDS), start=0.3 + 0.5 * j, end=0.7 + 0.5 * j) for j in range(14)]
        DATA[name].append(dict(id=f"{name}{i}", spk=f"s{i}", wav=np.random.randn(int(8 * 24000)).astype(np.float32) * 0.1,
                               segments=[(0, 8, " ".join(w["word"] for w in words))], words=words))
g.update(DATA=DATA, MAIN=["librispeech", "ami"])
exec(cell("from moshi.utils.sampling import sample_token"), g)
# generation writes only the model's own streams inside the window
U = DATA["librispeech"][0]
base, ustart = g["natural_seq"](g["mimi_codes"](U["wav"][:48000]), g["SIL"].repeat(1, 40))
g0 = ustart - g["LEAD"]; G = g["LEAD"] + 40
gen = g["generate"](base, g0, G, 0)
assert torch.equal(gen[9:], base[9:]) and torch.equal(gen[:, :g0], base[:, :g0]) and torch.equal(gen[:, g0 + G:], base[:, g0 + G:])
assert not torch.equal(gen[1:9, g0:g0 + G], base[1:9, g0:g0 + G])
# record/patch identities
codes, start = g["standard_seq"](g["mimi_codes"](U["wav"]))
a, b = start, start + 50
_, _, rec = g["run"](codes, (), record=(a, b))
t_plain, _, _ = g["run"](codes, (3,), 20)
t_patch, _, _ = g["run"](codes, (3,), 20, patch=(rec, a, b, {1, 2, 3}, "layer"))
assert torch.equal(t_plain[3], t_patch[3])
t_attn, _, _ = g["run"](codes, (3,), 20, patch=(rec, a, b, {2}, "attn"))
assert torch.equal(t_plain[3], t_attn[3])
exec(cell("COND_STD = "), g)
exec(cell("def score_excl"), g)
exec(cell("NAT, GEN_ROWS = {}, []"), g)
assert len(g["GEN"]) == 6
exec(cell("COND_NAT = "), g)
exec(cell("def response_pair"), g)
BR = g["BRES"]; l = BR[BR.cond.isin(["listen", "nat_listen"]) & (BR.metric == "kl")]
assert np.allclose(l.retention, 1.0) and len(l) == 4, l
exec(cell("PATCH_PAIRS = "), g)
P = g["PSUM"]; assert set(P.comp) == {"layer", "attn", "mlp"} and len(P) > 0
exec(cell("fig, axes = plt.subplots(1, 3"), g)
exec(cell("json.dump(CFG"), g)
print(sorted(os.listdir(f"{SCR}/v9_test")))
print("ALL V9 CHECKS PASSED")
# depformer: incremental (as in generate) == training forward on the same tokens
h = torch.randn(1, 1, 64); seq = torch.randint(0, 2000, (1, 17, 1)); seq[0, 0, 0] = 7
ref_logits = lm.forward_depformer_training(seq, lm.out_norm(h))
hn = lm.out_norm(h); ins = []
for k in range(8):
    prev = seq[:, 0] if k == 0 else seq[:, k]
    emb = lm.depformer_text_emb(prev) if k == 0 else lm.depformer_emb[k - 1](prev)
    ins.append(lm.depformer_in[k](hn) + emb)
    out = lm.depformer(torch.cat(ins, 1))
    assert torch.allclose(lm.linears[k](out[:, -1]), ref_logits[0, k, 0], atol=1e-4), k
print("DEPFORMER CHECK PASSED")
