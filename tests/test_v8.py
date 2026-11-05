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
exec(cell("CFG = dict(").replace('CFG["out_dir"] = f"{BASE}/{CFG[\'model\']}_v8"', f'CFG["out_dir"] = "{SCR}/v8_test"'), g)
g["CFG"].update(n_boot=100)
exec(cell("import json, math, random, re, tarfile"), g)
kw = dict(g["loaders"]._lm_kwargs, dim=64, num_heads=4, num_layers=3, depformer_dim=32, depformer_dim_feedforward=64, depformer_num_layers=1, depformer_num_heads=2, dep_q=16)
ref = g["LMModel"](device="cpu", dtype=torch.float32, **kw).eval()
for p in ref.parameters(): torch.nn.init.normal_(p, std=0.05)
save_file({k: v.contiguous().to(torch.bfloat16) for k, v in ref.state_dict().items()}, f"{SCR}/tiny.safetensors")
exec(cell("def load_lm_for_lens").split("DTYPE = pick_dtype()")[0], g)
g["DTYPE"] = torch.float32; g["lm"], g["LAYER_DEVS"] = g["load_lm_for_lens"](f"{SCR}/tiny.safetensors", kw, torch.float32)
class FakeMimi(torch.nn.Module):
    def __init__(s): super().__init__(); s.p = torch.nn.Parameter(torch.zeros(1)); s.sample_rate = 24000; s.frame_rate = 12.5
    def encode(s, x): return torch.randint(0, 2048, (1, 8, x.shape[-1] // 1920))
g["mimi"] = FakeMimi(); g["FRAME"], g["FPS"] = 1920, 12.5
rng = random.Random(0); WORDS = sorted({"".join(rng.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(300)})
VOCAB = ["<epad>", "<bos>", "<eos>", "<pad>"] + WORDS; IDX = {w: i for i, w in enumerate(VOCAB)}
class SP:
    def encode(s, x): return [IDX.get(w.lower(), 5) for w in x.split()]
    def id_to_piece(s, i): return "▁" + VOCAB[i]
g["sp"] = SP()
exec("import re\ndef norm_words(text):\n    return [w for w in (re.sub(r\"[^a-z']\", '', t) for t in text.lower().split()) if w.strip(\"'\")]", g)
DATA = {}
for name in ("librispeech", "ami"):
    DATA[name] = [dict(id=f"{name}{i}", spk=f"s{i}", wav=np.random.randn(int(12 * 24000)).astype(np.float32) * 0.1,
                       segments=[(0, 12, " ".join(rng.choice(WORDS) for _ in range(15)))]) for i in range(5)]
g.update(DATA=DATA, MAIN=["librispeech", "ami"])
exec(cell("def speech_shaped_noise"), g)
exec(cell("def first_word(codes"), g)
R = g["R"]; assert len(R) == 2 * 5 * len(g["CFG"]["conditions"]) and (R.kl >= -1e-5).all()
exec(cell("R[\"dlogp\"]"), g)
RES = g["RES"]; l = RES[(RES.cond == "listen") & (RES.metric == "kl")]
assert np.allclose(l.retention, 1.0)
exec(cell("json.dump(CFG"), g)
print("ALL V8 CHECKS PASSED")
