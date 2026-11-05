"""Smoke test for listen_in_text_v6.ipynb: tiny random model, planted effects, stubbed downloads."""
import os, sys, types, math, random, string
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

# fake GPT-2
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
exec(cell("CFG = dict(").replace('CFG["out_dir"] = f"{BASE}/{CFG[\'model\']}_v6"', f'CFG["out_dir"] = "{SCR}/v6_test"'), g)
g["CFG"].update(n_turns=8, tl_layers=[0, 2, 3], ab_layers=[2, 3], tl_steps=300, tl_ranks=[32],
                tl_batch=64, tl_lr=1e-2, n_boot=50, tl_extra_turns=4)
exec(cell("import json, math, random, re, tarfile"), g)
LMModel, loaders = g["LMModel"], g["loaders"]
kw = dict(loaders._lm_kwargs, dim=64, num_heads=4, num_layers=3, depformer_dim=32,
          depformer_dim_feedforward=64, depformer_num_layers=1, depformer_num_heads=2, dep_q=16)
ref = LMModel(device="cpu", dtype=torch.float32, **kw).eval()
for p in ref.parameters(): torch.nn.init.normal_(p, std=0.05)
path = f"{SCR}/tiny.safetensors"
save_file({k: v.contiguous().to(torch.bfloat16) for k, v in ref.state_dict().items()}, path)
exec(cell("def load_lm_for_lens").split("DTYPE = pick_dtype()")[0], g)
g["DTYPE"] = torch.float32
g["lm"], g["LAYER_DEVS"] = g["load_lm_for_lens"](path, kw, torch.float32)
g["N_LAYERS"] = 4
class FakeMimi(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1)); self.sample_rate = 24000; self.frame_rate = 12.5
    def encode(self, x): return torch.randint(0, 2048, (1, 8, x.shape[-1] // 1920))
g["mimi"] = FakeMimi(); g["FRAME"], g["FPS"] = 1920, 12.5

rng = random.Random(0)
WORDS = sorted({"".join(rng.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(300)})
VOCAB = ["<epad>", "<bos>", "<eos>", "<pad>"] + WORDS
IDX = {w: i for i, w in enumerate(VOCAB)}
class SP:
    def encode(self, s): return [IDX.get(w.lower(), 5) for w in s.split()]
    def id_to_piece(self, i): return "▁" + VOCAB[i] if i < len(VOCAB) else f"<{i}>"
g["sp"] = SP()
def mk(name, n):
    out = []
    for i in range(n):
        dur = 15.0
        words = [dict(word=rng.choice(WORDS), start=0.6 + 0.8 * j, end=1.0 + 0.8 * j) for j in range(17)]
        out.append(dict(id=f"{name}-{i}", spk=f"{name}-spk{i}", wav=np.random.randn(int(dur * 24000)).astype(np.float32) * 0.1,
                        segments=[(0, dur, " ".join(w["word"] for w in words))], words=words))
    return out
DATA = {"librispeech": mk("ls", 8), "ami": mk("ami", 8), "ls_dev-clean": mk("dev", 4), "ls_test-other": mk("oth", 4)}
g.update(DATA=DATA, MAIN=["librispeech", "ami"], ALIGNER_USED="test")

exec(cell("def speech_shaped_noise"), g)
w = DATA["librispeech"][0]["wav"]; nz = g["speech_shaped_noise"](w, 0)
assert abs(np.sqrt(np.mean(nz ** 2)) - np.sqrt(np.mean(w ** 2))) < 1e-3
c, t_, iv = g["own_streams"](DATA["librispeech"][1], 100, "own_noise")
assert torch.equal(c[:, :5], DATA["librispeech"][1]["codes_noise"][:, :5]) and (t_ == 3).all()

exec(cell("def lens_forward"), g)
RUNS = g["RUNS"]
assert RUNS[("librispeech", "talking_audio")][0]["own_words"] and not RUNS[("librispeech", "own_reversed")][0]["own_words"]

def plant(t, w, L=2):
    a, b = g["word_window"](t, w, 0)
    for p in range(a + 1, b + 1):
        t["top"][L, p, 0] = IDX[w["word"]]
for (name, cond), turns in RUNS.items():
    for t in turns:
        for wi, w in enumerate(t["words"]):
            if wi % 3 == 0: continue
            if cond in ("listen", "own_reversed", "own_quiet") or (cond == "own_noise" and wi % 2):
                plant(t, w)
        if cond == "talking_audio":
            for w in t["own_words"]: plant(t, w, L=3)       # the model 'hears itself' at layer 3
exec(cell("def score_all(name, cond, which"), g)
P = g["PAIRED"]
share = P[(P.dataset == "librispeech") & (P.layer == 2)].set_index("cond").eval("drop / listen")
print(share.round(2).to_dict())
assert share["talking_audio"] > 0.9 and abs(share["own_reversed"]) < 0.1 and abs(share["own_quiet"]) < 0.1 and 0.3 < share["own_noise"] < 0.7
exec(cell("own_curves = []"), g)
O = g["OWNC"]
assert O[(O.dataset == "librispeech") & (O.cond == "talking_audio") & (O.layer == 3)].excess.item() > 0.5

exec(cell("GAP = int(round"), g)
R = g["RESP"]
assert len(R) > 0 and {"kl_own", "kl_ctrl", "heard_own", "p_epad_clean"} <= set(R.columns)
# forced EPAD really sits in the text stream one frame before the read position
t = RUNS[("librispeech", "listen")][0]; b = t["user_start"] + t["n_user"]; e = b + g["GAP"]
assert e + 1 < t["T"]
exec(cell("def first_target_vec"), g)
X = g["CTX"]
assert len(X) > 0 and X.e_own.between(0, 1.0001).all() and X.e_ctrl.between(0, 1.0001).all()

E = torch.randn(len(VOCAB), 64)
for name, tf_ in g["TL_FRAMES"].items():
    words = [RUNS[(name, "listen")][ti]["words"][wi]["word"] for ti, wi, _ in tf_["meta"]]
    base = torch.stack([E[IDX[w]] for w in words])
    for L_ in tf_["h"]:
        tf_["h"][L_] = (base + 0.05 * torch.randn_like(base)).half()
with torch.no_grad():
    g["lm"].text_linear.weight[4:len(VOCAB)] *= 20
exec(cell("class LowRankLens"), g)
exec(cell("def prep(name)"), g)
T = g["TL"]; assert {"unseen_target", "ami_unseen_target"} <= set(T.test)
s = T[(T.test == "seen") & (T.layer == 2)].set_index(["lens", "rank"]).hit
assert s[("tuned", 32)] > s[("logit", 0)] + 0.3
exec(cell("np.savez_compressed(f\"{d}/top5.npz\""), g)
print(sorted(os.listdir(f"{SCR}/v6_test")))
print("ALL V6 CHECKS PASSED")
