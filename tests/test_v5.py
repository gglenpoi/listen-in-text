"""Smoke test for listen_in_text_v5.ipynb on a tiny random model with planted effects for each hypothesis."""
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

g = {}
exec(cell("CFG = dict(").replace('CFG["out_dir"] = f"{BASE}/{CFG[\'model\']}_v5"', f'CFG["out_dir"] = "{SCR}/v5_test"'), g)
g["CFG"].update(n_turns=8, tl_layers=[0, 2, 3], ab_layers=[2, 3], check_layer=3, tl_steps=300, tl_ranks=[32],
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
        out.append(dict(id=f"{name}-{i}", spk=f"{name}-spk{i}", wav=np.zeros(int(dur * 24000), np.float32),
                        segments=[(0, dur, " ".join(w["word"] for w in words))], words=words))
    return out
DATA = {"librispeech": mk("ls", 8), "ami": mk("ami", 8), "ls_dev-clean": mk("dev", 4), "ls_test-other": mk("oth", 4)}
g.update(DATA=DATA, MAIN=["librispeech", "ami"], ALIGNER_USED="test")

exec(cell("def own_streams"), g)
n = DATA["librispeech"][0]["codes_user"].shape[1]
c, t_, iv = g["own_streams"](DATA["librispeech"][1], n, "talking_partial")
assert iv == (n // 3, n // 3 + min(n // 3, n)) or iv[0] == n // 3, iv
assert (t_[:n // 3] == 3).all() and (t_[iv[0]:iv[1]] != 3).any() and (t_[iv[1]:] == 3).all()
c, t_, _ = g["own_streams"](DATA["librispeech"][1], n, "talking_audio"); assert (t_ == 3).all() and not torch.equal(c[:, :5], g["SIL"].repeat(1, 5))
c, t_, _ = g["own_streams"](DATA["librispeech"][1], n, "talking_text"); assert (t_ != 3).any() and torch.equal(c, g["SIL"].repeat(1, n))
codes, start = g["build_codes"](DATA["librispeech"][0]["codes_user"], c, t_, g["N_POST"])
assert codes.shape[1] == start + n + g["N_POST"]

exec(cell("def lens_forward"), g)
RUNS = g["RUNS"]
assert ("ls_dev-clean", "listen") in RUNS and ("ls_dev-clean", "talking") not in RUNS

# ---- plant effects at layer 2 ----
def plant(t, wi, w, ids_before=(), cond=""):
    a, b = g["word_window"](t, w, 0)
    for p in range(a + 1, b + 1):
        row = list(ids_before) + [IDX[w["word"]]]
        for k, tid in enumerate(row[:t["top"].shape[2]]):
            t["top"][2, p, k] = tid
for (name, cond), turns in RUNS.items():
    for t in turns:
        s0, s1 = t["own_interval"]
        own = [i for i in t["own_ids"]][:6]
        for wi, w in enumerate(t["words"]):
            if wi % 3 == 0: continue
            if cond == "listen": plant(t, wi, w)
            elif cond == "talking_text": plant(t, wi, w, ids_before=own[:5])       # pure top-k competition: own tokens in front
            elif cond == "talking": plant(t, wi, w, ids_before=own[:5]) if wi % 2 else None   # half suppressed + competition
            elif cond == "talking_partial":
                f0 = int(math.floor(w["start"] * 12.5)); f1 = int(math.ceil(w["end"] * 12.5))
                if f1 <= s0 or f0 >= s1: plant(t, wi, w)                            # suppressed only while the model talks
            # talking_audio: nothing planted = full suppression
exec(cell("def word_targets"), g)
P = g["PAIRED"]; L = 2
share = P[(P.dataset == "librispeech") & (P.layer == L)].set_index("cond").eval("drop / listen")
print(share.round(2).to_dict())
assert g["BEST"]["librispeech"] == 2
assert abs(share["talking_text"]) < 0.15, "own-token exclusion should remove pure competition"
assert share["talking_audio"] > 0.85 and 0.3 < share["talking"] < 0.7
C = g["CURVES"]
raw_txt = C[(C.dataset == "librispeech") & (C.cond == "talking_text") & (C.metric == "raw") & (C.layer == 2)].excess.item()
assert raw_txt < 0.2, raw_txt
exec(cell("cc = {\"listen\""), g)
exec(cell("def phase(r, interval)"), g)
D = g["DYNS"]; D = D[D.dataset == "librispeech"].set_index("bin")
print(D[["n", "drop"]].round(2).to_string())
assert D.loc["during", "drop"] > 0.5 and abs(D.loc["before", "drop"]) < 0.15

# ---- ablation mechanics ----
exec(cell("ALPHA = lm.out_norm.alpha"), g)
Q = g["basis"]({4, 5, 6, 7})["cpu"]
x = torch.randn(10, 64); y = x - (x @ Q) @ Q.T
W = (g["W_OUT"][[4, 5, 6, 7]].float() * g["ALPHA"]).T
assert torch.allclose(y @ W, torch.zeros(10, 4), atol=1e-4)
AB = g["AB"]
assert len(AB) > 0 and AB[["kl_own", "kl_ctrl", "mass_own", "nonpad_own"]].notna().all().all()
tt = g["RUNS"][("librispeech", "listen")][0]; a_, b_ = tt["user_start"], tt["user_start"] + tt["n_user"]
_, _, f_clean = g["lens_forward"](tt["codes"], 5, final_pos=list(range(a_, b_)))
_, f_abl = g["lens_forward"](tt["codes"], 5, ablate=(g["basis"](set(range(4, 60))), a_, b_), final_pos=list(range(a_, b_)), check_layer=3)
assert (f_clean - f_abl).abs().max() > 1e-2, "ablation had no effect"

# ---- tuned lens: make stored states learnable for seen words ----
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
T = g["TL"]
s = T[(T.test == "seen") & (T.layer == 2)].set_index(["lens", "rank"]).hit
print(s.round(2).to_dict())
assert s[("tuned", 32)] > s[("logit", 0)] + 0.3 and s[("tuned", 32)] > s[("shuffled", 32)] + 0.3
exec(cell("np.savez_compressed(f\"{d}/top5.npz\""), g)
print(sorted(os.listdir(f"{SCR}/v5_test")))
print("ALL V5 CHECKS PASSED")
