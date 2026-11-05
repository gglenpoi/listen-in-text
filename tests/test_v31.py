import sys, types, os, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import nbformat
SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp"); os.makedirs(SCR, exist_ok=True)
from nltk.corpus import cmudict
WORDS = sorted(w for w in cmudict.dict() if w.isalpha() and len(w) >= 3)[:31000]
IDX = {w: i + 4 for i, w in enumerate(WORDS)}
spm = types.ModuleType("sentencepiece")
class SPP:
    def __init__(self, *a): pass
    def get_piece_size(self): return 32000
    def id_to_piece(self, i): return "▁" + WORDS[i - 4] if 4 <= i < len(WORDS) + 4 else f"<{i}>"
    def encode(self, s): return [IDX.get(s.lower(), 31999)]
spm.SentencePieceProcessor = SPP; sys.modules["sentencepiece"] = spm
hh = types.ModuleType("huggingface_hub"); hh.hf_hub_download = lambda *a, **k: "x"; sys.modules["huggingface_hub"] = hh
st = types.ModuleType("sentence_transformers")
class ST:
    def __init__(self, *a, **k): pass
    def encode(self, words, **k):
        v = np.random.default_rng(0).normal(size=(len(words), 32)); return v / np.linalg.norm(v, axis=1, keepdims=True)
st.SentenceTransformer = ST; sys.modules["sentence_transformers"] = st
nb = nbformat.read(sys.argv[1], 4)
g = {}
os.chdir(SCR)
for c in nb.cells:
    if c.cell_type != "code" or "pip\", \"install\"" in c.source: continue
    src = c.source
    if "CFG = dict(" in src:
        src += "\nCFG.update(search_roots=['/mnt/user-data/uploads', '_tmp/v4_test'], out_dir=f'{BASE}/nearmiss31_test', n_boot=200)\nos.makedirs(CFG['out_dir'], exist_ok=True)"
    t0 = time.time(); exec(src, g); print(f"-- cell ok ({time.time()-t0:.0f}s): {src.splitlines()[0][:60]}")
print("V3 RUN COMPLETE")
