"""Builds nearmiss_v3.ipynb: CPU-only re-analysis of near-misses from v2 outputs (top5.npz + turns.json)."""
import sys
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Listen in Text — v3: что за «почти попадания»?

Ноутбук **не требует GPU**: он пересчитывает анализ по выходам v2 (`top5.npz`, `turns.json`, `config.json`) для обеих моделей.

**Два вопроса.**
1. **Контроль темы.** В v2 контрольное окно бралось из *другой* реплики. Смысловое сходство кандидатов могло отражать общую тему реплики, а не конкретное слово. Здесь добавлен второй контроль: окно той же длины из **той же реплики**, но не ближе 2 с к слову и без других вхождений этого слова.
2. **Классификация кандидатов.** Каждый кандидат из top-5 в окне слова (кроме самого слова) относится к одной категории:
   - **однокоренное** (city → cities, begun → began, know → knowledge): общая лемма или основа;
   - **омофон** (here → hear): совпадают фонемы CMUdict;
   - **по звуку** (word → world): фонетическая близость ≥ 0,5, смысловая < 0,6;
   - **по смыслу** (difficult → hard): смысловая близость ≥ 0,6, фонетическая < 0,5;
   - **и то и другое**, **прочее**.

   Для каждой категории считаем долю слов, у которых в окне есть хотя бы один такой кандидат, и вычитаем ту же долю в контрольных окнах. Плюс «чистые» меры: максимальное смысловое сходство среди кандидатов, *не похожих по звуку*, и максимальное звуковое среди *не похожих по смыслу*.

**Входные данные (Kaggle).** *Add Input → Datasets* → загрузите датасет с папками `personaplex_out` и `moshi_out` (нужны только `top5.npz`, `turns.json`, `config.json`). Ноутбук сам найдёт все `top5.npz` в `/kaggle/input`. Accelerator: **None**, Internet: **On**.
""")

md("## 1. Конфигурация")
code(r"""
import os
ON_KAGGLE = os.path.exists("/kaggle/working")
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")
BASE = "/kaggle/working" if ON_KAGGLE else ("/content" if ON_COLAB else os.getcwd())

CFG = dict(
    search_roots   = ["/kaggle/input", "/content", BASE],   # where to look for */top5.npz (recursively)
    fps            = 12.5,
    lag_frames     = 4,        # same window as v2: word frames + 320 ms
    min_word_chars = 4,
    n_ctrl         = 3,        # control windows per word, for each control type
    same_turn_gap_sec = 2.0,   # same-turn control must be at least this far from the word
    sem_thr        = 0.60,     # MiniLM cosine: 'similar in meaning'
    phon_thr       = 0.50,     # 1 - normalised phone edit distance: 'similar in sound'
    sem_low        = 0.30,     # 'not similar in meaning' for the pure phonetic measure
    phon_low       = 0.34,     # 'not similar in sound' for the pure semantic measure
    example_layer  = 28,
    ci_layers      = [12, 16, 20, 24, 26, 28, 30, 32],
    n_boot         = 1000,
    tokenizer_repo = "kyutai/moshiko-pytorch-bf16",   # same SentencePiece tokenizer as PersonaPlex, public
    seed           = 0,
    out_dir        = f"{BASE}/nearmiss_out",
)
os.makedirs(CFG["out_dir"], exist_ok=True)
""")

md("## 2. Установка")
code(r"""
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nltk", "sentence-transformers", "sentencepiece", "huggingface_hub"], check=True)
""")
code(r"""
import json, glob, math, random, re, zipfile, io, urllib.request
from functools import lru_cache
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
""")

md("## 3. Входные данные\nИщем все `top5.npz`; имя модели берём из соседнего `config.json`. Если найдены только zip-архивы v2, распаковываем их.")
code(r"""
def discover(roots):
    found = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for z in glob.glob(f"{root}/**/*_out.zip", recursive=True):
            dst = os.path.join(CFG["out_dir"], "unzipped", os.path.basename(z)[:-4])
            if not os.path.exists(dst):
                zipfile.ZipFile(z).extractall(dst)
        for p in glob.glob(f"{root}/**/top5.npz", recursive=True) + glob.glob(f"{CFG['out_dir']}/unzipped/**/top5.npz", recursive=True):
            d = os.path.dirname(p)
            if not os.path.exists(f"{d}/turns.json"):
                continue
            cfg = json.load(open(f"{d}/config.json")) if os.path.exists(f"{d}/config.json") else {}
            name = cfg.get("model", os.path.basename(d).replace("_out", ""))
            found.setdefault(name, d)
    return found

RUN_DIRS = discover(CFG["search_roots"])
assert RUN_DIRS, "no */top5.npz found: attach the v2 outputs as input"
RUNS = {}
for name, d in sorted(RUN_DIRS.items()):
    turns = json.load(open(f"{d}/turns.json"))
    z = np.load(f"{d}/top5.npz")
    for t in turns:
        t["top"] = z[t["id"]]                       # [L+1, T, 5] int32
    RUNS[name] = turns
    n_words = sum(len(t["words"]) for t in turns)
    print(f"{name:12s} {d} | {len(turns)} turns, {n_words} words, top shape {turns[0]['top'].shape}")
N_LAYERS = next(iter(RUNS.values()))[0]["top"].shape[0]
""")

md("## 4. Токенизатор и словарь кандидатов\nКандидат — кусок SentencePiece, который является целым словом: начинается с `▁`, только латинские буквы, длина ≥ 3.")
code(r"""
import sentencepiece
from huggingface_hub import hf_hub_download
sp = sentencepiece.SentencePieceProcessor(hf_hub_download(CFG["tokenizer_repo"], "tokenizer_spm_32k_3.model"))
SPECIAL_TEXT = {0, 1, 2, 3}

V = sp.get_piece_size()
CAND_WORDS, LUT = [], np.full(V + 1, -1, dtype=np.int64)
for i in range(V):
    p = sp.id_to_piece(i)
    w = p[1:]
    if p.startswith("▁") and len(w) >= 3 and w.isascii() and w.isalpha():
        LUT[i] = len(CAND_WORDS)
        CAND_WORDS.append(w.lower())
CAND_WORDS = np.array(CAND_WORDS)
print(len(CAND_WORDS), "whole-word candidate pieces")

def word_targets(word):
    ids = set()
    for v in {word, word.capitalize(), word.upper()}:
        enc = sp.encode(v)
        if enc:
            ids.add(enc[0])
    return ids - SPECIAL_TEXT
""")

md("## 5. Лексические ресурсы\nCMUdict (фонемы, все варианты произношения), WordNet + Porter (однокоренные), MiniLM (смысл). Если `nltk.download` не сработает, словари качаются напрямую с GitHub.")
code(r"""
import nltk
def ensure_nltk(pkg, path):
    try:
        nltk.data.find(path)
        return
    except LookupError:
        pass
    try:
        if nltk.download(pkg, quiet=True):
            nltk.data.find(path); return
    except Exception:
        pass
    target = os.path.join(os.path.expanduser("~"), "nltk_data", "corpora")
    os.makedirs(target, exist_ok=True)
    url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{pkg}.zip"
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extractall(target)
    nltk.data.find(path)

ensure_nltk("cmudict", "corpora/cmudict")
ensure_nltk("wordnet", "corpora/wordnet")
from nltk.corpus import cmudict, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
CMU = cmudict.dict()
STEM, LEMMA = PorterStemmer(), WordNetLemmatizer()

@lru_cache(maxsize=None)
def prons(w):
    return tuple(tuple(re.sub(r"\d", "", x) for x in p) for p in CMU.get(w, []))

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

@lru_cache(maxsize=None)
def lemmas(w):
    out = {w, STEM.stem(w)}
    for pos in "nvar":
        try:
            out.add(LEMMA.lemmatize(w, pos))
        except Exception:
            pass
    return frozenset(out)

def is_morph(w, c):
    if w.startswith(c) or c.startswith(w):
        return True
    return bool(lemmas(w) & lemmas(c))

@lru_cache(maxsize=None)
def phon(w, c):
    pw, pc = prons(w), prons(c)
    if not pw or not pc:
        return np.nan
    return max(1 - lev(a, b) / max(len(a), len(b)) for a in pw for b in pc)

def is_homophone(w, c):
    return bool(set(prons(w)) & set(prons(c)))

from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
""")

md("## 6. Окна: слово, контроль из другой реплики, контроль из той же реплики")
code(r"""
FPS, LAG = CFG["fps"], CFG["lag_frames"]

def word_window(t, w):
    f0 = int(math.floor(w["start"] * FPS)); f1 = int(math.ceil(w["end"] * FPS))
    a = t["user_start"] + f0 + 1
    b = min(t["user_start"] + f1 + 1 + LAG, t["top"].shape[1] - 1)
    return a, b

def make_jobs(turns, seed):
    rng = random.Random(seed)
    gap = int(CFG["same_turn_gap_sec"] * FPS)
    jobs = []
    for ti, t in enumerate(turns):
        others = [u for j, u in enumerate(turns) if j != ti]
        occ = {}
        for w in t["words"]:
            occ.setdefault(w["word"], []).append(word_window(t, w))
        for w in t["words"]:
            word = w["word"]
            if len(word) < CFG["min_word_chars"] or not word.isalpha():
                continue
            targets = word_targets(word)
            a, b = word_window(t, w)
            if b < a or not targets:
                continue
            n = b - a
            ctrl_other, ctrl_same = [], []
            for _ in range(50):
                if len(ctrl_other) >= CFG["n_ctrl"]:
                    break
                u = rng.choice(others)
                if word in u["text"].split():
                    continue
                lo, hi = u["user_start"] + 1, u["top"].shape[1] - 1 - n
                if hi > lo:
                    s = rng.randint(lo, hi); ctrl_other.append((u, s, s + n))
            lo, hi = t["user_start"] + 1, t["top"].shape[1] - 1 - n
            for _ in range(200):
                if len(ctrl_same) >= CFG["n_ctrl"] or hi <= lo:
                    break
                s = rng.randint(lo, hi)
                if abs((s + n / 2) - (a + n / 2)) < gap:
                    continue
                if any(not (s + n < oa or s > ob) for oa, ob in occ[word]):
                    continue                                   # must not overlap any occurrence of the same word
                ctrl_same.append((t, s, s + n))
            if len(ctrl_other) and len(ctrl_same):
                jobs.append(dict(word=word, targets=targets, real=(t, a, b), other=ctrl_other, same=ctrl_same))
    return jobs

JOBS = {name: make_jobs(turns, CFG["seed"]) for name, turns in RUNS.items()}
for name, j in JOBS.items():
    print(f"{name}: {len(j)} content words with both controls")
""")

md("## 7. Классификация кандидатов\nДля каждого слова, слоя и окна: набор кандидатов (без самого слова) → флаги категорий и «чистые» меры. Контрольные окна усредняются.")
code(r"""
CATS = ["morph", "homophone", "sound", "meaning", "both", "other"]

# embeddings for every candidate word and target word
vocab = sorted(set(CAND_WORDS.tolist()) | {j["word"] for jobs in JOBS.values() for j in jobs})
emb = st_model.encode(vocab, batch_size=512, normalize_embeddings=True, show_progress_bar=True)
EIDX = {w: i for i, w in enumerate(vocab)}
EMB = np.asarray(emb, dtype=np.float32)

@lru_cache(maxsize=None)
def classify(w, c):
    # returns (category, sem, phon) for target w and candidate c (c != w)
    s = float(EMB[EIDX[w]] @ EMB[EIDX[c]])
    p = phon(w, c)
    if is_morph(w, c):
        return "morph", s, p
    if is_homophone(w, c):
        return "homophone", s, p
    sound = (not np.isnan(p)) and p >= CFG["phon_thr"]
    mean = s >= CFG["sem_thr"]
    cat = "both" if (sound and mean) else "meaning" if mean else "sound" if sound else "other"
    return cat, s, p

def window_stats(top, L, a, b, word, target_ids):
    ids = top[L, a:b + 1, :].reshape(-1)
    ids = ids[~np.isin(ids, list(target_ids))]
    cidx = LUT[ids]
    cands = set(CAND_WORDS[cidx[cidx >= 0]].tolist()) - {word}
    out = {c: 0.0 for c in CATS}
    sem_any = phon_any = sem_pure = phon_pure = np.nan
    pairs = []
    for c in cands:
        cat, s, p = classify(word, c)
        out[cat] = 1.0
        pairs.append((c, cat))
        if cat == "morph":
            continue
        sem_any = s if np.isnan(sem_any) else max(sem_any, s)
        if not np.isnan(p):
            phon_any = p if np.isnan(phon_any) else max(phon_any, p)
            if p < CFG["phon_low"] and cat != "homophone":
                sem_pure = s if np.isnan(sem_pure) else max(sem_pure, s)
            if s < CFG["sem_low"]:
                phon_pure = p if np.isnan(phon_pure) else max(phon_pure, p)
    out.update(sem_any=sem_any, phon_any=phon_any, sem_pure=sem_pure, phon_pure=phon_pure, n_cands=len(cands))
    return out, pairs

METRICS = CATS + ["sem_any", "phon_any", "sem_pure", "phon_pure", "n_cands"]

def run_model(name):
    rows, examples = [], []
    for L in range(N_LAYERS):
        for wi, j in enumerate(JOBS[name]):
            t, a, b = j["real"]
            real, pairs = window_stats(t["top"], L, a, b, j["word"], j["targets"])
            if L == CFG["example_layer"]:
                examples += [(j["word"], c, cat) for c, cat in pairs]
            row = dict(model=name, layer=L, wi=wi, word=j["word"])
            for cond in ("other", "same"):
                stats = [window_stats(u["top"], L, s, e, j["word"], j["targets"])[0] for u, s, e in j[cond]]
                for m in METRICS:
                    vals = np.array([st[m] for st in stats], dtype=float)
                    row[f"{m}_{cond}"] = np.nanmean(vals) if not np.all(np.isnan(vals)) else np.nan
            for m in METRICS:
                row[f"{m}_real"] = real[m]
            rows.append(row)
    return pd.DataFrame(rows), examples

import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import time
RES, EXAMPLES = [], {}
for name in RUNS:
    t0 = time.time()
    df, ex = run_model(name)
    RES.append(df); EXAMPLES[name] = ex
    print(f"{name}: {len(df)} rows in {time.time() - t0:.0f}s")
RES = pd.concat(RES, ignore_index=True)
RES.to_csv(f"{CFG['out_dir']}/nearmiss_rows.csv.gz", index=False)
""")

md("## 8. Сводка: избыток над контролем\n`excess_same` — главный показатель (контроль из той же реплики, тема зафиксирована). `excess_other` — как в v2. 95% доверительные интервалы — бутстреп по словам.")
code(r"""
rng = np.random.default_rng(CFG["seed"])

def boot_ci(x, n_boot):
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan, np.nan
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

summ = []
for (name, L), g in RES.groupby(["model", "layer"]):
    for m in METRICS:
        real, oth, same = g[f"{m}_real"].values, g[f"{m}_other"].values, g[f"{m}_same"].values
        row = dict(model=name, layer=L, metric=m, real=np.nanmean(real), ctrl_other=np.nanmean(oth), ctrl_same=np.nanmean(same),
                   excess_other=np.nanmean(real - oth), excess_same=np.nanmean(real - same))
        if L in CFG["ci_layers"]:
            row["ci_lo"], row["ci_hi"] = boot_ci(real - same, CFG["n_boot"])
        summ.append(row)
SUMM = pd.DataFrame(summ)
SUMM.to_csv(f"{CFG['out_dir']}/nearmiss_summary.csv", index=False)

for name in RUNS:
    print(f"\n=== {name}: excess over SAME-turn control (95% CI) ===")
    d = SUMM[(SUMM.model == name) & SUMM.layer.isin(CFG["ci_layers"])]
    tab = d.assign(cell=lambda x: x.excess_same.map("{:+.3f}".format) + " [" + x.ci_lo.map("{:+.3f}".format) + ", " + x.ci_hi.map("{:+.3f}".format) + "]")
    print(tab.pivot(index="metric", columns="layer", values="cell").loc[METRICS].to_string())
""")

md("### Контроль темы: смысловой избыток относительно двух контролей\nЕсли смысловой избыток на средних слоях держится только против контроля из другой реплики и исчезает против контроля из той же, это была тема, а не слово.")
code(r"""
for name in RUNS:
    d = SUMM[(SUMM.model == name) & (SUMM.metric.isin(["sem_any", "sem_pure", "phon_any", "phon_pure"]))]
    print(f"\n{name}")
    print(d.pivot(index="layer", columns="metric", values="excess_other").add_suffix("_vs_other")
          .join(d.pivot(index="layer", columns="metric", values="excess_same").add_suffix("_vs_same"))
          .iloc[::2].round(3).to_string())
""")

md("## 9. Графики")
code(r"""
colors = dict(zip(CATS, ["#7f7f7f", "#9467bd", "#1f77b4", "#d62728", "#ff7f0e", "#c7c7c7"]))
names = list(RUNS)
fig, axes = plt.subplots(2, len(names), figsize=(7.5 * len(names), 8), squeeze=False, sharey="row")
for col, name in enumerate(names):
    d = SUMM[SUMM.model == name]
    ax = axes[0, col]
    for c in CATS:
        x = d[d.metric == c]
        ax.plot(x.layer, x.excess_same, color=colors[c], lw=2, label=c)
        ci = x.dropna(subset=["ci_lo"])
        ax.errorbar(ci.layer, ci.excess_same, yerr=[ci.excess_same - ci.ci_lo, ci.ci_hi - ci.excess_same],
                    fmt="none", ecolor=colors[c], capsize=2, lw=1)
    ax.axhline(0, color="k", lw=.8); ax.grid(alpha=.3)
    ax.set(title=f"{name}: near-miss categories (excess over same-turn control)", xlabel="layer",
           ylabel="Δ share of words with ≥1 such candidate")
    ax.legend(fontsize=8)
    ax = axes[1, col]
    for m, c, lab in (("sem_pure", "#d62728", "meaning, among sound-unlike cands"),
                      ("phon_pure", "#1f77b4", "sound, among meaning-unlike cands")):
        x = d[d.metric == m]
        ax.plot(x.layer, x.excess_same, color=c, lw=2, label=f"{lab} (vs same turn)")
        ax.plot(x.layer, x.excess_other, color=c, lw=1.5, ls=":", label=f"{lab} (vs other turn)")
    ax.axhline(0, color="k", lw=.8); ax.grid(alpha=.3)
    ax.set(title=f"{name}: pure meaning vs pure sound similarity", xlabel="layer", ylabel="max similarity, real − control")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/nearmiss.png", dpi=150); plt.show()
""")

md("## 10. Примеры на слое `example_layer`\nСамые частые пары «слово → кандидат» по категориям в окнах настоящих слов.")
code(r"""
for name, ex in EXAMPLES.items():
    ex = pd.DataFrame(ex, columns=["word", "cand", "cat"])
    ex.to_csv(f"{CFG['out_dir']}/examples_{name}_L{CFG['example_layer']}.csv", index=False)
    print(f"\n=== {name}, layer {CFG['example_layer']} ===")
    for c in ["homophone", "sound", "meaning", "both", "morph"]:
        top = ex[ex.cat == c].groupby(["word", "cand"]).size().sort_values(ascending=False).head(12)
        print(f"{c:>9s}: " + ", ".join(f"{w}→{k}" for (w, k) in top.index))
""")

md("## 11. Сохранение")
code(r"""
json.dump(CFG, open(f"{CFG['out_dir']}/config.json", "w"), indent=2)
import shutil
from IPython.display import FileLink, display
archive = shutil.make_archive(CFG["out_dir"], "zip", CFG["out_dir"])
print(archive, f"{os.path.getsize(archive) / 1e6:.1f} MB")
os.chdir(BASE); display(FileLink(os.path.basename(archive)))
""")

md(r"""
## Как читать

- **Категории (верхний ряд).** Положительный избыток с ДИ, не пересекающим ноль, значит, такие кандидаты появляются рядом со словом чаще, чем в случайном месте той же реплики. Сравните размер «по смыслу» и «по звуку» на слоях 24–30.
- **Чистые меры (нижний ряд).** Сплошная линия (контроль из той же реплики) ниже пунктирной (из другой) — часть сходства объяснялась темой. Если «чистый смысл» остаётся выше нуля против контроля той же реплики, смысловые соседи — свойство слова, а не темы.
- **Оговорки.** MiniLM на отдельных словах отчасти учитывает написание; пороги 0,6 и 0,5 условные — чувствительность стоит проверить, поменяв `sem_thr` и `phon_thr`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "nearmiss_v3.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
