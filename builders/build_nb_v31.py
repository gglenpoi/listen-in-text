"""Builds nearmiss_v31.ipynb by patching v3: stricter morphology, stop-word handling, threshold sweep, many runs."""
import sys
import nbformat as nbf

SRC = open(__file__.replace("build_nb_v31.py", "build_nb_v3.py")).read()
SRC = SRC.split("nb = nbf.v4.new_notebook()")[0]

def patch(old, new):
    global SRC
    assert SRC.count(old) == 1, (SRC.count(old), old[:70])
    SRC = SRC.replace(old, new)

patch("# Listen in Text — v3: что за «почти попадания»?", "# Listen in Text — v3.1: что за «почти попадания»? (исправленные фильтры)")
patch("**Входные данные (Kaggle).**", r"""**Что нового в v3.1.**
- Однокоренные строже: правило «одно слово — начало другого» только если короткое слово не короче 4 букв (в v3 their → the и consumption → con считались однокоренными).
- Служебные слова (NLTK stopwords) исключаются и как целевые слова, и как кандидаты (`exclude_stopwords`) — убирает that → the, with → which.
- Перебор порогов смысла и звука (§9b) — проверка, что выводы не держатся на одном пороге.
- Понимает выходы v4: каждая папка `<набор>_<условие>` — отдельный прогон (включая контроль `reversed`).

**Входные данные (Kaggle).**""")
patch("    example_layer  = 28,\n", """    example_layer  = 28,
    exclude_stopwords = True,
    run_include    = None,     # e.g. ["listen"]: only runs whose name contains one of these substrings
    sweep_sem      = [0.5, 0.6, 0.7],
    sweep_phon     = [0.4, 0.5, 0.6],
    sweep_layers   = [16, 20, 28],
""")
# run filter
patch("RUN_DIRS = discover(CFG[\"search_roots\"])\n",
      "RUN_DIRS = discover(CFG[\"search_roots\"])\nif CFG[\"run_include\"]:\n    RUN_DIRS = {k: v for k, v in RUN_DIRS.items() if any(s in k for s in CFG[\"run_include\"])}\n")
# stopwords resource + stricter morphology
patch('ensure_nltk("wordnet", "corpora/wordnet")\n', 'ensure_nltk("wordnet", "corpora/wordnet")\nensure_nltk("stopwords", "corpora/stopwords")\n')
patch("CMU = cmudict.dict()\n", "CMU = cmudict.dict()\nfrom nltk.corpus import stopwords\nSTOP = set(stopwords.words(\"english\")) | {\"would\", \"could\", \"upon\", \"shall\", \"also\", \"one\", \"us\"}\n")
patch("""def is_morph(w, c):
    if w.startswith(c) or c.startswith(w):
        return True
    return bool(lemmas(w) & lemmas(c))""",
"""def is_morph(w, c):
    short, long_ = sorted((w, c), key=len)
    if len(short) >= 4 and long_.startswith(short):
        return True
    return bool(lemmas(w) & lemmas(c))""")
# target words: skip stopwords
patch("""            if len(word) < CFG["min_word_chars"] or not word.isalpha():
                continue
            targets = word_targets(word)""",
"""            if len(word) < CFG["min_word_chars"] or not word.isalpha():
                continue
            if CFG["exclude_stopwords"] and word in STOP:
                continue
            targets = word_targets(word)""")
# classification: thresholds as arguments, cached raw features
patch("""@lru_cache(maxsize=None)
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
    return cat, s, p""",
"""@lru_cache(maxsize=None)
def raw_feats(w, c):
    return is_morph(w, c), is_homophone(w, c), float(EMB[EIDX[w]] @ EMB[EIDX[c]]), phon(w, c)

def classify(w, c, sem_thr=None, phon_thr=None):
    # returns (category, sem, phon) for target w and candidate c (c != w)
    sem_thr = CFG["sem_thr"] if sem_thr is None else sem_thr
    phon_thr = CFG["phon_thr"] if phon_thr is None else phon_thr
    morph, homo, s, p = raw_feats(w, c)
    if morph:
        return "morph", s, p
    if homo:
        return "homophone", s, p
    sound = (not np.isnan(p)) and p >= phon_thr
    mean = s >= sem_thr
    cat = "both" if (sound and mean) else "meaning" if mean else "sound" if sound else "other"
    return cat, s, p""")
patch("""def window_stats(top, L, a, b, word, target_ids):
    ids = top[L, a:b + 1, :].reshape(-1)
    ids = ids[~np.isin(ids, list(target_ids))]
    cidx = LUT[ids]
    cands = set(CAND_WORDS[cidx[cidx >= 0]].tolist()) - {word}""",
"""def window_stats(top, L, a, b, word, target_ids, sem_thr=None, phon_thr=None):
    ids = top[L, a:b + 1, :].reshape(-1)
    ids = ids[~np.isin(ids, list(target_ids))]
    cidx = LUT[ids]
    cands = set(CAND_WORDS[cidx[cidx >= 0]].tolist()) - {word}
    if CFG["exclude_stopwords"]:
        cands -= STOP""")
patch("""        cat, s, p = classify(word, c)""", """        cat, s, p = classify(word, c, sem_thr, phon_thr)""")
# plots: one row per run
patch("""names = list(RUNS)
fig, axes = plt.subplots(2, len(names), figsize=(7.5 * len(names), 8), squeeze=False, sharey="row")
for col, name in enumerate(names):
    d = SUMM[SUMM.model == name]
    ax = axes[0, col]""",
"""names = list(RUNS)
fig, axes = plt.subplots(len(names), 2, figsize=(15, 3.8 * len(names)), squeeze=False)
for col, name in enumerate(names):
    d = SUMM[SUMM.model == name]
    ax = axes[col, 0]""")
patch("""    ax.legend(fontsize=8)
    ax = axes[1, col]""", """    ax.legend(fontsize=8)
    ax = axes[col, 1]""")
# threshold sweep, inserted before the examples section
patch('md("## 10. Примеры на слое `example_layer`', r'''md("""## 9b. Устойчивость к порогам
Пересчёт избытка (контроль из той же реплики) для категорий «по смыслу» и «по звуку» при разных порогах. Только для прогонов условия listen (или для всех, если условий нет). Вывод держится, если знак и порядок величин не меняются.""")
code(r"""
def sweep_model(name, layers, sem_thr, phon_thr):
    acc = {c: [] for c in ("meaning", "sound")}
    for L in layers:
        for j in JOBS[name]:
            t, a, b = j["real"]
            real, _ = window_stats(t["top"], L, a, b, j["word"], j["targets"], sem_thr, phon_thr)
            same = [window_stats(u["top"], L, s, e, j["word"], j["targets"], sem_thr, phon_thr)[0] for u, s, e in j["same"]]
            for c in acc:
                acc[c].append((L, real[c] - np.mean([x[c] for x in same])))
    return acc

sweep_runs = [n for n in RUNS if "listen" in n or "|" not in n]   # listen runs (v4) and v2-style runs
rows = []
for name in sweep_runs:
    for st_ in CFG["sweep_sem"]:
        for pt_ in CFG["sweep_phon"]:
            acc = sweep_model(name, CFG["sweep_layers"], st_, pt_)
            for c, vals in acc.items():
                v = pd.DataFrame(vals, columns=["layer", "x"]).groupby("layer").x.mean()
                for L, x in v.items():
                    rows.append(dict(run=name, sem_thr=st_, phon_thr=pt_, cat=c, layer=L, excess_same=x))
SWEEP = pd.DataFrame(rows, columns=["run", "sem_thr", "phon_thr", "cat", "layer", "excess_same"])
SWEEP.to_csv(f"{CFG['out_dir']}/threshold_sweep.csv", index=False)
for name in sweep_runs:
    print(f"\n=== {name}: excess over same-turn control by threshold ===")
    d = SWEEP[SWEEP.run == name]
    if d.empty:
        print("  (no sweep layers in this run)"); continue
    print(d.pivot_table(index=["cat", "sem_thr", "phon_thr"], columns="layer", values="excess_same").round(3).to_string())
""")

md("## 10. Примеры на слое `example_layer`''')

patch("    for L in range(N_LAYERS):\n", "    for L in range(RUNS[name][0][\"top\"].shape[0]):\n")
patch("    for L in layers:\n        for j in JOBS[name]:", "    for L in [x for x in layers if x < RUNS[name][0][\"top\"].shape[0]]:\n        for j in JOBS[name]:")
patch("""    d = SUMM[(SUMM.model == name) & SUMM.layer.isin(CFG["ci_layers"])]
""", """    d = SUMM[(SUMM.model == name) & SUMM.layer.isin(CFG["ci_layers"])]
    if d.empty:
        print("  (no layers from ci_layers in this run)"); continue
""")
patch("examples_{name}_L", "examples_{name.replace('|', '__')}_L")
exec(SRC)
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "nearmiss_v31.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
