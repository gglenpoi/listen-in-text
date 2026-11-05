"""Builds listen_in_text_v2.ipynb by patching the v1 cell list and appending the new analyses."""
import re, sys
import nbformat as nbf

SRC = open(__file__.replace("build_nb_v2.py", "build_nb.py")).read()
SRC = SRC.split("nb = nbf.v4.new_notebook()")[0]          # keep only cell definitions


def patch(old, new, count=1):
    global SRC
    assert SRC.count(old) == count, (SRC.count(old), old[:80])
    SRC = SRC.replace(old, new)


# ---------- header ----------
patch("# Pilot: Do Full-Duplex Speech Models Listen in Text?",
      "# Do Full-Duplex Speech Models Listen in Text? — v2")
patch("**Критерий go/no-go:**", r"""**Что нового в v2.**
- Переключатель модели: `CFG["model"] = "personaplex"` или `"moshi"` (базовый Moshi от Kyutai, лицензию принимать не нужно).
- §12 **Время:** появляется ли слово до своего начала (предсказание), во время или после (узнавание).
- §13 **Смысл или звук:** близки ли ошибочные кандидаты к слову по смыслу (hard ↔ difficult) или по звучанию (know ↔ no).
- §11 Probe с отложенными дикторами и контролем по длительности.
- Сохраняются top-5 по всем слоям (`top5.npz`) — анализы можно пересчитывать без GPU.

**Критерий go/no-go (из v1):**""")

# ---------- config with model presets ----------
patch('    model_repo   = "nvidia/personaplex-7b-v1",  # or "kyutai/moshiko-pytorch-bf16" for the Moshi baseline (see note at the end)\n',
      '    model        = "personaplex",   # "personaplex" | "moshi"  (run the notebook once per model)\n')
patch('    text_prompt  = "You enjoy having a good conversation.",  # PersonaPlex prompt used for FullDuplexBench\n', "")
patch('    out_dir      = f"{BASE}/pilot_out",\n', "")
patch('    seed         = 0,\n)\n', r'''    seed         = 0,
    save_topk    = 5,          # top-k kept on disk for offline re-analysis
    offsets      = (-8, 20),   # §12: frames relative to word onset/offset (80 ms each)
    sem_topk     = 5,          # §13: candidates per position
    sem_layers   = list(range(0, 33, 2)),
)
PRESETS = {
    # PersonaPlex: system prompt as in its README; user stream carries a sine tone during prompts (as in LMGen)
    "personaplex": dict(model_repo="nvidia/personaplex-7b-v1",
                        text_prompt="You enjoy having a good conversation.", prefix_user="sine"),
    # Moshi was trained without system prompts: short silence on both streams, then the user speaks
    "moshi":       dict(model_repo="kyutai/moshiko-pytorch-bf16", text_prompt="", prefix_user="silence"),
}
CFG.update(PRESETS[CFG["model"]])
CFG["out_dir"] = f"{BASE}/{CFG['model']}_out"
print(CFG["model_repo"], "->", CFG["out_dir"])
''')

# ---------- install ----------
patch('"openai-whisper", "soundfile", "tqdm"]', '"openai-whisper", "soundfile", "tqdm", "nltk", "sentence-transformers", "scikit-learn"]')

# ---------- weights: tolerate different LM file names across repos ----------
patch("moshi_path = hf_hub_download(repo, loaders.MOSHI_NAME)",
      """from huggingface_hub import list_repo_files
files = list_repo_files(repo)
lm_file = loaders.MOSHI_NAME if loaders.MOSHI_NAME in files else next(
    f for f in files if f.endswith(".safetensors") and not f.startswith("tokenizer"))
moshi_path = hf_hub_download(repo, lm_file)""")
patch("- Примите лицензию модели на https://huggingface.co/nvidia/personaplex-7b-v1",
      "- Для PersonaPlex примите лицензию на https://huggingface.co/nvidia/personaplex-7b-v1 (для Moshi не нужно)")

# ---------- sequence builder: prefix filler depends on the model ----------
patch("def build_codes(user_codes, prompt_ids, n_sil):", "def build_codes(user_codes, prompt_ids, n_sil, prefix_user=\"sine\"):")
patch("    sine = torch.as_tensor(SINE_TOKENS).view(8, 1)\n",
      "    sine = torch.as_tensor(SINE_TOKENS).view(8, 1)\n    filler = sine if prefix_user == \"sine\" else sil\n")
patch("    for _ in range(n_sil): frame(pad, sil, sine)\n    for tok in prompt_ids:  frame(tok, sil, sine)\n    for _ in range(n_sil): frame(pad, sil, sine)\n",
      "    for _ in range(n_sil): frame(pad, sil, filler)\n    for tok in prompt_ids:  frame(tok, sil, filler)\n    for _ in range(n_sil): frame(pad, sil, filler)\n")
patch('build_codes(mimi_codes(t["wav"]), prompt_ids, CFG["silence_frames"])',
      'build_codes(mimi_codes(t["wav"]), prompt_ids, CFG["silence_frames"], CFG["prefix_user"])')
patch("Префикс повторяет PersonaPlex: тишина → системный промпт → тишина (голосовой промпт в пилоте пропускаем).",
      "Префикс для PersonaPlex: тишина → системный промпт → тишина (голосовой промпт пропускаем); для Moshi — только тишина.")

# ---------- plots titled by model ----------
patch("plt.suptitle(f\"{CFG['model_repo']}: is the user's word decodable via the text head?\")",
      "plt.suptitle(f\"{CFG['model']}: is the user's word decodable via the text head?\")")

# ---------- probe: speaker-held-out, more classes, duration baseline ----------
old_probe = SRC[SRC.index('md("### Быстрый probe'):SRC.index('md(r"""\n## Как читать результат')]
SRC = SRC.replace(old_probe, r'''md("""### Probe: идентичность слова по слоям
В v1 probe был на 7 словах и ничего не различал. Здесь: все слова с ≥ 5 примерами, PCA до 64, разбиение с **отложенными дикторами** (GroupKFold), контроль — модель только на длительности слова. Ожидание по v1: слово различимо уже на слое 0 (токены Mimi несут фонетику), поэтому probe отвечает на вопрос «есть ли информация», а logit lens — «в каком она виде».""")
code(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

m = pd.DataFrame(meta)
m["spk"] = m.turn.str.split("-").str[0]
vc = m[m.word.str.len() >= 4].word.value_counts()
classes = vc[vc >= 5].index
mask = m.word.isin(classes).values
y, groups = m.word[mask].values, m.spk[mask].values
print(f"{len(classes)} classes, {mask.sum()} samples, majority {pd.Series(y).value_counts(normalize=True).max():.2f}")
dur = (m.end - m.start)[mask].values[:, None]
base = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), dur, y,
                       cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
print(f"duration-only baseline: {base:.3f}")
probe = []
for L in CFG["save_layers"]:
    X = np.stack(feats[L])[mask].astype(np.float32)
    n_pc = min(64, int(len(y) * 0.8) - 1, X.shape[1])   # PCA must fit inside each training fold
    pipe = make_pipeline(StandardScaler(), PCA(n_pc, random_state=0), LogisticRegression(max_iter=3000, C=0.1))
    acc = cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=groups).mean()
    probe.append(dict(layer=L, acc_speaker_heldout=acc, duration_baseline=base))
    print(f"layer {L:2d}: speaker-held-out acc {acc:.3f}")
pd.DataFrame(probe).to_csv(f"{CFG['out_dir']}/probe_word_identity.csv", index=False)
""")

''')

# ---------- new analyses, inserted before the closing markdown ----------
NEW = r'''
md(r"""## 12. Время: предсказание или узнавание?
Привязываем анализ к началу (и отдельно к концу) каждого содержательного слова. В позиции `p = старт_реплики + f0 + 1 + d` модель услышала семантические токены пользователя до кадра `f0 + d` включительно, т. е. примерно `40 + 80·d` мс от начала слова (кадры по 80 мс, точность ±40 мс). Попадание считаем **в одной позиции**, без окна. Контроль — то же слово в случайной позиции другой реплики.

- Попадания при `d < 0` (слово ещё не прозвучало) — **предсказание** следующего слова.
- Рост внутри слова — **узнавание по ходу**; пик после конца — узнавание с задержкой.""")
code(r"""
OFFS = np.arange(CFG["offsets"][0], CFG["offsets"][1] + 1)

def hits_at(top, positions, targets, k):
    # top [L+1, T, K]; positions int array; returns [L+1, len(positions)] float with NaN where invalid
    T = top.shape[1]
    out = np.full((top.shape[0], len(positions)), np.nan)
    ok = (positions >= 0) & (positions < T)
    if ok.any():
        win = top[:, positions[ok], :k]
        tt = torch.tensor(sorted(targets), dtype=win.dtype)
        out[:, ok] = torch.isin(win, tt).any(-1).numpy()
    return out

def locked_curves(turns, anchor, k=5, min_chars=4, n_ctrl=3, seed=0):
    rng = random.Random(seed)
    real, ctrl = [], []
    for ti, t in enumerate(turns):
        others = [u for j, u in enumerate(turns) if j != ti]
        for w in t["words"]:
            targets = word_targets(w["word"])
            if len(w["word"]) < min_chars or not targets:
                continue
            f = int(math.floor(w["start"] * FPS)) if anchor == "start" else int(math.ceil(w["end"] * FPS)) - 1
            pos = t["user_start"] + f + 1 + OFFS
            r = hits_at(t["top"], pos, targets, k)
            r[:, pos <= t["user_start"]] = np.nan            # before the user turn started
            cs = []
            for _ in range(n_ctrl):
                u = rng.choice(others)
                if w["word"] in u["text"].split():
                    continue
                lo, hi = u["user_start"] + 1 - OFFS.min(), u["top"].shape[1] - 1 - OFFS.max()
                if hi <= lo:
                    continue
                cs.append(hits_at(u["top"], rng.randint(lo, hi) + OFFS, targets, k))
            if cs:
                real.append(r); ctrl.append(np.nanmean(np.stack(cs), 0))
    return np.stack(real), np.stack(ctrl)

timing = {}
for anchor in ("start", "end"):
    R, C = locked_curves(turns, anchor)
    timing[anchor] = (np.nanmean(R, 0), np.nanmean(C, 0), len(R))
    print(anchor, "-locked:", len(R), "words")
np.savez_compressed(f"{CFG['out_dir']}/timing.npz", offsets=OFFS,
                    **{f"{a}_{s}": v[i] for a, v in timing.items() for i, s in enumerate(("real", "ctrl"))})
""")
code(r"""
best_layer = int(curves[(curves.k == 5) & (curves.subset == "content")].sort_values("real").iloc[-1].layer)
ms = OFFS * 80 + 40          # audio heard relative to the anchor, frame-quantised (±40 ms)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for col, anchor in enumerate(("start", "end")):
    real, ctrl, n = timing[anchor]
    diff = real - ctrl
    ax = axes[0, col]
    im = ax.imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-.4, vmax=.4,
                   extent=[ms[0] - 40, ms[-1] + 40, -.5, diff.shape[0] - .5])
    ax.axvline(0, color="k", lw=.8)
    ax.set_title(f"hit@5 (real − control), locked to word {anchor} (n={n})")
    ax.set_xlabel(f"ms of audio heard relative to word {anchor}"); ax.set_ylabel("layer")
    plt.colorbar(im, ax=ax)
    ax = axes[1, col]
    show = dict.fromkeys([best_layer, max(best_layer - 6, 0), real.shape[0] // 2])
    for L, c in zip(show, ("C0", "C1", "C2")):
        ax.plot(ms, real[L], color=c, label=f"layer {L}")
        ax.plot(ms, ctrl[L], color=c, ls=":", alpha=.7)
    ax.axvline(0, color="k", lw=.8); ax.grid(alpha=.3)
    ax.set_xlabel(f"ms relative to word {anchor}"); ax.set_ylabel("hit@5 (dotted = control)"); ax.legend(fontsize=8)
plt.suptitle(f"{CFG['model']}: when does the user's word become readable?")
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/timing.png", dpi=150); plt.show()

real, ctrl, _ = timing["start"]
d = real[best_layer] - ctrl[best_layer]
pre = d[ms <= 0]
print(f"layer {best_layer}: excess hit@5 before onset (prediction) mean {np.nanmean(pre):.3f}, max {np.nanmax(pre):.3f}")
half = np.nanmax(d) / 2
print(f"excess reaches half-max at ~{ms[np.argmax(d >= half)]} ms after onset; peak at {ms[np.nanargmax(d)]} ms")
""")

md(r"""## 13. Смысл или звук?
Для каждого содержательного слова берём кандидатов из top-5 в его окне (только целые слова от 3 букв, исключая само слово и однокоренные с общим префиксом ≥ 4 букв). Для каждого кандидата считаем:
- **смысловую близость** — косинус эмбеддингов `all-MiniLM-L6-v2`;
- **звуковую близость** — 1 − нормированное расстояние Левенштейна между фонемами CMUdict.

Берём максимум по кандидатам и сравниваем с контрольным окном той же длины из другой реплики. Если на поздних слоях растёт смысловой избыток, а звуковой нет, модель переводит речь в значение, а не в написание.""")
code(r"""
import nltk
from functools import lru_cache
nltk.download("cmudict", quiet=True)
from nltk.corpus import cmudict
CMU = cmudict.dict()

@lru_cache(maxsize=None)
def phones(w):
    p = CMU.get(w)
    return tuple(re.sub(r"\d", "", x) for x in p[0]) if p else None

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

@lru_cache(maxsize=None)
def phon_sim(a, b):
    pa, pb = phones(a), phones(b)
    if not pa or not pb:
        return np.nan
    return 1 - lev(pa, pb) / max(len(pa), len(pb))

from sentence_transformers import SentenceTransformer
st_dev = "cuda:0" if torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0] / 1e9 > 1.5 else "cpu"
st_model = SentenceTransformer("all-MiniLM-L6-v2", device=st_dev)
def embed_words(words):
    words = sorted(set(words))
    vecs = st_model.encode(words, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    return dict(zip(words, vecs))
""")
code(r"""
CAND = {}                                   # vocab id -> lowercase whole word, for usable pieces only
for i in range(sp.get_piece_size()):
    p = sp.id_to_piece(i)
    if p.startswith("▁") and p[1:].isalpha() and len(p) >= 4 and p[1:].isascii():
        CAND[i] = p[1:].lower()

def related(c, w):
    return c == w or c.startswith(w) or w.startswith(c) or len(os.path.commonprefix([c, w])) >= 4

def window_cands(top, a, b, L, k):
    ids = top[L, a:b + 1, :k].reshape(-1).tolist()
    return {CAND[i] for i in ids if i in CAND}

def sem_phon_rows(turns, layers, k, min_chars=4, seed=0):
    rng = random.Random(seed)
    jobs = []
    for ti, t in enumerate(turns):
        others = [u for j, u in enumerate(turns) if j != ti]
        for w in t["words"]:
            if len(w["word"]) < min_chars or not w["word"].isalpha():
                continue
            a, b = window(t, w, CFG["lag_frames"])
            if b < a:
                continue
            u = rng.choice(others)
            if w["word"] in u["text"].split():
                continue
            lo, hi = u["user_start"] + 1, u["top"].shape[1] - 1 - (b - a)
            if hi <= lo:
                continue
            s = rng.randint(lo, hi)
            jobs.append((w["word"], t, a, b, u, s, s + (b - a)))
    vocab = {j[0] for j in jobs}
    for L in layers:
        for word, t, a, b, u, s, e in jobs:
            vocab |= window_cands(t["top"], a, b, L, k) | window_cands(u["top"], s, e, L, k)
    E = embed_words(vocab)
    rows = []
    for L in layers:
        for word, t, a, b, u, s, e in jobs:
            for cond, cands in (("real", window_cands(t["top"], a, b, L, k)), ("control", window_cands(u["top"], s, e, L, k))):
                cands = sorted(c for c in cands if not related(c, word))
                if not cands:
                    continue
                sims = [float(E[word] @ E[c]) for c in cands]
                phs = [phon_sim(word, c) for c in cands]
                i_sem = int(np.argmax(sims))
                i_ph = None if all(np.isnan(phs)) else int(np.nanargmax(phs))
                rows.append(dict(layer=L, cond=cond, word=word, n_cands=len(cands),
                                 sem=sims[i_sem], sem_cand=cands[i_sem],
                                 phon=np.nan if i_ph is None else phs[i_ph],
                                 phon_cand=None if i_ph is None else cands[i_ph]))
    return pd.DataFrame(rows)

sp_df = sem_phon_rows(turns, CFG["sem_layers"], CFG["sem_topk"])
sp_df.to_csv(f"{CFG['out_dir']}/sem_phon_rows.csv", index=False)
agg = sp_df.groupby(["layer", "cond"])[["sem", "phon", "n_cands"]].mean().unstack("cond")
summary = pd.DataFrame({
    "sem_real": agg[("sem", "real")], "sem_ctrl": agg[("sem", "control")],
    "phon_real": agg[("phon", "real")], "phon_ctrl": agg[("phon", "control")],
    "cands_real": agg[("n_cands", "real")], "cands_ctrl": agg[("n_cands", "control")],
})
summary["sem_excess"] = summary.sem_real - summary.sem_ctrl
summary["phon_excess"] = summary.phon_real - summary.phon_ctrl
summary.to_csv(f"{CFG['out_dir']}/sem_phon_summary.csv")
print(summary.round(3).to_string())
""")
code(r"""
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(summary.index, summary.sem_excess, "-o", label="semantic excess (MiniLM cosine)")
ax.plot(summary.index, summary.phon_excess, "-s", label="phonetic excess (CMUdict)")
ax.axhline(0, color="k", lw=.8); ax.grid(alpha=.3)
ax.set_xlabel("layer"); ax.set_ylabel("max similarity: real − control")
ax.set_title(f"{CFG['model']}: are near-misses similar in meaning or in sound?")
ax.legend(); plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/sem_phon.png", dpi=150); plt.show()

L = int(summary.sem_excess.idxmax())
ex = sp_df[(sp_df.layer == L) & (sp_df.cond == "real")].sort_values("sem", ascending=False)
print(f"strongest semantic near-misses at layer {L} (word -> candidate):")
for _, r in ex.head(15).iterrows():
    print(f"  {r['word']:>14s} -> {r['sem_cand']:<14s} sem {r['sem']:.2f} | closest in sound: {r['phon_cand']} ({r['phon']:.2f})")
""")

md("## 14. Сохранение и архив\nТоп-5 по всем слоям и разметка реплик — для пересчёта любых анализов без GPU. Всё упаковывается в один zip.")
code(r"""
np.savez_compressed(f"{CFG['out_dir']}/top5.npz",
                    **{t["id"]: t["top"][:, :, :CFG["save_topk"]].numpy().astype(np.int32) for t in turns})
json.dump([dict(id=t["id"], user_start=t["user_start"], text=t["text"], words=t["words"]) for t in turns],
          open(f"{CFG['out_dir']}/turns.json", "w"))
json.dump(CFG, open(f"{CFG['out_dir']}/config.json", "w"), indent=2)

import shutil
from IPython.display import FileLink, display
archive = shutil.make_archive(CFG["out_dir"], "zip", CFG["out_dir"])
print(archive, f"{os.path.getsize(archive) / 1e6:.1f} MB")
for f in sorted(os.listdir(CFG["out_dir"])):
    print(f"  {f:28s} {os.path.getsize(os.path.join(CFG['out_dir'], f)) / 1e6:8.2f} MB")
os.chdir(BASE)
display(FileLink(os.path.basename(archive)))
""")

'''
SRC = SRC.replace('md(r"""\n## Как читать результат', NEW + 'md(r"""\n## Как читать результат', 1)

# ---------- closing notes ----------
patch("1. Сравнение с базовым Moshi: `CFG[\"model_repo\"] = \"kyutai/moshiko-pytorch-bf16\"` (та же архитектура; проверьте имена файлов в репозитории, у Moshi depformer меньше, но для lens он не нужен). Вопрос: изменило ли дообучение PersonaPlex то, как модель слушает.",
      "1. Прогон с `CFG[\"model\"] = \"moshi\"` и сравнение: изменило ли дообучение PersonaPlex то, как модель слушает.")

exec(SRC)
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "listen_in_text_v2.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
