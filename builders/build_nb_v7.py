"""Builds listen_in_text_v7.ipynb — the final paper run."""
import sys
import nbformat as nbf

HERE = __file__.replace("build_nb_v7.py", "")
def load_cells(fname):
    src = open(HERE + fname).read().rsplit("nb = nbf.v4.new_notebook()", 1)[0]
    ns = {"__file__": HERE + fname}
    exec(src, ns)
    return [c.source for c in ns["cells"] if c.cell_type == "code"]
V5, V6 = load_cells("build_nb_v5.py"), load_cells("build_nb_v6.py")

def pick(cells_, marker):
    hits = [s for s in cells_ if marker in s]
    assert len(hits) == 1, marker
    return hits[0]

def sub(s, old, new):
    assert s.count(old) == 1, old[:70]
    return s.replace(old, new)

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Do Full-Duplex Speech Models Listen in Text? — v7 (финальный прогон)

Все основные результаты статьи в одном воспроизводимом прогоне, 80 реплик на набор, плюс три новых проверки.

| # | Вопрос | Как |
|---|---|---|
| R1 | Транскрипция собеседника и её подавление собственным голосом | listen, собственный канал: речь / обращённая речь / шум со спектром речи (как v6, вдвое больше данных) |
| R2 | **Эхо**: выключает ли слушание утечка *речи самого собеседника* в собственный канал? | `own_echo`: речь собеседника, задержка 200 мс, −10 дБ — как при плохом эхоподавлении |
| R3 | **Порог** голосового «шлюза» | собственная речь на −40 / −30 / −20 / −10 / 0 дБ |
| R4 | Модель «слышит» себя | чтение слов собственного канала |
| R5 | Обобщаемый код со слоя 12 | tuned lens, строгий тест на невиданных целевых токенах |
| R6 | **Используется ли ранний код?** | Вынужденный ответ (EPAD через 0,5 с) при абляции: (а) поздних направлений logit lens (слои 22–30, как v6), (б) **ранних направлений tuned lens (слои 12–30)**. Против: чужие слова. **Позитивный контроль**: речь собеседника заменена тишиной |

В конце — `paper_numbers.json` и `paper_table.csv`: все числа для текста статьи с 95% интервалами.

Kaggle: GPU T4×2, Internet On, `HF_TOKEN`. ~1,5–2 ч на модель (можно запустить через Save & Run All и закрыть вкладку).
""")

md("## 1. Конфигурация")
code(r"""
import os
ON_KAGGLE = os.path.exists("/kaggle/working")
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")
BASE = "/kaggle/working" if ON_KAGGLE else ("/content" if ON_COLAB else os.getcwd())

GAINS_DB = [-40, -30, -20, -10]
CFG = dict(
    model          = "personaplex",              # "personaplex" | "moshi"
    datasets       = ["librispeech", "ami"],
    conditions     = ["listen", "talking_audio", "own_reversed", "own_noise", "own_echo"] + [f"own_gain{g}" for g in GAINS_DB],
    echo_delay_sec = 0.2,
    echo_gain_db   = -10,
    n_turns        = 80,
    turn_sec       = 20.0,
    gap_sec        = 0.3,
    aligner        = "mms",
    whisper_model  = "small.en",
    silence_frames = 6,
    post_sec       = 1.5,
    resp_gap_sec   = 0.5,
    topk           = 20,
    lag_frames     = 4,
    n_control      = 5,
    min_word_chars = 4,
    n_boot         = 1000,
    # tuned lens (R5) — also provides the early directions for R6
    tl_extra       = ["dev-clean", "test-other"],
    tl_extra_turns = 60,
    tl_layers      = [0, 4, 8, 12, 16, 20, 24, 28, 32],
    tl_rank        = 128,
    tl_steps       = 800,
    tl_batch       = 512,
    tl_lr          = 1e-3,
    tl_wd          = 1e-2,
    tl_train_frac  = 0.7,
    # R6 causal
    causal_turns   = 60,                         # per dataset
    ab_late        = list(range(22, 31)),        # logit-lens directions
    ab_early       = list(range(12, 31)),        # tuned-lens directions (lens of the nearest trained layer below)
    tuned_check    = 20,                         # tuned-lens readout layer for the manipulation check
    dtype          = "auto",
    seed           = 0,
)
PRESETS = {
    "personaplex": dict(model_repo="nvidia/personaplex-7b-v1",
                        text_prompt="You enjoy having a good conversation.", prefix_user="sine"),
    "moshi":       dict(model_repo="kyutai/moshiko-pytorch-bf16", text_prompt="", prefix_user="silence"),
}
CFG.update(PRESETS[CFG["model"]])
CFG["ab_layers"] = CFG["ab_late"]                # used by lens_forward (v5) for the late ablation
CFG["out_dir"] = f"{BASE}/{CFG['model']}_v7"
os.makedirs(CFG["out_dir"], exist_ok=True)
print(CFG["model_repo"], "->", CFG["out_dir"])
""")

md("## 2. Установка, модель, данные, границы слов")
code(pick(V5, "import sys, subprocess"))
code(pick(V5, "import json, math, random, re, tarfile"))
code(pick(V5, "from huggingface_hub import hf_hub_download, list_repo_files"))
code(pick(V5, "def load_lm_for_lens"))
code(pick(V5, "def librispeech_turns"))
code(pick(V5, "def whisper_align(turns)"))

md("## 3. Варианты собственного канала")
v6_own = pick(V6, "def speech_shaped_noise")
v6_own = sub(v6_own, 'OWN_SRC = {"talking_audio": "codes_user", "own_reversed": "codes_rev", "own_noise": "codes_noise", "own_quiet": "codes_quiet"}',
             'OWN_SRC = {"talking_audio": "codes_user", "own_reversed": "codes_rev", "own_noise": "codes_noise"}\n'
             'OWN_SRC.update({f"own_gain{g}": f"codes_gain{g}" for g in GAINS_DB})')
v6_own = sub(v6_own, '''    src = other[OWN_SRC[cond]]''', '''    src = self_[ "codes_echo"] if cond == "own_echo" else other[OWN_SRC[cond]]''')
v6_own = sub(v6_own, "def own_streams(other, n, cond):", "def own_streams(other, n, cond, self_=None):")
v6_own = sub(v6_own, '''            t["codes_quiet"] = mimi_codes(t["wav"] * CFG["quiet_gain"])''',
             '''            for gdb in GAINS_DB:
                t[f"codes_gain{gdb}"] = mimi_codes(t["wav"] * 10 ** (gdb / 20))
            d = int(CFG["echo_delay_sec"] * 24000)
            echo = np.concatenate([np.zeros(d, np.float32), t["wav"][:-d]]) * 10 ** (CFG["echo_gain_db"] / 20)
            t["codes_echo"] = mimi_codes(echo.astype(np.float32))''')
code(v6_own)

md("## 4. Прямой проход")
fwd = pick(V6, "RUNS, TL_FRAMES = {}, {}")
fwd = sub(fwd, "own_codes, own_text, interval = own_streams(other, n, cond)", "own_codes, own_text, interval = own_streams(other, n, cond, self_=t)")
fwd = sub(fwd, 'if cond in ("talking_audio", "own_quiet") else []', 'if cond in OWN_WORD_CONDS else []')
fwd = sub(fwd, "RUNS, TL_FRAMES = {}, {}", 'OWN_WORD_CONDS = ["talking_audio"] + [f"own_gain{g}" for g in GAINS_DB]\nRUNS, TL_FRAMES = {}, {}')
code(fwd)

md("## 5. R1–R3: подавление по вариантам собственного канала")
code(pick(V6, "def score_all(name, cond, which").replace("_v6.csv", "_v7.csv"))
code(r'''
GAIN_CONDS = [f"own_gain{g}" for g in GAINS_DB] + ["talking_audio"]
GAIN_X = GAINS_DB + [0]
fig, axes = plt.subplots(1, 3, figsize=(20, 4.4))
for name, mk in zip(MAIN, ("o", "s")):
    L = BEST[name]
    p = PAIRED[(PAIRED.dataset == name) & (PAIRED.layer == L)].set_index("cond")
    sh = p["drop"] / p.listen; lo = p.ci_lo / p.listen; hi = p.ci_hi / p.listen
    conds = ["talking_audio", "own_reversed", "own_noise", "own_echo"]
    x = np.arange(len(conds)) + (0.2 if mk == "s" else -0.2)
    axes[0].bar(x, sh[conds], 0.4, yerr=[sh[conds] - lo[conds], hi[conds] - sh[conds]], capsize=3, label=name)
    axes[1].errorbar(GAIN_X, sh[GAIN_CONDS], yerr=[sh[GAIN_CONDS] - lo[GAIN_CONDS], hi[GAIN_CONDS] - sh[GAIN_CONDS]], marker=mk, capsize=3, label=name)
    d = CURVES[(CURVES.dataset == name) & (CURVES.cond == "listen")]
    axes[2].plot(d.layer, d.excess, lw=2, label=f"{name}: user words (listen)")
    axes[2].fill_between(d.layer, d.ci_lo, d.ci_hi, alpha=.15)
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(["own speech", "own speech reversed", "speech-shaped noise", "echo of the user"])
axes[0].set(title=f"{CFG['model']}: share of user-word readout lost", ylabel="drop / listen")
axes[1].set(title="dose-response: level of own speech", xlabel="own-channel gain, dB", ylabel="drop / listen")
axes[2].set(title="user-word readout by layer (listen)", xlabel="layer", ylabel="hit@5 excess")
for a in axes: a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/suppression_v7.png", dpi=150); plt.show()
''')

md("## 6. R4: модель слышит себя")
own = pick(V6, "own_curves = []")
own = sub(own, 'for cond in ("talking_audio", "own_quiet"):\n        rows = score_all', 'for cond in OWN_WORD_CONDS:\n        rows = score_all')
own = sub(own, 'cc = {"listen": "C0", "talking_audio": "C1", "own_reversed": "C7", "own_noise": "C5", "own_quiet": "C2"}',
          'cc = {c: f"C{i}" for i, c in enumerate(CFG["conditions"])}')
own = sub(own, '    for cond in ("talking_audio", "own_quiet"):\n        d = OWNC', '    for cond in ["talking_audio"]:\n        d = OWNC')
own = own.replace("own_channel_v6", "own_channel_v7")
code(own)

md("## 7. R5: tuned lens (строгий тест) — обученные линзы сохраняются для R6")
code(pick(V5, "class LowRankLens"))
code(r'''
def prep(name):
    f = frame_table(name)
    f = f[(f.target >= 0) & (f.word.str.len() >= CFG["min_word_chars"])].copy()
    return f

FT = prep("librispeech")
spks = sorted(FT.spk.unique()); random.Random(CFG["seed"]).shuffle(spks)
train_spk = set(spks[:int(len(spks) * CFG["tl_train_frac"])])
train_sets = [("librispeech", FT[FT.spk.isin(train_spk)])] + [(n, prep(n)) for n in TL_FRAMES if n.startswith("ls_")]
test = FT[~FT.spk.isin(train_spk)]
seen = set().union(*[set(f.word) for _, f in train_sets])
seen_targets = set(np.concatenate([f.target.values for _, f in train_sets]).tolist())
tests = {"seen": test[test.word.isin(seen)], "unseen": test[~test.word.isin(seen)], "unseen_target": test[~test.target.isin(seen_targets)]}
if "ami" in TL_FRAMES:
    fa = prep("ami"); tests["ami_unseen_target"] = fa[~fa.target.isin(seen_targets)]
print("train frames:", sum(len(f) for _, f in train_sets), "| test words:", {k: v.groupby(["ti", "wi"]).ngroups for k, v in tests.items()})

def mean_ci(v):
    v = np.asarray(v, float)
    if not len(v): return np.nan, np.nan, np.nan
    lo, hi = boot_ci(v, 500)
    return v.mean(), lo, hi

LENSES, tl_rows = {}, []
for L in CFG["tl_layers"]:
    Htr = torch.cat([TL_FRAMES[n]["h"][L][torch.as_tensor(f.fi.values)] for n, f in train_sets])
    ytr = torch.tensor(np.concatenate([f.target.values for _, f in train_sets]))
    lens = train_lens(Htr, ytr, CFG["tl_steps"], CFG["seed"], CFG["tl_rank"])
    shuf = train_lens(Htr, ytr, CFG["tl_steps"], CFG["seed"], CFG["tl_rank"], shuffle=True)
    for tag, ft in tests.items():
        src = "ami" if tag.startswith("ami") else "librispeech"
        for lname, ln in (("logit", None), ("tuned", lens), ("shuffled", shuf)):
            m, lo, hi = mean_ci(word_hit(ln, TL_FRAMES[src]["h"][L], ft))
            tl_rows.append(dict(layer=L, lens=lname, test=tag, hit=m, ci_lo=lo, ci_hi=hi))
    LENSES[L] = lens.eval()
    del shuf, Htr; torch.cuda.empty_cache()
    print(f"layer {L} done")
TL = pd.DataFrame(tl_rows); TL.to_csv(f"{CFG['out_dir']}/tuned_lens_v7.csv", index=False)
for tag in tests:
    print(f"--- {tag}"); print(TL[TL.test == tag].pivot_table(index="layer", columns="lens", values="hit").round(3).to_string())
fig, axes = plt.subplots(1, len(tests), figsize=(5.5 * len(tests), 4), squeeze=False)
for ax, tag in zip(axes[0], tests):
    for ln, g in TL[TL.test == tag].groupby("lens"):
        ax.plot(g.layer, g.hit, "-o", ms=3, label=ln, ls=":" if ln == "shuffled" else "-"); ax.fill_between(g.layer, g.ci_lo, g.ci_hi, alpha=.12)
    ax.set(title=f"{CFG['model']}: {tag}", xlabel="layer", ylabel="word hit@5"); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/tuned_lens_v7.png", dpi=150); plt.show()
''')

md(r"""## 8. R6: используется ли транскрипция для ответа?
Вынужденный ответ: через 0,5 с после реплики в собственный текст модели ставится EPAD, на следующем кадре читается распределение первого слова ответа. Условия:
- **clean**;
- **silence** — позитивный контроль: речь собеседника заменена тишиной (мера обязана почувствовать потерю информации);
- **late_own / late_ctrl** — вырезаем направления logit lens (слои 22–30) для слов этой реплики / чужих слов;
- **early_own / early_ctrl** — вырезаем направления tuned lens (слои 12–30): для слоя `l` используется линза ближайшего обученного слоя `≤ l`, направление токена `t` — `w_t + U(V w_t)`, где `w_t` — строка текстовой головы с весами `out_norm`.

Проверка манипуляции: чтение слов собеседника logit lens на лучшем слое и tuned lens на слое `tuned_check`.""")
code(pick(V5, "ALPHA = lm.out_norm.alpha").split("ab_rows = []")[0] + r'''
GAP = int(round(CFG["resp_gap_sec"] * FPS))
rng_ab = random.Random(CFG["seed"])
TRAINED = sorted(LENSES)

def content_ids(words):
    ids = set()
    for w in words:
        if len(w["word"]) >= CFG["min_word_chars"]:
            ids |= word_targets(w["word"])
    return ids

def wvecs(ids):
    return W_OUT[torch.tensor(sorted(ids), device=W_OUT.device)].float() * ALPHA.to(W_OUT.device)   # [k, D]

def layer_bases(ids, kind):
    # {layer l (1-based output index): Q on that layer's device}
    W = wvecs(ids)
    out = {}
    for l in (CFG["ab_late"] if kind == "late" else CFG["ab_early"]):
        dev = LAYER_DEVS[l - 1]
        if kind == "late":
            D = W.T
        else:
            lens = LENSES[max(x for x in TRAINED if x <= l)]
            U, V = lens.U.detach().float(), lens.V.detach().float()
            D = (W + (W @ V.T) @ U.T).T                      # columns: w_t + U (V w_t)
        Q, _ = torch.linalg.qr(D)
        out[l] = Q.to(dev)
    return out

@torch.no_grad()
def forward_ablate(codes, bases, a, b, final_pos, check_logit, check_tuned):
    first, last = LAYER_DEVS[0], LAYER_DEVS[-1]
    codes = codes.unsqueeze(0).to(first)
    initial = lm._get_initial_token().to(first)
    seq = torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], 2)[:, :, :-1]
    x = lm.embed_codes(seq)
    special = torch.tensor(SPECIAL_TEXT, device=last)
    chk = {}
    def top5(h):
        logits = lm.text_linear(lm.out_norm(h.to(last))).float()[0]
        logits[:, special] = -float("inf")
        return logits.topk(5, -1).indices.int().cpu()
    for i, layer in enumerate(lm.transformer.layers):
        x = layer(x.to(LAYER_DEVS[i]))
        l = i + 1
        if bases is not None and l in bases:
            q = bases[l]
            seg = x[0, a:b].float()
            x[0, a:b] = (seg - (seg @ q) @ q.T).to(x.dtype)
        if l == check_logit:
            chk["logit"] = top5(x)
        if l == check_tuned:
            h = x[0].to(last).float()
            logits = lens_logits(LENSES[check_tuned](h))
            chk["tuned"] = logits.topk(5, -1).indices.int().cpu()
    final = torch.log_softmax(lm.text_linear(lm.out_norm(x.to(last))).float()[0, final_pos], -1).cpu()
    return chk, final

def read_hits_win(top5, t):
    hits = []
    for w in t["words"]:
        if len(w["word"]) < CFG["min_word_chars"]: continue
        a, b = word_window(t, w, CFG["lag_frames"])
        if b < a: continue
        tg = torch.tensor(sorted(word_targets(w["word"])), dtype=top5.dtype)
        hits.append(bool(torch.isin(top5[a:b + 1], tg).any()))
    return np.mean(hits) if hits else np.nan

resp = []
t0 = time.time()
for name in MAIN:
    turns = RUNS[(name, "listen")][:CFG["causal_turns"]]
    pool_all = [content_ids(u["words"]) for u in RUNS[(name, "listen")]]
    for ti, t in enumerate(turns):
        own = pool_all[ti]
        pool = set().union(*[p for j, p in enumerate(pool_all) if j != ti]) - own
        if len(own) < 3 or len(pool) < len(own): continue
        ctrl = set(rng_ab.sample(sorted(pool), len(own)))
        a, b = t["user_start"], t["user_start"] + t["n_user"]
        e = b + GAP
        codes = t["codes"].clone(); codes[0, e] = 0
        sil = codes.clone(); sil[9:, a:b] = SIL                # positive control: user stream silenced
        pos = list(range(b, e + 2))
        own_t, ctrl_t = torch.tensor(sorted(own)), torch.tensor(sorted(ctrl))
        row = dict(dataset=name, ti=ti, k=len(own))
        runs_ = {"clean": (codes, None), "silence": (sil, None),
                 "late_own": (codes, layer_bases(own, "late")), "late_ctrl": (codes, layer_bases(ctrl, "late")),
                 "early_own": (codes, layer_bases(own, "early")), "early_ctrl": (codes, layer_bases(ctrl, "early"))}
        firsts = {}
        for tag, (c_, bases) in runs_.items():
            chk, lp = forward_ablate(c_, bases, a, b, pos, BEST[name], CFG["tuned_check"])
            first = lp[-1]; firsts[tag] = first
            row[f"heard_{tag}"] = first.exp()[own_t].sum().item()
            row[f"other_{tag}"] = first.exp()[ctrl_t].sum().item()
            row[f"top1_{tag}"] = int(first.argmax())
            row[f"read_logit_{tag}"] = read_hits_win(chk["logit"], t)
            row[f"read_tuned_{tag}"] = read_hits_win(chk["tuned"], t)
        for tag in runs_:
            if tag != "clean":
                row[f"kl_{tag}"] = torch.sum(firsts["clean"].exp() * (firsts["clean"] - firsts[tag])).item()
        resp.append(row)
    print(f"{name}: causal tests done ({time.time() - t0:.0f}s)")
RESP = pd.DataFrame(resp); RESP.to_csv(f"{CFG['out_dir']}/causal_v7.csv", index=False)

def lr(g, tag):
    return np.log((g[f"heard_{tag}"] + 1e-9) / (g.heard_clean + 1e-9))
CAUSAL = []
for name, g in RESP.groupby("dataset"):
    print(f"\n=== {name}: n={len(g)} turns; first answer word: P(heard) clean {g.heard_clean.mean():.4f}, P(other) {g.other_clean.mean():.4f}")
    print("  manipulation check (readout of heard words):")
    for tag in ("clean", "silence", "late_own", "late_ctrl", "early_own", "early_ctrl"):
        print(f"    {tag:11s} logit L{BEST[name]} {g[f'read_logit_{tag}'].mean():.3f} | tuned L{CFG['tuned_check']} {g[f'read_tuned_{tag}'].mean():.3f}")
    for label, own_tag, ctrl_tag in (("silence vs clean", "silence", None), ("late own vs ctrl", "late_own", "late_ctrl"), ("early own vs ctrl", "early_own", "early_ctrl")):
        d = lr(g, own_tag) - (lr(g, ctrl_tag) if ctrl_tag else 0); lo, hi = boot_ci(d.values)
        k = g[f"kl_{own_tag}"] - (g[f"kl_{ctrl_tag}"] if ctrl_tag else 0); klo, khi = boot_ci(k.values)
        t1 = np.mean(g[f"top1_{own_tag}"] != g.top1_clean)
        print(f"  {label:18s} log P(heard) {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | KL {k.mean():+.4f} [{klo:+.4f}, {khi:+.4f}] | top-1 changed {t1:.0%}")
        CAUSAL.append(dict(dataset=name, contrast=label, logP_heard=d.mean(), logP_lo=lo, logP_hi=hi, kl=k.mean(), kl_lo=klo, kl_hi=khi, top1_changed=t1))
CAUSAL = pd.DataFrame(CAUSAL); CAUSAL.to_csv(f"{CFG['out_dir']}/causal_summary_v7.csv", index=False)
''')

md("## 9. Числа для статьи\nОдна таблица со всеми ключевыми результатами и 95% интервалами (`paper_table.csv`, `paper_numbers.json`).")
code(r'''
rows = []
for name in MAIN:
    L = BEST[name]
    c = CURVES[(CURVES.dataset == name) & (CURVES.cond == "listen") & (CURVES.layer == L)].iloc[0]
    rows.append(dict(model=CFG["model"], dataset=name, result="R1 listen readout hit@5 excess", layer=L, value=c.excess, lo=c.ci_lo, hi=c.ci_hi, n=c.n_words))
    p = PAIRED[(PAIRED.dataset == name) & (PAIRED.layer == L)].set_index("cond")
    for cond in CFG["conditions"][1:]:
        r = p.loc[cond]
        rows.append(dict(model=CFG["model"], dataset=name, result=f"R1-R3 share lost: {cond}", layer=L, value=r["drop"] / r.listen,
                         lo=r.ci_lo / r.listen, hi=r.ci_hi / r.listen, n=r.n_words))
    if len(OWNC):
        o = OWNC[(OWNC.dataset == name) & (OWNC.cond == "talking_audio") & (OWNC.layer == L)]
        if len(o):
            o = o.iloc[0]
            rows.append(dict(model=CFG["model"], dataset=name, result="R4 own-channel readout (own speech)", layer=L, value=o.excess, lo=o.ci_lo, hi=o.ci_hi, n=o.n_words))
for tag in tests:
    for L in (12, 28):
        for ln in ("logit", "tuned", "shuffled"):
            r = TL[(TL.test == tag) & (TL.layer == L) & (TL.lens == ln)]
            if len(r):
                r = r.iloc[0]
                rows.append(dict(model=CFG["model"], dataset=tag, result=f"R5 {ln} lens word hit@5", layer=L, value=r.hit, lo=r.ci_lo, hi=r.ci_hi, n=np.nan))
for _, r in CAUSAL.iterrows():
    rows.append(dict(model=CFG["model"], dataset=r.dataset, result=f"R6 {r.contrast}: log P(heard words) in answer", layer=np.nan,
                     value=r.logP_heard, lo=r.logP_lo, hi=r.logP_hi, n=np.nan))
    rows.append(dict(model=CFG["model"], dataset=r.dataset, result=f"R6 {r.contrast}: KL of answer", layer=np.nan,
                     value=r.kl, lo=r.kl_lo, hi=r.kl_hi, n=np.nan))
PAPER = pd.DataFrame(rows)
PAPER.to_csv(f"{CFG['out_dir']}/paper_table.csv", index=False)
json.dump(PAPER.to_dict("records"), open(f"{CFG['out_dir']}/paper_numbers.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
print(PAPER.round(3).to_string(index=False))
''')

md("## 10. Сохранение")
save = pick(V6, 'np.savez_compressed(f"{d}/top5.npz"')
code(save)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "listen_in_text_v7.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
