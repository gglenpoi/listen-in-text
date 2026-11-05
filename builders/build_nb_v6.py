"""Builds listen_in_text_v6.ipynb. Reuses v5 setup cells; new conditions, causal tests and stricter tuned lens."""
import sys
import nbformat as nbf

HERE = __file__.replace("build_nb_v6.py", "")
_v5 = open(HERE + "build_nb_v5.py").read().rsplit("nb = nbf.v4.new_notebook()", 1)[0]
_ns = {"__file__": HERE + "build_nb_v5.py"}
exec(_v5, _ns)
V5 = [c.source for c in _ns["cells"] if c.cell_type == "code"]
_v4 = open(HERE + "build_nb_v4.py").read().rsplit("nb = nbf.v4.new_notebook()", 1)[0]
_ns4 = {"__file__": HERE + "build_nb_v4.py"}
exec(_v4, _ns4)
V4 = [c.source for c in _ns4["cells"] if c.cell_type == "code"]

def pick(cells_, marker):
    hits = [s for s in cells_ if marker in s]
    assert len(hits) == 1, marker
    return hits[0]

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Do Full-Duplex Speech Models Listen in Text? — v6

v5 установил: собственная речь модели подавляет «чтение» собеседника (PersonaPlex теряет ~70%, Moshi ~40%), подавляет именно **свой звук**, а не свой текст, подавление держится только пока модель говорит. Абляция полностью убрала чтение, но поведенческая мера не сработала: после реплики модель с вероятностью ~1 выдаёт PAD. Tuned lens на 4× данных нашёл обобщаемый текстовый код уже с 12-го слоя.

| # | Вопрос | Как проверяем |
|---|---|---|
| H10 | Что в своём звуке подавляет слушание: **слова** или **активность канала**? | Свой канал: речь (как v5) / та же речь **обращённая** / **шум со спектром речи** / речь **на −20 дБ** |
| H11 | Модель «слушает» **себя**? | Чтение слов, звучащих в своём канале, по слоям |
| H9a | Использует ли модель транскрипцию **для ответа**? | **Вынужденный ответ:** через 0,5 с после реплики в своём текстовом потоке ставим EPAD (начало слова) и смотрим распределение первого слова ответа: KL и вероятность услышанных слов при абляции своих направлений против чужих |
| H9b | Использует ли модель транскрипцию **как контекст**? | Вырезаем направления слов **первой половины** реплики (только в первой половине) и смотрим чтение слов **второй половины**; по предсказуемости слов (GPT-2); с контролем перекрытия подпространств |
| H5″ | Код с 12-го слоя — словесный или только «звук → первый кусок слова»? | Строгий тест tuned lens: слова, чей **целевой токен ни разу не встречался в обучении** |

Kaggle: GPU T4×2, Internet On, `HF_TOKEN`. ~60–80 мин на модель.
""")

md("## 1. Конфигурация")
code(r"""
import os
ON_KAGGLE = os.path.exists("/kaggle/working")
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")
BASE = "/kaggle/working" if ON_KAGGLE else ("/content" if ON_COLAB else os.getcwd())

CFG = dict(
    model          = "personaplex",              # "personaplex" | "moshi"
    datasets       = ["librispeech", "ami"],
    conditions     = ["listen", "talking_audio", "own_reversed", "own_noise", "own_quiet"],
    quiet_gain     = 0.1,                        # -20 dB
    n_turns        = 40,
    turn_sec       = 20.0,
    gap_sec        = 0.3,
    aligner        = "mms",
    whisper_model  = "small.en",
    silence_frames = 6,
    post_sec       = 1.5,
    resp_gap_sec   = 0.5,                        # H9a: EPAD forced this long after the user stops
    topk           = 20,
    lag_frames     = 4,
    n_control      = 5,
    min_word_chars = 4,
    n_boot         = 1000,
    ab_layers      = list(range(22, 31)),
    surprisal_lm   = "gpt2",
    # H5'' tuned lens
    tl_extra       = ["dev-clean", "test-other"],
    tl_extra_turns = 60,
    tl_layers      = [0, 4, 8, 12, 16, 20, 24, 28, 32],
    tl_ranks       = [128],
    tl_steps       = 800,
    tl_batch       = 512,
    tl_lr          = 1e-3,
    tl_wd          = 1e-2,
    tl_train_frac  = 0.7,
    dtype          = "auto",
    seed           = 0,
)
PRESETS = {
    "personaplex": dict(model_repo="nvidia/personaplex-7b-v1",
                        text_prompt="You enjoy having a good conversation.", prefix_user="sine"),
    "moshi":       dict(model_repo="kyutai/moshiko-pytorch-bf16", text_prompt="", prefix_user="silence"),
}
CFG.update(PRESETS[CFG["model"]])
CFG["out_dir"] = f"{BASE}/{CFG['model']}_v6"
os.makedirs(CFG["out_dir"], exist_ok=True)
print(CFG["model_repo"], "->", CFG["out_dir"])
""")

md("## 2. Установка и модель")
code(pick(V5, "import sys, subprocess"))
code(pick(V5, "import json, math, random, re, tarfile"))
code(pick(V5, "from huggingface_hub import hf_hub_download, list_repo_files"))
code(pick(V5, "def load_lm_for_lens"))

md("## 3. Данные и границы слов")
code(pick(V5, "def librispeech_turns"))
code(pick(V5, "def whisper_align(turns)"))

md(r"""## 4. Варианты своего канала (H10)
Для реплики `other`, которую модель «произносит» поверх собеседника, кодируем четыре версии звука: исходную речь, ту же речь задом наперёд (тот же голос и спектр, нет слов), шум с тем же долговременным спектром (случайные фазы: нет ни слов, ни ритма речи) и речь на −20 дБ. Свой текстовый поток во всех вариантах — PAD.""")
code(r"""
@torch.no_grad()
def mimi_codes(wav):
    pad = (-len(wav)) % FRAME
    x = torch.from_numpy(np.pad(wav, (0, pad))).to(next(mimi.parameters()).device).view(1, 1, -1)
    return mimi.encode(x)[0].cpu()

def speech_shaped_noise(wav, seed):
    X = np.fft.rfft(wav)
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(X))
    y = np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(wav)).astype(np.float32)
    return y * (np.sqrt(np.mean(wav ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))

def wrap_system(text):
    return f"<system> {text.strip()} <system>"

SIL = torch.as_tensor(SILENCE_TOKENS).view(8, 1)
OWN_SRC = {"talking_audio": "codes_user", "own_reversed": "codes_rev", "own_noise": "codes_noise", "own_quiet": "codes_quiet"}

def own_streams(other, n, cond):
    codes = SIL.repeat(1, n).clone()
    text = np.full(n, 3, dtype=np.int64)
    if cond == "listen":
        return codes, text, (0, 0)
    src = other[OWN_SRC[cond]]
    L = min(n, src.shape[1])
    codes[:, :L] = src[:, :L]
    return codes, text, (0, L)

def build_codes(user_codes, own_codes, own_text, n_post):
    pad = 3
    filler = torch.as_tensor(SINE_TOKENS).view(8, 1) if CFG["prefix_user"] == "sine" else SIL
    cols = []
    def frame(tt, ma, ua): cols.append(torch.cat([torch.tensor([[int(tt)]]), ma, ua], 0))
    for _ in range(CFG["silence_frames"]): frame(pad, SIL, filler)
    for tok in PROMPT_IDS: frame(tok, SIL, filler)
    for _ in range(CFG["silence_frames"]): frame(pad, SIL, filler)
    start = len(cols)
    for i in range(user_codes.shape[1]):
        frame(own_text[i], own_codes[:, i:i + 1], user_codes[:, i:i + 1])
    for _ in range(n_post): frame(pad, SIL, SIL)
    return torch.cat(cols, 1).long(), start

PROMPT_IDS = sp.encode(wrap_system(CFG["text_prompt"])) if CFG["text_prompt"] else []
N_POST = int(round(CFG["post_sec"] * FPS))
for name, turns in DATA.items():
    for i, t in enumerate(turns):
        t["codes_user"] = mimi_codes(t["wav"])
        if name in MAIN:
            t["codes_rev"] = mimi_codes(np.ascontiguousarray(t["wav"][::-1]))
            t["codes_noise"] = mimi_codes(speech_shaped_noise(t["wav"], CFG["seed"] + i))
            t["codes_quiet"] = mimi_codes(t["wav"] * CFG["quiet_gain"])
print("ok, post frames:", N_POST)
""")

md("## 5. Прямой проход")
fwd = pick(V5, "def lens_forward")
fwd_defs = fwd.split("RUNS, TL_FRAMES = {}, {}")[0]
code(fwd_defs + r'''
RUNS, TL_FRAMES = {}, {}
t0 = time.time()
for name, turns in DATA.items():
    conds = CFG["conditions"] if name in MAIN else ["listen"]
    for cond in conds:
        out = []
        for ti, t in enumerate(turns):
            other = turns[(ti + 1) % len(turns)]
            n = t["codes_user"].shape[1]
            own_codes, own_text, interval = own_streams(other, n, cond)
            codes, start = build_codes(t["codes_user"], own_codes, own_text, N_POST)
            # words that actually sound in the model's own channel (H11): only for the forward-speech variants
            own_words = [w for w in other["words"] if w["end"] * FPS < interval[1]] if cond in ("talking_audio", "own_quiet") else []
            rt = dict(id=t["id"], spk=t["spk"], user_start=start, n_user=n, T=codes.shape[1], words=t["words"],
                      text=" ".join(w["word"] for w in t["words"]), own_interval=interval, own_ids=[],
                      own_words=own_words, codes=codes if cond == "listen" else None)
            keep_pos, meta = None, []
            if cond == "listen":
                pos = []
                for wi, w in enumerate(t["words"]):
                    a, b = word_window(rt, w, 0)
                    for p in range(a, b + 1):
                        pos.append(p); meta.append((ti, wi, p - a))
                keep_pos = torch.tensor(pos, dtype=torch.long)
            rt["top"], kept, _ = lens_forward(codes, CFG["topk"], keep_pos, CFG["tl_layers"] if cond == "listen" else ())
            if cond == "listen":
                TL_FRAMES.setdefault(name, dict(meta=[], h={L: [] for L in CFG["tl_layers"]}))
                TL_FRAMES[name]["meta"] += meta
                for L, h in kept.items():
                    TL_FRAMES[name]["h"][L].append(h)
            out.append(rt)
        RUNS[(name, cond)] = out
        print(f"{name}/{cond}: {len(out)} turns ({time.time() - t0:.0f}s)")
for name in TL_FRAMES:
    TL_FRAMES[name]["h"] = {L: torch.cat(v) for L, v in TL_FRAMES[name]["h"].items()}
''')

md(r"""## 6. H10: что в своём канале подавляет слушание
Парное сравнение с listen для тех же слов на лучшем слое: доля потерянного чтения для каждого варианта своего канала.""")
met = pick(V5, "def word_targets(word)")
met_defs = met.split("# words usable in every condition")[0]
code(met_defs + r'''
boot_rng = np.random.default_rng(CFG["seed"])
def boot_ci(x, n=None):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 10: return np.nan, np.nan
    idx = boot_rng.integers(0, len(x), (n or CFG["n_boot"], len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def score_all(name, cond, which="user", seed=0):
    turns = RUNS[(name, cond)]
    rng = random.Random(seed)
    rows = []
    for ti, t in enumerate(turns):
        others = [u for j, u in enumerate(turns) if j != ti]
        words = t["words"] if which == "user" else t["own_words"]
        for wi, w in enumerate(words):
            targets = word_targets(w["word"])
            a, b = word_window(t, w, CFG["lag_frames"])
            r = best_rank(t["top"], a, b, targets)
            if r is None: continue
            c = []
            for _ in range(CFG["n_control"]):
                u = rng.choice(others)
                if w["word"] in u["text"].split(): continue
                n = b - a; lo, hi = u["user_start"] + 1, u["user_start"] + u["n_user"] - n
                if hi <= lo: continue
                s = rng.randint(lo, hi)
                c.append(best_rank(u["top"], s, s + n, targets))
            if c:
                rows.append(dict(ti=ti, wi=wi, word=w["word"], n_chars=len(w["word"]), raw=r, c_raw=np.stack(c)))
    return rows

def excess_curve(rows, min_chars):
    sel = [r for r in rows if r["n_chars"] >= min_chars]
    real = np.stack([r["raw"] < 5 for r in sel]).astype(float)
    ctrl = np.stack([(r["c_raw"] < 5).mean(0) for r in sel])
    out = []
    for L in range(real.shape[1]):
        lo, hi = boot_ci(real[:, L] - ctrl[:, L])
        out.append(dict(layer=L, real=real[:, L].mean(), control=ctrl[:, L].mean(), excess=(real[:, L] - ctrl[:, L]).mean(),
                        ci_lo=lo, ci_hi=hi, n_words=len(sel)))
    return pd.DataFrame(out)

SCORES = {(n, c): score_all(n, c, "user", CFG["seed"]) for n in MAIN for c in CFG["conditions"]}
curves, paired = [], []
for (name, cond), rows in SCORES.items():
    curves.append(excess_curve(rows, CFG["min_word_chars"]).assign(dataset=name, cond=cond))
CURVES = pd.concat(curves, ignore_index=True)
BEST = {n: int(CURVES[(CURVES.dataset == n) & (CURVES.cond == "listen")].sort_values("excess").iloc[-1].layer) for n in MAIN}
for name in MAIN:
    base = {(r["ti"], r["wi"]): r for r in SCORES[(name, "listen")] if r["n_chars"] >= CFG["min_word_chars"]}
    for cond in CFG["conditions"][1:]:
        pairs = [(base[(r["ti"], r["wi"])], r) for r in SCORES[(name, cond)] if (r["ti"], r["wi"]) in base]
        for L in range(N_LAYERS):
            lis = np.array([b_["raw"][L] < 5 for b_, r in pairs], float)
            d = lis - np.array([r["raw"][L] < 5 for b_, r in pairs], float)
            lo, hi = boot_ci(d)
            paired.append(dict(dataset=name, cond=cond, layer=L, listen=lis.mean(), drop=d.mean(), ci_lo=lo, ci_hi=hi, n_words=len(pairs)))
PAIRED = pd.DataFrame(paired)
CURVES.to_csv(f"{CFG['out_dir']}/hit_curves_v6.csv", index=False); PAIRED.to_csv(f"{CFG['out_dir']}/paired_drop_v6.csv", index=False)
print("best listen layer:", BEST)
for name in MAIN:
    p = PAIRED[(PAIRED.dataset == name) & (PAIRED.layer == BEST[name])].set_index("cond")
    p = p.assign(share_lost=p["drop"] / p.listen, share_lo=p.ci_lo / p.listen, share_hi=p.ci_hi / p.listen)
    print(f"\n{name}, layer {BEST[name]}:"); print(p[["listen", "drop", "share_lost", "share_lo", "share_hi", "n_words"]].round(3).to_string())
''')

md("## 7. H11: слышит ли модель себя?\nЧтение слов, звучащих в собственном канале модели (`talking_audio`, `own_quiet`), по слоям — по той же метрике, что и для собеседника.")
code(r'''
own_curves = []
for name in MAIN:
    for cond in ("talking_audio", "own_quiet"):
        rows = score_all(name, cond, "own", CFG["seed"])
        if rows:
            own_curves.append(excess_curve(rows, CFG["min_word_chars"]).assign(dataset=name, cond=cond))
OWNC = pd.concat(own_curves, ignore_index=True) if own_curves else pd.DataFrame()
OWNC.to_csv(f"{CFG['out_dir']}/own_channel_readout.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
cc = {"listen": "C0", "talking_audio": "C1", "own_reversed": "C7", "own_noise": "C5", "own_quiet": "C2"}
for ax, name in zip(axes, MAIN):
    for cond in CFG["conditions"]:
        d = CURVES[(CURVES.dataset == name) & (CURVES.cond == cond)]
        ax.plot(d.layer, d.excess, color=cc[cond], lw=2, label=f"user words: {cond}")
    for cond in ("talking_audio", "own_quiet"):
        d = OWNC[(OWNC.dataset == name) & (OWNC.cond == cond)] if len(OWNC) else []
        if len(d):
            ax.plot(d.layer, d.excess, color=cc[cond], lw=1.5, ls="--", label=f"OWN-channel words: {cond}")
    ax.axhline(0, color="k", lw=.8); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set(title=f"{CFG['model']} / {name}", xlabel="layer", ylabel="hit@5 excess over control")
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/own_channel_v6.png", dpi=150); plt.show()
for name in MAIN:
    if len(OWNC):
        d = OWNC[(OWNC.dataset == name)].sort_values("excess").groupby("cond").tail(1)
        print(name, "max own-channel readout:", d[["cond", "layer", "excess", "ci_lo", "ci_hi"]].round(3).to_dict("records"))
''')

md(r"""## 8. H9a: вынужденный ответ
Через `resp_gap_sec` после конца реплики в собственный текстовый поток модели ставим EPAD («сейчас начнётся слово»). Распределение текстовой головы на следующем кадре — это первое слово ответа. Сравниваем чистый прогон с двумя абляциями (на слоях 22–30, в кадрах речи собеседника): направления слов **этой** реплики против того же числа направлений слов **других** реплик.""")
code(pick(V5, "ALPHA = lm.out_norm.alpha").split("ab_rows = []")[0] + r'''
GAP = int(round(CFG["resp_gap_sec"] * FPS))
rng_ab = random.Random(CFG["seed"])
def content_ids(words):
    ids = set()
    for w in words:
        if len(w["word"]) >= CFG["min_word_chars"]:
            ids |= word_targets(w["word"])
    return ids

resp = []
t0 = time.time()
for name in MAIN:
    turns = RUNS[(name, "listen")]
    pool_all = [content_ids(u["words"]) for u in turns]
    for ti, t in enumerate(turns):
        own = pool_all[ti]
        pool = set().union(*[p for j, p in enumerate(pool_all) if j != ti]) - own
        if len(own) < 3 or len(pool) < len(own): continue
        ctrl = set(rng_ab.sample(sorted(pool), len(own)))
        a, b = t["user_start"], t["user_start"] + t["n_user"]
        e = b + GAP
        codes = t["codes"].clone(); codes[0, e] = 0              # EPAD forced in the model's own text stream
        pos = list(range(b, e + 2))
        _, _, clean = lens_forward(codes, 5, final_pos=pos)
        row = dict(dataset=name, ti=ti, k=len(own))
        own_t, ctrl_t = torch.tensor(sorted(own)), torch.tensor(sorted(ctrl))
        def summ(lp, tag):
            first = lp[-1]                                       # distribution of the first word of the answer
            row[f"p_epad_{tag}"] = lp[:-1, 0].exp().mean().item()   # turn-taking: P(EPAD) before the forced onset
            row[f"heard_{tag}"] = first.exp()[own_t].sum().item()
            row[f"other_{tag}"] = first.exp()[ctrl_t].sum().item()
            row[f"top1_{tag}"] = int(first.argmax())
            return first
        f_clean = summ(clean, "clean")
        for tag, ids in (("own", own), ("ctrl", ctrl)):
            _, fin = lens_forward(codes, 5, ablate=(basis(ids), a, b), final_pos=pos)
            f = summ(fin, tag)
            row[f"kl_{tag}"] = torch.sum(f_clean.exp() * (f_clean - f)).item()
        resp.append(row)
    print(f"{name}: forced-response ablation done ({time.time() - t0:.0f}s)")
RESP = pd.DataFrame(resp); RESP.to_csv(f"{CFG['out_dir']}/forced_response.csv", index=False)

for name, g in RESP.groupby("dataset"):
    print(f"\n=== {name}: n={len(g)} turns, ~{g.k.mean():.0f} directions ===")
    print(f"  clean: P(heard words) in first answer word {g.heard_clean.mean():.4f} | P(other words) {g.other_clean.mean():.4f} "
          f"| P(EPAD) before onset {g.p_epad_clean.mean():.4f}")
    print("  first answer words (clean, top-1):", ", ".join(sp.id_to_piece(int(i)) for i in g.top1_clean.value_counts().index[:15]))
    d = g.kl_own - g.kl_ctrl; lo, hi = boot_ci(d.values)
    print(f"  KL clean->ablated: own {g.kl_own.mean():.4f} | ctrl {g.kl_ctrl.mean():.4f} | own-ctrl {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | own>ctrl in {np.mean(d > 0):.0%} of turns")
    for m in ("heard", "p_epad"):
        d = (g[f"{m}_own"] - g[f"{m}_clean"]) - (g[f"{m}_ctrl"] - g[f"{m}_clean"]); lo, hi = boot_ci(d.values)
        print(f"  {m:7s} Δown {np.mean(g[f'{m}_own'] - g[f'{m}_clean']):+.5f} | Δctrl {np.mean(g[f'{m}_ctrl'] - g[f'{m}_clean']):+.5f} | own-ctrl {d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]")
    r = np.log((g.heard_own + 1e-9) / (g.heard_clean + 1e-9)) - np.log((g.heard_ctrl + 1e-9) / (g.heard_clean + 1e-9))
    lo, hi = boot_ci(r.values)
    print(f"  log-ratio of P(heard words), own vs ctrl ablation: {r.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  (<0: the answer forgets what was heard)")
''')

md(r"""## 9. H9b: транскрипция как контекст
В каждой реплике: слова первой половины (до середины) и второй (начинаются хотя бы через 160 мс после середины). Вырезаем направления слов **первой половины** только в кадрах первой половины (слои 22–30) и смотрим чтение слов **второй половины** на лучшем слое. Контроль — столько же направлений чужих слов. Чтобы исключить прямое перекрытие подпространств, для каждого слова второй половины считаем, какая доля его вектора лежит в вырезанном подпространстве (`e_own`, `e_ctrl`), и показываем результат отдельно для слов с малым перекрытием. Отдельно — по предсказуемости слов (сюрприз GPT-2).""")
fact = pick(V4, "from wordfreq import zipf_frequency")
surp_defs = fact.split("FACT = []")[0].replace("from wordfreq import zipf_frequency\n", "")
code(surp_defs + r'''
def first_target_vec(word):
    enc = sp.encode(word)
    return (W_OUT[enc[0]].float() * ALPHA.to(W_OUT.device)) if enc else None

def energy(Qd, v):
    q = Qd[str(v.device)] if str(v.device) in Qd else next(iter(Qd.values())).to(v.device)
    return float(((q.T @ v) ** 2).sum() / (v @ v))

ctx = []
t0 = time.time()
for name in MAIN:
    turns = RUNS[(name, "listen")]
    L = BEST[name]
    pool_all = [content_ids(u["words"]) for u in turns]
    for ti, t in enumerate(turns):
        mid = t["n_user"] // 2
        first = [w for w in t["words"] if math.ceil(w["end"] * FPS) <= mid]
        second = [(wi, w) for wi, w in enumerate(t["words"]) if math.floor(w["start"] * FPS) >= mid + 2 and len(w["word"]) >= CFG["min_word_chars"]]
        ids_first = content_ids(first) - content_ids([w for _, w in second])
        pool = set().union(*[p for j, p in enumerate(pool_all) if j != ti]) - pool_all[ti]
        if len(ids_first) < 3 or len(second) < 3 or len(pool) < len(ids_first): continue
        ids_ctrl = set(rng_ab.sample(sorted(pool), len(ids_first)))
        Qo, Qc = basis(ids_first), basis(ids_ctrl)
        a = t["user_start"]
        chk_own, _ = lens_forward(t["codes"], 5, ablate=(Qo, a, a + mid + 1), check_layer=L)
        chk_ctrl, _ = lens_forward(t["codes"], 5, ablate=(Qc, a, a + mid + 1), check_layer=L)
        surp = word_surprisal([w["word"] for w in t["words"]])
        for wi, w in second:
            tg = torch.tensor(sorted(word_targets(w["word"])), dtype=torch.int32)
            if len(tg) == 0: continue
            s0, s1 = word_window(t, w, CFG["lag_frames"])
            hit = lambda top5: float(torch.isin(top5[s0:s1 + 1], tg).any())
            v = first_target_vec(w["word"])
            ctx.append(dict(dataset=name, ti=ti, wi=wi, word=w["word"], surprisal=surp[wi],
                            clean=hit(t["top"][L, :, :5]), own=hit(chk_own), ctrl=hit(chk_ctrl),
                            e_own=energy(Qo, v), e_ctrl=energy(Qc, v)))
    print(f"{name}: context ablation done ({time.time() - t0:.0f}s)")
CTX = pd.DataFrame(ctx).dropna(); CTX.to_csv(f"{CFG['out_dir']}/context_ablation.csv", index=False)
del slm; gc.collect(); torch.cuda.empty_cache()

def ctx_report(g, label):
    d = (g.own - g.clean) - (g.ctrl - g.clean); lo, hi = boot_ci(d.values)
    print(f"  {label:34s} n={len(g):4d} clean {g.clean.mean():.3f} | own {g.own.mean():.3f} | ctrl {g.ctrl.mean():.3f} | own-ctrl {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
for name, g in CTX.groupby("dataset"):
    print(f"\n=== {name}: readout of second-half words after ablating first-half words ===")
    print(f"  subspace overlap: e_own {g.e_own.mean():.4f} vs e_ctrl {g.e_ctrl.mean():.4f}")
    ctx_report(g, "all words")
    thr = g.e_ctrl.quantile(0.75)
    ctx_report(g[g.e_own <= thr], "low overlap (e_own <= ctrl 75th pct)")
    g = g.assign(tert=pd.qcut(g.surprisal, 3, labels=["predictable", "medium", "surprising"]))
    for tert, h in g.groupby("tert", observed=True):
        ctx_report(h, f"surprisal: {tert}")
''')

md(r"""## 10. H5″: строгий tuned lens
Как в v5 (обучение на dev-clean + test-other + 70% дикторов test-clean), плюс тест **unseen_target**: слова, чей первый токен ни разу не был целевым при обучении. Если линза читает и их, код с 12-го слоя — не просто запомненное соответствие «начало звучания → кусок слова» для знакомых кусков.""")
code(pick(V5, "class LowRankLens"))
tl = pick(V5, "def prep(name)")
tl = tl.replace('''tests = {"seen": test[test.word.isin(seen)], "unseen": test[~test.word.isin(seen)]}''',
                '''seen_targets = set(np.concatenate([f.target.values for _, f in train_sets]).tolist())
tests = {"seen": test[test.word.isin(seen)], "unseen": test[~test.word.isin(seen)],
         "unseen_target": test[~test.target.isin(seen_targets)]}''')
tl = tl.replace('''    fa = prep("ami"); tests["ami_unseen"] = fa[~fa.word.isin(seen)]''',
                '''    fa = prep("ami"); tests["ami_unseen"] = fa[~fa.word.isin(seen)]; tests["ami_unseen_target"] = fa[~fa.target.isin(seen_targets)]''')
tl = tl.replace('print(TL[TL.test == "unseen"].pivot_table(index="layer", columns=["lens", "rank"], values="hit").round(3).to_string())',
                'for tag in tests:\n    print(f"--- {tag}"); print(TL[TL.test == tag].pivot_table(index="layer", columns=["lens", "rank"], values="hit").round(3).to_string())')
tl = tl.replace("tuned_lens_v5", "tuned_lens_v6")
assert "unseen_target" in tl and "ami_unseen_target" in tl and "tuned_lens_v6" in tl and 'for tag in tests:' in tl
code(tl)

md("## 11. Сохранение")
code(r"""
for (name, cond), turns in RUNS.items():
    if name not in MAIN: continue
    d = f"{CFG['out_dir']}/{name}_{cond}"
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(f"{d}/top5.npz", **{t["id"]: t["top"][:, :, :5].numpy().astype(np.int32) for t in turns})
    json.dump([dict(id=t["id"], spk=t["spk"], user_start=t["user_start"], n_user=t["n_user"], text=t["text"], words=t["words"],
                    own_words=t["own_words"], own_interval=list(t["own_interval"])) for t in turns], open(f"{d}/turns.json", "w"))
    json.dump(dict(CFG, model=f"{CFG['model']}|{name}|{cond}"), open(f"{d}/config.json", "w"), indent=2)
json.dump(dict(CFG, aligner_used=ALIGNER_USED), open(f"{CFG['out_dir']}/config.json", "w"), indent=2)
import shutil
from IPython.display import FileLink, display
archive = shutil.make_archive(CFG["out_dir"], "zip", CFG["out_dir"])
print(archive, f"{os.path.getsize(archive) / 1e6:.1f} MB")
os.chdir(BASE); display(FileLink(os.path.basename(archive)))
""")

md(r"""
## Как читать
- **H10:** если `own_reversed` и `own_noise` подавляют так же, как речь, — дело в активности своего канала; если слабее — в словах своей речи. `own_quiet` показывает, зависит ли эффект от громкости.
- **H11:** пунктирные кривые: читает ли модель слова собственного канала. Ноль при сильном подавлении собеседника = «не слушает ни себя, ни его».
- **H9a:** главное — `log-ratio of P(heard words)` < 0 с интервалом ниже нуля и `KL own-ctrl` > 0: без транскрипции ответ «забывает» услышанное.
- **H9b:** `own-ctrl` < 0 для слов второй половины (особенно предсказуемых) при малом перекрытии подпространств: транскрипция ранних слов служит контекстом для узнавания следующих.
- **H5″:** `unseen_target` у tuned lens выше logit lens и перемешанного контроля со слоя ~12.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "listen_in_text_v6.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
