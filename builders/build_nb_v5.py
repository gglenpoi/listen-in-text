"""Builds listen_in_text_v5.ipynb. Reuses v4 setup cells (install, model loading, MMS alignment) and replaces the rest."""
import sys
import nbformat as nbf

V4 = open(__file__.replace("build_nb_v5.py", "build_nb_v4.py")).read().split("nb = nbf.v4.new_notebook()")[0]
_ns = {}
exec(V4, _ns)
V4_CODE = [c.source for c in _ns["cells"] if c.cell_type == "code"]
def v4_cell(marker):
    hits = [s for s in V4_CODE if marker in s]
    assert len(hits) == 1, marker
    return hits[0]

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Do Full-Duplex Speech Models Listen in Text? — v5

v4 показал: когда модель сама говорит, транскрипция собеседника почти исчезает (H4). v5 проверяет, правда ли это, и как это устроено, плюс две проверки, которых не хватало для статьи.

| # | Вопрос | Как проверяем |
|---|---|---|
| H4′ | Подавление при собственной речи — не артефакт конкуренции за top-k? | Ранг слова собеседника считается **без собственных токенов модели**; парное сравнение «то же слово: слушает / говорит» |
| H7 | Что подавляет: свой **звук** или свой **текст**? | `talking_audio` (только своя речь, текст PAD) и `talking_text` (только свой текст, аудио — тишина) |
| H8 | Возвращается ли транскрипция, когда модель замолкает, и как быстро? | `talking_partial`: модель говорит только среднюю треть реплики собеседника; попадания до / во время / после, по времени с момента, как модель замолчала |
| H9 | Модель **использует** эту транскрипцию? | Абляция: на слоях 22–30 вырезаем из скрытого состояния направления текстовых токенов **услышанных** слов и смотрим на прогноз модели после конца реплики (что и когда она скажет). Контроль — вырезать столько же направлений **чужих** слов |
| H5′ | Нет раннего текстового кода — или линзе не хватало данных? | Tuned lens на ~4× большем объёме (LibriSpeech dev-clean + test-other), два ранга, проверка на невиданных словах с интервалами |

**Kaggle:** GPU T4×2, Internet On, `HF_TOKEN`. ~60–80 мин на модель. Начните с `personaplex`, затем `moshi`.
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
    conditions     = ["listen", "talking", "talking_audio", "talking_text", "talking_partial"],
    n_turns        = 40,
    turn_sec       = 20.0,
    gap_sec        = 0.3,
    aligner        = "mms",
    whisper_model  = "small.en",
    silence_frames = 6,
    post_sec       = 1.5,        # user silence appended after each turn: where the model would answer (H9)
    topk           = 20,
    lag_frames     = 4,
    n_control      = 5,
    min_word_chars = 4,
    n_boot         = 500,
    # H9 causal ablation
    ab_layers      = list(range(22, 31)),
    check_layer    = 30,         # lens readout after ablation (manipulation check)
    # H5' tuned lens with more data
    tl_extra       = ["dev-clean", "test-other"],
    tl_extra_turns = 60,         # per extra subset
    tl_layers      = [0, 4, 8, 12, 16, 20, 22, 24, 26, 28, 30, 32],
    tl_ranks       = [32, 128],
    tl_steps       = 600,
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
CFG["out_dir"] = f"{BASE}/{CFG['model']}_v5"
os.makedirs(CFG["out_dir"], exist_ok=True)
print(CFG["model_repo"], "->", CFG["out_dir"])
""")

md("## 2. Установка и модель")
code(v4_cell("import sys, subprocess"))
code(v4_cell("import json, math, random, re, tarfile"))
code(v4_cell("from huggingface_hub import hf_hub_download, list_repo_files"))
code(v4_cell("def load_lm_for_lens"))

md("## 3. Данные\nОсновные наборы — LibriSpeech test-clean и AMI IHM (как в v4). Для tuned lens дополнительно LibriSpeech dev-clean и test-other (только условие listen); их дикторы не пересекаются с test-clean.")
data_cell = v4_cell("def librispeech_turns")
data_cell = data_cell.replace(
    'def librispeech_turns(n_turns, turn_sec, gap_sec, sr_out=24000):\n    root = Path(f"{BASE}/librispeech")\n    if not (root / "LibriSpeech" / "test-clean").exists():\n        root.mkdir(parents=True, exist_ok=True)\n        tgz = root / "test-clean.tar.gz"\n        urllib.request.urlretrieve("https://www.openslr.org/resources/12/test-clean.tar.gz", tgz)\n        with tarfile.open(tgz) as t:\n            t.extractall(root)\n    chapters = sorted((root / "LibriSpeech" / "test-clean").glob("*/*"))',
    'def librispeech_turns(n_turns, turn_sec, gap_sec, sr_out=24000, subset="test-clean"):\n    root = Path(f"{BASE}/librispeech")\n    if not (root / "LibriSpeech" / subset).exists():\n        root.mkdir(parents=True, exist_ok=True)\n        tgz = root / f"{subset}.tar.gz"\n        urllib.request.urlretrieve(f"https://www.openslr.org/resources/12/{subset}.tar.gz", tgz)\n        with tarfile.open(tgz) as t:\n            t.extractall(root)\n        tgz.unlink()\n    chapters = sorted((root / "LibriSpeech" / subset).glob("*/*"))')
data_cell = data_cell.replace('turns.append(dict(id=f"ls-{ch.parent.name}-{ch.name}"', 'turns.append(dict(id=f"ls-{subset}-{ch.parent.name}-{ch.name}"')
assert 'subset="test-clean"' in data_cell and "ls-{subset}-" in data_cell
data_cell = data_cell.replace('''assert DATA, "no dataset loaded"''', '''assert DATA, "no dataset loaded"
MAIN = [n for n in CFG["datasets"] if n in DATA]
for sub in CFG["tl_extra"]:
    try:
        DATA[f"ls_{sub}"] = librispeech_turns(CFG["tl_extra_turns"], CFG["turn_sec"], CFG["gap_sec"], subset=sub)
        print(f"ls_{sub}: {len(DATA[f'ls_{sub}'])} turns (tuned lens only)")
    except Exception as e:
        print(f"!! ls_{sub} skipped: {type(e).__name__}: {e}")''')
code(data_cell)

md("## 4. Границы слов (MMS)")
code(v4_cell("def whisper_align(turns)"))

md(r"""## 5. Входы для пяти условий
Собственный поток модели во время речи собеседника (`other` — другая реплика того же набора):

| условие | своё аудио | свой текст | когда |
|---|---|---|---|
| listen | тишина | PAD | — |
| talking | речь `other` | текст `other` | вся реплика |
| talking_audio | речь `other` | PAD | вся реплика |
| talking_text | тишина | текст `other` | вся реплика |
| talking_partial | речь `other` | текст `other` | средняя треть |

После реплики во всех условиях добавляется `post_sec` тишины собеседника — это место, где модель начала бы отвечать (H9).""")
code(r"""
@torch.no_grad()
def mimi_codes(wav):
    pad = (-len(wav)) % FRAME
    x = torch.from_numpy(np.pad(wav, (0, pad))).to(next(mimi.parameters()).device).view(1, 1, -1)
    return mimi.encode(x)[0].cpu()

def wrap_system(text):
    return f"<system> {text.strip()} <system>"

SIL = torch.as_tensor(SILENCE_TOKENS).view(8, 1)

def text_track(words, n_frames, shift=0):
    track = np.full(n_frames, 3, dtype=np.int64)
    nxt = 0
    for w in words:
        f = max(int(math.floor(w["start"] * FPS)) + shift, nxt)
        if f >= n_frames: break
        ids = sp.encode(w["word"])
        if f - 1 >= 0 and track[f - 1] == 3: track[f - 1] = 0
        for k, tid in enumerate(ids):
            if f + k < n_frames: track[f + k] = tid
        nxt = f + len(ids)
    return track

def own_streams(other, n, cond):
    # model-own audio [8, n] and text [n] while the user speaks, plus the frame interval where the model 'talks'
    codes = SIL.repeat(1, n).clone()
    text = np.full(n, 3, dtype=np.int64)
    if cond == "listen":
        return codes, text, (0, 0)
    s0, s1 = (n // 3, 2 * n // 3) if cond == "talking_partial" else (0, n)
    L = min(s1 - s0, other["codes_user"].shape[1])
    if cond in ("talking", "talking_audio", "talking_partial"):
        codes[:, s0:s0 + L] = other["codes_user"][:, :L]
    if cond in ("talking", "talking_text", "talking_partial"):
        text[s0:s0 + L] = text_track(other["words"], L)
    return codes, text, (s0, s0 + L)

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
    for t in turns:
        t["codes_user"] = mimi_codes(t["wav"])
print("ok, post frames:", N_POST)
""")

md("## 6. Прямой проход\nТоп-20 текстовой головы на каждом слое; для listen — скрытые состояния кадров слов (для tuned lens).")
code(r"""
@torch.no_grad()
def lens_forward(codes, topk, keep_pos=None, keep_layers=(), ablate=None, final_pos=None, check_layer=None):
    # ablate: (Q dict device->[D,k], a, b): remove span(Q) from the residual at positions [a, b) after layers in ab_layers
    first, last = LAYER_DEVS[0], LAYER_DEVS[-1]
    codes = codes.unsqueeze(0).to(first)
    initial = lm._get_initial_token().to(first)
    seq = torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], 2)[:, :, :-1]
    x = lm.embed_codes(seq)
    special = torch.tensor(SPECIAL_TEXT, device=last)
    def lens(h):
        logits = lm.text_linear(lm.out_norm(h.to(last))).float()[0]
        logits[:, special] = -float("inf")
        return logits.topk(topk, -1).indices.int().cpu()
    tops, kept, check = [], {}, None
    def keep(L, h):
        if keep_pos is not None and L in keep_layers:
            kept[L] = h[0, keep_pos].to(torch.float16).cpu()
    if ablate is None:
        tops.append(lens(x)); keep(0, x)
    for i, layer in enumerate(lm.transformer.layers):
        x = layer(x.to(LAYER_DEVS[i]))
        if ablate is not None and (i + 1) in CFG["ab_layers"]:
            Q, a, b = ablate
            q = Q[str(x.device)]
            seg = x[0, a:b].float()
            x[0, a:b] = (seg - (seg @ q) @ q.T).to(x.dtype)
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"non-finite activations after layer {i}")
        if ablate is None:
            tops.append(lens(x)); keep(i + 1, x)
        elif check_layer == i + 1:
            check = lens(x)[:, :5]
    final = None
    if final_pos is not None:
        final = torch.log_softmax(lm.text_linear(lm.out_norm(x.to(last))).float()[0, final_pos], -1).cpu()
    if ablate is not None:
        return check, final
    return torch.stack(tops), kept, final

def word_window(t, w, lag):
    f0 = int(math.floor(w["start"] * FPS)); f1 = int(math.ceil(w["end"] * FPS))
    return t["user_start"] + f0 + 1, min(t["user_start"] + f1 + 1 + lag, t["T"] - 1)

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
            rt = dict(id=t["id"], spk=t["spk"], user_start=start, n_user=n, T=codes.shape[1], words=t["words"],
                      text=" ".join(w["word"] for w in t["words"]), own_interval=interval,
                      own_ids=sorted(set(int(i) for i in own_text) - set(SPECIAL_TEXT)), codes=codes if cond == "listen" else None)
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
    print(name, "tuned-lens frames:", len(TL_FRAMES[name]["meta"]))
""")

md(r"""## 7. H4′ и H7: попадания без собственных токенов модели
Для каждого слова собеседника — лучший ранг среди токенов, **не входящих** в собственный текст модели в этой реплике (`excl`), и обычный ранг (`raw`). Слова, первый токен которых совпадает с собственным текстом модели, исключаются во всех условиях. Главный показатель — **парная разница с listen для тех же слов**: какая доля чтения теряется, когда модель говорит.""")
code(r"""
def word_targets(word):
    ids = set()
    for v in {word, word.capitalize(), word.upper()}:
        enc = sp.encode(v)
        if enc: ids.add(enc[0])
    return ids - set(SPECIAL_TEXT)

def best_rank(top, a, b, targets, excl=()):
    if b < a or not targets: return None
    win = top[:, a:b + 1, :]
    hit = torch.isin(win, torch.tensor(sorted(targets), dtype=win.dtype))
    K = top.shape[2]
    if excl:
        valid = ~torch.isin(win, torch.tensor(sorted(excl), dtype=win.dtype))
        rank = valid.long().cumsum(-1) - 1
        ranks = torch.where(hit & valid, rank, torch.full_like(rank, K))
    else:
        ranks = torch.where(hit, torch.arange(K).view(1, 1, K).expand_as(win), torch.full_like(win, K))
    return ranks.amin(dim=(1, 2)).numpy()

# words usable in every condition of a dataset: their target must not collide with any own text of that turn
BLOCK = {}
for name in MAIN:
    for cond in CFG["conditions"]:
        for t in RUNS[(name, cond)]:
            for wi, w in enumerate(t["words"]):
                if word_targets(w["word"]) & set(t["own_ids"]):
                    BLOCK.setdefault(name, set()).add((t["id"], wi))

def score_all(name, cond, seed=0):
    turns = RUNS[(name, cond)]
    rng = random.Random(seed)
    rows = []
    for ti, t in enumerate(turns):
        others = [u for j, u in enumerate(turns) if j != ti]
        for wi, w in enumerate(t["words"]):
            if (t["id"], wi) in BLOCK.get(name, set()): continue
            targets = word_targets(w["word"])
            a, b = word_window(t, w, CFG["lag_frames"])
            r_raw = best_rank(t["top"], a, b, targets)
            if r_raw is None: continue
            r_exc = best_rank(t["top"], a, b, targets, t["own_ids"])
            c_raw, c_exc = [], []
            for _ in range(CFG["n_control"]):
                u = rng.choice(others)
                if w["word"] in u["text"].split(): continue
                n = b - a; lo, hi = u["user_start"] + 1, u["user_start"] + u["n_user"] - n
                if hi <= lo: continue
                s = rng.randint(lo, hi)
                c_raw.append(best_rank(u["top"], s, s + n, targets))
                c_exc.append(best_rank(u["top"], s, s + n, targets, u["own_ids"]))
            if c_raw:
                f0 = int(math.floor(w["start"] * FPS)); f1 = int(math.ceil(w["end"] * FPS))
                rows.append(dict(ti=ti, wi=wi, word=w["word"], n_chars=len(w["word"]), f0=f0, f1=f1,
                                 raw=r_raw, exc=r_exc, c_raw=np.stack(c_raw), c_exc=np.stack(c_exc)))
    return rows

boot_rng = np.random.default_rng(CFG["seed"])
def boot_ci(x, n=None):
    x = x[~np.isnan(x)]
    if len(x) < 10: return np.nan, np.nan
    idx = boot_rng.integers(0, len(x), (n or CFG["n_boot"], len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

SCORES = {(n, c): score_all(n, c, CFG["seed"]) for n in MAIN for c in CFG["conditions"]}
curves, paired = [], []
for (name, cond), rows in SCORES.items():
    sel = [r for r in rows if r["n_chars"] >= CFG["min_word_chars"]]
    base = {(r["ti"], r["wi"]): r for r in SCORES[(name, "listen")]}
    for metric, key, ckey in (("raw", "raw", "c_raw"), ("excl", "exc", "c_exc")):
        real = np.stack([r[key] < 5 for r in sel]).astype(float)
        ctrl = np.stack([(r[ckey] < 5).mean(0) for r in sel])
        for L in range(real.shape[1]):
            lo, hi = boot_ci(real[:, L] - ctrl[:, L])
            curves.append(dict(dataset=name, cond=cond, metric=metric, layer=L, real=real[:, L].mean(),
                               control=ctrl[:, L].mean(), excess=(real[:, L] - ctrl[:, L]).mean(), ci_lo=lo, ci_hi=hi, n_words=len(sel)))
    if cond != "listen":
        pairs = [(base[(r["ti"], r["wi"])], r) for r in sel if (r["ti"], r["wi"]) in base]
        d = np.stack([(b_["exc"] < 5).astype(float) - (r["exc"] < 5).astype(float) for b_, r in pairs])
        lis = np.stack([(b_["exc"] < 5).astype(float) for b_, r in pairs])
        for L in range(d.shape[1]):
            lo, hi = boot_ci(d[:, L])
            paired.append(dict(dataset=name, cond=cond, layer=L, listen=lis[:, L].mean(), drop=d[:, L].mean(),
                               ci_lo=lo, ci_hi=hi, n_words=len(pairs)))
CURVES, PAIRED = pd.DataFrame(curves), pd.DataFrame(paired)
CURVES.to_csv(f"{CFG['out_dir']}/hit_curves_v5.csv", index=False)
PAIRED.to_csv(f"{CFG['out_dir']}/paired_drop_v5.csv", index=False)
BEST = {n: int(CURVES[(CURVES.dataset == n) & (CURVES.cond == "listen") & (CURVES.metric == "excl")].sort_values("excess").iloc[-1].layer) for n in MAIN}
print("best listen layer:", BEST)
for n in MAIN:
    L = BEST[n]
    d = CURVES[(CURVES.dataset == n) & (CURVES.layer == L)].pivot_table(index="cond", columns="metric", values="excess")
    p = PAIRED[(PAIRED.dataset == n) & (PAIRED.layer == L)].set_index("cond")[["listen", "drop", "ci_lo", "ci_hi"]]
    p["share_lost"] = p["drop"] / p["listen"]
    print(f"\n{n}, layer {L}: excess over control (raw vs own-tokens-excluded)"); print(d.round(3).to_string())
    print("paired drop vs listen (same words, own tokens excluded):"); print(p.round(3).to_string())
""")
code(r"""
cc = {"listen": "C0", "talking": "C3", "talking_audio": "C1", "talking_text": "C2", "talking_partial": "C4"}
fig, axes = plt.subplots(1, len(MAIN) + 1, figsize=(7 * (len(MAIN) + 1), 4.2), squeeze=False)
for ax, name in zip(axes[0], MAIN):
    for cond in CFG["conditions"]:
        d = CURVES[(CURVES.dataset == name) & (CURVES.cond == cond) & (CURVES.metric == "excl")]
        ax.plot(d.layer, d.excess, color=cc[cond], lw=2, label=cond)
        ax.fill_between(d.layer, d.ci_lo, d.ci_hi, color=cc[cond], alpha=.15)
        if cond != "listen":
            r = CURVES[(CURVES.dataset == name) & (CURVES.cond == cond) & (CURVES.metric == "raw")]
            ax.plot(r.layer, r.excess, color=cc[cond], lw=1, ls=":")
    ax.axhline(0, color="k", lw=.8); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set(title=f"{CFG['model']} / {name}: user word in top-5 (own tokens excluded; dotted = raw)", xlabel="layer", ylabel="hit@5 excess")
ax = axes[0][-1]
for i, name in enumerate(MAIN):
    p = PAIRED[(PAIRED.dataset == name) & (PAIRED.layer == BEST[name])]
    x = np.arange(len(p)) + i * 0.3
    ax.bar(x, p["drop"] / p["listen"], width=0.3, label=name)
    ax.set_xticks(np.arange(len(p)) + 0.15); ax.set_xticklabels(p.cond, rotation=20)
ax.set(title="share of listen readout lost (same words, best layer)", ylabel="drop / listen"); ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/conditions_v5.png", dpi=150); plt.show()
""")

md(r"""## 8. H8: возвращается ли транскрипция, когда модель замолкает
`talking_partial`: модель говорит в средней трети реплики собеседника. Для каждого слова — фаза (до / во время / после) и время с момента, как модель замолчала. Сравнение парное: то же слово в `listen` и в `talking_partial` (без собственных токенов).""")
code(r"""
def phase(r, interval):
    s0, s1 = interval
    if r["f1"] <= s0: return "before", np.nan
    if r["f0"] < s1: return "during", np.nan
    return "after", (r["f0"] - s1) / FPS

dyn = []
for name in MAIN:
    L = BEST[name]
    base = {(r["ti"], r["wi"]): r for r in SCORES[(name, "listen")]}
    turns = RUNS[(name, "talking_partial")]
    for r in SCORES[(name, "talking_partial")]:
        if r["n_chars"] < CFG["min_word_chars"] or (r["ti"], r["wi"]) not in base: continue
        ph, since = phase(r, turns[r["ti"]]["own_interval"])
        b = base[(r["ti"], r["wi"])]
        dyn.append(dict(dataset=name, phase=ph, since=since, listen=float(b["exc"][L] < 5), partial=float(r["exc"][L] < 5)))
DYN = pd.DataFrame(dyn)
DYN["bin"] = DYN.phase
after = DYN.phase == "after"
DYN.loc[after, "bin"] = pd.cut(DYN.since[after], [-0.01, 0.5, 1.0, 2.0, 99], labels=["after 0–0.5 s", "after 0.5–1 s", "after 1–2 s", "after >2 s"]).astype(str)
order = ["before", "during", "after 0–0.5 s", "after 0.5–1 s", "after 1–2 s", "after >2 s"]
rows = []
for (name, b), g in DYN.groupby(["dataset", "bin"]):
    d = (g.listen - g.partial).values
    lo, hi = boot_ci(d)
    rows.append(dict(dataset=name, bin=b, n=len(g), listen=g.listen.mean(), partial=g.partial.mean(), drop=d.mean(), ci_lo=lo, ci_hi=hi))
DYNS = pd.DataFrame(rows); DYNS["bin"] = pd.Categorical(DYNS.bin, order); DYNS = DYNS.sort_values(["dataset", "bin"])
DYNS.to_csv(f"{CFG['out_dir']}/partial_dynamics.csv", index=False)
print(DYNS.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(9, 4))
for i, (name, g) in enumerate(DYNS.groupby("dataset")):
    x = np.arange(len(g)) + i * 0.35
    ax.bar(x, g["drop"], width=0.35, yerr=[g["drop"] - g.ci_lo, g.ci_hi - g["drop"]], capsize=3, label=name)
    ax.set_xticks(np.arange(len(g)) + 0.17); ax.set_xticklabels(g.bin.astype(str), rotation=15)
ax.axhline(0, color="k", lw=.8); ax.grid(alpha=.3); ax.legend()
ax.set(title=f"{CFG['model']}: readout lost vs listen, by phase of the model's own speech", ylabel="hit@5 drop (listen − partial)")
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/partial_dynamics.png", dpi=150); plt.show()
""")

md(r"""## 9. H9: использует ли модель эту транскрипцию? (абляция)
Для каждой реплики (условие listen) на слоях `ab_layers` и только в кадрах речи собеседника вырезаем из скрытого состояния подпространство, натянутое на выходные векторы текстовой головы (с учётом весов `out_norm`) для **слов этой реплики**. Контроль — столько же направлений слов **из других реплик**. Сравниваем с чистым прогоном:
- **проверка манипуляции:** чтение слов собеседника на слое `check_layer`;
- **что модель собирается сказать:** KL между прогнозами текстовой головы в `post_sec` тишины после реплики (чистый vs абляция) и **вероятность слов реплики** в этих прогнозах;
- **когда:** вероятность не-PAD (начать говорить).

Если вырезание *своих* направлений меняет ответ сильнее, чем *чужих*, модель использует транскрипцию.""")
code(r"""
ALPHA = lm.out_norm.alpha.detach().float().view(-1).to(LAYER_DEVS[-1])
W_OUT = lm.text_linear.weight

def basis(ids):
    W = (W_OUT[torch.tensor(sorted(ids), device=W_OUT.device)].float() * ALPHA.to(W_OUT.device)).T
    Q, _ = torch.linalg.qr(W)
    return {d: Q.to(d) for d in sorted(set(LAYER_DEVS))}

def turn_ids(t):
    ids = set()
    for w in t["words"]:
        if len(w["word"]) >= CFG["min_word_chars"]:
            ids |= word_targets(w["word"])
    return ids

def read_hits(check, t):
    # share of the turn's content words readable (top-5) at check_layer, word window as in §7
    hits = []
    for w in t["words"]:
        if len(w["word"]) < CFG["min_word_chars"]: continue
        a, b = word_window(t, w, CFG["lag_frames"])
        if b < a: continue
        tg = torch.tensor(sorted(word_targets(w["word"])), dtype=check.dtype)
        hits.append(bool(torch.isin(check[a:b + 1], tg).any()))
    return np.mean(hits) if hits else np.nan

ab_rows = []
rng_ab = random.Random(CFG["seed"])
t0 = time.time()
for name in MAIN:
    turns = RUNS[(name, "listen")]
    for ti, t in enumerate(turns):
        own = turn_ids(t)
        pool = set().union(*[turn_ids(u) for j, u in enumerate(turns) if j != ti]) - own
        if len(own) < 3 or len(pool) < len(own): continue
        ctrl = set(rng_ab.sample(sorted(pool), len(own)))
        a, b = t["user_start"], t["user_start"] + t["n_user"]
        post = list(range(b, t["T"]))
        clean_check = t["top"][CFG["check_layer"], :, :5]
        res = {}
        _, _, clean = lens_forward(t["codes"], 5, final_pos=post)
        for tag, ids in (("own", own), ("ctrl", ctrl)):
            check, fin = lens_forward(t["codes"], 5, ablate=(basis(ids), a, b), final_pos=post, check_layer=CFG["check_layer"])
            res[tag] = (check, fin)
        own_t = torch.tensor(sorted(own))
        def mass(lp): return lp.exp()[:, own_t].sum(-1).mean().item()
        def nonpad(lp): return 1 - lp.exp()[:, [0, 3]].sum(-1).mean().item()
        row = dict(dataset=name, ti=ti, k=len(own), read_clean=read_hits(clean_check, t),
                   mass_clean=mass(clean), nonpad_clean=nonpad(clean))
        for tag, (check, fin) in res.items():
            row[f"read_{tag}"] = read_hits(check, t)
            row[f"kl_{tag}"] = torch.sum(clean.exp() * (clean - fin), -1).mean().item()
            row[f"mass_{tag}"] = mass(fin)
            row[f"nonpad_{tag}"] = nonpad(fin)
        ab_rows.append(row)
    print(f"{name}: ablation done ({time.time() - t0:.0f}s)")
AB = pd.DataFrame(ab_rows); AB.to_csv(f"{CFG['out_dir']}/ablation.csv", index=False)

for name, g in AB.groupby("dataset"):
    print(f"\n=== {name} (n={len(g)} turns, ~{g.k.mean():.0f} directions) ===")
    for m, lab in (("read", "readout of heard words at check layer"), ("kl", "KL clean→ablated after the turn"),
                   ("mass", "prob. of heard words after the turn"), ("nonpad", "P(start speaking) after the turn")):
        if m == "kl":
            o, c = g.kl_own.values, g.kl_ctrl.values
            d = o - c; lo, hi = boot_ci(d)
            print(f"  {lab:45s} own {o.mean():.4f} | ctrl {c.mean():.4f} | own−ctrl {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
        else:
            cl, o, c = g[f"{m}_clean"].values, g[f"{m}_own"].values, g[f"{m}_ctrl"].values
            d = (o - cl) - (c - cl); lo, hi = boot_ci(d)
            print(f"  {lab:45s} clean {cl.mean():.4f} | own {o.mean():.4f} | ctrl {c.mean():.4f} | Δown−Δctrl {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
""")

md(r"""## 10. H5′: tuned lens на большем объёме данных
Обучение: все дикторы dev-clean и test-other плюс 70% дикторов test-clean. Проверка: оставшиеся дикторы test-clean (знакомые / **невиданные** в обучении слова) и AMI (только невиданные слова). Два ранга; 95% интервалы бутстрепом по словам.""")
code(v4_cell("class LowRankLens").split("train_name = ")[0].replace(
    "opt = torch.optim.AdamW(lens.parameters(), lr=CFG[\"tl_lr\"], weight_decay=1e-2)",
    "opt = torch.optim.AdamW(lens.parameters(), lr=CFG[\"tl_lr\"], weight_decay=CFG[\"tl_wd\"])").replace(
    "def train_lens(H, y, steps, seed, shuffle=False):\n    g = torch.Generator().manual_seed(seed)\n    if shuffle: y = y[torch.randperm(len(y), generator=g)]\n    lens = LowRankLens(H.shape[1], CFG[\"tl_rank\"]).to(LAST)",
    "def train_lens(H, y, steps, seed, rank, shuffle=False):\n    g = torch.Generator().manual_seed(seed)\n    if shuffle: y = y[torch.randperm(len(y), generator=g)]\n    lens = LowRankLens(H.shape[1], rank).to(LAST)").replace(
    "    return f.groupby([\"ti\", \"wi\"]).hit.max().mean() if len(f) else np.nan",
    "    return f.groupby([\"ti\", \"wi\"]).hit.max().values").replace(
    "    if len(ft) == 0:\n        return np.nan", "    if len(ft) == 0:\n        return np.array([])"))
TL_CELL = [c for c in cells if "def train_lens(H, y, steps, seed, rank, shuffle=False)" in c.source]
assert len(TL_CELL) == 1 and "weight_decay=CFG[\"tl_wd\"]" in TL_CELL[0].source and "return np.array([])" in TL_CELL[0].source and ".hit.max().values" in TL_CELL[0].source
code(r"""
def prep(name):
    f = frame_table(name)
    f = f[(f.target >= 0) & (f.word.str.len() >= CFG["min_word_chars"])].copy()
    f["src"] = name
    return f

FT = prep("librispeech")
spks = sorted(FT.spk.unique()); random.Random(CFG["seed"]).shuffle(spks)
train_spk = set(spks[:int(len(spks) * CFG["tl_train_frac"])])
train_sets = [("librispeech", FT[FT.spk.isin(train_spk)])] + [(n, prep(n)) for n in TL_FRAMES if n.startswith("ls_")]
test = FT[~FT.spk.isin(train_spk)]
seen = set().union(*[set(f.word) for _, f in train_sets])
tests = {"seen": test[test.word.isin(seen)], "unseen": test[~test.word.isin(seen)]}
if "ami" in TL_FRAMES:
    fa = prep("ami"); tests["ami_unseen"] = fa[~fa.word.isin(seen)]
print("train frames:", sum(len(f) for _, f in train_sets), "| word types seen in training:", len(seen),
      "| test words:", {k: v.groupby(["ti", "wi"]).ngroups for k, v in tests.items()})

def mean_ci(v):
    v = np.asarray(v, float)
    lo, hi = boot_ci(v, 300) if len(v) else (np.nan, np.nan)
    return (v.mean() if len(v) else np.nan), lo, hi

tl_rows = []
for L in CFG["tl_layers"]:
    Htr = torch.cat([TL_FRAMES[n]["h"][L][torch.as_tensor(f.fi.values)] for n, f in train_sets])
    ytr = torch.tensor(np.concatenate([f.target.values for _, f in train_sets]))
    for tag, ft in tests.items():
        src = "ami" if tag.startswith("ami") else "librispeech"
        m, lo, hi = mean_ci(word_hit(None, TL_FRAMES[src]["h"][L], ft))
        tl_rows.append(dict(layer=L, lens="logit", rank=0, test=tag, hit=m, ci_lo=lo, ci_hi=hi))
    for rank in CFG["tl_ranks"]:
        lens = train_lens(Htr, ytr, CFG["tl_steps"], CFG["seed"], rank)
        shuf = train_lens(Htr, ytr, CFG["tl_steps"], CFG["seed"], rank, shuffle=True)
        for tag, ft in tests.items():
            src = "ami" if tag.startswith("ami") else "librispeech"
            for lname, ln in (("tuned", lens), ("shuffled", shuf)):
                m, lo, hi = mean_ci(word_hit(ln, TL_FRAMES[src]["h"][L], ft))
                tl_rows.append(dict(layer=L, lens=lname, rank=rank, test=tag, hit=m, ci_lo=lo, ci_hi=hi))
        del lens, shuf; torch.cuda.empty_cache()
    print(f"layer {L} done")
TL = pd.DataFrame(tl_rows); TL.to_csv(f"{CFG['out_dir']}/tuned_lens_v5.csv", index=False)
print(TL[TL.test == "unseen"].pivot_table(index="layer", columns=["lens", "rank"], values="hit").round(3).to_string())

fig, axes = plt.subplots(1, len(tests), figsize=(6 * len(tests), 4), squeeze=False)
for ax, tag in zip(axes[0], tests):
    for (ln, rk), g in TL[TL.test == tag].groupby(["lens", "rank"]):
        lab = "logit lens" if ln == "logit" else f"{ln} r={rk}"
        ax.plot(g.layer, g.hit, "-o", ms=3, label=lab, ls=":" if ln == "shuffled" else "-")
        ax.fill_between(g.layer, g.ci_lo, g.ci_hi, alpha=.12)
    ax.set(title=f"{CFG['model']}: {tag} words", xlabel="layer", ylabel="word hit@5"); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/tuned_lens_v5.png", dpi=150); plt.show()
""")

md("## 11. Сохранение\nПапки формата v2 для v3.1 (`<набор>_<условие>`, в `turns.json` есть `own_ids`), таблицы, графики, архив.")
code(r"""
for (name, cond), turns in RUNS.items():
    if name not in MAIN: continue
    d = f"{CFG['out_dir']}/{name}_{cond}"
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(f"{d}/top5.npz", **{t["id"]: t["top"][:, :, :5].numpy().astype(np.int32) for t in turns})
    json.dump([dict(id=t["id"], spk=t["spk"], user_start=t["user_start"], n_user=t["n_user"], text=t["text"], words=t["words"],
                    own_ids=t["own_ids"], own_interval=list(t["own_interval"])) for t in turns], open(f"{d}/turns.json", "w"))
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
- **H4′**: если `excl` для talking почти так же низко, как `raw`, подавление настоящее, а не конкуренция за top-k. Главная цифра — `share_lost` в парном сравнении.
- **H7**: `talking_audio` против `talking_text` — что отключает слушание: собственный звук или собственный текст.
- **H8**: столбцы «after»: насколько быстро потеря возвращается к нулю после того, как модель замолчала.
- **H9**: `read` для own должен сильно упасть, для ctrl — нет (манипуляция сработала). Затем ключевое: `kl` и `mass` для own больше, чем для ctrl, с интервалом, не включающим ноль.
- **H5′**: tuned lens на невиданных словах выше logit lens и перемешанного контроля на слоях < 22 — значит, ранний текстовый код есть; если нет — вывод v4 подтверждён на большем объёме.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "listen_in_text_v5.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
