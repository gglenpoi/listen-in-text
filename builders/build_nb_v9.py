"""Builds listen_in_text_v9.ipynb: natural self-generated overlap, two-talker control, and the locus of the voice gate."""
import sys
import nbformat as nbf

HERE = __file__.replace("build_nb_v9.py", "")
def load_cells(fname):
    src = open(HERE + fname).read().rsplit("nb = nbf.v4.new_notebook()", 1)[0]
    ns = {"__file__": HERE + fname}
    exec(src, ns)
    return [c.source for c in ns["cells"] if c.cell_type == "code"]
V5, V7 = load_cells("build_nb_v5.py"), load_cells("build_nb_v7.py")
def pick(cells_, marker):
    hits = [s for s in cells_ if marker in s]
    assert len(hits) == 1, marker
    return hits[0]

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Do Full-Duplex Speech Models Listen in Text? — v9: естественное перекрытие и механизм шлюза

Три вопроса, которые зададут рецензенты:

1. **Естественное перекрытие (§5).** До сих пор в собственный канал подставлялась чужая реплика. Здесь модель **сама генерирует ответ** (свой голос + свой inner monologue), и на её речь накладывается реплика собеседника. Сохраняется ли шлюз — в чтении и в поведении?
2. **Своя речь или просто два голоса? (§4)** Вторая речь подмешивается **в канал собеседника** (0 и −10 дБ). Если чтение падает гораздо меньше, чем при речи в собственном канале, шлюз специфичен для собственного канала, а не просто «коктейль-эффект».
3. **Где работает шлюз (§7).** Подстановка вкладов слоёв (внимание / MLP / весь слой) из прогона «слушает» в прогон «говорит» по блокам слоёв: какие блоки восстанавливают чтение собеседника.

Плюс поведенческий тест (§6) для всех новых условий, со слушающими по содержательным словам (без служебных).

Kaggle: GPU T4×2, Internet On, `HF_TOKEN`. Время: ~50–60 мин на модель (генерация — основная часть).
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
    n_turns        = 48,                          # readout block (20 s turns)
    turn_sec       = 20.0,
    gap_sec        = 0.3,
    mix_db         = [0, -10],                    # second talker mixed into the USER channel
    n_beh          = 40,                          # turns for the behavioural test (standard block)
    n_nat          = 20,                          # natural-overlap turns per dataset
    nat_turn_sec   = 10.0,                        # user turn length in the natural block
    trig_sec       = 6.0,                         # the user utterance the model answers
    nat_lead_sec   = 0.5,                         # user starts this long after the model's reply window opens
    temp           = 0.8, top_k = 250,            # audio sampling (Moshi defaults)
    temp_text      = 0.7, top_k_text = 25,
    gen_tries      = 3,
    min_speech_frac= 0.3,
    min_gen_words  = 3,
    n_patch        = 12,
    patch_blocks   = [[1, 8], [9, 16], [17, 20], [21, 24], [25, 28], [1, 28]],
    silence_frames = 6,
    post_sec       = 1.5,
    resp_gap_sec   = 0.5,
    topk           = 20,
    lag_frames     = 4,
    n_control      = 5,
    min_word_chars = 4,
    n_boot         = 1000,
    aligner        = "mms",
    whisper_model  = "small.en",
    tl_extra       = [],
    tl_extra_turns = 0,
    dtype          = "auto",
    seed           = 0,
)
PRESETS = {
    "personaplex": dict(model_repo="nvidia/personaplex-7b-v1",
                        text_prompt="You enjoy having a good conversation.", prefix_user="sine"),
    "moshi":       dict(model_repo="kyutai/moshiko-pytorch-bf16", text_prompt="", prefix_user="silence"),
}
CFG.update(PRESETS[CFG["model"]])
CFG["out_dir"] = f"{BASE}/{CFG['model']}_v9"
os.makedirs(CFG["out_dir"], exist_ok=True)
print(CFG["model_repo"], "->", CFG["out_dir"])
""")

md("## 2. Установка, модель (вместе с depformer для генерации), данные, границы слов")
code(pick(V5, "import sys, subprocess"))
code(pick(V5, "import json, math, random, re, tarfile"))
code(pick(V5, "from huggingface_hub import hf_hub_download, list_repo_files"))
code(r"""
SPECIAL_TEXT = [0, 1, 2, 3]   # EPAD, BOS, EOS, PAD

def pick_dtype():
    if CFG["dtype"] != "auto":
        return {"fp16": torch.float16, "bf16": torch.bfloat16}[CFG["dtype"]]
    major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    return torch.bfloat16 if major >= 8 else torch.float16

def layer_devices(n_layers):
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0: return ["cpu"] * n_layers
    if n_gpu == 1: return ["cuda:0"] * n_layers
    half = n_layers // 2
    return ["cuda:0"] * half + ["cuda:1"] * (n_layers - half)

def load_lm_full(path, lm_kwargs, dtype):
    # like v5-v8, but the depformer is kept (needed to generate the model's own speech)
    with torch.device("meta"):
        lm = LMModel(device="meta", dtype=dtype, **dict(lm_kwargs))
    devs = layer_devices(len(lm.transformer.layers))
    ref = lm.state_dict()
    def target_device(key):
        m = re.match(r"transformer\.layers\.(\d+)\.", key)
        if m: return devs[int(m.group(1))]
        if key.startswith(("emb.", "text_emb.")): return devs[0]
        return devs[-1]
    def cast(key, t):
        return t.to(device=target_device(key), dtype=torch.float32 if key.endswith(".alpha") else dtype)
    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key not in ref: continue
            t = f.get_tensor(key)
            if t.shape != ref[key].shape:          # depformer trained with fewer steps: repeat along dim 0
                reps = math.ceil(ref[key].shape[0] / t.shape[0])
                t = torch.cat([t] * reps, 0)[:ref[key].shape[0]]
                assert t.shape == ref[key].shape, (key, t.shape, ref[key].shape)
            sd[key] = cast(key, t)
    for key in ref:                                # missing per-codebook modules 8..15: copy 0..7
        if key in sd: continue
        m = re.search(r"(gating|linears|depformer_in|depformer_emb)\.(\d+)\.", key)
        if m and int(m.group(2)) >= 8:
            src = key[:m.start()] + f"{m.group(1)}.{int(m.group(2)) - 8}." + key[m.end():]
            if src in sd:
                sd[key] = sd[src].clone(); continue
        assert not key.startswith(("transformer.", "text_linear", "emb.", "text_emb.", "out_norm")), f"missing {key}"
        sd[key] = torch.zeros(ref[key].shape, device=target_device(key), dtype=dtype)
        print("!! zero-filled", key)
    lm.load_state_dict(sd, strict=False, assign=True)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm, devs

DTYPE = pick_dtype()
lm, LAYER_DEVS = load_lm_full(moshi_path, dict(loaders._lm_kwargs, dep_q=16), DTYPE)
N_LAYERS = len(lm.transformer.layers) + 1
mimi = loaders.get_mimi(mimi_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
FRAME = int(mimi.sample_rate / mimi.frame_rate); FPS = mimi.frame_rate
print("dtype", DTYPE, "| devices", sorted(set(LAYER_DEVS)))
""")
code(pick(V5, "def librispeech_turns"))
code(pick(V5, "def whisper_align(turns)"))

md("## 3. Последовательности, прямой проход с подстановкой, генерация")
code(r"""
from moshi.utils.sampling import sample_token

@torch.no_grad()
def mimi_codes(wav):
    pad = (-len(wav)) % FRAME
    x = torch.from_numpy(np.pad(wav, (0, pad))).to(next(mimi.parameters()).device).view(1, 1, -1)
    return mimi.encode(x)[0].cpu()

def rms(x): return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-9)

def fit(src, n, fill):
    out = fill.repeat(1, n).clone()
    L = min(n, src.shape[1]); out[:, :L] = src[:, :L]
    return out

PAD = 3
SIL = torch.as_tensor(SILENCE_TOKENS).view(8, 1)
FILL = torch.as_tensor(SINE_TOKENS).view(8, 1) if CFG["prefix_user"] == "sine" else SIL
PROMPT_IDS = sp.encode(f"<system> {CFG['text_prompt'].strip()} <system>") if CFG["text_prompt"] else []
N_POST = int(round(CFG["post_sec"] * FPS))
GAP = int(round(CFG["resp_gap_sec"] * FPS))
LEAD = int(round(CFG["nat_lead_sec"] * FPS))

def seg(n, text=None, own=None, user=None):
    t = torch.full((1, n), PAD, dtype=torch.long) if text is None else torch.as_tensor(text, dtype=torch.long).view(1, n)
    return torch.cat([t, SIL.repeat(1, n) if own is None else own.long(), SIL.repeat(1, n) if user is None else user.long()], 0)

def prefix():
    S = CFG["silence_frames"]
    text = [PAD] * S + list(PROMPT_IDS) + [PAD] * S
    return seg(len(text), text=text, user=FILL.repeat(1, len(text)))

def standard_seq(user, own=None):
    P = prefix(); n = user.shape[1]
    return torch.cat([P, seg(n, own=own, user=user), seg(N_POST)], 1), P.shape[1]

def natural_seq(trig, user, own_audio=None, own_text=None):
    # [prefix][trigger, model silent][reply window: LEAD frames + user turn][post]
    P = prefix(); nT, nU = trig.shape[1], user.shape[1]; W = LEAD + nU
    win_user = torch.cat([SIL.repeat(1, LEAD), user], 1)
    own = None if own_audio is None else own_audio[:, :W]
    text = None if own_text is None else own_text[:W]
    codes = torch.cat([P, seg(nT, user=trig), seg(W, text=text, own=own, user=win_user), seg(N_POST)], 1)
    return codes, P.shape[1] + nT + LEAD

@torch.no_grad()
def run(codes, read_layers=(), topk=20, final_pos=None, record=None, patch=None):
    # record=(a,b): return per-layer attention / MLP updates at frames [a,b)
    # patch=(deltas, a, b, layers, comp): replace those updates by the recorded ones
    first, last = LAYER_DEVS[0], LAYER_DEVS[-1]
    c = codes.unsqueeze(0).to(first)
    initial = lm._get_initial_token().to(first)
    seq = torch.cat([initial, _delay_sequence(lm.delays, c, initial)], 2)[:, :, :-1]
    x = lm.embed_codes(seq)
    special = torch.tensor(SPECIAL_TEXT, device=last)
    tops, rec = {}, {}
    def lens(h):
        logits = lm.text_linear(lm.out_norm(h.to(last))).float()[0]
        logits[:, special] = -float("inf")
        return logits.topk(topk, -1).indices.int().cpu()
    if 0 in read_layers: tops[0] = lens(x)
    for i, layer in enumerate(lm.transformer.layers):
        l = i + 1
        x = x.to(LAYER_DEVS[i])
        h1 = layer._sa_block(x)
        if record is not None:
            a, b = record; dA = (h1[0, a:b] - x[0, a:b]).clone()
        if patch is not None and l in patch[3] and patch[4] in ("attn", "layer"):
            d, a, b = patch[0], patch[1], patch[2]
            h1[0, a:b] = x[0, a:b] + d[l][0].to(h1.device)
        x2 = layer._ff_block(h1)
        if record is not None:
            rec[l] = (dA, (x2[0, a:b] - h1[0, a:b]).clone())
        if patch is not None and l in patch[3] and patch[4] in ("mlp", "layer"):
            d, a, b = patch[0], patch[1], patch[2]
            x2[0, a:b] = h1[0, a:b] + d[l][1].to(x2.device)
        x = x2
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"non-finite activations after layer {l}")
        if l in read_layers: tops[l] = lens(x)
    final = None
    if final_pos is not None:
        final = torch.log_softmax(lm.text_linear(lm.out_norm(x.to(last))).float()[0, final_pos], -1).cpu()
    return tops, final, rec

def stack_tops(tops):
    return torch.stack([tops[l] for l in sorted(tops)])

@torch.no_grad()
def generate(base, g0, G, seed):
    # samples the model's own text + 8 audio codebooks for frames [g0, g0+G); the user stream stays as given
    torch.manual_seed(seed)
    codes = base.clone(); D = lm.delays
    first, last = LAYER_DEVS[0], LAYER_DEVS[-1]
    initial = lm._get_initial_token().to(first)
    for t in range(g0, g0 + G + max(D)):
        c = codes[:, :t + 1].unsqueeze(0).to(first)
        seq = torch.cat([initial, _delay_sequence(D, c, initial)], 2)[:, :, :t + 1]
        x = lm.embed_codes(seq)
        for i, layer in enumerate(lm.transformer.layers):
            x = layer(x.to(LAYER_DEVS[i]))
        h = lm.out_norm(x[:, -1:].to(last))
        text = sample_token(lm.text_linear(h).float(), True, CFG["temp_text"], CFG["top_k_text"]).view(1, 1)
        prev, ins, toks = text, [], []
        for k in range(8):
            emb = lm.depformer_text_emb(prev) if k == 0 else lm.depformer_emb[k - 1](prev)
            li = 0
            if lm.depformer_multi_linear:
                sch = getattr(lm, "depformer_weights_per_step_schedule", None)
                li = sch[k] if sch is not None else k
            ins.append(lm.depformer_in[li](h) + emb)
            out = lm.depformer(torch.cat(ins, 1))
            prev = sample_token(lm.linears[k](out[:, -1:]).float(), True, CFG["temp"], CFG["top_k"]).view(1, 1)
            toks.append(prev)
        for k, tok in enumerate([text] + toks):
            tt = t - D[k]
            if g0 <= tt < g0 + G:
                codes[k, tt] = int(tok)
    return codes

def word_window(t, w, lag):
    f0 = int(math.floor(w["start"] * FPS)); f1 = int(math.ceil(w["end"] * FPS))
    return t["user_start"] + f0 + 1, min(t["user_start"] + f1 + 1 + lag, t["T"] - 1)

def crop(t, sec):
    n = int(sec * 24000)
    return dict(t, id=t["id"] + f"-{sec:g}s", wav=t["wav"][:n].copy(), words=[w for w in t["words"] if w["end"] <= sec - 0.05])

STOP = set('''a about above after again against all am an and any are as at be because been before being below between both but by can
did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is
it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what
when where which while who whom why will with you your yours yourself yourselves yeah okay well also like know think going really would
could shall might must right mean thing things gonna um uh hmm'''.split())

def content_ids(words):
    ids = set()
    for w in words:
        if len(w["word"]) >= CFG["min_word_chars"] and w["word"] not in STOP:
            enc = sp.encode(w["word"])
            if enc and enc[0] not in SPECIAL_TEXT:
                ids.add(enc[0])
    return ids
print("prefix frames:", prefix().shape[1], "| post", N_POST, "| gap", GAP, "| lead", LEAD)
""")

md("## 4. Блок чтения: слушает / речь в своём канале / второй голос в канале собеседника")
code(r"""
COND_STD = ["listen", "talking_audio"] + [f"mix{d}" for d in CFG["mix_db"]]
ALL_LAYERS = tuple(range(N_LAYERS))

def mixed(t, other, db):
    o = np.zeros_like(t["wav"]); L = min(len(o), len(other["wav"])); o[:L] = other["wav"][:L]
    o = o * (rms(t["wav"]) / rms(o[:L])) * 10 ** (db / 20)
    return (t["wav"] + o).astype(np.float32), o.astype(np.float32)

for name in MAIN:
    turns = DATA[name]
    for i, t in enumerate(turns):
        other = turns[(i + 1) % len(turns)]
        t["codes_user"] = mimi_codes(t["wav"])
        for d in CFG["mix_db"]:
            m, o = mixed(t, other, d)
            t[f"codes_mix{d}"], t[f"codes_mixother{d}"] = mimi_codes(m), mimi_codes(o)

def std_codes(t, other, cond, user_on=True):
    n = t["codes_user"].shape[1]
    if cond.startswith("mix"):
        user = t[f"codes_{cond}"] if user_on else t[f"codes_mixother{cond[3:]}"]
        return standard_seq(fit(user, n, SIL))
    user = t["codes_user"] if user_on else SIL.repeat(1, n)
    own = fit(other["codes_user"], n, SIL) if cond == "talking_audio" else None
    return standard_seq(user, own)

RUNS = {}
t0 = time.time()
for name in MAIN:
    turns = DATA[name]
    for cond in COND_STD:
        out = []
        for ti, t in enumerate(turns):
            codes, start = std_codes(t, turns[(ti + 1) % len(turns)], cond)
            tops, _, _ = run(codes, ALL_LAYERS, CFG["topk"])
            out.append(dict(id=t["id"], user_start=start, n_user=t["codes_user"].shape[1], T=codes.shape[1], words=t["words"],
                            text=" ".join(w["word"] for w in t["words"]), top=stack_tops(tops), codes=codes, own_ids=set()))
        RUNS[(name, cond)] = out
        print(f"{name}/{cond}: {len(out)} turns ({time.time() - t0:.0f}s)")
""")
code(pick(V7, "def score_all(name, cond, which").split("SCORES = {")[0] + r'''

def score_excl(name, cond, seed=0):
    # like score_all, but the model's own text tokens of that turn are removed from the ranking (natural block)
    turns = RUNS[(name, cond)]; rng = random.Random(seed); rows = []
    for ti, t in enumerate(turns):
        others = [u for j, u in enumerate(turns) if j != ti]
        ex = t["own_ids"]
        for wi, w in enumerate(t["words"]):
            targets = word_targets(w["word"]) - ex
            a, b = word_window(t, w, CFG["lag_frames"])
            r = best_rank(t["top"], a, b, targets, excl=ex)
            if r is None: continue
            c = []
            for _ in range(CFG["n_control"]):
                u = rng.choice(others)
                if w["word"] in u["text"].split(): continue
                n = b - a; lo, hi = u["user_start"] + 1, u["user_start"] + u["n_user"] - n
                if hi <= lo: continue
                s = rng.randint(lo, hi)
                c.append(best_rank(u["top"], s, s + n, targets, excl=ex))
            if c:
                rows.append(dict(ti=ti, wi=wi, word=w["word"], n_chars=len(w["word"]), raw=r, c_raw=np.stack(c)))
    return rows

def paired_drop(name, base_cond, cond, scores, tag):
    base = {(r["ti"], r["wi"]): r for r in scores[(name, base_cond)] if r["n_chars"] >= CFG["min_word_chars"]}
    pairs = [(base[(r["ti"], r["wi"])], r) for r in scores[(name, cond)] if (r["ti"], r["wi"]) in base]
    out = []
    for L in range(N_LAYERS):
        lis = np.array([b_["raw"][L] < 5 for b_, r in pairs], float)
        d = lis - np.array([r["raw"][L] < 5 for b_, r in pairs], float)
        lo, hi = boot_ci(d)
        out.append(dict(dataset=name, base=base_cond, cond=cond, metric=tag, layer=L, listen=lis.mean(), drop=d.mean(),
                        ci_lo=lo, ci_hi=hi, n_words=len(pairs)))
    return out

SCORES = {(n, c): score_all(n, c, "user", CFG["seed"]) for n in MAIN for c in COND_STD}
CURVES = pd.concat([excess_curve(r, CFG["min_word_chars"]).assign(dataset=n, cond=c) for (n, c), r in SCORES.items()], ignore_index=True)
BEST = {n: int(CURVES[(CURVES.dataset == n) & (CURVES.cond == "listen")].sort_values("excess").iloc[-1].layer) for n in MAIN}
PAIRED = pd.DataFrame([row for n in MAIN for c in COND_STD[1:] for row in paired_drop(n, "listen", c, SCORES, "top5")])
print("best listen layer:", BEST)
for n in MAIN:
    p = PAIRED[(PAIRED.dataset == n) & (PAIRED.layer == BEST[n])].set_index("cond")
    print(f"\n{n}, layer {BEST[n]}: share of readout lost")
    print(p.assign(share=p["drop"] / p.listen, lo=p.ci_lo / p.listen, hi=p.ci_hi / p.listen)[["listen", "share", "lo", "hi", "n_words"]].round(3).to_string())
''')

md(r"""## 5. Естественное перекрытие: модель отвечает своим голосом, собеседник вступает поверх

Для каждой реплики-цели `U` (10 с) берётся другая реплика-«вопрос» (6 с). Модель слушает вопрос и **свободно генерирует ответ** (текст + 8 кодбуков аудио, сэмплирование как в Moshi). Ответы, где модель почти не говорит, перегенерируются (до 3 попыток). Затем через 0,5 с после начала ответа на канал собеседника подаётся `U`. Условия (одинаковый контекст):
- `nat_listen` — собственный канал молчит;
- `nat_self` — собственный ответ модели: её голос **и её текст** (полностью естественное состояние);
- `nat_self_audio` — только её голос, текст = PAD;
- `nat_other` — голос другого человека (как в v7), текст = PAD.
""")
code(r"""
NAT, GEN_ROWS = {}, []
t0 = time.time()
for name in MAIN:
    turns = DATA[name]; N = len(turns); nat = []
    for i in range(min(CFG["n_nat"], N // 2)):
        U = crop(turns[i], CFG["nat_turn_sec"]); trig = crop(turns[N - 1 - i], CFG["trig_sec"])
        U["codes_user"] = mimi_codes(U["wav"]); trig["codes_user"] = mimi_codes(trig["wav"])
        other = crop(turns[(i + 1) % N], CFG["nat_turn_sec"]); other["codes_user"] = mimi_codes(other["wav"])
        W = LEAD + U["codes_user"].shape[1]
        base, ustart = natural_seq(trig["codes_user"], SIL.repeat(1, U["codes_user"].shape[1]))
        g0 = ustart - LEAD
        best = None
        for k in range(CFG["gen_tries"]):
            g = generate(base, g0, W, CFG["seed"] + 1000 * i + k)
            txt, aud = g[0, g0:g0 + W].clone(), g[1:9, g0:g0 + W].clone()
            nwords = int(((txt != 0) & (txt != 3)).sum())
            try:
                wav = mimi.decode(aud.unsqueeze(0).to(next(mimi.parameters()).device))[0, 0].float().cpu().numpy()
                fr = wav[:len(wav) // FRAME * FRAME].reshape(-1, FRAME)
                speech = float((np.sqrt((fr[LEAD:] ** 2).mean(1)) > 0.01).mean())
            except Exception:
                wav, speech = None, float("nan")
            cand = dict(txt=txt, aud=aud, wav=wav, nwords=nwords, speech=speech, tries=k + 1)
            if best is None or (nwords, np.nan_to_num(speech)) > (best["nwords"], np.nan_to_num(best["speech"])):
                best = cand
            if nwords >= CFG["min_gen_words"] and not (speech < CFG["min_speech_frac"]):
                break
        ok = best["nwords"] >= CFG["min_gen_words"] and not (best["speech"] < CFG["min_speech_frac"])
        reply = sp.decode([int(x) for x in best["txt"] if int(x) not in SPECIAL_TEXT]) if hasattr(sp, "decode") else ""
        GEN_ROWS.append(dict(dataset=name, i=i, ok=ok, n_text_tokens=best["nwords"], speech_frac=best["speech"], tries=best["tries"], reply=reply))
        if best["wav"] is not None and i < 4:
            sf.write(f"{CFG['out_dir']}/gen_{name}_{i}.wav", best["wav"], 24000)
        if ok:
            nat.append(dict(U=U, trig=trig, other=other, txt=best["txt"], aud=best["aud"]))
        print(f"{name} #{i}: ok={ok} words={best['nwords']} speech={best['speech']:.2f} tries={best['tries']} | {reply[:70]!r} ({time.time() - t0:.0f}s)")
    NAT[name] = nat
GEN = pd.DataFrame(GEN_ROWS); GEN.to_csv(f"{CFG['out_dir']}/generation_v9.csv", index=False)
print(GEN.groupby("dataset")[["ok", "n_text_tokens", "speech_frac"]].mean().round(2))
""")
code(r"""
COND_NAT = ["nat_listen", "nat_self", "nat_self_audio", "nat_other"]
def nat_codes(item, cond, user_on=True):
    U = item["U"]; n = U["codes_user"].shape[1]; W = LEAD + n
    user = U["codes_user"] if user_on else SIL.repeat(1, n)
    if cond == "nat_listen": return natural_seq(item["trig"]["codes_user"], user)
    if cond == "nat_self": return natural_seq(item["trig"]["codes_user"], user, item["aud"], item["txt"])
    if cond == "nat_self_audio": return natural_seq(item["trig"]["codes_user"], user, item["aud"])
    return natural_seq(item["trig"]["codes_user"], user, fit(item["other"]["codes_user"], W, SIL))

for name in MAIN:
    for cond in COND_NAT:
        out = []
        for item in NAT[name]:
            codes, start = nat_codes(item, cond)
            tops, _, _ = run(codes, ALL_LAYERS, CFG["topk"])
            own_ids = {int(x) for x in item["txt"] if int(x) not in SPECIAL_TEXT}
            U = item["U"]
            out.append(dict(id=U["id"], user_start=start, n_user=U["codes_user"].shape[1], T=codes.shape[1], words=U["words"],
                            text=" ".join(w["word"] for w in U["words"]), top=stack_tops(tops), codes=codes, own_ids=own_ids))
        RUNS[(name, cond)] = out
for n in MAIN:
    for c in COND_NAT:
        SCORES[(n, c)] = score_all(n, c, "user", CFG["seed"])
        SCORES[(n, c + "|excl")] = score_excl(n, c, CFG["seed"])
CURVES = pd.concat([CURVES] + [excess_curve(SCORES[(n, c)], CFG["min_word_chars"]).assign(dataset=n, cond=c)
                               for n in MAIN for c in COND_NAT if len(SCORES[(n, c)]) >= 10], ignore_index=True)
rows = []
for n in MAIN:
    if len(SCORES[(n, "nat_listen")]) < 10: continue
    for c in COND_NAT[1:]:
        rows += paired_drop(n, "nat_listen", c, SCORES, "top5")
        rows += paired_drop(n, "nat_listen|excl", c + "|excl", SCORES, "top5_excl_own")
PAIRED = pd.concat([PAIRED, pd.DataFrame(rows)], ignore_index=True)
PAIRED["cond"] = PAIRED["cond"].str.replace("|excl", "", regex=False)
CURVES.to_csv(f"{CFG['out_dir']}/hit_curves_v9.csv", index=False); PAIRED.to_csv(f"{CFG['out_dir']}/paired_drop_v9.csv", index=False)
for n in MAIN:
    p = PAIRED[(PAIRED.dataset == n) & (PAIRED.layer == BEST[n])].set_index(["cond", "metric"])
    print(f"\n{n}, layer {BEST[n]}: share of readout lost (natural block vs nat_listen; standard vs listen)")
    print(p.assign(share=p["drop"] / p.listen, lo=p.ci_lo / p.listen, hi=p.ci_hi / p.listen)[["listen", "share", "lo", "hi", "n_words"]].round(3).to_string())
""")

md(r"""## 6. Поведение: влияние собеседника на ответ (KL и содержательные слова)
`retention = KL(собеседник есть ‖ собеседник заглушён)` в условии / то же при слушании. Для `mix*` «заглушён» = в канале собеседника остаётся только второй голос. Вероятность услышанных слов считается только по **содержательным** словам (без служебных).""")
code(r"""
def response_pair(codes_on, codes_off, start, n_user, heard):
    out = {}
    for tag, codes in (("on", codes_on), ("off", codes_off)):
        c = codes.clone(); e = start + n_user + GAP
        c[0, e] = 0                                         # forced EPAD: the reply starts on the next frame
        _, lp, _ = run(c, (), final_pos=[e + 1])
        out[tag] = lp[-1]
    on, off = out["on"], out["off"]
    h = torch.tensor(sorted(heard)) if heard else None
    return dict(kl=float(torch.sum(on.exp() * (on - off))),
                logp_on=float(torch.logsumexp(on[h], 0)) if h is not None else np.nan,
                logp_off=float(torch.logsumexp(off[h], 0)) if h is not None else np.nan,
                flip=float(on.argmax() != off.argmax()))

BEH = []
t0 = time.time()
for name in MAIN:
    turns = DATA[name]
    for ti, t in enumerate(turns[:CFG["n_beh"]]):
        other = turns[(ti + 1) % len(turns)]; heard = content_ids(t["words"])
        for cond in COND_STD:
            on, start = std_codes(t, other, cond, True); off, _ = std_codes(t, other, cond, False)
            BEH.append(dict(dataset=name, block="standard", ti=ti, cond=cond, **response_pair(on, off, start, t["codes_user"].shape[1], heard)))
    for ti, item in enumerate(NAT[name]):
        heard = content_ids(item["U"]["words"])
        for cond in COND_NAT:
            on, start = nat_codes(item, cond, True); off, _ = nat_codes(item, cond, False)
            BEH.append(dict(dataset=name, block="natural", ti=ti, cond=cond, **response_pair(on, off, start, item["U"]["codes_user"].shape[1], heard)))
    print(f"{name}: behaviour done ({time.time() - t0:.0f}s)")
B = pd.DataFrame(BEH); B["dlogp"] = B.logp_on - B.logp_off
B.to_csv(f"{CFG['out_dir']}/behaviour_rows_v9.csv", index=False)

rng_b = np.random.default_rng(CFG["seed"])
res = []
for (name, block), g in B.groupby(["dataset", "block"]):
    base_c = "listen" if block == "standard" else "nat_listen"
    base = g[g.cond == base_c].set_index("ti")
    if len(base) < 3: continue
    for cond, h in g.groupby("cond"):
        h = h.set_index("ti").loc[base.index]
        for metric in ("kl", "flip", "dlogp"):
            num, den = h[metric].values, base[metric].values
            ok = ~(np.isnan(num) | np.isnan(den)); num, den = num[ok], den[ok]
            idx = rng_b.integers(0, len(num), (CFG["n_boot"], len(num)))
            ratio = num[idx].mean(1) / den[idx].mean(1)
            dd = num[idx].mean(1) - den[idx].mean(1)
            res.append(dict(dataset=name, block=block, cond=cond, metric=metric, value=num.mean(), listen=den.mean(),
                            retention=num.mean() / den.mean(), lo=np.percentile(ratio, 2.5), hi=np.percentile(ratio, 97.5),
                            diff=num.mean() - den.mean(), diff_lo=np.percentile(dd, 2.5), diff_hi=np.percentile(dd, 97.5), n=len(num)))
BRES = pd.DataFrame(res); BRES.to_csv(f"{CFG['out_dir']}/behaviour_v9.csv", index=False)
print(BRES[BRES.metric.isin(["kl", "dlogp"])].pivot_table(index=["dataset", "block", "cond"], columns="metric",
      values=["value", "retention", "lo", "hi"]).round(3).to_string())
""")

md(r"""## 7. Где работает шлюз: подстановка вкладов слоёв из «слушает» в «говорит»
Для блока слоёв и компонента (внимание / MLP / весь слой) вклады в остаточный поток на кадрах реплики собеседника берутся из прогона `listen` и подставляются в прогон с речью в собственном канале. `restored = (hit_patch − hit_talk) / (hit_listen − hit_talk)` на лучшем слое чтения: 1 — чтение восстановлено полностью.""")
code(r"""
PATCH_PAIRS = [("listen", "talking_audio"), ("nat_listen", "nat_self")]
COMPS = ["layer", "attn", "mlp"]
prow = []
t0 = time.time()
for name in MAIN:
    Lr = BEST[name]
    for base_c, cond in PATCH_PAIRS:
        R0, R1 = RUNS[(name, base_c)], RUNS[(name, cond)]
        for ti in range(min(CFG["n_patch"], len(R0))):
            t0_, t1_ = R0[ti], R1[ti]
            a = t1_["user_start"]; b = min(a + t1_["n_user"] + CFG["lag_frames"] + 2, t1_["T"])
            _, _, deltas = run(t0_["codes"], (), record=(a, b))
            wins = []
            for wi, w in enumerate(t1_["words"]):
                tg = word_targets(w["word"])
                if len(w["word"]) < CFG["min_word_chars"] or not tg: continue
                wa, wb = word_window(t1_, w, CFG["lag_frames"])
                wins.append((wi, wa, wb, tg))
            def hits(top):
                return [float(best_rank(top, wa, wb, tg)[0] < 5) for _, wa, wb, tg in wins]
            h_lis = hits(t0_["top"][Lr:Lr + 1]); h_talk = hits(t1_["top"][Lr:Lr + 1])
            for blk in CFG["patch_blocks"]:
                layers = set(range(blk[0], min(blk[1], Lr) + 1))
                for comp in COMPS:
                    tops, _, _ = run(t1_["codes"], (Lr,), CFG["topk"], patch=(deltas, a, b, layers, comp))
                    hp = hits(tops[Lr][None])
                    for (wi, *_), x0, x1, xp in zip(wins, h_lis, h_talk, hp):
                        prow.append(dict(dataset=name, cond=cond, ti=ti, wi=wi, block=f"{blk[0]}-{blk[1]}", comp=comp,
                                         hit_listen=x0, hit_talk=x1, hit_patch=xp))
            del deltas
        print(f"{name}/{cond}: patching done ({time.time() - t0:.0f}s)")
PR = pd.DataFrame(prow); PR.to_csv(f"{CFG['out_dir']}/patching_rows_v9.csv", index=False)
rng_p = np.random.default_rng(CFG["seed"]); summ = []
for (name, cond, blk, comp), g in PR.groupby(["dataset", "cond", "block", "comp"]):
    v = g[["hit_listen", "hit_talk", "hit_patch"]].values
    idx = rng_p.integers(0, len(v), (CFG["n_boot"], len(v)))
    m = v[idx].mean(1); gap_ = m[:, 0] - m[:, 1]
    rr = (m[:, 2] - m[:, 1]) / np.where(np.abs(gap_) < 1e-9, np.nan, gap_)
    mm = v.mean(0)
    summ.append(dict(dataset=name, cond=cond, block=blk, comp=comp, hit_listen=mm[0], hit_talk=mm[1], hit_patch=mm[2],
                     restored=(mm[2] - mm[1]) / (mm[0] - mm[1]) if mm[0] != mm[1] else np.nan,
                     lo=np.nanpercentile(rr, 2.5), hi=np.nanpercentile(rr, 97.5), n_words=len(v)))
PSUM = pd.DataFrame(summ); PSUM.to_csv(f"{CFG['out_dir']}/patching_v9.csv", index=False)
print(PSUM.pivot_table(index=["dataset", "cond", "block"], columns="comp", values="restored").round(2).to_string())
""")

md("## 8. Сводка и графики")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(20, 4.6))
for k, name in enumerate(MAIN):
    p = PAIRED[(PAIRED.dataset == name) & (PAIRED.layer == BEST[name]) & (PAIRED.metric == "top5")].set_index("cond")
    order = [c for c in COND_STD[1:] + COND_NAT[1:] if c in p.index]
    s = p.loc[order, "drop"] / p.loc[order, "listen"]
    lo = p.loc[order, "ci_lo"] / p.loc[order, "listen"]; hi = p.loc[order, "ci_hi"] / p.loc[order, "listen"]
    x = np.arange(len(order)) + (k - 0.5) * 0.38
    axes[0].bar(x, s, 0.38, yerr=[s - lo, hi - s], capsize=3, label=name)
    axes[0].set_xticks(range(len(order))); axes[0].set_xticklabels(order, rotation=25)
    b = BRES[(BRES.dataset == name) & (BRES.metric == "kl") & ~BRES.cond.isin(["listen", "nat_listen"])].set_index("cond")
    ob = [c for c in COND_STD[1:] + COND_NAT[1:] if c in b.index]
    axes[1].bar(np.arange(len(ob)) + (k - 0.5) * 0.38, b.loc[ob, "retention"], 0.38,
                yerr=[b.loc[ob, "retention"] - b.loc[ob, "lo"], b.loc[ob, "hi"] - b.loc[ob, "retention"]], capsize=3, label=name)
    axes[1].set_xticks(range(len(ob))); axes[1].set_xticklabels(ob, rotation=25)
    q = PSUM[(PSUM.dataset == name) & (PSUM.cond == "talking_audio") & (PSUM.comp == "layer")].set_index("block").loc[[f"{a}-{b_}" for a, b_ in CFG["patch_blocks"]]]
    axes[2].errorbar(np.arange(len(q)) + (k - 0.5) * 0.15, q.restored, yerr=[q.restored - q.lo, q.hi - q.restored], fmt="o", capsize=3, label=name)
    axes[2].set_xticks(range(len(q))); axes[2].set_xticklabels(q.index)
axes[0].set(title=f"{CFG['model']}: share of user readout lost", ylabel="drop / listen")
axes[1].axhline(1, color="k", ls=":"); axes[1].set(title="user influence on reply (KL retention)", ylabel="relative to listening")
axes[2].axhline(0, color="k", lw=.8); axes[2].axhline(1, color="k", ls=":"); axes[2].set(title="readout restored by patching layer block (speech in own channel)", ylabel="restored")
for a in axes: a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/summary_v9.png", dpi=150); plt.show()

NUM = {"best_layer": BEST, "generation": GEN.groupby("dataset")[["ok", "n_text_tokens", "speech_frac"]].mean().round(3).to_dict()}
for n in MAIN:
    p = PAIRED[(PAIRED.dataset == n) & (PAIRED.layer == BEST[n])]
    for _, r in p.iterrows():
        dv = lambda x: float(np.float64(x) / np.float64(r.listen)) if r.listen else float("nan")
        NUM[f"{n}|share_lost|{r.cond}|{r.metric}"] = [round(dv(r["drop"]), 3), round(dv(r.ci_lo), 3), round(dv(r.ci_hi), 3), int(r.n_words)]
for _, r in BRES.iterrows():
    NUM[f"{r.dataset}|beh|{r.cond}|{r.metric}"] = [round(r.retention, 3), round(r.lo, 3), round(r.hi, 3), round(r.value, 3), int(r.n)]
for _, r in PSUM.iterrows():
    NUM[f"{r.dataset}|patch|{r.cond}|{r.block}|{r.comp}"] = [round(r.restored, 3), round(r.lo, 3), round(r.hi, 3)]
json.dump(NUM, open(f"{CFG['out_dir']}/paper_numbers_v9.json", "w"), indent=1, default=float)
print(len(NUM), "numbers saved")
""")

md("## 9. Сохранение")
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
- **§4 `mix0` / `mix-10`**: если потеря чтения заметно меньше, чем у `talking_audio`, шлюз специфичен для собственного канала (не «коктейль-эффект»).
- **§5 `nat_self`**: шлюз при естественной собственной речи; `top5_excl_own` исключает конкуренцию собственных слов модели в топ-5.
- **§6**: `retention` < 1 — модель меньше учитывает собеседника; сравнить `nat_self` и `nat_other`.
- **§7**: блок с большим `restored` — там шлюз действует; `attn` vs `mlp` показывает, переносится ли подавление вниманием из кадров собственной речи или вычисляется локально.
- Файлы `gen_*.wav` — послушать, что модель говорила (проверка естественности).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "listen_in_text_v9.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
