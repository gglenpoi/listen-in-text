"""Builds listen_in_text_v4.ipynb: one GPU session tests a batch of hypotheses and saves everything for CPU re-analysis."""
import sys
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Do Full-Duplex Speech Models Listen in Text? — v4 (пакет гипотез)

Один прогон на модель проверяет сразу несколько гипотез и сохраняет всё для пересчёта на CPU (ноутбук v3.1).

| # | Гипотеза | Как проверяем |
|---|---|---|
| H1 | Эффект не артефакт читаемой речи | Разговорная речь **AMI** (гарнитуры участников совещаний) наряду с LibriSpeech |
| H2 | Слово узнаётся до своего конца | Точные границы слов: **forced alignment MMS** по эталонному тексту вместо Whisper |
| H3 | Сигнал лексический, а не акустический | **Обращённая во времени речь**: тот же спектр и голос, но слов нет. Ожидаем ноль |
| H4 | Модель «слушает текстом», даже когда говорит сама | Условие **talking**: собственный поток модели — чужая речь со своим текстом, поверх речи пользователя |
| H5 | Текстовое представление появляется раньше 21-го слоя, logit lens его просто не видит | **Tuned lens**: обученная низкоранговая поправка на каждом слое; проверка на **отложенных дикторах и невиданных словах** + перенос на AMI |
| H6 | Предсказуемые и частые слова читаются раньше и лучше | **Факторы слов**: частотность (wordfreq), предсказуемость (сюрприз GPT-2), длительность; регрессия и предсказание до начала слова |

Плюс бутстреп-интервалы для всех кривых. Все выходы раскладываются в папки формата v2 (`<набор>_<условие>/top5.npz, turns.json, config.json`), их сразу понимает v3.1.

**Kaggle:** GPU T4×2, Internet On, секрет `HF_TOKEN`. Время: ~45–70 мин на модель (из них ~15 мин — скачивание весов). Если AMI или MMS не загрузятся, ноутбук не упадёт: он сообщит об этом и продолжит без них.
""")

md("## 1. Конфигурация")
code(r"""
import os
ON_KAGGLE = os.path.exists("/kaggle/working")
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")
BASE = "/kaggle/working" if ON_KAGGLE else ("/content" if ON_COLAB else os.getcwd())

CFG = dict(
    model         = "personaplex",          # "personaplex" | "moshi"
    datasets      = ["librispeech", "ami"],  # H1
    conditions    = ["listen", "reversed", "talking"],  # H3, H4
    n_turns       = 40,                     # per dataset
    turn_sec      = 20.0,
    gap_sec       = 0.3,
    aligner       = "mms",                  # H2: "mms" (gold text) | "whisper" (fallback, ASR words)
    whisper_model = "small.en",
    silence_frames= 6,
    topk          = 20,
    lag_frames    = 4,
    n_control     = 5,
    min_word_chars= 4,
    offsets       = (-8, 20),
    n_boot        = 500,
    # H5 tuned lens
    tl_layers     = list(range(0, 33, 2)),
    tl_rank       = 128,
    tl_steps      = 400,
    tl_batch      = 512,
    tl_lr         = 1e-3,
    tl_train_frac = 0.7,                    # share of speakers used for training
    # H6 word factors
    surprisal_lm  = "gpt2",
    dtype         = "auto",
    seed          = 0,
)
PRESETS = {
    "personaplex": dict(model_repo="nvidia/personaplex-7b-v1",
                        text_prompt="You enjoy having a good conversation.", prefix_user="sine"),
    "moshi":       dict(model_repo="kyutai/moshiko-pytorch-bf16", text_prompt="", prefix_user="silence"),
}
CFG.update(PRESETS[CFG["model"]])
CFG["out_dir"] = f"{BASE}/{CFG['model']}_v4"
os.makedirs(CFG["out_dir"], exist_ok=True)
print(CFG["model_repo"], "->", CFG["out_dir"])
""")

md("## 2. Установка и загрузка модели")
code(r"""
import sys, subprocess
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["NO_CUDA_GRAPH"] = "1"
if not os.path.exists("personaplex"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/personaplex.git"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "sentencepiece", "sphn", "openai-whisper",
                "soundfile", "tqdm", "wordfreq", "datasets"], check=True)
sys.path.insert(0, os.path.abspath("personaplex/moshi"))
""")
code(r"""
import json, math, random, re, tarfile, urllib.request, gc, io, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import sentencepiece, soundfile as sf
from scipy.signal import resample_poly
from safetensors import safe_open
from moshi.models import loaders
from moshi.models.lm import LMModel, SILENCE_TOKENS, SINE_TOKENS, _delay_sequence
warnings.filterwarnings("ignore", category=RuntimeWarning)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
""")
code(r"""
from huggingface_hub import hf_hub_download, list_repo_files
if not os.environ.get("HF_TOKEN"):
    try:
        if ON_KAGGLE:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        elif ON_COLAB:
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception as e:
        print("could not read HF_TOKEN secret:", e)
repo = CFG["model_repo"]
files = list_repo_files(repo)
lm_file = loaders.MOSHI_NAME if loaders.MOSHI_NAME in files else next(
    f for f in files if f.endswith(".safetensors") and not f.startswith("tokenizer"))
moshi_path = hf_hub_download(repo, lm_file)
mimi_path  = hf_hub_download(repo, loaders.MIMI_NAME)
sp = sentencepiece.SentencePieceProcessor(hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME))
""")
code(r"""
SPECIAL_TEXT = [0, 1, 2, 3]   # EPAD, BOS, EOS, PAD
SKIP_PREFIXES = ("depformer", "linears.")

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

def load_lm_for_lens(path, lm_kwargs, dtype):
    with torch.device("meta"):
        lm = LMModel(device="meta", dtype=dtype, **dict(lm_kwargs))
    devs = layer_devices(len(lm.transformer.layers))
    expected = set(lm.state_dict().keys())
    def target_device(key):
        m = re.match(r"transformer\.layers\.(\d+)\.", key)
        if m: return devs[int(m.group(1))]
        if key.startswith(("emb.", "text_emb.")): return devs[0]
        return devs[-1]
    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(SKIP_PREFIXES) or key not in expected:
                continue
            t = f.get_tensor(key)
            sd[key] = t.to(device=target_device(key), dtype=torch.float32 if key.endswith(".alpha") else dtype)
    missing = [k for k in expected if k not in sd and not k.startswith(SKIP_PREFIXES)]
    assert not missing, f"missing weights: {missing[:5]}"
    lm.load_state_dict(sd, strict=False, assign=True)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm, devs

DTYPE = pick_dtype()
lm, LAYER_DEVS = load_lm_for_lens(moshi_path, dict(loaders._lm_kwargs, dep_q=16), DTYPE)
N_LAYERS = len(lm.transformer.layers) + 1
mimi = loaders.get_mimi(mimi_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
FRAME = int(mimi.sample_rate / mimi.frame_rate); FPS = mimi.frame_rate
print("dtype", DTYPE, "| devices", sorted(set(LAYER_DEVS)))
""")

md("## 3. Данные (H1)\nКаждая реплика хранит эталонные фразы `segments = [(t0, t1, text)]` — по ним делается выравнивание. LibriSpeech test-clean: подряд идущие фразы одной главы. AMI IHM (тест): подряд идущие фразы одного участника совещания в его гарнитуре, паузы сокращены до `gap_sec`.")
code(r"""
def norm_words(text):
    return [w for w in (re.sub(r"[^a-z']", "", t) for t in text.lower().split()) if w.strip("'")]

def librispeech_turns(n_turns, turn_sec, gap_sec, sr_out=24000):
    root = Path(f"{BASE}/librispeech")
    if not (root / "LibriSpeech" / "test-clean").exists():
        root.mkdir(parents=True, exist_ok=True)
        tgz = root / "test-clean.tar.gz"
        urllib.request.urlretrieve("https://www.openslr.org/resources/12/test-clean.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(root)
    chapters = sorted((root / "LibriSpeech" / "test-clean").glob("*/*"))
    random.Random(CFG["seed"]).shuffle(chapters)
    turns = []
    for ch in chapters:
        trans = dict(l.split(" ", 1) for l in next(ch.glob("*.trans.txt")).read_text().splitlines())
        pieces, segs, t = [], [], 0.0
        for uid in sorted(trans):
            wav, sr = sf.read(ch / f"{uid}.flac", dtype="float32")
            wav = resample_poly(wav, sr_out // 1000, sr // 1000).astype(np.float32)
            segs.append((t, t + len(wav) / sr_out, trans[uid]))
            pieces += [wav, np.zeros(int(gap_sec * sr_out), np.float32)]
            t += len(wav) / sr_out + gap_sec
            if t >= turn_sec:
                break
        turns.append(dict(id=f"ls-{ch.parent.name}-{ch.name}", spk=f"ls-{ch.parent.name}", wav=np.concatenate(pieces), segments=segs))
        if len(turns) >= n_turns:
            break
    return turns

def ami_turns(n_turns, turn_sec, gap_sec, sr_out=24000, max_items=20000):
    from datasets import load_dataset, Audio
    ds = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    by_spk = {}
    for k, ex in enumerate(ds):
        if k >= max_items: break
        key = (ex["meeting_id"], ex["speaker_id"])
        by_spk.setdefault(key, []).append(ex)
    turns = []
    for (meeting, spk), exs in sorted(by_spk.items()):
        exs.sort(key=lambda e: e["begin_time"])
        pieces, segs, t = [], [], 0.0
        for ex in exs:
            if len(norm_words(ex["text"])) < 3:
                continue
            a = ex["audio"]
            wav, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")
            if wav.ndim > 1: wav = wav.mean(1)
            wav = resample_poly(wav, sr_out, sr).astype(np.float32)
            segs.append((t, t + len(wav) / sr_out, ex["text"]))
            pieces += [wav, np.zeros(int(gap_sec * sr_out), np.float32)]
            t += len(wav) / sr_out + gap_sec
            if t >= turn_sec:
                turns.append(dict(id=f"ami-{meeting}-{spk}-{len(turns)}", spk=f"ami-{spk}", wav=np.concatenate(pieces), segments=segs))
                pieces, segs, t = [], [], 0.0
                if sum(1 for x in turns if x["spk"] == f"ami-{spk}") >= 3:
                    break                                    # at most 3 turns per speaker for diversity
        if len(turns) >= n_turns:
            break
    return turns

DATA = {}
for name in CFG["datasets"]:
    try:
        DATA[name] = (librispeech_turns if name == "librispeech" else ami_turns)(CFG["n_turns"], CFG["turn_sec"], CFG["gap_sec"])
        print(f"{name}: {len(DATA[name])} turns, {sum(len(t['wav']) for t in DATA[name]) / 24000 / 60:.1f} min, "
              f"{len({t['spk'] for t in DATA[name]})} speakers")
    except Exception as e:
        print(f"!! {name} skipped: {type(e).__name__}: {e}")
assert DATA, "no dataset loaded"
""")

md("## 4. Границы слов (H2)\nMMS forced alignment (torchaudio) по эталонному тексту каждой фразы. Если MMS недоступен, используется Whisper (слова из распознавания).")
code(r"""
def whisper_align(turns):
    import whisper
    free = torch.cuda.mem_get_info(0)[0] / 1e9 if torch.cuda.is_available() else 0
    dev = "cuda:0" if free > 3 else "cpu"
    wm = whisper.load_model(CFG["whisper_model"], device=dev)
    for t in turns:
        res = wm.transcribe(resample_poly(t["wav"], 2, 3).astype(np.float32), language="en",
                            word_timestamps=True, condition_on_previous_text=False, fp16=(dev != "cpu"))
        t["words"] = [dict(word=re.sub(r"[^a-z']", "", w["word"].lower()), start=float(w["start"]), end=float(w["end"]))
                      for s in res["segments"] for w in s.get("words", [])]
        t["words"] = [w for w in t["words"] if w["word"].strip("'")]
    del wm; gc.collect(); torch.cuda.empty_cache()

def mms_align(turns):
    import torchaudio
    bundle = torchaudio.pipelines.MMS_FA
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = bundle.get_model(with_star=False).to(dev).eval()
    tok, aligner = bundle.get_tokenizer(), bundle.get_aligner()
    for t in turns:
        words = []
        for t0, t1, text in t["segments"]:
            ws = norm_words(text)
            if not ws:
                continue
            seg = t["wav"][int(t0 * 24000):int(t1 * 24000)]
            wav16 = torch.from_numpy(resample_poly(seg, 2, 3).astype(np.float32))[None].to(dev)
            with torch.inference_mode():
                emission, _ = model(wav16)
            spans = aligner(emission[0], tok(ws))
            ratio = wav16.shape[1] / emission.shape[1] / 16000
            words += [dict(word=w, start=t0 + s[0].start * ratio, end=t0 + s[-1].end * ratio) for w, s in zip(ws, spans)]
        t["words"] = words
    del model; gc.collect(); torch.cuda.empty_cache()

ALIGNER_USED = CFG["aligner"]
for name, turns in DATA.items():
    try:
        if CFG["aligner"] != "mms": raise RuntimeError("whisper requested")
        mms_align(turns)
    except Exception as e:
        print(f"!! MMS alignment failed for {name} ({type(e).__name__}: {e}); falling back to Whisper")
        ALIGNER_USED = "whisper"
        whisper_align(turns)
    print(name, sum(len(t["words"]) for t in turns), "words; e.g.", turns[0]["words"][:4])
CFG["aligner_used"] = ALIGNER_USED
""")

md("## 5. Сборка входов для трёх условий\n- **listen**: пользователь говорит, модель молчит (тишина + PAD), как в v2.\n- **reversed** (H3): речь пользователя обращена во времени; границы слов отражены.\n- **talking** (H4): одновременно с пользователем модель «говорит» речью из другой реплики того же набора; её текстовый поток содержит эту речь (кусочки слов с началом слова, EPAD перед словом), как в обучающих данных Moshi.")
code(r"""
@torch.no_grad()
def mimi_codes(wav):
    pad = (-len(wav)) % FRAME
    x = torch.from_numpy(np.pad(wav, (0, pad))).to(next(mimi.parameters()).device).view(1, 1, -1)
    return mimi.encode(x)[0].cpu()

def wrap_system(text):
    return f"<system> {text.strip()} <system>"

def text_track(words, n_frames):
    # model-own text stream: word pieces starting at the word onset frame, EPAD right before, PAD elsewhere
    track = np.full(n_frames, 3, dtype=np.int64)
    nxt = 0
    for w in words:
        f = max(int(math.floor(w["start"] * FPS)), nxt)
        if f >= n_frames: break
        ids = sp.encode(w["word"])
        if f - 1 >= 0 and track[f - 1] == 3: track[f - 1] = 0
        for k, tid in enumerate(ids):
            if f + k < n_frames: track[f + k] = tid
        nxt = f + len(ids)
    return track

def build_codes(user_codes, prompt_ids, n_sil, prefix_user, model_codes=None, model_text=None):
    pad = 3
    sil = torch.as_tensor(SILENCE_TOKENS).view(8, 1)
    filler = torch.as_tensor(SINE_TOKENS).view(8, 1) if prefix_user == "sine" else sil
    cols = []
    def frame(tt, ma, ua): cols.append(torch.cat([torch.tensor([[int(tt)]]), ma, ua], 0))
    for _ in range(n_sil): frame(pad, sil, filler)
    for tok in prompt_ids: frame(tok, sil, filler)
    for _ in range(n_sil): frame(pad, sil, filler)
    start = len(cols)
    for i in range(user_codes.shape[1]):
        own = model_codes[:, i:i + 1] if model_codes is not None and i < model_codes.shape[1] else sil
        txt = model_text[i] if model_text is not None and i < len(model_text) else pad
        frame(txt, own, user_codes[:, i:i + 1])
    return torch.cat(cols, 1).long(), start

def reverse_words(words, dur):
    return sorted([dict(word=w["word"], start=dur - w["end"], end=dur - w["start"]) for w in words], key=lambda w: w["start"])

prompt_ids = sp.encode(wrap_system(CFG["text_prompt"])) if CFG["text_prompt"] else []
for name, turns in DATA.items():
    for t in turns:
        t["codes_user"] = mimi_codes(t["wav"])
        t["codes_rev"] = mimi_codes(np.ascontiguousarray(t["wav"][::-1]))
        t["dur"] = len(t["wav"]) / 24000
print("ok")
""")

md("## 6. Прямой проход: logit lens + скрытые состояния для tuned lens")
code(r"""
@torch.no_grad()
def lens_forward(codes, topk, keep_pos=None, keep_layers=()):
    first, last = LAYER_DEVS[0], LAYER_DEVS[-1]
    codes = codes.unsqueeze(0).to(first)
    initial = lm._get_initial_token().to(first)
    seq = torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], 2)[:, :, :-1]
    x = lm.embed_codes(seq)
    special = torch.tensor(SPECIAL_TEXT, device=last)
    def lens(h):
        logits = lm.text_linear(lm.out_norm(h.to(last))).float()[0]
        own_argmax = logits.argmax(-1)
        logits[:, special] = -float("inf")
        return logits.topk(topk, -1).indices.int().cpu(), own_argmax.cpu()
    tops, kept = [], {}
    def keep(L, h):
        if keep_pos is not None and L in keep_layers:
            kept[L] = h[0, keep_pos].to(torch.float16).cpu()
    top, _ = lens(x); tops.append(top); keep(0, x)
    for i, layer in enumerate(lm.transformer.layers):
        x = layer(x.to(LAYER_DEVS[i]))
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"non-finite activations after layer {i}")
        top, own = lens(x); tops.append(top); keep(i + 1, x)
    return torch.stack(tops), own, kept

def word_window(t, w, lag):
    f0 = int(math.floor(w["start"] * FPS)); f1 = int(math.ceil(w["end"] * FPS))
    return t["user_start"] + f0 + 1, min(t["user_start"] + f1 + 1 + lag, t["T"] - 1)

RUNS, TL_FRAMES = {}, {}
t0 = time.time()
for name, turns in DATA.items():
    for ci, cond in enumerate(CFG["conditions"]):
        out = []
        for ti, t in enumerate(turns):
            if cond == "talking":
                other = turns[(ti + 1) % len(turns)]
                model_codes = other["codes_user"]
                model_text = text_track(other["words"], t["codes_user"].shape[1])
                codes, start = build_codes(t["codes_user"], prompt_ids, CFG["silence_frames"], CFG["prefix_user"], model_codes, model_text)
                words = t["words"]
            else:
                user = t["codes_rev"] if cond == "reversed" else t["codes_user"]
                codes, start = build_codes(user, prompt_ids, CFG["silence_frames"], CFG["prefix_user"])
                words = reverse_words(t["words"], t["dur"]) if cond == "reversed" else t["words"]
            rt = dict(id=t["id"], spk=t["spk"], user_start=start, T=codes.shape[1], words=words,
                      text=" ".join(w["word"] for w in words))
            keep_pos, meta = None, []
            if cond == "listen":
                pos = []
                for wi, w in enumerate(words):
                    a, b = word_window(rt, w, 0)
                    for p in range(a, b + 1):
                        pos.append(p); meta.append((ti, wi, p - a))
                keep_pos = torch.tensor(pos, dtype=torch.long)
            rt["top"], own, kept = lens_forward(codes, CFG["topk"], keep_pos, CFG["tl_layers"] if cond == "listen" else ())
            rt["pad_share"] = float(((own[start:] == 3) | (own[start:] == 0)).float().mean())
            if cond == "listen":
                TL_FRAMES.setdefault(name, dict(meta=[], h={L: [] for L in CFG["tl_layers"]}))
                TL_FRAMES[name]["meta"] += meta
                for L, h in kept.items():
                    TL_FRAMES[name]["h"][L].append(h)
            out.append(rt)
        RUNS[(name, cond)] = out
        print(f"{name}/{cond}: {len(out)} turns, model PAD share while user speaks {np.mean([r['pad_share'] for r in out]):.2f} "
              f"({time.time() - t0:.0f}s)")
for name in TL_FRAMES:
    TL_FRAMES[name]["h"] = {L: torch.cat(v) for L, v in TL_FRAMES[name]["h"].items()}
    print(name, "tuned-lens frames:", len(TL_FRAMES[name]["meta"]))
""")

md("## 7. Попадания по слоям с бутстреп-интервалами (H1, H3, H4)")
code(r"""
def word_targets(word):
    ids = set()
    for v in {word, word.capitalize(), word.upper()}:
        enc = sp.encode(v)
        if enc: ids.add(enc[0])
    return ids - set(SPECIAL_TEXT)

def best_rank(top, a, b, targets):
    if b < a or not targets: return None
    win = top[:, a:b + 1, :]
    hit = torch.isin(win, torch.tensor(sorted(targets), dtype=win.dtype))
    k = top.shape[2]
    ranks = torch.where(hit, torch.arange(k).view(1, 1, k).expand_as(win), torch.full_like(win, k))
    return ranks.amin(dim=(1, 2)).numpy()

def score_all(turns, seed=0):
    rng = random.Random(seed)
    rows = []
    for ti, t in enumerate(turns):
        others = [u for j, u in enumerate(turns) if j != ti]
        for wi, w in enumerate(t["words"]):
            targets = word_targets(w["word"])
            a, b = word_window(t, w, CFG["lag_frames"])
            r = best_rank(t["top"], a, b, targets)
            if r is None: continue
            ctrl = []
            for _ in range(CFG["n_control"]):
                u = rng.choice(others)
                if w["word"] in u["text"].split(): continue
                n = b - a; lo, hi = u["user_start"] + 1, u["T"] - 1 - n
                if hi <= lo: continue
                s = rng.randint(lo, hi)
                ctrl.append(best_rank(u["top"], s, s + n, targets))
            if ctrl:
                rows.append(dict(ti=ti, wi=wi, turn=t["id"], spk=t["spk"], word=w["word"], n_chars=len(w["word"]),
                                 start=w["start"], dur=w["end"] - w["start"], real=r, ctrl=np.stack(ctrl)))
    return rows

boot_rng = np.random.default_rng(CFG["seed"])
def boot_ci(x, n):
    idx = boot_rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

SCORES, curves = {}, []
for (name, cond), turns in RUNS.items():
    rows = score_all(turns, CFG["seed"]); SCORES[(name, cond)] = rows
    for subset, mc in (("all", 0), ("content", CFG["min_word_chars"])):
        sel = [r for r in rows if r["n_chars"] >= mc]
        for k in (1, 5, 20):
            real = np.stack([r["real"] < k for r in sel]).astype(float)
            ctrl = np.stack([(r["ctrl"] < k).mean(0) for r in sel])
            for L in range(real.shape[1]):
                lo, hi = boot_ci(real[:, L] - ctrl[:, L], CFG["n_boot"]) if (k == 5 and subset == "content") else (np.nan, np.nan)
                curves.append(dict(dataset=name, cond=cond, subset=subset, k=k, layer=L, real=real[:, L].mean(),
                                   control=ctrl[:, L].mean(), excess=(real[:, L] - ctrl[:, L]).mean(),
                                   ci_lo=lo, ci_hi=hi, n_words=len(sel)))
CURVES = pd.DataFrame(curves)
CURVES.to_csv(f"{CFG['out_dir']}/hit_curves_v4.csv", index=False)
peak = CURVES[(CURVES.k == 5) & (CURVES.subset == "content")].sort_values("excess").groupby(["dataset", "cond"]).tail(1)
print(peak[["dataset", "cond", "layer", "real", "control", "excess", "ci_lo", "ci_hi", "n_words"]].round(3).to_string(index=False))
""")
code(r"""
fig, axes = plt.subplots(1, len(DATA), figsize=(7 * len(DATA), 4.2), squeeze=False)
cc = {"listen": "C0", "reversed": "C7", "talking": "C3"}
for ax, name in zip(axes[0], DATA):
    for cond in CFG["conditions"]:
        d = CURVES[(CURVES.dataset == name) & (CURVES.cond == cond) & (CURVES.k == 5) & (CURVES.subset == "content")]
        if d.empty: continue
        ax.plot(d.layer, d.excess, color=cc[cond], lw=2, label=cond)
        ax.fill_between(d.layer, d.ci_lo, d.ci_hi, color=cc[cond], alpha=.2)
    ax.axhline(0, color="k", lw=.8); ax.grid(alpha=.3)
    ax.set(title=f"{CFG['model']} / {name}: user word in top-5 (excess over control, 95% CI)", xlabel="layer", ylabel="hit@5 excess")
    ax.legend()
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/conditions.png", dpi=150); plt.show()
""")

md("## 8. Время (H2): с точными границами слов")
code(r"""
OFFS = np.arange(CFG["offsets"][0], CFG["offsets"][1] + 1)
MS = OFFS * 80 + 40

def hits_at(top, positions, targets, k=5):
    out = np.full((top.shape[0], len(positions)), np.nan)
    ok = (positions >= 0) & (positions < top.shape[1])
    if ok.any():
        win = top[:, positions[ok], :k]
        out[:, ok] = torch.isin(win, torch.tensor(sorted(targets), dtype=win.dtype)).any(-1).numpy()
    return out

def locked(turns, anchor, seed=0, n_ctrl=3):
    rng = random.Random(seed)
    R, C, keys = [], [], []
    for ti, t in enumerate(turns):
        others = [u for j, u in enumerate(turns) if j != ti]
        for wi, w in enumerate(t["words"]):
            targets = word_targets(w["word"])
            if len(w["word"]) < CFG["min_word_chars"] or not targets: continue
            f = int(math.floor(w["start"] * FPS)) if anchor == "start" else int(math.ceil(w["end"] * FPS)) - 1
            pos = t["user_start"] + f + 1 + OFFS
            r = hits_at(t["top"], pos, targets); r[:, pos <= t["user_start"]] = np.nan
            cs = []
            for _ in range(n_ctrl):
                u = rng.choice(others)
                if w["word"] in u["text"].split(): continue
                lo, hi = u["user_start"] + 1 - OFFS.min(), u["T"] - 1 - OFFS.max()
                if hi > lo: cs.append(hits_at(u["top"], rng.randint(lo, hi) + OFFS, targets))
            if cs:
                R.append(r); C.append(np.nanmean(np.stack(cs), 0)); keys.append((ti, wi))
    return np.stack(R), np.stack(C), keys

TIMING = {}
for name in DATA:
    for anchor in ("start", "end"):
        TIMING[(name, anchor)] = locked(RUNS[(name, "listen")], anchor)
np.savez_compressed(f"{CFG['out_dir']}/timing_v4.npz", offsets=OFFS,
                    **{f"{n}_{a}_{s}": np.nanmean(v[i], 0) for (n, a), v in TIMING.items() for i, s in enumerate(("real", "ctrl"))})

BEST = {name: int(CURVES[(CURVES.dataset == name) & (CURVES.cond == "listen") & (CURVES.k == 5) & (CURVES.subset == "content")]
                  .sort_values("excess").iloc[-1].layer) for name in DATA}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, anchor in zip(axes, ("start", "end")):
    for name in DATA:
        R, C, _ = TIMING[(name, anchor)]
        L = BEST[name]
        d = R[:, L] - C[:, L]
        m = np.nanmean(d, 0)
        lo, hi = np.array([boot_ci(d[:, j][~np.isnan(d[:, j])], 300) for j in range(d.shape[1])]).T
        ax.plot(MS, m, lw=2, label=f"{name} (layer {L})"); ax.fill_between(MS, lo, hi, alpha=.2)
        print(f"{name} {anchor}-locked, layer {L}: peak {MS[np.nanargmax(m)]} ms; "
              f"pre-anchor (≤-200 ms) mean excess {np.nanmean(m[MS <= -200]):+.3f}")
    ax.axvline(0, color="k", lw=.8); ax.grid(alpha=.3); ax.legend()
    ax.set(title=f"locked to word {anchor} ({ALIGNER_USED} boundaries)", xlabel=f"ms of audio heard relative to word {anchor}",
           ylabel="hit@5 excess over control")
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/timing_v4.png", dpi=150); plt.show()
""")

md("## 9. Tuned lens (H5)\nНа каждом слое из `tl_layers` учим поправку `h + (h·U)·V + b` (ранг `tl_rank`, старт с нуля = обычный logit lens) так, чтобы текстовая голова модели выдавала первый токен слова, которое пользователь произносит в этот момент. Обучение — на `tl_train_frac` дикторов LibriSpeech (или первого набора). Проверка:\n- **отложенные дикторы, знакомые слова**;\n- **отложенные дикторы, невиданные слова** — главная мера: обобщает ли поправка перевод «звук → текст», а не запоминает классы;\n- **перенос на AMI** (если загружен);\n- контроль: та же поправка, обученная на **перемешанных** метках.\n\nМетрика — доля слов, у которых в окне слова хотя бы на одном кадре цель в top-5.")
code(r"""
class LowRankLens(torch.nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(d, r) * 0.01)
        self.V = torch.nn.Parameter(torch.zeros(r, d))
        self.b = torch.nn.Parameter(torch.zeros(d))
    def forward(self, h):
        # work on the RMS-normalised state: residual norms differ by orders of magnitude across layers,
        # and the model's out_norm is scale-invariant, so at init this is exactly the logit lens
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return h + (h @ self.U) @ self.V + self.b

LAST = LAYER_DEVS[-1]
SPECIAL_MASK = torch.zeros(lm.text_linear.out_features, dtype=torch.bool, device=LAST)
SPECIAL_MASK[SPECIAL_TEXT] = True

def lens_logits(h):
    logits = lm.text_linear(lm.out_norm(h.to(DTYPE).unsqueeze(0)))[0].float()
    return logits.masked_fill(SPECIAL_MASK, -1e4)

def frame_table(name):
    turns = RUNS[(name, "listen")]
    meta = TL_FRAMES[name]["meta"]
    rows = []
    for fi, (ti, wi, off) in enumerate(meta):
        w = turns[ti]["words"][wi]["word"]
        enc = sp.encode(w)
        rows.append(dict(fi=fi, ti=ti, wi=wi, spk=turns[ti]["spk"], word=w, target=enc[0] if enc else -1))
    return pd.DataFrame(rows)

def train_lens(H, y, steps, seed, shuffle=False):
    g = torch.Generator().manual_seed(seed)
    if shuffle: y = y[torch.randperm(len(y), generator=g)]
    lens = LowRankLens(H.shape[1], CFG["tl_rank"]).to(LAST)
    opt = torch.optim.AdamW(lens.parameters(), lr=CFG["tl_lr"], weight_decay=1e-2)
    for step in range(steps):
        idx = torch.randint(0, len(y), (CFG["tl_batch"],), generator=g)
        h = H[idx].to(LAST).float()
        loss = torch.nn.functional.cross_entropy(lens_logits(lens(h)), y[idx].to(LAST))
        opt.zero_grad(); loss.backward(); opt.step()
    return lens

@torch.no_grad()
def word_hit(lens, H, ft, k=5):
    # fraction of words (grouped by ti, wi) with target in top-k on any frame
    if len(ft) == 0:
        return np.nan
    hits = []
    for i in range(0, len(ft), 2048):
        h = H[torch.as_tensor(ft.fi.values[i:i + 2048])].to(LAST).float()
        h = lens(h) if lens is not None else h
        top = lens_logits(h).topk(k, -1).indices.cpu()
        hits.append((top == torch.tensor(ft.target.values[i:i + 2048]).view(-1, 1)).any(-1))
    f = ft.assign(hit=torch.cat(hits).numpy())
    return f.groupby(["ti", "wi"]).hit.max().mean() if len(f) else np.nan

train_name = "librispeech" if "librispeech" in TL_FRAMES else next(iter(TL_FRAMES))
FT = frame_table(train_name)
FT = FT[(FT.target >= 0) & (FT.word.str.len() >= CFG["min_word_chars"])]
spks = sorted(FT.spk.unique()); random.Random(CFG["seed"]).shuffle(spks)
train_spk = set(spks[:int(len(spks) * CFG["tl_train_frac"])])
tr, te = FT[FT.spk.isin(train_spk)], FT[~FT.spk.isin(train_spk)]
seen_types = set(tr.word)
te_seen, te_unseen = te[te.word.isin(seen_types)], te[~te.word.isin(seen_types)]
transfer = {n: frame_table(n) for n in TL_FRAMES if n != train_name}
transfer = {n: f[(f.target >= 0) & (f.word.str.len() >= CFG["min_word_chars"]) & ~f.word.isin(seen_types)] for n, f in transfer.items()}
print(f"train frames {len(tr)} | test words seen-type {te_seen.groupby(['ti','wi']).ngroups}, "
      f"unseen-type {te_unseen.groupby(['ti','wi']).ngroups} | transfer: " + ", ".join(f"{n} {f.groupby(['ti','wi']).ngroups}" for n, f in transfer.items()))

tl_rows = []
for L in CFG["tl_layers"]:
    H = TL_FRAMES[train_name]["h"][L]
    y = torch.tensor(tr.target.values)
    Htr = H[torch.as_tensor(tr.fi.values)]
    lens = train_lens(Htr, y, CFG["tl_steps"], CFG["seed"])
    shuf = train_lens(Htr, y, CFG["tl_steps"], CFG["seed"], shuffle=True)
    row = dict(layer=L)
    for tag, ft in (("seen", te_seen), ("unseen", te_unseen)):
        row[f"logit_{tag}"] = word_hit(None, H, ft)
        row[f"tuned_{tag}"] = word_hit(lens, H, ft)
        row[f"shuffled_{tag}"] = word_hit(shuf, H, ft)
    for n, ft in transfer.items():
        Hn = TL_FRAMES[n]["h"][L]
        row[f"logit_{n}"] = word_hit(None, Hn, ft)
        row[f"tuned_{n}"] = word_hit(lens, Hn, ft)
    tl_rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    del lens, shuf; torch.cuda.empty_cache()
TL = pd.DataFrame(tl_rows); TL.to_csv(f"{CFG['out_dir']}/tuned_lens.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(TL.layer, TL.logit_unseen, "-o", color="C7", label="logit lens (unseen words)")
ax.plot(TL.layer, TL.tuned_unseen, "-o", color="C0", label="tuned lens (unseen words)")
ax.plot(TL.layer, TL.tuned_seen, "--", color="C0", label="tuned lens (seen words)")
ax.plot(TL.layer, TL.shuffled_unseen, ":", color="C3", label="tuned on shuffled labels")
for n in transfer:
    ax.plot(TL.layer, TL[f"tuned_{n}"], "-s", color="C2", label=f"tuned lens → {n} (unseen words)")
ax.set(title=f"{CFG['model']}: when is the heard word linearly readable as text?", xlabel="layer",
       ylabel="word hit@5 (held-out speakers)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/tuned_lens.png", dpi=150); plt.show()
""")

md("## 10. Факторы слов (H6)\nДля содержательных слов (условие listen): частотность (Zipf, wordfreq), сюрприз по GPT-2 при предыдущем тексте реплики (биты), длительность, длина. Исход — попадание в top-5 на лучшем слое. Логистическая регрессия на стандартизованных признаках, интервалы бутстрепом. Отдельно: есть ли чтение **до начала слова** у самых предсказуемых слов.")
code(r"""
from wordfreq import zipf_frequency
from transformers import AutoTokenizer, AutoModelForCausalLM
lm_tok = AutoTokenizer.from_pretrained(CFG["surprisal_lm"])
lm_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
slm = AutoModelForCausalLM.from_pretrained(CFG["surprisal_lm"]).to(lm_dev).eval()

@torch.no_grad()
def word_surprisal(words):
    text = " ".join(words)
    enc = lm_tok(text, return_offsets_mapping=True, return_tensors="pt")
    ids = torch.cat([torch.tensor([[lm_tok.bos_token_id]]), enc.input_ids], 1).to(lm_dev)[:, :1024]
    logp = torch.log_softmax(slm(ids).logits[0, :-1].float(), -1)
    tok_s = -logp.gather(1, ids[0, 1:, None])[:, 0].cpu().numpy() / np.log(2)
    offs = enc.offset_mapping[0].numpy()[:len(tok_s)]
    out, pos = [], 0
    for w in words:
        s, e = pos, pos + len(w); pos = e + 1
        m = (offs[:, 0] < e) & (offs[:, 1] > s)
        out.append(float(tok_s[m].sum()) if m.any() else np.nan)
    return out

FACT = []
for name in DATA:
    turns = RUNS[(name, "listen")]
    surp = {ti: word_surprisal([w["word"] for w in t["words"]]) for ti, t in enumerate(turns)}
    L = BEST[name]
    for r in SCORES[(name, "listen")]:
        if r["n_chars"] < CFG["min_word_chars"]: continue
        FACT.append(dict(dataset=name, ti=r["ti"], wi=r["wi"], word=r["word"], spk=r["spk"],
                         hit=float(r["real"][L] < 5), ctrl=float((r["ctrl"][:, L] < 5).mean()),
                         surprisal=surp[r["ti"]][r["wi"]], zipf=zipf_frequency(r["word"], "en"),
                         dur=r["dur"], n_chars=r["n_chars"], start=r["start"]))
FACT = pd.DataFrame(FACT).dropna()
FACT.to_csv(f"{CFG['out_dir']}/word_factors.csv", index=False)
del slm; gc.collect(); torch.cuda.empty_cache()

from sklearn.linear_model import LogisticRegression
feats = ["surprisal", "zipf", "dur", "n_chars", "start"]
for name, d in FACT.groupby("dataset"):
    X = ((d[feats] - d[feats].mean()) / d[feats].std()).values; y = d.hit.values
    if y.min() == y.max():
        print(f"{name}: all words have the same outcome, regression skipped"); continue
    coef = LogisticRegression(C=10, max_iter=2000).fit(X, y).coef_[0]
    rng = np.random.default_rng(0); boots = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        boots.append(LogisticRegression(C=10, max_iter=2000).fit(X[i], y[i]).coef_[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f"\n{name} (n={len(d)}, hit rate {y.mean():.2f}, layer {BEST[name]}): standardized log-odds")
    for f_, c, a, b in zip(feats, coef, lo, hi):
        print(f"  {f_:>10s} {c:+.2f}  [{a:+.2f}, {b:+.2f}]")
    d = d.assign(tert=pd.qcut(d.surprisal, 3, labels=["predictable", "medium", "surprising"]))
    print(d.groupby("tert", observed=True)[["hit", "ctrl"]].mean().round(3).to_string())

    R, C, keys = TIMING[(name, "start")]
    kd = {k: i for i, k in enumerate(keys)}
    L = BEST[name]
    pre = MS <= -120
    for tert, g in d.groupby("tert", observed=True):
        idx = [kd[k] for k in zip(g.ti, g.wi) if k in kd]
        ex = np.nanmean(R[idx][:, L][:, pre] - C[idx][:, L][:, pre], 1)
        print(f"  pre-onset (≤ -120 ms) excess, {tert:>11s}: {np.nanmean(ex):+.3f}  (n={len(idx)})")
""")

md("## 11. Сохранение\nПапки в формате v2 для каждого набора и условия (их читает v3.1), сводные таблицы и графики, один архив.")
code(r"""
for (name, cond), turns in RUNS.items():
    d = f"{CFG['out_dir']}/{name}_{cond}"
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(f"{d}/top5.npz", **{t["id"]: t["top"][:, :, :5].numpy().astype(np.int32) for t in turns})
    json.dump([dict(id=t["id"], spk=t["spk"], user_start=t["user_start"], text=t["text"], words=t["words"]) for t in turns],
              open(f"{d}/turns.json", "w"))
    json.dump(dict(CFG, model=f"{CFG['model']}|{name}|{cond}"), open(f"{d}/config.json", "w"), indent=2)
json.dump(CFG, open(f"{CFG['out_dir']}/config.json", "w"), indent=2)

import shutil
from IPython.display import FileLink, display
archive = shutil.make_archive(CFG["out_dir"], "zip", CFG["out_dir"])
print(archive, f"{os.path.getsize(archive) / 1e6:.1f} MB")
os.chdir(BASE); display(FileLink(os.path.basename(archive)))
""")

md(r"""
## Как читать

- **H1** — кривая `ami/listen` с интервалом выше нуля на поздних слоях: эффект не артефакт LibriSpeech.
- **H2** — пик на графике времени до конца слова (при границах MMS).
- **H3** — `reversed` около нуля на всех слоях: сигнал лексический.
- **H4** — `talking` против `listen`: если сигнал сохраняется, модель «слушает текстом» и во время своей речи; если падает — собственная речь вытесняет транскрипцию собеседника.
- **H5** — `tuned lens (unseen words)` выше `logit lens` и выше перемешанного контроля на слоях раньше 21-го: текстовое представление появляется раньше, чем видит logit lens.
- **H6** — знаки коэффициентов (ожидаем: сюрприз −, частотность +) и чтение до начала слова у предсказуемых слов.

Пришлите архив `<модель>_v4.zip`; папки `<набор>_<условие>` можно сразу прогнать через v3.1.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "listen_in_text_v4.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
