"""Builds pilot_listen_in_text.ipynb from the cell list below."""
import nbformat as nbf
import sys

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Pilot: Do Full-Duplex Speech Models Listen in Text?

**Вопрос пилота.** PersonaPlex (и Moshi) имеет текстовый поток только для *своей* речи. Речь пользователя модель получает исключительно как аудиотокены Mimi. Проверяем: становятся ли слова пользователя декодируемыми как текстовые токены в промежуточных слоях временного трансформера (logit lens через текстовую голову модели)?

**Схема.**
1. Пользовательская речь (LibriSpeech test-clean, склеенная в «реплики» по ~20 с) → токены Mimi → пользовательский поток модели.
2. Свой поток модели: тишина + PAD (модель молча слушает), перед этим — системный промпт, как в PersonaPlex.
3. Один teacher-forced прямой проход, на выходе каждого слоя: `out_norm` → `text_linear` → top-k токенов.
4. Для каждого слова пользователя (тайм-коды из Whisper): попал ли его токен в top-k в окне вокруг слова. Контроль: то же слово в окне из *другой* реплики (уровень случайного совпадения).

**Железо.** Ноутбук сам определяет Kaggle или Colab. Kaggle 2×T4 (модель делится 16+16 слоёв, fp16) или одна GPU ≥ 24 ГБ (L4/A100 в Colab Pro, лучше bf16). Время: ~10–20 мин на загрузку весов, затем секунды на реплику.

**Перед запуском.**
- Примите лицензию модели на https://huggingface.co/nvidia/personaplex-7b-v1
- Kaggle: *Add-ons → Secrets* → `HF_TOKEN`; включите Internet и GPU T4×2.
- Colab: значок ключа слева → секрет `HF_TOKEN` с доступом для ноутбука; *Runtime → Change runtime type* → L4 или A100 (бесплатная T4 на пределе по памяти).

**Критерий go/no-go:** если на каком-то слое hit@5 для реальных окон устойчиво выше контроля (скажем, в 2+ раза при сотнях слов), сигнал есть — идём в полноценный эксперимент.
""")

md("## 1. Конфигурация")
code(r"""
import os
ON_KAGGLE = os.path.exists("/kaggle/working")
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")
BASE = "/kaggle/working" if ON_KAGGLE else ("/content" if ON_COLAB else os.getcwd())
print("platform:", "kaggle" if ON_KAGGLE else "colab" if ON_COLAB else "local", "| base:", BASE)

CFG = dict(
    model_repo   = "nvidia/personaplex-7b-v1",  # or "kyutai/moshiko-pytorch-bf16" for the Moshi baseline (see note at the end)
    n_turns      = 40,        # number of ~20 s user turns (40 turns ≈ 2500 words)
    turn_sec     = 20.0,      # target length of one concatenated user turn
    gap_sec      = 0.3,       # silence inserted between LibriSpeech utterances
    text_prompt  = "You enjoy having a good conversation.",  # PersonaPlex prompt used for FullDuplexBench
    silence_frames = 6,       # 0.5 s spacer, as in PersonaPlex (12.5 Hz * 0.5)
    topk         = 50,        # how many lens tokens to keep per layer and frame
    lag_frames   = 4,         # extra frames after word end allowed (4 * 80 ms = 320 ms)
    n_control    = 5,         # random control windows per word
    min_word_chars = 4,       # 'content words' subset: words with >= 4 letters
    save_layers  = [0, 4, 8, 12, 16, 20, 24, 28, 32],  # hidden states kept for probes (0 = embeddings)
    whisper_model = "small.en",
    dtype        = "auto",    # "auto": bf16 if GPU supports it natively (sm80+), else fp16
    out_dir      = f"{BASE}/pilot_out",
    seed         = 0,
)
""")

md("## 2. Установка\nНе ставим `pip install moshi/.`: пакет PersonaPlex пинит `torch<2.5` и сломает окружение Kaggle. Достаточно добавить исходники в `sys.path`.")
code(r"""
import os, sys, subprocess
os.environ["NO_TORCH_COMPILE"] = "1"   # avoid torch.compile on T4
os.environ["NO_CUDA_GRAPH"] = "1"

if not os.path.exists("personaplex"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/personaplex.git"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "sentencepiece", "sphn", "openai-whisper", "soundfile", "tqdm"], check=True)
sys.path.insert(0, os.path.abspath("personaplex/moshi"))
os.makedirs(CFG["out_dir"], exist_ok=True)
""")

code(r"""
import json, math, random, re, tarfile, urllib.request, gc
from pathlib import Path
import numpy as np, pandas as pd, torch
import sentencepiece, soundfile as sf
from scipy.signal import resample_poly
from safetensors import safe_open

from moshi.models import loaders
from moshi.models.lm import LMModel, SILENCE_TOKENS, SINE_TOKENS, _delay_sequence

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
print(torch.__version__, "GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
""")

md("## 3. Загрузка весов\nТокен берётся из Kaggle Secrets (или переменной окружения `HF_TOKEN`).")
code(r"""
from huggingface_hub import hf_hub_download
if not os.environ.get("HF_TOKEN"):
    try:
        if ON_KAGGLE:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        elif ON_COLAB:
            from google.colab import userdata          # Colab: key icon on the left -> HF_TOKEN, enable notebook access
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception as e:
        print("could not read HF_TOKEN secret:", e)
assert os.environ.get("HF_TOKEN"), "Set HF_TOKEN (Kaggle Secrets / Colab Secrets / env) after accepting the model license"

repo = CFG["model_repo"]
moshi_path = hf_hub_download(repo, loaders.MOSHI_NAME)
mimi_path  = hf_hub_download(repo, loaders.MIMI_NAME)
tok_path   = hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)
sp = sentencepiece.SentencePieceProcessor(tok_path)
print(moshi_path, os.path.getsize(moshi_path) / 1e9, "GB")
""")

md("## 4. Модель на двух GPU\nГрузим веса потензорно (низкое потребление RAM), пропускаем depformer — для logit lens он не нужен. Слои делятся между GPU, проход по слоям делаем вручную.")
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
    if n_gpu == 0:
        return ["cpu"] * n_layers
    if n_gpu == 1:
        return ["cuda:0"] * n_layers
    half = n_layers // 2
    return ["cuda:0"] * half + ["cuda:1"] * (n_layers - half)

def load_lm_for_lens(path, lm_kwargs, dtype):
    kw = dict(lm_kwargs)
    with torch.device("meta"):
        lm = LMModel(device="meta", dtype=dtype, **kw)
    n_layers = len(lm.transformer.layers)
    devs = layer_devices(n_layers)
    first, last = devs[0], devs[-1]
    expected = set(lm.state_dict().keys())

    def target_device(key):
        m = re.match(r"transformer\.layers\.(\d+)\.", key)
        if m:
            return devs[int(m.group(1))]
        if key.startswith(("emb.", "text_emb.")):
            return first
        return last   # out_norm, text_linear

    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(SKIP_PREFIXES) or key not in expected:
                continue
            t = f.get_tensor(key)
            keep_fp32 = key.endswith(".alpha")   # RMSNorm gains are fp32 in Moshi
            sd[key] = t.to(device=target_device(key), dtype=torch.float32 if keep_fp32 else dtype)
    missing = [k for k in expected if k not in sd and not k.startswith(SKIP_PREFIXES)]
    assert not missing, f"missing weights: {missing[:5]}"
    lm.load_state_dict(sd, strict=False, assign=True)
    lm.eval()
    return lm, devs

DTYPE = pick_dtype()
lm_kwargs = dict(loaders._lm_kwargs, dep_q=16)
lm, LAYER_DEVS = load_lm_for_lens(moshi_path, lm_kwargs, DTYPE)
print("dtype:", DTYPE, "| layers:", len(lm.transformer.layers), "| devices:", sorted(set(LAYER_DEVS)))
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} allocated {torch.cuda.memory_allocated(i)/1e9:.1f} GB")
""")

md("## 5. Данные: «реплики» пользователя из LibriSpeech test-clean\nСклеиваем подряд идущие фразы одной главы до ~20 с. Это чтение, а не разговор — для пилота достаточно; разговорные данные — следующий шаг.")
code(r"""
DATA = Path(f"{BASE}/librispeech")
if not (DATA / "LibriSpeech" / "test-clean").exists():
    DATA.mkdir(parents=True, exist_ok=True)
    tgz = DATA / "test-clean.tar.gz"
    urllib.request.urlretrieve("https://www.openslr.org/resources/12/test-clean.tar.gz", tgz)
    with tarfile.open(tgz) as t:
        t.extractall(DATA)

def librispeech_turns(root, n_turns, turn_sec, gap_sec, sr_out=24000):
    chapters = sorted((root / "LibriSpeech" / "test-clean").glob("*/*"))
    random.shuffle(chapters)
    turns = []
    for ch in chapters:
        trans = {}
        for line in next(ch.glob("*.trans.txt")).read_text().splitlines():
            uid, txt = line.split(" ", 1)
            trans[uid] = txt
        pieces, text, dur = [], [], 0.0
        for uid in sorted(trans):
            wav, sr = sf.read(ch / f"{uid}.flac", dtype="float32")
            wav = resample_poly(wav, sr_out // 1000, sr // 1000).astype(np.float32)  # 16k -> 24k
            pieces += [wav, np.zeros(int(gap_sec * sr_out), np.float32)]
            text.append(trans[uid].lower())
            dur += len(wav) / sr_out + gap_sec
            if dur >= turn_sec:
                break
        turns.append(dict(id=ch.parent.name + "-" + ch.name, wav=np.concatenate(pieces), text=" ".join(text)))
        if len(turns) >= n_turns:
            break
    return turns

turns = librispeech_turns(DATA, CFG["n_turns"], CFG["turn_sec"], CFG["gap_sec"])
print(len(turns), "turns,", sum(len(t["wav"]) for t in turns) / 24000 / 60, "min")
""")

md("## 6. Пословные тайм-коды (Whisper)\nWhisper грузится на GPU, после разметки выгружается.")
code(r"""
import whisper
free_gb = torch.cuda.mem_get_info(0)[0] / 1e9 if torch.cuda.is_available() else 0
wdev = "cuda:0" if free_gb > 3 else "cpu"     # single T4: model leaves too little room, Whisper goes to CPU
print(f"free GPU memory {free_gb:.1f} GB -> whisper on {wdev}")
wmodel = whisper.load_model(CFG["whisper_model"], device=wdev)
for t in turns:
    audio16 = resample_poly(t["wav"], 2, 3).astype(np.float32)   # 24k -> 16k
    res = wmodel.transcribe(audio16, language="en", word_timestamps=True, condition_on_previous_text=False, fp16=(wdev != "cpu"))
    t["words"] = [
        dict(word=re.sub(r"[^\w']", "", w["word"]).lower(), start=float(w["start"]), end=float(w["end"]))
        for seg in res["segments"] for w in seg.get("words", [])
    ]
    t["words"] = [w for w in t["words"] if w["word"]]
del wmodel; gc.collect(); torch.cuda.empty_cache()
print(sum(len(t["words"]) for t in turns), "words; example:", turns[0]["words"][:5])
""")

md("## 7. Токены Mimi и сборка последовательности\nПотоки модели: `0` — свой текст, `1..8` — своё аудио, `9..16` — аудио пользователя. Префикс повторяет PersonaPlex: тишина → системный промпт → тишина (голосовой промпт в пилоте пропускаем). Пока говорит пользователь, свой поток модели — тишина + PAD.")
code(r"""
mimi = loaders.get_mimi(mimi_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
FRAME = int(mimi.sample_rate / mimi.frame_rate)   # 1920 samples = 80 ms
FPS = mimi.frame_rate                              # 12.5

@torch.no_grad()
def mimi_codes(wav):
    pad = (-len(wav)) % FRAME
    x = torch.from_numpy(np.pad(wav, (0, pad))).to(next(mimi.parameters()).device).view(1, 1, -1)
    return mimi.encode(x)[0].cpu()                 # [8, frames]

def wrap_system(text):
    return f"<system> {text.strip()} <system>"

def build_codes(user_codes, prompt_ids, n_sil):
    # Returns undelayed codes [17, T] and the frame index where the user turn starts.
    pad = 3
    sil = torch.as_tensor(SILENCE_TOKENS).view(8, 1)
    sine = torch.as_tensor(SINE_TOKENS).view(8, 1)
    cols = []
    def frame(text_tok, model_audio, user_audio):
        cols.append(torch.cat([torch.tensor([[text_tok]]), model_audio, user_audio], 0))
    for _ in range(n_sil): frame(pad, sil, sine)
    for tok in prompt_ids:  frame(tok, sil, sine)
    for _ in range(n_sil): frame(pad, sil, sine)
    start = len(cols)
    for i in range(user_codes.shape[1]):
        frame(pad, sil, user_codes[:, i:i + 1])
    return torch.cat(cols, 1).long(), start

prompt_ids = sp.encode(wrap_system(CFG["text_prompt"])) if CFG["text_prompt"] else []
for t in turns:
    t["codes"], t["user_start"] = build_codes(mimi_codes(t["wav"]), prompt_ids, CFG["silence_frames"])
print("example sequence:", turns[0]["codes"].shape, "user starts at frame", turns[0]["user_start"])
""")

md("## 8. Прямой проход с logit lens\nВход строится так же, как в `LMModel.forward_train`: задержки потоков + начальный токен. Позиция `p` видит семантические токены пользователя до кадра `p-1`. На выходе каждого слоя (и после эмбеддингов) применяем `out_norm` + `text_linear`, спецтокены маскируем, сохраняем top-k.")
code(r"""
@torch.no_grad()
def lens_forward(lm, devs, codes, topk, save_layers=()):
    # codes: [17, T] undelayed. Returns topk ids [L+1, T, k] (int32, cpu) and saved hidden states.
    first, last = devs[0], devs[-1]
    codes = codes.unsqueeze(0).to(first)
    initial = lm._get_initial_token().to(first)
    delayed = _delay_sequence(lm.delays, codes, initial)
    seq = torch.cat([initial, delayed], dim=2)[:, :, :-1]
    x = lm.embed_codes(seq)                                   # [1, T, D]
    special = torch.tensor(SPECIAL_TEXT, device=last)

    def lens(h):
        h = h.to(last)
        logits = lm.text_linear(lm.out_norm(h)).float()[0]     # [T, V]
        logits[:, special] = -float("inf")
        return logits.topk(topk, dim=-1).indices.int().cpu()

    tops, saved = [lens(x)], {}
    if 0 in save_layers: saved[0] = x[0].to(torch.float16).cpu()
    for i, layer in enumerate(lm.transformer.layers):
        x = layer(x.to(devs[i]))
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"non-finite activations after layer {i}: try dtype='bf16' on an sm80+ GPU")
        tops.append(lens(x))
        if i + 1 in save_layers: saved[i + 1] = x[0].to(torch.float16).cpu()
    return torch.stack(tops), saved

import time
t0 = time.time()
for k, t in enumerate(turns):
    t["top"], t["hid"] = lens_forward(lm, LAYER_DEVS, t["codes"], CFG["topk"], CFG["save_layers"])
    if k == 0: print(f"first turn: {time.time()-t0:.1f}s, top shape {tuple(t['top'].shape)}")
print(f"done in {time.time()-t0:.0f}s")
""")

md("## 9. Метрика: слово пользователя в top-k\nДля слова в кадрах `[f0, f1)` смотрим позиции `[start+f0+1, start+f1+1+lag]`. Целевые токены — первый SentencePiece-кусок слова в вариантах регистра. Контроль — то же слово в окне той же длины из другой реплики.")
code(r"""
def word_targets(word):
    ids = set()
    for v in {word, word.capitalize(), word.upper()}:
        enc = sp.encode(v)
        if enc:
            ids.add(enc[0])
    return ids - set(SPECIAL_TEXT)

def window(t, w, lag):
    f0 = int(math.floor(w["start"] * FPS)); f1 = int(math.ceil(w["end"] * FPS))
    a = t["user_start"] + f0 + 1
    b = min(t["user_start"] + f1 + 1 + lag, t["top"].shape[1] - 1)
    return a, b

def best_rank(top, a, b, targets):
    # top: [L+1, T, k]. Best (lowest) rank of any target in positions [a, b], per layer; k if absent.
    if b < a or not targets:
        return None
    win = top[:, a:b + 1, :]                                    # [L+1, n, k]
    hit = torch.zeros_like(win, dtype=torch.bool)
    for tid in targets:
        hit |= win == tid
    k = top.shape[2]
    ranks = torch.where(hit, torch.arange(k).view(1, 1, k).expand_as(win), torch.full_like(win, k))
    return ranks.amin(dim=(1, 2)).numpy()                      # [L+1]

def score_all(turns, lag, n_control, seed=0):
    rng = random.Random(seed)
    rows = []
    for ti, t in enumerate(turns):
        others = [u for j, u in enumerate(turns) if j != ti]
        for w in t["words"]:
            targets = word_targets(w["word"])
            a, b = window(t, w, lag)
            r = best_rank(t["top"], a, b, targets)
            if r is None:
                continue
            ctrl = []
            for _ in range(n_control):
                u = rng.choice(others)
                if w["word"] in u["text"].split():
                    continue                                    # control turn must not contain the word
                n = b - a
                lo, hi = u["user_start"] + 1, u["top"].shape[1] - 1 - n
                if hi <= lo:
                    continue
                s = rng.randint(lo, hi)
                ctrl.append(best_rank(u["top"], s, s + n, targets))
            if not ctrl:
                continue
            rows.append(dict(turn=t["id"], word=w["word"], n_chars=len(w["word"]),
                             real=r, ctrl=np.stack(ctrl)))
    return rows

rows = score_all(turns, CFG["lag_frames"], CFG["n_control"], CFG["seed"])
print(len(rows), "scored words")
""")

code(r"""
def hit_curves(rows, k, min_chars=0):
    sel = [r for r in rows if r["n_chars"] >= min_chars]
    real = np.stack([r["real"] < k for r in sel]).mean(0)
    ctrl = np.stack([(r["ctrl"] < k).mean(0) for r in sel]).mean(0)
    return real, ctrl, len(sel)

curves = []
for k in (1, 5, 20):
    for subset, mc in (("all", 0), ("content", CFG["min_word_chars"])):
        real, ctrl, n = hit_curves(rows, k, mc)
        for layer in range(len(real)):
            curves.append(dict(k=k, subset=subset, layer=layer, real=real[layer], control=ctrl[layer], n_words=n))
curves = pd.DataFrame(curves)
curves.to_csv(f"{CFG['out_dir']}/hit_curves.csv", index=False)

best = curves[(curves.k == 5) & (curves.subset == "content")].assign(ratio=lambda d: d.real / d.control.clip(lower=1e-6))
print(best.sort_values("real", ascending=False).head(8).to_string(index=False))
""")

md("## 10. Графики")
code(r"""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
for ax, k in zip(axes, (1, 5, 20)):
    for subset, style in (("all", "--"), ("content", "-")):
        d = curves[(curves.k == k) & (curves.subset == subset)]
        ax.plot(d.layer, d.real, style, color="C0", label=f"user word ({subset})")
        ax.plot(d.layer, d.control, style, color="C7", label=f"control ({subset})")
    ax.set_title(f"hit@{k}"); ax.set_xlabel("layer (0 = embeddings)"); ax.grid(alpha=.3)
axes[0].set_ylabel("fraction of words"); axes[0].legend(fontsize=8)
plt.suptitle(f"{CFG['model_repo']}: is the user's word decodable via the text head?")
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/hit_curves.png", dpi=150); plt.show()
""")

md("### Санити-чек: что модель «сказала бы» сама\nНа последнем слое без маскировки спецтокенов модель, которая слушает, должна почти всегда выдавать PAD. Если нет — вход собран неправильно.")
code(r"""
@torch.no_grad()
def final_argmax(lm, devs, codes):
    first, last = devs[0], devs[-1]
    codes = codes.unsqueeze(0).to(first)
    initial = lm._get_initial_token().to(first)
    seq = torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], 2)[:, :, :-1]
    x = lm.embed_codes(seq)
    for i, layer in enumerate(lm.transformer.layers):
        x = layer(x.to(devs[i]))
    return lm.text_linear(lm.out_norm(x.to(last)))[0].argmax(-1).cpu()

am = final_argmax(lm, LAYER_DEVS, turns[0]["codes"])
user = am[turns[0]["user_start"]:]
print("PAD/EPAD share during user turn:", float(((user == 3) | (user == 0)).float().mean()))
print("non-pad pieces:", [sp.id_to_piece(int(i)) for i in user if int(i) not in SPECIAL_TEXT][:40])
""")

md("### Примеры: что видно в top-5 на лучшем слое")
code(r"""
L = int(best.sort_values("real", ascending=False).iloc[0].layer)
t = turns[0]
for w in t["words"][:25]:
    a, b = window(t, w, CFG["lag_frames"])
    pieces = []
    for p in range(a, b + 1):
        pieces += [sp.id_to_piece(int(i)) for i in t["top"][L, p, :5]]
    hit = bool(word_targets(w["word"]) & {sp.piece_to_id(x) for x in pieces})
    print(f"{'✓' if hit else ' '} {w['word']:>14s} | {' '.join(dict.fromkeys(pieces))[:110]}")
""")

md("## 11. Сохранение для probes\nСредние скрытые состояния по окну каждого слова на выбранных слоях — для следующего шага (линейные probes на идентичность слова, границы слов, просодию).")
code(r"""
feats = {L: [] for L in CFG["save_layers"]}
meta = []
for t in turns:
    for w in t["words"]:
        a, b = window(t, w, 0)
        if b < a:
            continue
        for L in CFG["save_layers"]:
            feats[L].append(t["hid"][L][a:b + 1].float().mean(0).numpy().astype(np.float16))
        meta.append(dict(turn=t["id"], word=w["word"], start=w["start"], end=w["end"]))
np.savez_compressed(f"{CFG['out_dir']}/word_feats.npz", **{f"layer{L}": np.stack(v) for L, v in feats.items()})
pd.DataFrame(meta).to_csv(f"{CFG['out_dir']}/word_meta.csv", index=False)
json.dump(CFG, open(f"{CFG['out_dir']}/config.json", "w"), indent=2)
print("saved", len(meta), "words x", len(feats), "layers")
""")

md("### Быстрый probe: идентичность частых слов по слоям\nЛогистическая регрессия на 30 самых частых словах (≥ 4 букв), 5-fold CV. Контроль — перемешанные метки.")
code(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

m = pd.DataFrame(meta)
vc = m[m.word.str.len() >= 4].word.value_counts()
top_words = vc[vc >= 10].head(30).index   # >= 10 examples per class for 5-fold CV
print(len(top_words), "word classes")
mask = m.word.isin(top_words).values
y = m.word[mask].values
probe = []
for L in CFG["save_layers"]:
    X = np.stack(feats[L])[mask].astype(np.float32)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    acc = cross_val_score(clf, X, y, cv=5).mean()
    acc_shuf = cross_val_score(clf, X, np.random.permutation(y), cv=5).mean()
    probe.append(dict(layer=L, acc=acc, shuffled=acc_shuf))
    print(f"layer {L:2d}: acc {acc:.3f} | shuffled {acc_shuf:.3f}")
pd.DataFrame(probe).to_csv(f"{CFG['out_dir']}/probe_word_identity.csv", index=False)
""")

md(r"""
## Как читать результат и что дальше

- **Сигнал есть**, если на средних/поздних слоях `real` заметно выше `control` (hit@5, content words). Пик по слоям — главный результат пилота: где появляется «неявная транскрипция».
- **Санити-чек** с долей PAD должен быть высоким; иначе — проблема со сборкой входа.
- **Probe** может показать идентичность слова даже там, где logit lens молчит: информация есть, но не в «текстовых координатах».

**Следующие шаги:**
1. Сравнение с базовым Moshi: `CFG["model_repo"] = "kyutai/moshiko-pytorch-bf16"` (та же архитектура; проверьте имена файлов в репозитории, у Moshi depformer меньше, но для lens он не нужен). Вопрос: изменило ли дообучение PersonaPlex то, как модель слушает.
2. Разговорные данные вместо чтения (Fisher/CANDOR или DuplexChat), русский — через FastDialogueSidon.
3. Textless-контроль: маленькая модель без текстового потока + те же probes.
4. Tuned lens вместо logit lens для ранних слоёв.

**Ограничения пилота:** нет голосового промпта; свой поток модели принудительно молчит; LibriSpeech — читаемая речь; тайм-коды Whisper, а не forced alignment.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "pilot_listen_in_text.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
