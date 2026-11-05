"""Builds listen_in_text_v8.ipynb: behavioural test of the voice gate (forced response under own-channel speech)."""
import sys
import nbformat as nbf

HERE = __file__.replace("build_nb_v8.py", "")
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
# Do Full-Duplex Speech Models Listen in Text? — v8: поведенческий тест «голосового шлюза»

v7 показал: голос в собственном канале модели подавляет **текстовое представление** собеседника, но это представление почти не используется для ответа. Значит, нужно прямо проверить, теряет ли **ответ** информацию об услышанном, когда модель говорит одновременно с собеседником.

**Дизайн.** Для каждой реплики и каждого варианта собственного канала (как в v7) делаем два прогона: собеседник говорит / собеседник заглушён. Через 0,5 с после реплики в собственный текст модели ставится EPAD; распределение первого слова ответа сравнивается:
- **влияние собеседника** `U(c) = KL(p[c, собеседник есть] ‖ p[c, собеседник заглушён])`;
- **вероятность услышанных слов** `Δlog P(c) = log P_heard[c, есть] − log P_heard[c, заглушён]`.

**Сохранность информации** `retention(c) = U(c) / U(listen)` (и то же для `Δlog P`): 1 — ответ зависит от слов собеседника так же, как когда модель молча слушала; 0 — ответ не зависит от них вовсе. Сравнение с заглушённым собеседником внутри того же условия убирает влияние самой собственной речи на ответ.

Выравнивание слов не нужно (используется эталонный текст), прогон ~30–45 мин на модель. Kaggle: GPU T4×2, Internet On, `HF_TOKEN`.
""")

md("## 1. Конфигурация")
code(r"""
import os
ON_KAGGLE = os.path.exists("/kaggle/working")
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")
BASE = "/kaggle/working" if ON_KAGGLE else ("/content" if ON_COLAB else os.getcwd())

GAINS_DB = [-30, -10]
CFG = dict(
    model          = "personaplex",              # "personaplex" | "moshi"
    datasets       = ["librispeech", "ami"],
    conditions     = ["listen", "talking_audio", "own_reversed", "own_noise", "own_echo", "own_gain-30", "own_gain-10"],
    echo_delay_sec = 0.2,
    echo_gain_db   = -10,
    n_turns        = 60,
    turn_sec       = 20.0,
    gap_sec        = 0.3,
    silence_frames = 6,
    post_sec       = 1.5,
    resp_gap_sec   = 0.5,
    min_word_chars = 4,
    n_boot         = 2000,
    tl_extra       = [],                          # no tuned-lens data needed here
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
CFG["out_dir"] = f"{BASE}/{CFG['model']}_v8"
os.makedirs(CFG["out_dir"], exist_ok=True)
print(CFG["model_repo"], "->", CFG["out_dir"])
""")

md("## 2. Установка, модель, данные")
code(pick(V5, "import sys, subprocess"))
code(pick(V5, "import json, math, random, re, tarfile"))
code(pick(V5, "from huggingface_hub import hf_hub_download, list_repo_files"))
code(pick(V5, "def load_lm_for_lens"))
code(pick(V5, "def librispeech_turns"))

md("## 3. Варианты собственного канала (как в v7)")
code(pick(V7, "def speech_shaped_noise"))

md("## 4. Вынужденный ответ: собеседник есть / заглушён")
code(r"""
SPECIAL_T = None
@torch.no_grad()
def first_word(codes, final_pos):
    first, last = LAYER_DEVS[0], LAYER_DEVS[-1]
    c = codes.unsqueeze(0).to(first)
    initial = lm._get_initial_token().to(first)
    seq = torch.cat([initial, _delay_sequence(lm.delays, c, initial)], 2)[:, :, :-1]
    x = lm.embed_codes(seq)
    for i, layer in enumerate(lm.transformer.layers):
        x = layer(x.to(LAYER_DEVS[i]))
    if not torch.isfinite(x).all():
        raise FloatingPointError("non-finite activations")
    return torch.log_softmax(lm.text_linear(lm.out_norm(x.to(last))).float()[0, final_pos], -1).cpu()

def turn_ids(t):
    ids = set()
    for _, _, text in t["segments"]:
        for w in norm_words(text):
            if len(w) >= CFG["min_word_chars"]:
                enc = sp.encode(w)
                if enc and enc[0] not in (0, 1, 2, 3):
                    ids.add(enc[0])
    return ids

GAP = int(round(CFG["resp_gap_sec"] * FPS))
rows = []
t0 = time.time()
for name in MAIN:
    turns = DATA[name]
    ids_all = [turn_ids(t) for t in turns]
    for ti, t in enumerate(turns):
        heard = torch.tensor(sorted(ids_all[ti]))
        if len(heard) < 3: continue
        other = turns[(ti + 1) % len(turns)]
        n = t["codes_user"].shape[1]
        silent_user = SIL.repeat(1, n)
        for cond in CFG["conditions"]:
            own_codes, own_text, _ = own_streams(other, n, cond, self_=t)
            out = {}
            for user_on in (True, False):
                codes, start = build_codes(t["codes_user"] if user_on else silent_user, own_codes, own_text, N_POST)
                b = start + n; e = b + GAP
                codes[0, e] = 0                                     # forced EPAD: the answer starts on the next frame
                lp = first_word(codes, list(range(b, e + 2)))
                out[user_on] = lp
            on, off = out[True], out[False]
            rows.append(dict(dataset=name, ti=ti, cond=cond,
                             kl=torch.sum(on[-1].exp() * (on[-1] - off[-1])).item(),
                             logp_on=float(torch.logsumexp(on[-1][heard], 0)), logp_off=float(torch.logsumexp(off[-1][heard], 0)),
                             p_epad_on=on[:-1, 0].exp().mean().item(), p_epad_off=off[:-1, 0].exp().mean().item(),
                             top1_on=int(on[-1].argmax()), top1_off=int(off[-1].argmax())))
    print(f"{name}: done ({time.time() - t0:.0f}s)")
R = pd.DataFrame(rows); R.to_csv(f"{CFG['out_dir']}/gate_behaviour_rows.csv", index=False)
""")

md("## 5. Сохранность информации об услышанном")
code(r"""
boot = np.random.default_rng(CFG["seed"])
R["dlogp"] = R.logp_on - R.logp_off
res = []
for name, g in R.groupby("dataset"):
    base = g[g.cond == "listen"].set_index("ti")
    for cond, h in g.groupby("cond"):
        h = h.set_index("ti").loc[base.index]
        for metric in ("kl", "dlogp"):
            num, den = h[metric].values, base[metric].values
            idx = boot.integers(0, len(num), (CFG["n_boot"], len(num)))
            ratios = num[idx].mean(1) / den[idx].mean(1)
            res.append(dict(dataset=name, cond=cond, metric=metric, value=num.mean(), listen=den.mean(),
                            retention=num.mean() / den.mean(), lo=np.percentile(ratios, 2.5), hi=np.percentile(ratios, 97.5), n=len(num)))
        res.append(dict(dataset=name, cond=cond, metric="p_epad_on", value=h.p_epad_on.mean(), listen=base.p_epad_on.mean(),
                        retention=np.nan, lo=np.nan, hi=np.nan, n=len(h)))
RES = pd.DataFrame(res); RES.to_csv(f"{CFG['out_dir']}/gate_behaviour.csv", index=False)
pd.set_option("display.width", 200)
for name in MAIN:
    d = RES[(RES.dataset == name) & RES.metric.isin(["kl", "dlogp"])]
    print(f"\n=== {name}: information about the user's words retained in the answer (1 = as when listening) ===")
    print(d.pivot_table(index="cond", columns="metric", values=["value", "retention", "lo", "hi"]).round(3).to_string())
    e = RES[(RES.dataset == name) & (RES.metric == "p_epad_on")].set_index("cond").value
    print("P(EPAD) before the forced onset (readiness to speak):", e.round(4).to_dict())

order = [c for c in CFG["conditions"]]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for ax, metric, lab in zip(axes, ("kl", "dlogp"), ("KL(user on ‖ user silenced)", "Δ log P(heard words)")):
    for k, name in enumerate(MAIN):
        d = RES[(RES.dataset == name) & (RES.metric == metric)].set_index("cond").loc[order]
        x = np.arange(len(order)) + (k - 0.5) * 0.35
        ax.bar(x, d.retention, 0.35, yerr=[d.retention - d.lo, d.hi - d.retention], capsize=3, label=name)
    ax.axhline(1, color="k", lw=.8, ls=":"); ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20)
    ax.set(title=f"{CFG['model']}: retention of user information in the answer ({lab})", ylabel="relative to listen")
    ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(f"{CFG['out_dir']}/gate_behaviour.png", dpi=150); plt.show()
""")

md("## 6. Сохранение")
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
- `retention` ≈ 1 для `own_*`: модель продолжает учитывать слова собеседника, хотя их текстовое представление подавлено → «слышит, но не транскрибирует».
- `retention` заметно < 1 для речи и обращённой речи, но ≈ 1 для шума и эха (как в v7) → голосовой шлюз имеет поведенческие последствия.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "listen_in_text_v8.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
