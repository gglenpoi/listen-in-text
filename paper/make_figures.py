"""Builds every figure (PDF, embedded Type-42 Times-metric (Liberation Serif) fonts) and LaTeX table of the paper from data/."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); F = os.path.join(HERE, "figures"); T = os.path.join(HERE, "tables")
os.makedirs(F, exist_ok=True); os.makedirs(T, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.3, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
TEXTW, COLW = 6.3, 3.03            # ACL: 16 cm text width, 7.7 cm column
MODELS = {"moshi": "Moshi", "personaplex": "PersonaPlex"}
MC = {"moshi": "#c4602b", "personaplex": "#2c5fa8"}
DS = {"librispeech": "LibriSpeech", "ami": "AMI"}
DLS = {"librispeech": "-", "ami": "--"}

def best_layer(m, ds):
    C = pd.read_csv(f"{D}/{m}/hit_curves_v7.csv")
    return int(C[(C.dataset == ds) & (C.cond == "listen")].sort_values("excess").iloc[-1].layer)

def save(fig, name):
    fig.savefig(f"{F}/{name}.pdf"); plt.close(fig)

# ---------------- Figure 1: where and when the user's words become text ----------------
fig, ax = plt.subplots(1, 3, figsize=(TEXTW, 1.75), gridspec_kw=dict(wspace=0.38))
for m in MODELS:
    C = pd.read_csv(f"{D}/{m}/hit_curves_v7.csv")
    for ds in DS:
        d = C[(C.dataset == ds) & (C.cond == "listen")]
        ax[0].plot(d.layer, d.excess, DLS[ds], color=MC[m], label=f"{MODELS[m]}, {DS[ds]}")
        ax[0].fill_between(d.layer, d.ci_lo, d.ci_hi, color=MC[m], alpha=.15, lw=0)
ax[0].axhline(0, color="k", lw=.5)
ax[0].set(xlabel="layer", ylabel="hit@5 excess over control", title="(a) user word readable via text head")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
for m in MODELS:
    TL = pd.read_csv(f"{D}/{m}/tuned_lens_v7.csv")
    for lens, ls, lab in (("tuned", "-", "tuned lens"), ("logit", ":", "logit lens"), ("shuffled", "--", "shuffled labels")):
        g = TL[(TL.test == "unseen_target") & (TL.lens == lens)]
        ax[1].plot(g.layer, g.hit, ls, color=MC[m], marker="o", ms=1.6)
        if lens == "tuned":
            ax[1].fill_between(g.layer, g.ci_lo, g.ci_hi, color=MC[m], alpha=.15, lw=0)
ax[1].set(xlabel="layer", ylabel="word hit@5", title="(b) never-trained target tokens")
from matplotlib.lines import Line2D
ax[1].legend([Line2D([], [], color="0.3", ls=l) for l in ("-", ":", "--")], ["tuned", "logit", "shuffled"], loc="upper left", handlelength=1.8)
for m in MODELS:
    z = np.load(f"{D}/{m}/timing_v4.npz"); ms = z["offsets"] * 80 + 40
    for ds in DS:
        ax[2].plot(ms, z[f"{ds}_end_real"][29] - z[f"{ds}_end_ctrl"][29], DLS[ds], color=MC[m], label=f"{MODELS[m]}, {DS[ds]}")
ax[2].axvline(0, color="k", lw=.5)
ax[2].set(xlabel="time rel. to word end (ms)", ylabel="hit@5 excess (layer 29)", title="(c) recognised before the word ends", xlim=(-650, 1000))
save(fig, "fig1_readout")

# ---------------- Figure 2: the voice gate ----------------
fig, ax = plt.subplots(1, 3, figsize=(TEXTW, 1.9), gridspec_kw=dict(wspace=0.36, width_ratios=[1.25, 1, 1]))
conds = ["talking_audio", "own_reversed", "own_noise", "own_echo"]
labels = ["speech", "reversed", "noise", "echo"]
x = np.arange(len(conds)); w = 0.2; k = 0
for m in MODELS:
    for ds in DS:
        L = best_layer(m, ds)
        P = pd.read_csv(f"{D}/{m}/paired_drop_v7.csv"); p = P[(P.dataset == ds) & (P.layer == L)].set_index("cond").loc[conds]
        s, lo, hi = p["drop"] / p.listen, p.ci_lo / p.listen, p.ci_hi / p.listen
        ax[0].bar(x + (k - 1.5) * w, s, w, yerr=[s - lo, hi - s], color=MC[m], alpha=1 if ds == "librispeech" else .55,
                  error_kw=dict(lw=.6, capsize=1.2), label=f"{MODELS[m]}, {DS[ds]}"); k += 1
ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylim(0, 1)
ax[0].set(ylabel="share of readout lost", title="(a) own-channel content")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
gx = [-40, -30, -20, -10, 0]
gc = ["own_gain-40", "own_gain-30", "own_gain-20", "own_gain-10", "talking_audio"]
for m in MODELS:
    for ds in DS:
        L = best_layer(m, ds)
        P = pd.read_csv(f"{D}/{m}/paired_drop_v7.csv"); p = P[(P.dataset == ds) & (P.layer == L)].set_index("cond").loc[gc]
        s, lo, hi = p["drop"] / p.listen, p.ci_lo / p.listen, p.ci_hi / p.listen
        ax[1].errorbar(gx, s, yerr=[s - lo, hi - s], fmt=DLS[ds], marker="o", ms=2.5, color=MC[m], lw=1, capsize=1.2, label=f"{MODELS[m]}, {DS[ds]}")
ax[1].set(xlabel="own-channel speech level (dB)", ylabel="share of readout lost", title="(b) own-speech level", ylim=(0, 1))
cond5 = ["talking_audio", "talking_text"]; lab5 = ["own audio\nonly", "own text\nonly"]
k = 0
for m in MODELS:
    C5 = pd.read_csv(f"{D}/{m}/hit_curves_v5.csv"); P5 = pd.read_csv(f"{D}/{m}/paired_drop_v5.csv")
    for ds in DS:
        L = int(C5[(C5.dataset == ds) & (C5.cond == "listen") & (C5.metric == "excl")].sort_values("excess").iloc[-1].layer)
        p = P5[(P5.dataset == ds) & (P5.layer == L)].set_index("cond").loc[cond5]
        s, lo, hi = p["drop"] / p.listen, p.ci_lo / p.listen, p.ci_hi / p.listen
        ax[2].bar(np.arange(2) + (k - 1.5) * w, s, w, yerr=[s - lo, hi - s], color=MC[m], alpha=1 if ds == "librispeech" else .55,
                  error_kw=dict(lw=.6, capsize=1.2)); k += 1
ax[2].set_xticks(range(2)); ax[2].set_xticklabels(lab5); ax[2].set_ylim(0, 1)
ax[2].set(ylabel="share of readout lost", title="(c) own audio vs. own text")
save(fig, "fig2_gate")

# ---------------- Figure 3: causal tests ----------------
fig, ax = plt.subplots(1, 2, figsize=(COLW, 1.75), gridspec_kw=dict(wspace=0.6))
contr = ["silence vs clean", "early own vs ctrl", "late own vs ctrl"]
clab = ["silence\nuser", "early\nablation", "late\nablation"]
k = 0
for m in MODELS:
    S = pd.read_csv(f"{D}/{m}/causal_summary_v7.csv")
    for ds in DS:
        s = S[S.dataset == ds].set_index("contrast").loc[contr]
        xx = np.arange(3) + (k - 1.5) * 0.2
        kl = s.kl.clip(lower=1e-4); lo = s.kl_lo.clip(lower=1e-5); hi = s.kl_hi
        ax[0].bar(xx, kl, 0.2, yerr=[kl - lo, hi - kl], color=MC[m], alpha=1 if ds == "librispeech" else .55, error_kw=dict(lw=.5, capsize=1))
        ax[1].bar(xx, s.logP_heard, 0.2, yerr=[s.logP_heard - s.logP_lo, s.logP_hi - s.logP_heard], color=MC[m],
                  alpha=1 if ds == "librispeech" else .55, error_kw=dict(lw=.5, capsize=1), label=f"{MODELS[m]}, {DS[ds]}")
        k += 1
ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 2)
for a in ax:
    a.set_xticks(range(3)); a.set_xticklabels(clab, fontsize=6)
ax[0].set(ylabel="KL of first answer word", title="(a) change of the answer")
ax[1].axhline(0, color="k", lw=.5)
ax[1].set(ylabel=r"$\Delta\log P$(heard words)", title="(b) heard words in answer")
fig.legend(*ax[1].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
save(fig, "fig3_causal")


# ---------------- Figure 4: behavioural consequence of the gate (v8) ----------------
fig, ax = plt.subplots(figsize=(COLW, 1.75))
bc = ["talking_audio", "own_reversed", "own_noise", "own_gain-30", "own_gain-10"]
bl = ["speech", "reversed", "noise", "speech\n$-30$ dB", "speech\n$-10$ dB"]
k = 0
for m in MODELS:
    G = pd.read_csv(f"{D}/{m}/gate_behaviour.csv")
    for ds in DS:
        g = G[(G.dataset == ds) & (G.metric == "kl")].set_index("cond").loc[bc]
        ax.bar(np.arange(len(bc)) + (k - 1.5) * 0.2, g.retention, 0.2, yerr=[g.retention - g.lo, g.hi - g.retention],
               color=MC[m], alpha=1 if ds == "librispeech" else .55, error_kw=dict(lw=.5, capsize=1), label=f"{MODELS[m]}, {DS[ds]}"); k += 1
ax.axhline(1, color="k", lw=.6, ls=":")
ax.set_xticks(range(len(bc))); ax.set_xticklabels(bl, fontsize=6)
ax.set(ylabel="user influence on reply\n(rel. to listening)", ylim=(0, 1.45))
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
save(fig, "fig4_gate_behaviour")


# ---------------- Figure 5: two-talker control, natural overlap, and where the gate is set (v9) ----------------
fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 1.85), gridspec_kw=dict(wspace=0.28, width_ratios=[1.35, 1]))
cv = [("talking_audio", "own ch.\n0 dB"), ("mix0", "user ch.\n0 dB"), ("mix-10", "user ch.\n$-10$ dB"),
      ("nat_self", "own reply\n(voice+text)"), ("nat_self_audio", "own reply\n(voice)"), ("nat_other", "other\nspeaker")]
k = 0
for m in MODELS:
    P9 = pd.read_csv(f"{D}/{m}/v9/paired_drop_v9.csv")
    for ds in DS:
        p = P9[(P9.dataset == ds) & (P9.layer == 29) & (P9.metric == "top5")].set_index("cond").loc[[c for c, _ in cv]]
        s, lo, hi = p["drop"] / p.listen, p.ci_lo / p.listen, p.ci_hi / p.listen
        ax[0].bar(np.arange(len(cv)) + (k - 1.5) * 0.2, s, 0.2, yerr=[s - lo, hi - s], color=MC[m], alpha=1 if ds == "librispeech" else .55,
                  error_kw=dict(lw=.5, capsize=1), label=f"{MODELS[m]}, {DS[ds]}"); k += 1
ax[0].axvline(2.5, color="0.5", lw=.6, ls="--")
ax[0].set_xticks(range(len(cv))); ax[0].set_xticklabels([l for _, l in cv], fontsize=5.8); ax[0].set_ylim(0, 1.05)
ax[0].set(ylabel="share of readout lost", title="(a) second voice: own vs. user channel $|$ natural overlap")
blocks = ["1-8", "9-16", "17-20", "21-24", "25-28", "1-28"]
k = 0
for m in MODELS:
    S9 = pd.read_csv(f"{D}/{m}/v9/patching_v9.csv")
    for ds in DS:
        q = S9[(S9.dataset == ds) & (S9.cond == "talking_audio") & (S9.comp == "layer")].set_index("block").loc[blocks]
        ax[1].errorbar(np.arange(len(blocks)) + (k - 1.5) * 0.12, q.restored, yerr=[q.restored - q.lo, q.hi - q.restored], fmt="o", ms=2.5,
                       color=MC[m], alpha=1 if ds == "librispeech" else .55, lw=.8, capsize=1); k += 1
ax[1].axhline(1, color="k", lw=.6, ls=":"); ax[1].axhline(0, color="k", lw=.5)
ax[1].set_xticks(range(len(blocks))); ax[1].set_xticklabels(blocks, fontsize=6); ax[1].set_ylim(-0.1, 1.15)
ax[1].set(xlabel="layers patched from listening", ylabel="share of readout restored", title="(b) where the gate is set")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
save(fig, "fig5_v9")

# ---------------- Appendix figure: recovery after the model stops talking ----------------
fig, ax = plt.subplots(figsize=(COLW, 1.6))
order = ["before", "during", "after 0–0.5 s", "after 0.5–1 s", "after 1–2 s", "after >2 s"]
k = 0
for m in MODELS:
    Dd = pd.read_csv(f"{D}/{m}/partial_dynamics.csv")
    for ds in DS:
        g = Dd[Dd.dataset == ds].set_index("bin").loc[order]
        xx = np.arange(len(order)) + (k - 1.5) * 0.15
        lo = g.ci_lo.fillna(g["drop"]); hi = g.ci_hi.fillna(g["drop"])
        ax.errorbar(xx, g["drop"], yerr=[g["drop"] - lo, hi - g["drop"]], fmt="o", ms=2.5, color=MC[m], alpha=1 if ds == "librispeech" else .55,
                    lw=.8, capsize=1, label=f"{MODELS[m]}, {DS[ds]}"); k += 1
ax.axhline(0, color="k", lw=.5); ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=25, fontsize=6)
ax.set(ylabel="hit@5 drop vs listening"); fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
save(fig, "figA_recovery")

# ---------------- Table: near-miss categories ----------------
S = pd.read_csv(f"{D}/nearmiss_summary.csv")
def cell(model, metric, L=28):
    r = S[(S.model == model) & (S.metric == metric) & (S.layer == L)]
    if not len(r): return "--"
    r = r.iloc[0]
    star = "$^{*}$" if (r.ci_lo > 0 or r.ci_hi < 0) else ""
    return f"{r.excess_same:+.3f}{star}"
cats = [("morph", "same lemma"), ("sound", "similar sound"), ("meaning", "similar meaning"), ("both", "sound + meaning"), ("homophone", "homophone")]
runs = [("moshi|librispeech|listen", "Moshi LS"), ("moshi|ami|listen", "Moshi AMI"), ("personaplex|librispeech|listen", "PPlex LS"),
        ("personaplex|ami|listen", "PPlex AMI")]
lines = [r"\begin{tabular}{l" + "r" * len(runs) + "}", r"\toprule", "Candidate & " + " & ".join(n for _, n in runs) + r" \\", r"\midrule"]
for c, lab in cats:
    lines.append(lab + " & " + " & ".join(cell(r, c) for r, _ in runs) + r" \\")
lines.append(r"\midrule")
lines.append(r"meaning (sound-unlike) & " + " & ".join(cell(r, "sem_pure") for r, _ in runs) + r" \\")
lines.append(r"sound (meaning-unlike) & " + " & ".join(cell(r, "phon_pure") for r, _ in runs) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{T}/nearmiss.tex", "w").write("\n".join(lines))

# ---------------- Table: word factors ----------------
feats = [("surprisal", "surprisal (GPT-2)"), ("zipf", "frequency (Zipf)"), ("dur", "duration"), ("n_chars", "length"), ("start", "position in turn")]
cols, coefs = [], {}
rng = np.random.default_rng(0)
for m in MODELS:
    Fw = pd.read_csv(f"{D}/{m}/word_factors.csv")
    for ds in DS:
        g = Fw[Fw.dataset == ds]; fs = [f for f, _ in feats]
        X = ((g[fs] - g[fs].mean()) / g[fs].std()).values; y = g.hit.values
        c = LogisticRegression(C=10, max_iter=3000).fit(X, y).coef_[0]
        B = np.array([LogisticRegression(C=10, max_iter=3000).fit(X[i], y[i]).coef_[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(300))])
        lo, hi = np.percentile(B, [2.5, 97.5], 0)
        coefs[(m, ds)] = (c, lo, hi, len(y)); cols.append((m, ds))
lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
         "Predictor & " + " & ".join(f"{'PPlex' if m == 'personaplex' else 'Moshi'} {'LS' if ds == 'librispeech' else 'AMI'}" for m, ds in cols) + r" \\", r"\midrule"]
for i, (f, lab) in enumerate(feats):
    row = []
    for key in cols:
        c, lo, hi, n = coefs[key]
        row.append(f"{c[i]:+.2f}" + ("$^{*}$" if lo[i] > 0 or hi[i] < 0 else ""))
    lines.append(lab + " & " + " & ".join(row) + r" \\")
lines += [r"\midrule", "$n$ words & " + " & ".join(str(coefs[k][3]) for k in cols) + r" \\", r"\bottomrule", r"\end{tabular}"]
open(f"{T}/factors.tex", "w").write("\n".join(lines))

# ---------------- numbers used in the text ----------------
NUM = {}
for m in MODELS:
    P = pd.read_csv(f"{D}/{m}/paper_table.csv")
    for _, r in P.iterrows():
        NUM[f"{m}|{r.dataset}|{r.result}|{'' if pd.isna(r.layer) else int(r.layer)}"] = [round(float(r.value), 3), round(float(r.lo), 3) if not pd.isna(r.lo) else None,
                                                                                         round(float(r.hi), 3) if not pd.isna(r.hi) else None]
json.dump(NUM, open(f"{HERE}/numbers.json", "w"), indent=1)
print("figures:", sorted(os.listdir(F)), "| tables:", sorted(os.listdir(T)))
