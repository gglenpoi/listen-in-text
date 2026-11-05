# Do Full-Duplex Speech Models Listen in Text?

**A latent transcript of the user and a voice gate that closes it.**
Code, notebooks, results and paper sources for the study by Mikhail Kulik (Independent Researcher).

Full-duplex speech models such as [Moshi](https://arxiv.org/abs/2410.00037) and [PersonaPlex](https://arxiv.org/abs/2602.06053) listen and speak at the same time, but emit text only for their own speech. Using logit and tuned lenses on their 32-layer backbones, we find:

| Finding | Result |
|---|---|
| Latent transcript of the user | The text head reads the user's current word in layers 21–30 for 46–48% of content words above control (LibriSpeech and AMI), while the model's output stays silent. It vanishes for time-reversed speech. |
| Timing and content | The word becomes readable ~150 ms before it ends. Predictable and frequent words are read out more often. Errors share form (lemma or sound) rather than meaning. A tuned lens finds a word code from layer 12, including target tokens never seen in training. |
| Voice gate | While the model speaks, 50–77% of the readout of the user is lost. The gate is triggered by speech-like audio in the model's **own** channel: time-reversed speech and the model's own generated replies trigger it, but noise, an echo of the user, or a second talker in the user's channel hardly do. |
| Locus of the gate | Patching layers 1–8 from the listening run restores 80–92% of the readout, long before the transcript becomes readable (layer 21). |
| Readable is not used | Ablating the latent words changes the model's reply by only 1–6% of the effect of muting the user. The gate's behavioural effect is weaker and less specific than its representational effect. |

All experiments ran on free Kaggle notebooks (2× NVIDIA T4, ~13 GPU-hours in total).

## Repository layout

```
notebooks/   Kaggle notebooks, in the order they were run (v1 pilot … v9)
builders/    Python scripts that generate each notebook (build_nb_v9.py -> listen_in_text_v9.ipynb)
tests/       CPU smoke tests: tiny random model, planted effects, stubbed downloads
paper/       LaTeX sources (ACL format), figure script, result tables used in the paper
  data/      per-model CSV/JSON outputs of the notebooks (v4–v9)
```

| Notebook | What it tests |
|---|---|
| `v1_pilot`, `listen_in_text_v2` | Logit-lens readout of the user, timing, semantic vs phonetic neighbours |
| `nearmiss_v3`, `nearmiss_v31` | Near-miss categories with same-turn control and threshold sweep (CPU) |
| `listen_in_text_v4` | AMI replication, MMS alignment, reversed speech, talking condition, tuned lens, word factors |
| `listen_in_text_v5` | Own audio vs own text, own-token exclusion, causal ablation |
| `listen_in_text_v6` | Own reversed speech / noise / gain, own-channel readout, forced response |
| `listen_in_text_v7` | Final readout, gate and causal runs used in the paper |
| `listen_in_text_v8` | Behavioural test of the gate (forced reply) |
| `listen_in_text_v9` | Natural overlap with the model's own generated speech, two-talker control, activation patching |

## Reproducing

**Notebooks.** Open a notebook on Kaggle with *GPU T4 ×2* and *Internet on*. Add a Hugging Face token as the secret `HF_TOKEN` (the PersonaPlex weights require accepting the licence). Set `model = "personaplex"` or `"moshi"` in the first cell, then run all cells. Each run writes a zip of its outputs to `/kaggle/working`.

**Figures and tables of the paper.**
```bash
cd paper
pip install numpy pandas matplotlib scikit-learn
python make_figures.py          # writes figures/*.pdf and tables/*.tex from data/
latexmk -pdf preprint.tex       # named preprint; main.tex builds the anonymous review version
```

**Tests.** The tests need the `moshi` package from the PersonaPlex repository on `PYTHONPATH`, then run e.g. `python tests/test_v9.py notebooks/listen_in_text_v9.ipynb`.

## Data and models

- [LibriSpeech](https://www.openslr.org/12) (CC BY 4.0)
- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (CC BY 4.0)
- [Moshi](https://github.com/kyutai-labs/moshi)
- [PersonaPlex](https://github.com/NVIDIA/personaplex)

Model weights are downloaded from Hugging Face under their respective licences and are not redistributed here.

## Paper

Preprint PDF: [`paper/preprint.pdf`](paper/preprint.pdf) (arXiv link to follow).

## Citation

```bibtex
@misc{kulik2026listenintext,
  title  = {Do Full-Duplex Speech Models Listen in Text? A Latent Transcript of the User and a Voice Gate That Closes It},
  author = {Kulik, Mikhail},
  year   = {2026},
  note   = {Preprint}
}
```

## Licence

Code: MIT (see `LICENSE`). Result tables are derived from CC BY 4.0 corpora.
