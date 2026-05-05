# Runtime Assets And Training Artifacts

This repo is intended to run inference from a fresh clone, but not every local
training artifact is checked in.

## Included

These files are included because they are required for normal inference:

| File | Needed for | Notes |
| --- | --- | --- |
| `auto_icon_vectorizer/runtime/nn-seg-results/best-gated-unet.pt` | stroke/outline icon mask inference | Included model checkpoint. |
| `auto_icon_vectorizer/runtime/nn-seg-results/best-filled-silhouette-unet.pt` | filled icon mask inference | Included model checkpoint. |
| `auto_icon_vectorizer/runtime/package.json` and `package-lock.json` | installing the Node Potrace wrapper | Run `python3 -m auto_icon_vectorizer.install_runtime` after cloning. |
| `auto_icon_vectorizer/runtime/*.py` | feature extraction, mask cleanup, inference, tracing | Included runtime code. |

## Not Included

These are intentionally excluded:

| Artifact | Why it is not included | How to regenerate |
| --- | --- | --- |
| `auto_icon_vectorizer/runtime/node_modules/` | generated dependency folder | `python3 -m auto_icon_vectorizer.install_runtime` |
| `auto_icon_vectorizer/runtime/nn-seg-results/feature-cache-v2.npz` | large generated stroke training feature cache | generated automatically by `train_aux_fusion_icon_segmenter.py` |
| `auto_icon_vectorizer/runtime/nn-seg-results/filled-silhouette-feature-cache-v1.npz` | large generated filled training feature cache | generated automatically by `train_filled_silhouette_segmenter.py` |
| `auto_icon_vectorizer/runtime/truth-stress-eval/` | local stroke-training corpus used during development | generate a public synthetic replacement with `scripts/generate_public_training_corpus.py` |
| `auto_icon_vectorizer/runtime/nn-training-corpus-v2/` | optional extra local stroke-training corpus | optional; use your own labeled reports if needed |
| diagnostic sheets and TSV summaries | generated during local experiments | rerun the training/eval scripts |

The feature cache files are not needed for inference. They only speed up
retraining by avoiding repeated feature extraction.

## Fresh Clone Inference Setup

```bash
git clone https://github.com/jaydenbarnescs-tech/auto-icon-vectorizer.git
cd auto-icon-vectorizer
python3 -m pip install -e .
python3 -m auto_icon_vectorizer.install_runtime
python3 -m auto_icon_vectorizer.doctor
```

If the doctor command reports a missing native Cairo dependency, install Cairo
for your platform:

```bash
# macOS
brew install cairo

# Debian/Ubuntu
sudo apt-get install libcairo2
```

Then run a smoke test:

```bash
auto-icon-vectorizer examples/sample-ai-icon-crop.png \
  --out-prefix out/sample-icon \
  --json out/sample-icon.json
```

The command prints SVG to stdout and writes `out/sample-icon.svg`,
`out/sample-icon-mask.png`, `out/sample-icon-source.png`, and diagnostics JSON.

## Regenerating Training Artifacts

Normal users do not need this. Use these commands only if you want to retrain or
modify the neural networks.

### Stroke / Outline Branch

The stroke branch expects a report file at:

```text
auto_icon_vectorizer/runtime/truth-stress-eval/latest-run.json
```

Each report points to:

- `sourceCrop`: RGB crop image
- `truthIcon`: RGBA image where alpha marks the icon mask
- `id`, `icon`, `backgroundMode`: metadata used for splits and diagnostics

Generate a public synthetic corpus:

```bash
python3 scripts/generate_public_training_corpus.py --count 240
```

Then retrain:

```bash
cd auto_icon_vectorizer/runtime
python3 train_aux_fusion_icon_segmenter.py --force-features
```

This creates:

- `nn-seg-results/feature-cache-v2.npz`
- `nn-seg-results/best-gated-unet.pt`
- `nn-seg-results/training-history.json`
- diagnostic PNG/TSV files

### Filled / Silhouette Branch

The filled branch has its own synthetic generator built into the training
script. It does not require `truth-stress-eval`.

```bash
cd auto_icon_vectorizer/runtime
python3 train_filled_silhouette_segmenter.py --force-features
```

This creates:

- `nn-seg-results/filled-silhouette-feature-cache-v1.npz`
- `nn-seg-results/best-filled-silhouette-unet.pt`
- `nn-seg-results/filled-silhouette-training-history.json`

## If You Have Your Own Training Data

Create `auto_icon_vectorizer/runtime/truth-stress-eval/latest-run.json` with
this shape:

```json
{
  "reports": [
    {
      "id": "s01-000",
      "icon": "check-circle",
      "backgroundMode": "ai-noisy",
      "sourceCrop": "/absolute/path/to/source.png",
      "truthIcon": "/absolute/path/to/truth-rgba.png"
    }
  ]
}
```

`truthIcon` should be an RGBA image. The alpha channel is treated as the ground
truth mask.

## What An AI Agent Should Do First

1. Run `python3 -m auto_icon_vectorizer.doctor`.
2. If `node_modules/potrace` is missing, run
   `python3 -m auto_icon_vectorizer.install_runtime`.
3. Do not look for `.npz` feature caches unless retraining is requested.
4. Do not retrain unless the included `.pt` checkpoints are missing or the user
   explicitly wants a new model.
5. For retraining, generate or provide `truth-stress-eval/latest-run.json`
   before running `train_aux_fusion_icon_segmenter.py`.
