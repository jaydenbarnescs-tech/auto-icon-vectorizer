# Runtime

This directory contains the inference runtime used by `auto_icon_vectorizer`.

Files:

- `trace_icon_component.py`: mask cleanup, Potrace invocation, SVG normalization,
  SVG rendering, and visual-diff scoring helpers.
- `train_aux_fusion_icon_segmenter.py`: stroke gated U-Net feature stack and
  model architecture.
- `train_filled_silhouette_segmenter.py`: filled silhouette feature stack,
  model architecture, synthetic generator, and inference cleanup.
- `apply_svm_connections.py`: visual-diff helpers plus earlier SVM connection
  experiments; SVM is not part of the default renderer.
- `generate_spectral_evidence_bank.py`: evidence-map helpers imported by the
  stroke model feature stack.
- `nn-seg-results/best-gated-unet.pt`: stroke model checkpoint.
- `nn-seg-results/best-filled-silhouette-unet.pt`: filled model checkpoint.
- `package.json` / `package-lock.json`: Node Potrace dependency.

Not included:

- `node_modules/`: generated with `python3 -m auto_icon_vectorizer.install_runtime`.
- `nn-seg-results/*.npz`: generated feature caches used only for retraining.
- `truth-stress-eval/` and `nn-training-corpus-v2/`: local training corpora.

See `docs/RUNTIME_ASSETS.md` for setup, doctor checks, and retraining commands.
