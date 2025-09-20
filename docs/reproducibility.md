# Reproducibility

- Fix random seeds (e.g., 42) for splits and model initialization.
- Use **GroupKFold** by `participant_id_global` to avoid leakage.
- Keep **frozen** manifests and split assignments under `Frozen Basic Data/vX_YYYY-MM-DD/`.
- Record library versions in `requirements.txt`. Consider exporting `pip freeze > outputs/pip_freeze.txt`.
