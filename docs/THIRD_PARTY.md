# Upstream dependencies

- SingLEM source: https://github.com/ttlabtuat/SingLEM ; see `configs/model/singlem.yaml` for the recorded revision and checkpoint layout. The detector uses a frozen pretrained SingLEM encoder.
- CBraMod source: https://github.com/wjq-learning/CBraMod ; see `configs/model/cbramod.yaml` for the recorded revision and checkpoint layout.
- BIOT is used by related follow-up router scripts, configured under `configs/` and loaded from external `third_party/BIOT`.
- Scientific packages and runtime dependencies are listed in `pyproject.toml` and the requirements files.

Upstream repositories, pretrained weights and datasets are not vendored. Obtain them from their maintainers and follow their licenses. Manuscript files are excluded from this public code release.

Historical oracle runtime observed locally: Python 3.11, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, joblib 1.6.0. These observations are provenance, not a complete portable lockfile or a guarantee that installing latest packages reproduces saved models exactly.
