# ---- Base: pinned, minimal build environment ----
FROM python:3.11-slim-bookworm AS builder

WORKDIR /qlib

# Build-time only toolchain for Cython extensions (dropped in runtime image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py MANIFEST.in README.md ./
COPY qlib ./qlib

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir cython "numpy>=1.24.0" \
    && python -m pip wheel --no-cache-dir --no-deps -w /wheels .

# ---- Runtime: minimal, non-root, no build toolchain ----
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /qlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r qlib && useradd -r -g qlib -d /qlib qlib

COPY --from=builder /wheels /wheels
COPY pyproject.toml README.md ./
COPY qlib ./qlib
COPY examples ./examples

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels /root/.cache \
    && chown -R qlib:qlib /qlib

USER qlib

ENV PYTHONUNBUFFERED=1 \
    QLIB_PROVIDER_URI=/data/qlib_data \
    QLIB_REGION=cn

# Example (mount data + env at runtime, never bake secrets into the image):
#   docker build -t qlib_image -f ./Dockerfile .
#   docker run --rm -v ~/.qlib/qlib_data:/data/qlib_data --env-file .env qlib_image \
#       python -m qlib.cli.run examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

CMD ["python", "-c", "import qlib; print('qlib container ready:', qlib.__version__)"]

