# Rolexl Qlib fork: Courage Strict V1

This repository is the only active implementation workspace for `courage_strict_v1`.
The upstream framework remote is `https://github.com/microsoft/qlib.git`; the maintained fork remote is
`https://github.com/Rolexl/qlib.git`.

## Active route

- evidence route: `docs/courage_strict_v1/evidence/COURAGE_TECHNICAL_ROUTE_STRICT_20260818.md`;
- evidence SHA-256: `ef6534868cd42e554f6dc7903e35459b42ecd21c020cf83e7340ab81938d9a91`;
- implementation decisions: `examples/courage_strict_v1/configs/courage_strict_c1_implementation_decisions_v1.json`;
- implementation-decision SHA-256: `baf373d37918b9ef91efd026e25e9f55bb720b8749ce698c4de0c11de651fe45`.

This is the new V1 implementation. Its canonical machine identity is exactly `courage_strict_v1`; do not append
`reproduction`, `migration`, `ACOT`, or another project qualifier. Historical Courage evidence and later engineering
decisions remain provenance only and must not become runtime dependencies.

## Storage

- external source data: `/data1/lxl/workspace/datasets/.tmp/dataset`;
- Xianyu source: `/data1/lxl/workspace/datasets/.tmp/dataset/xianyu`;
- governed generated data: `/data1/lxl/workspace/datasets/.tmp/qlib/data/courage_strict_v1`;
- experiment artifacts: `/data1/lxl/workspace/datasets/.tmp/qlib/artifacts/courage_strict_v1`;
- project virtual environment: `/data1/lxl/workspace/datasets/.tmp/qlib/.venv`.

No active code, configuration, catalog, command or generated artifact may resolve an ACOT path. The external raw
vendor dataset remains read-only at `/data1/lxl/workspace/datasets/.tmp/dataset/xianyu`; source facts imported during
the one-time transfer must be copied under this repository's `data/courage_strict_v1/source/` identity.

Do not commit raw data, generated Qlib bins, checkpoints, predictions, or credentials. Never create or use paths
named `AQuantLab` or `AQuantLab_raw`.

## Project boundary

Only Courage Strict V1 evidence, contracts, PIT/data kernels, V1 features/labels, Qlib dataset/model integration,
tests, and new-run reports belong here. Do not import V2, V3, Long Alpha LA0--LA10, old authorizations, old
scalers, old checkpoints, optimizer state, predictions, strategy, backtest, or trading code.

All learned state must be fitted from scratch by this Qlib V1. April 2026 is Development replay only;
May is historical-consumed evidence and June remains sealed. ACC is reported as a diagnostic, not a selection gate.
