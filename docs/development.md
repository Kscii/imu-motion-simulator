# Development and validation

## Environment

```bash
uv sync --frozen --group dev
uv run imu-sim --help
```

Optional dependencies:

```bash
uv sync --extra motion
uv sync --extra cloud
uv sync --extra control
```

Python support is currently limited to 3.12.

## Checks

```bash
uv run --frozen pytest -q
uv run --frozen imu-sim docs check
uv build --offline
git diff --check
```

Tests that compare independent sibling implementations are optional. Set
`IMU_COLLECTOR_REPO`, `IMU_BENCHMARK_REPO` or `IMU_LEGACY_PYTHON` to
enable those checks; no workstation-specific path is assumed.

## Kinematic processing

Validate plans before processing:

```bash
uv run imu-sim pipeline validate PLAN.json
```

Every real run uses separately licensed assets through `--library-root`.
Use `--through convert`, `sensors` or `review` to stop at a declared
stage. Reusing an output directory is allowed only for the same immutable plan
and input identities.

## Production

Copy `configs/production/ubuntu-example.yaml` to a private location and
replace all placeholder paths and cloud identifiers. Run:

```bash
imu-sim production validate JOB.yaml
imu-sim production run JOB.yaml
imu-sim production status JOB.yaml
```

The durable upload profile separates local computation from network transfer.
Backpressure is based on pending bytes and reserved free disk space. Cloud
credentials come from the normal client-library environment and must never be
placed in the YAML or committed.

The optional control service and dashboard bind loopback by default. Exposing
them beyond the host requires a separate authentication and deployment review.

## Evidence boundaries

Unit tests establish code behavior. Real-data runs establish compatibility with
specific licensed inputs. Browser observation establishes visible playback.
Real-device comparison establishes only the tested sensor/layout conditions.
Publication and clinical claims require separate acceptance.
