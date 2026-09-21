# IMU Motion Simulator

IMU Motion Simulator converts motion-capture sources into canonical SMPL+H
motion, derives traceable virtual IMU signals, and produces browser review
bundles and HDF5 3.3 datasets.

```text
motion source
  -> canonical SMPL+H motion-v2
  -> named sensor layout and sensors-v2
  -> machine quality gate
  -> human review
  -> immutable HDF5 3.3 snapshot
```

The default path is kinematic: it reproduces the recorded motion directly.
It does not estimate muscle forces, contact forces, autonomous balance, or
clinical validity.

## Features

- AMASS SMPL+H G and explicit 16-beta Stage-II adapters
- representative BVH import support
- preserved source clock, gender, shape and DMPL provenance
- named virtual IMU layouts with convergence checks
- machine QA separated from human acceptance
- interactive Three.js review and on-demand Blender MP4 rendering
- resumable per-clip production with local or GCS publication
- persistent upload queues for temporary network outages
- immutable reviewed and machine-QA-only HDF5 3.3 exports

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- separately licensed motion/model assets for real processing

The repository does not include AMASS recordings, SMPL-family models, personal
data, generated datasets, or cloud credentials.

## Install

For development:

```bash
git clone https://github.com/Kscii/imu-motion-simulator.git
cd imu-motion-simulator
uv sync --frozen --group dev
uv run imu-sim --help
```

Optional capabilities are installed explicitly:

```bash
uv sync --extra motion   # SMPL+H/C3D processing
uv sync --extra cloud    # Google Cloud Storage publication
uv sync --extra control  # local production API and dashboard
```

## Minimal workflow

Validate a frozen kinematic plan:

```bash
uv run imu-sim pipeline validate configs/pipeline/kinematic-accad-v3.json
```

Run it with a private asset library:

```bash
uv run imu-sim pipeline run configs/pipeline/kinematic-accad-v3.json \
  --library-root /path/to/library \
  --checkout . \
  --output /path/to/new-study
```

Serve a generated review bundle:

```bash
uv run imu-sim review serve /path/to/example.review --open
```

The production job template is
[configs/production/ubuntu-example.yaml](configs/production/ubuntu-example.yaml).
Copy it outside the repository and provide your own paths, storage bucket and
project. No default configuration writes to a real cloud target.

## Data contracts

- [Kinematic motion and sensor contracts](docs/contracts/kinematic-v2.md)
- [HDF5 3.3 delivery contract](docs/contracts/imu-hdf5-v3.3.md)
- [Architecture](docs/architecture.md)
- [Asset setup](docs/data-assets.md)
- [Development and validation](docs/development.md)
- [Current status](STATUS.md)

Machine QA only checks declared structural and numerical rules. It does not
replace visual review, source-label verification, real-device comparison, or
scientific validation.

## License

Project code is licensed under the [MIT License](LICENSE). Bundled Three.js
modules remain under their upstream MIT license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Motion-capture datasets and body models are not covered by this repository's
license. Users must obtain and use them under their respective terms.
