# Project status

The public baseline is version 0.1.0 and exposes the direct kinematic
motion-to-IMU path.

## Available

- canonical SMPL+H motion-v2 conversion;
- explicit selections and named sensor layouts;
- ideal and calibrated sensors-v2 derivation;
- machine QA and candidate-corpus generation;
- Three.js review bundles and on-demand MP4 rendering;
- resumable per-clip production and persistent upload queues;
- reviewed snapshots and machine-QA-only HDF5 3.3 exports;
- local production API and read-only dashboard.

## Maturity

The core contracts, test fixtures and offline package build are automated.
The API is still pre-1.0 and may change between minor versions.

The repository does not ship source datasets or SMPL-family assets. Real-data
runs require separately licensed inputs and explicit local configuration.

Machine acceptance is not human acceptance. Generated IMU has not been
clinically validated and must not be represented as medical ground truth.
