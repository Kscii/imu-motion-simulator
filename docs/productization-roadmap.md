# Productization roadmap

## Components

1. **Python SDK/CLI** converts sources, derives virtual IMU, applies machine QA,
   publishes candidates and builds HDF5 outputs.
2. **Review integration** consumes immutable candidate commits and stores
   independent human quality and label revisions.
3. **Local production control** exposes job state, queue depth, errors and
   explicit export operations without embedding annotation decisions.

## Completed foundation

- content-bound motion, selection, sensor and review artifacts;
- per-clip resumable production;
- local and GCS publication stores;
- persistent upload queue with disk backpressure;
- candidate/review/snapshot contracts;
- streamed HDF5 3.3 snapshot and machine-QA-only export paths;
- loopback control API and read-only dashboard.

## Next steps

- stabilize the public Python API before version 1.0;
- add more independently validated source adapters;
- expand paired real-IMU comparisons by layout and activity;
- benchmark short-lived cloud CPU workers against local production;
- add multi-producer coordination only when more than one producer is required;
- publish versioned example datasets only when redistribution rights and
  acceptance evidence are complete.

No public release includes source motion, body models, personal data or cloud
credentials by default.
