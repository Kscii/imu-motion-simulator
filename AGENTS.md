# Repository guidance

This repository converts motion-capture sources to canonical SMPL+H motion,
named-layout virtual IMU, review bundles and HDF5 3.3 datasets.

## Start here

1. Read `README.md` and `STATUS.md`.
2. Open only the relevant document linked from the README.
3. Verify changing facts with code, tests and generated artifacts.
4. Keep raw data, licensed models, credentials and generated studies outside
   Git.

## Invariants

- The main flow is `source -> motion-v2 -> selection -> sensors-v2 -> review
  -> HDF5 3.3`.
- Preserve the source clock, gender, 16 shape coefficients and SMPL+H joint
  identity.
- Missing Stage-II DMPL must be declared as `disabled-zero`; do not invent it.
- Sensor layout and profile identities are explicit.
- Motion, sensors, review and video artifacts bind their parent hashes.
- Machine QA does not imply human, scientific or medical acceptance.
- Never overwrite source files or immutable published artifacts.
- Public examples must use placeholder paths and storage identities.

## Validation

```bash
uv run --frozen pytest -q
uv run --frozen imu-sim docs check
uv build --frozen
```

Report code tests, real-data checks, visual review and publication validation
separately. A local commit does not authorize a push, release or data
publication.
