# Project scope

## Goal

Use heterogeneous motion-capture sources to reproduce recorded human motion,
derive traceable virtual IMU, and provide synchronized visual review and HDF5
outputs.

## Version 0.1 scope

- canonical SMPL+H motion with 52 joints;
- source gender, 16 shape coefficients and original timing;
- native DMPL when present and explicit `disabled-zero` for supported
  Stage-II inputs without DMPL;
- named sensor layouts and versioned signal profiles;
- automatic structural/numerical QA followed by independent human review;
- HDF5 3.3 delivery with immutable provenance.

The repository implements direct kinematic reproduction. It does not infer
muscle activation, joint torque, contact forces, soft-tissue motion, autonomous
balance or unrecorded actions.

## Acceptance layers

| Layer | Question |
|---|---|
| Contract | Are schema, hashes, clocks and parent references closed? |
| Numerical | Are rotations, derivatives and resampling finite and stable? |
| Visual | Can pose, direction, floor and sensor placement be judged? |
| Data | Do source, selection, label and synchronized IMU agree? |
| Scientific | Does comparison with real IMU support a specific claim? |
| Publication | Do the licenses and intended use permit distribution? |

Passing one layer never implies the later layers.
