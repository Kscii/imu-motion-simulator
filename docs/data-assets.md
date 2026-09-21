# Data and model assets

Source datasets, body models and generated studies are intentionally separate
from the source repository. A typical private library is:

```text
~/.local/share/imu-motion-simulator/library/
├── datasets/
├── models/
├── software/
├── runtime/
└── metadata/
```

The location is configurable; it is not a required absolute path.

## Required responsibilities

Users must obtain motion-capture datasets and SMPL-family assets directly from
their licensors. This repository's MIT license does not grant rights to those
files.

Keep the following outside Git:

- original dataset archives and download receipts;
- SMPL, SMPL+H, SMPL-X, DMPL and MANO files;
- generated motion, sensor, review and HDF5 outputs;
- videos, cloud credentials and service-account files.

The public acquisition manifest contains only examples whose public download
and checksum were independently recorded. It is not a blanket redistribution
license.

## Supported input roles

- SMPL+H motion archives provide pose, root translation, gender and shape.
- DMPL archives provide dynamic surface coefficients when licensed and
  available.
- BABEL contributes label candidates; candidates are not final labels.
- real paired IMU/motion data supports scientific comparison.
- BVH support demonstrates the adapter boundary and requires an explicit joint
  mapping.

Use a new output directory whenever input content, model, layout, profile or
policy changes.
