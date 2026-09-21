# Active decisions

| ID | Decision |
|---|---|
| DEC-001 | Version 0.1 uses direct kinematic motion reproduction as the default path. |
| DEC-002 | motion-v2 is the authoritative high-dimensional motion artifact. |
| DEC-003 | Selection, IMU, review and video bind immutable parent hashes. |
| DEC-004 | Layouts and signal profiles are named and versioned; ideal signals are never overwritten by calibration. |
| DEC-005 | Structural/numerical machine QA only controls candidate eligibility; human acceptance is separate. |
| DEC-006 | Source-fit discontinuities are advisory evidence and do not silently rewrite motion. |
| DEC-007 | HDF5 3.3 snapshots are immutable and include only eligible frozen revisions. |
| DEC-008 | MP4 is generated on demand and is not required for every candidate. |
| DEC-009 | A persistent local upload queue is sufficient for a single producer; a distributed broker is deferred until multiple producers require coordination. |
| DEC-010 | Raw data, licensed models, credentials and private generated artifacts do not enter the public repository. |
| DEC-011 | Public production examples use placeholders and require explicit user configuration before cloud writes. |
