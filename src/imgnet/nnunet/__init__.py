"""Build nnU-Net v2 raw datasets from med-imagenet query results.

Submodules:

- `imgnet.nnunet.config`: the JSON config model.
- `imgnet.nnunet.plan`: picks scans/masks, splits and case IDs (no I/O).
- `imgnet.nnunet.images`: writes one case (runs in worker processes).
- `imgnet.nnunet.build`: runs the whole build.

This package deliberately imports nothing at import time, so worker
processes that only need `imgnet.nnunet.images` start quickly.
"""
