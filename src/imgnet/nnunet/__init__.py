"""Build nnU-Net v2 raw datasets from med-imagenet query results.

Submodules:

- `imgnet.nnunet.dataset_config`: the JSON config model.
- `imgnet.nnunet.case_selection`: picks scans/masks, splits and case IDs (no I/O).
- `imgnet.nnunet.case_writer`: writes one case (runs in worker processes).
- `imgnet.nnunet.dataset_builder`: runs the whole build.

This package deliberately imports nothing at import time, so worker
processes that only need `imgnet.nnunet.case_writer` start quickly.
"""
