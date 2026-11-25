# Building nnU-Net datasets

`imgnet nnunet` turns downloaded NIfTI collections into an
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) raw dataset:

```console
imgnet query -c "Abdomen Atlas 3" -c CancerVerse -c MSWAL -o query_results
imgnet download query_results/query_results.csv -o imgnet_output
imgnet nnunet config.json --dry-run   # check which cases would be written
imgnet nnunet config.json -j 16       # write the dataset
```

It writes `<nnunet_raw_dir>/DatasetXXX_<name>/` containing:

| Path                  | Contents                                           |
| --------------------- | -------------------------------------------------- |
| `imagesTr/`           | `<case>_0000<file_ending>` training scans          |
| `labelsTr/`           | `<case><file_ending>` training labels              |
| `imagesTs/`           | test scans                                         |
| `labelsTs/`           | test labels (ignored by nnU-Net, used for eval)    |
| `dataset.json`        | counts only cases that were written successfully   |
| `imgnet_manifest.csv` | one row per sample: case ID, split, status, reason, notes |

## Config

Relative paths are resolved against the config file's directory. This example
builds a CT dataset of liver and kidney tumors from three CT collections:

```json
{
  "input_dir": "imgnet_output/srcdata",
  "query_results_csv": "query_results/query_results.csv",
  "nnunet_raw_dir": "nnUNet_raw",
  "dataset_id": 1,
  "dataset_name": "AbdomenLesions",
  "labels": {
    "background": 0,
    "liver tumor": 1,
    "kidney tumor": 2
  },
  "collections": [
    {
      "collection": "Abdomen Atlas 3",
      "method": "combine",
      "intensity_map": {"liver_lesion": 1, "kidney_lesion": 2}
    },
    {
      "collection": "CancerVerse",
      "method": "combine",
      "intensity_map": {"liver_lesion": 1, "kidney_lesion": 2},
      "group_column": "Patient ID"
    },
    {
      "collection": "MSWAL",
      "method": "remap",
      "intensity_map": {"3": 1, "4": 2}
    }
  ]
}
```

| Field                    | Default   | Meaning                                                      |
| ------------------------ | --------- | ------------------------------------------------------------ |
| `input_dir`              | required  | Download root; files are read from `<input_dir>/<Collection>/<filepath>` |
| `query_results_csv`      | required  | `query_results.csv` from `imgnet query`                      |
| `nnunet_raw_dir`         | required  | Your `nnUNet_raw` folder                                     |
| `dataset_id`             | required  | 1-999; must not be used by another folder in `nnunet_raw_dir` |
| `dataset_name`           | required  | Letters, digits, `_` and `-`                                 |
| `labels`                 | required  | Must contain `"background": 0`; values consecutive from 0    |
| `collections`            | required  | One entry per collection, see below                          |
| `channel_name`           | `"CT"`    | Written to `channel_names["0"]`; nnU-Net picks the normalisation from it |
| `file_ending`            | `.nii.gz` | `.nii.gz`, `.nii`, `.mha` or `.nrrd`                         |
| `test_size`              | `0.2`     | Fraction of samples (or groups) in the test set              |
| `random_state`           | `42`      | Seed for random splits (non-negative)                        |
| `prefix_with_collection` | `true`    | Case IDs are `<Collection>_<SampleID>`                       |
| `scan_link_mode`         | `"copy"`  | `"symlink"` links scans instead of copying them              |
| `geometry_tolerance`     | `0.001`   | Max mask/scan spacing, origin or direction difference        |
| `num_workers`            | all CPUs  | Worker processes; `-j` overrides it                          |

### Collection entries

| Field                | Default      | Meaning                                                    |
| -------------------- | ------------ | ---------------------------------------------------------- |
| `collection`         | required     | Name as it appears in the `Collection` column              |
| `method`             | required     | `combine` or `remap`                                       |
| `intensity_map`      | required     | combine: ROI name -> label. remap: mask value -> label     |
| `selector`           | `{}`         | Row filter for scans and masks                             |
| `scan_selector`      | `{}`         | Extra row filter for scans                                 |
| `mask_selector`      | `{}`         | Extra row filter for masks                                 |
| `roi_column`         | `"ROINames"` | combine: column holding each mask's ROI name               |
| `allow_missing_rois` | `false`      | combine: keep samples lacking some ROIs                    |
| `group_column`       | none         | Patient column; a patient's samples stay in one split      |
| `split`              | none         | Use the collection's own split column                      |

In MSWAL, mask value 3 is liver tumor and 4 is kidney tumor. MSWAL's own
`Split` column is `Tr` for every sample, so it is split randomly like the
others; see [Splits](#splits) for using a collection's own split column.

All scans in a dataset should be the same modality, because the dataset has one
input channel. If the selected scans' `Modality` values are mixed, or don't
match `channel_name`, a warning is logged.

**combine** merges binary masks (any non-zero voxel) into one label map. Where
ROIs overlap, the one listed later in `intensity_map` wins. By default a sample
missing any listed ROI is skipped, because a missing mask file would otherwise
be written as "no lesion". An ROI that has no masks anywhere in the collection
(usually a typo) stops the build and lists the ROI names that do exist.

**remap** takes a single multi-label mask and maps its values. Keys are whole
numbers (`"1"` or `"1.0"`). Values not in `intensity_map` become 0. This is
normal when you only want some of a mask's labels, so it is reported once per
collection, with per-case values in the manifest's `notes` column.
Setting `roi_column` or `allow_missing_rois` to anything but its default on a
remap entry is an error.

## Selectors

Selectors pick rows by any column of the query results, so no changes to the
index are needed. Every entry must match:

| Selector                                 | Matches rows where                |
| ---------------------------------------- | --------------------------------- |
| `{"Phase": "C+V"}`                       | `Phase` is `C+V`                  |
| `{"Phase": ["C+V", "C+A"]}`              | `Phase` is either value           |
| `{"filepath": {"regex": "_venous\\.nii"}}` | the regex is found in `filepath` |

Strings are compared with the CSV text exactly. Whole numbers also match values
written with a trailing `.0` (`{"ImageID": 1}` matches `1` and `1.0`, which
`imgnet query` writes when it combines collections), and other numbers are
compared numerically. Missing values never match.

After the selectors, each sample must have exactly one scan, and for `remap`
exactly one mask (for `combine`, one mask per ROI). If any sample has more, the
build stops before writing anything and lists examples, so you can add a
selector. For example, LLD-MMRI has one scan and mask per phase, so it needs
`"selector": {"Phase": "C+V"}`.

## Splits

By default each collection is split randomly with `test_size` and
`random_state`. The test set has `test_size` of the samples (or groups),
rounded up, but always leaves at least one for training. Set `group_column` to
a patient ID column to keep a patient's samples on one side.

Each sample (or group) gets a fixed pseudo-random rank from the seed, the
collection name and its own ID, and the lowest ranks go to the test set. So
splits are stable across rebuilds:

- adding or removing a sample moves at most one other sample (or group) across
  the split, so a test set never silently leaks into training;
- adding, removing or reordering other collections never changes a
  collection's split.

`split` uses the collection's own split column instead, for example
`{"column": "Split", "train": ["Tr"], "test": ["Ts"]}`. Every value must be
listed in `train` or `test`. If the column is missing from the query results
or has no values for that collection, a warning is logged (suggesting close
column names) and a random split is used, grouped by `group_column` if set.
When the split column is used, `group_column` is ignored, with a warning.

## Safety checks

- **Geometry:** each mask must have the same size as its scan, and spacing,
  origin and direction within `geometry_tolerance`. Differences within the
  tolerance are snapped to the scan's values; anything larger fails the case.
- **Failed cases:** a failed case is left out of `dataset.json` and its files
  are removed, including when a worker process dies (for example out of
  memory); then the remaining cases are reported as not written, with a hint
  to rerun with a lower `--num-workers`. The command exits with status 1 if any
  case failed.
- **Verified output:** after writing, every output folder is listed and must
  hold exactly the files of the cases in `dataset.json`, so `numTraining`
  matches the files on disk.
- **Case IDs:** IDs must be unique ignoring letter case, because macOS and
  Windows file systems would otherwise overwrite one case with another.
- **Missing files:** samples whose files are not on disk are skipped and listed
  in the manifest. A collection that ends up with no cases at all stops the
  build; if no files were found, check that `input_dir` points at the
  download's `srcdata` folder.
- **Existing output:** an existing, non-empty dataset folder is never written
  into, and a dataset ID used by another folder in `nnunet_raw_dir` is
  refused. Use `--overwrite` to delete and rebuild the folder.
- **Dry run:** `--dry-run` runs every check that doesn't write (case
  selection, splits, existing folder, dataset ID), so a passing dry run means
  the real run won't stop on them.
