# Getting Started

## Installation

```console
pip install med-imagenet
```

```console
imgnet --help
```

## Quick Start

### 1. Update the dataset index

On first use, Med-ImageNet downloads its collection index from HuggingFace.
You can also force an update:

```console
imgnet update-index
```

### 2. Browse available collections

```console
imgnet collections summary
```

Get details on a specific collection:

```console
imgnet collections info 4D-Lung
```

### 3. Query for datasets

Find CT scans with associated RTSTRUCTs in the 4D-Lung collection:

```console
imgnet query -c 4D-Lung -m CT,RTSTRUCT
```

### 4. Download and process

Pipe query results directly into the download command:

```console
imgnet query -c 4D-Lung -m CT,RTSTRUCT | imgnet download -p
```

The `-p` flag runs the Med-ImageTools autopipeline on downloaded DICOM files,
producing AI-ready NIfTI outputs.

## Output Layout

```
imgnet_output_YYYYMMDD_HHMMSS/
├── srcdata/                                        # Raw downloaded DICOM files
│   └── 4D-Lung/
│       └── 119_HM10395/                            # Patient ID
│           └── <StudyInstanceUID>/
│               ├── CT_<SeriesInstanceUID>/          # DICOM series folders
│               │   └── *.dcm
│               └── RTSTRUCT_<SeriesInstanceUID>/
│                   └── *.dcm
├── procdata/                                       # AI-ready processed outputs
│   └── 4D-Lung/
│       ├── 4D-Lung_index.csv                       # Per-collection metadata index
│       ├── 0000__119_HM10395/                      # Subject folder
│       │   ├── CT_17647495/
│       │   │   └── CT.nii.gz                       # Processed CT
│       │   └── RTSTRUCT_62947.72/
│       │       ├── ROI__[Carina_c10].nii.gz        # Processed RTSTRUCT 
│       │       ├── ROI__[LN2_c10].nii.gz
│       │       ├── ROI__[LN_c10].nii.gz
│       │       ├── ROI__[Tumor_c10].nii.gz
│       │       └── ROI__[Vertebra_c10].nii.gz
│       ├── 0001__119_HM10395/
│       │   └── ...
│       └── ...                                     
```
