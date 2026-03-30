# Methodology

## Overview

Med-ImageNet addresses the gap between raw, heterogeneous public oncology
imaging and the harmonized, well-documented datasets that AI practitioners
need. This page describes the principles and processes that guide data
collection, curation, and publication.

## Data Collection and Curation

### Source Selection

Collections are drawn from publicly accessible, institutional-quality
archives — primarily [TCIA](https://www.cancerimagingarchive.net/) via
the [Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/)
(IDC), with additional sources on S3, Zenodo, HuggingFace, and Dropbox.
Each collection's backend and file type are recorded in a validated
`source.json` manifest using Pydantic-enforced schemas.

### Indexing

Each collection is indexed via a per-collection `index.csv` that records
available series along with modality, body part, and study-level metadata.
For DICOM-native collections, series are identified by `SeriesInstanceUID`
and a companion `crawl_db.json` stores granular DICOM tag-level detail,
enabling fine-grained filtering via tag-based query rules. For
NIfTI-native collections, the index records file paths and modality
labels directly, supporting the same query interface without requiring
DICOM headers. Both index types are published as a versioned Hugging Face
dataset and updated via commit-tracked snapshots.

### Preprocessing

For DICOM-native collections, the optional processing flag
(`imgnet download -p`) passes raw data through the
[med-imagetools](https://github.com/bhklab/med-imagetools) Autopipeline,
which performs DICOM ingestion, voxel harmonization, intensity
normalization, and conversion to NIfTI (`.nii.gz`) with an accompanying
CSV index under `procdata/`. NIfTI-native collections are already in an
analysis-ready format and are downloaded directly, bypassing the
conversion pipeline.

## Standards and Formats

Med-ImageNet supports two internationally recognized medical imaging
formats as first-class data types:

- **DICOM** — the standard acquisition format for clinical imaging,
  used for raw data from TCIA/IDC and other clinical archives. DICOM's
  rich header metadata enables tag-based query rules for fine-grained
  series selection.
- **NIfTI** (`.nii.gz`) — a volumetric format widely adopted in
  neuroimaging and AI research, used both as the output of
  Med-ImageNet's preprocessing pipeline and as the native format for
  collections that are published directly in NIfTI (e.g., from
  HuggingFace or Zenodo sources).

Tabular metadata uses CSV, structured configuration uses JSON, and query
interoperability between CLI commands is achieved through a compact,
reproducible token format (msgpack + zlib + base64).

## FAIR Principles

The platform's architecture embodies the FAIR principles:

| Principle | Implementation |
|---|---|
| **Findable** | All collections are indexed by SeriesInstanceUID, modality, and body part; the index is published on Hugging Face with persistent identifiers. |
| **Accessible** | Open-source CLI and Python API; data retrieved from public archives via standard protocols (HTTPS, S3). |
| **Interoperable** | Both DICOM and NIfTI are domain-standard formats with broad tooling support; the unified query interface abstracts over format differences so users interact with a single API regardless of source file type. Validated Pydantic schemas enforce consistent structure across heterogeneous sources. |
| **Reusable** | GPL-3.0 licensed software; versioned index snapshots; configurable query rules enable adaptation to diverse research questions. |

