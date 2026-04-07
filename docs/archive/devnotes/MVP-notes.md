# scaled down minimum viable product

date: 2025-06-19

inspiration: `idc-index`

```console
idc-index download <PatientID>/<SeriesInstanceUID>/<Collection>
```

```console
med-imagenet download <PatientID>/<SeriesInstanceUID>/<Collection>
```

- if they have username and password for private data

```console
med-imagenet download  <PatientID>/<SeriesInstanceUID>/<Collection> --nbia_username <x> --nbia_password <x> 
```

- if they only want modalities

```console
med-imagenet download <Collection> --modalities "CT,RTSTRUCT" --run-<autopipeline>
```

- this should only download rtstructs with GTVp *and* their references
```console
med-imagenet download <Collection> --query RTSTRUCTS where "GTVp" in "ROINames" \
  # optionally, if i.e CT,PT,RTSTRUCT and CT,RTSTRUCT both exist, and we only want the CT,RTSTRUCT:
  --modalities "CT,RTSTRUCT"
```


complicated but useful approach:

```console
med-imagenet query <Collection> <QUERY> | med-imagenet download <--from-file> 
```

download stuff:
- check if the data already exists
- be able to download 'newer' series if projects are ongoing


database building:
- use something like tcia's [updated series endpoint](https://github.com/jjjermiah/nbia-toolkit/blob/d8f4bb401584d3d94072ed8708eddc5936b5fda2/src/nbiatoolkit/utils/nbia_endpoints.py#L41) to only update the new series every month or so
- to prevent running a huge job

# TODO:

- Setup releases on this repo for every database scraped from TCIA (crawled)
  - options for this:
    - download everything and crawl :(
    - be able to scrape from the tcia 'GetDicomTags' Endpoint > create DICOM file (without pixel data)
- Setup a way to query the crawl db
  - MVP: some minimal query options, to be extended in the future
- Use the output of the query with NBIAToolkit to download


## Stretch-goal

create a quick website using streamlit that uses the crawl db and lets users investigate data 
- query the db really quickly
- plot some metrics
- potentially generate manifest easily








