# GeoRocData

GEOROC database downloader and SQLite converter.

Automatically downloads, processes, and builds a SQLite database from
[GEOROC](https://georoc.mpch-mainz.gwdg.de/georoc/) precompiled datasets.

## Features

- **Download** all GEOROC precompiled CSV data from the new georoc.eu DOI system
- **Legacy fallback** to georoc.mpch-mainz.gwdg.de for older data
- **Fix** CSV quote errors that GEOROC data commonly has
- **Split** each CSV into data, abbreviations, and references sections
- **Build** SQLite database with deduplication

## Installation

```bash
pip install georocdata
```

## Usage

### Command Line

```bash
# Full update: download + process + build DB
georocdata

# Only download
georocdata --download-only

# Only process existing CSVs + build DB
georocdata --process-only

# Use legacy georoc.mpch-mainz.gwdg.de source
georocdata --source legacy

# Both sources
georocdata --source both

# Skip steps
georocdata --skip-fix --skip-split

# Custom output directory and DB name
georocdata --output-dir ./my_data --db-name my_georoc.db
```

### Python API

```python
from georocdata import GEOROCDownloader, GEOROCProcessor

# Download
dl = GEOROCDownloader(output_dir="./data")
dl.download_new_source()

# Process
proc = GEOROCProcessor(data_dir="./data")
proc.fix_all_csvs()
proc.split_all_csvs()
proc.build_database(db_path="./GeoRoc.db")
```

## Data Sources

GEOROC provides 14 precompiled dataset categories:

| Category | DOI |
|----------|-----|
| Archaean Cratons | 10.25625/1KRR1P |
| Complex Volcanic Settings | 10.25625/1VOFM5 |
| Continental Flood Basalts | 10.25625/WSTPOX |
| Convergent Margins | 10.25625/PVFZCE |
| Intraplate Volcanic Rocks | 10.25625/RZZ9VM |
| Oceanic Plateaus | 10.25625/JRZIJF |
| Ocean Basin Flood Basalts | 10.25625/AVLFC2 |
| Ocean Island Groups | 10.25625/WFJZKY |
| Seamounts | 10.25625/JUQK7N |
| Rift Volcanics | 10.25625/KAIVCT |
| Rocks | 10.25625/2JETOA |
| Minerals | 10.25625/SGFTFN |
| Melt Inclusions | 10.25625/7JW6XU |
| Sample Metadata | 10.25625/4EZ7ID |

## License

GPLv3+