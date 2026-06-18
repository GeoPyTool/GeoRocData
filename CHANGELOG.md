# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-18

### Added

- **New GEOROC data source support**: Downloads from `georoc.eu` DOI-based system
  (`data.goettingen-research-online.de`), replacing the deprecated
  `georoc.mpch-mainz.gwdg.de/Csv_Downloads/` individual CSV downloads.
  The new source provides data generated from the GEOROC database as of 2026-06-01.
- **14 precompiled dataset categories** with DOI mapping:
  - Archaean Cratons (`10.25625/1KRR1P`)
  - Complex Volcanic Settings (`10.25625/1VOFM5`)
  - Continental Flood Basalts (`10.25625/WSTPOX`)
  - Convergent Margins (`10.25625/PVFZCE`)
  - Intraplate Volcanic Rocks (`10.25625/RZZ9VM`)
  - Oceanic Plateaus (`10.25625/JRZIJF`)
  - Ocean Basin Flood Basalts (`10.25625/AVLFC2`)
  - Ocean Island Groups (`10.25625/WFJZKY`)
  - Seamounts (`10.25625/JUQK7N`)
  - Rift Volcanics (`10.25625/KAIVCT`)
  - Rocks (`10.25625/2JETOA`)
  - Minerals (`10.25625/SGFTFN`)
  - Melt Inclusions (`10.25625/7JW6XU`)
  - Sample Metadata (`10.25625/4EZ7ID`)
- **Python package** `georocdata` published to PyPI (https://pypi.org/project/georocdata/)
- **CLI entry point** `georocdata` command with full argument support
- **Python API**: `GEOROCDownloader` and `GEOROCProcessor` classes for programmatic use
- **CSV quote fixing**: Automatically detects and fixes isolated quote errors common in
  GEOROC CSV files
- **CSV section splitting**: Splits each GEOROC CSV into three sections (data,
  abbreviations, references) based on `Abbreviations:` and `References:` markers
- **SQLite database builder**: Merges all processed CSV data into a single SQLite
  database with automatic deduplication and incremental import tracking
- **References database**: Separate `Ref.db` for references data
- **Legacy source fallback**: `--source legacy` option to download from the old
  `georoc.mpch-mainz.gwdg.de` server (still using `links.sh` URL list)
- **Skip-step flags**: `--skip-fix`, `--skip-split`, `--skip-db` for selective execution
- **Download-only and process-only modes**: `--download-only` and `--process-only`

### Changed

- **Data source migrated**: The old GEOROC website
  (`georoc.mpch-mainz.gwdg.de`) has been superseded by `georoc.eu` with a new
  DOI-based download system. File names now include date prefixes and DOI identifiers
  (e.g., `2026-06-1KRR1P_ALDAN_SHIELD_ARCHEAN.csv` instead of `ALDAN_SHIELD_-_ARCHEAN.csv`).
- **Category naming**: `Inclusions_comp` → `Melt_Inclusions`,
  `Intraplate_Volcanics_comp` → `Intraplate_Volcanic_Rocks`,
  `Archean_Cratons_comp` → `Archaean_Cratons` (matching new GEOROC naming)

### Fixed

- Fixed CSV parsing with `ISO-8859-1` and `Windows-1252` encoding fallback
- Fixed regex for parsing GEOROC metadata pages to handle both DOI link formats
- Fixed PyPI classifier validation (`Topic :: Scientific/Engineering` instead of
  `Topic :: Scientific/Engineering :: Earth Sciences`)

### Deprecated

- `links.sh` and `back.sh` scripts are deprecated in favor of the `georocdata` CLI
- `DB_Gen/0_Get_Data_Fix_Quota_Error.py` through `4_REF_to_DB.py` scripts are
  superseded by the `georocdata` package

## [0.1.0] - 2026-06-18

### Added

- Initial package structure with `src/georocdata/` layout
- `pyproject.toml` with setuptools build backend
- CLI via `georocdata` command
- Python API via `from georocdata import GEOROCDownloader, GEOROCProcessor`
- Published to PyPI as `georocdata`

[0.2.0]: https://github.com/GeoPyTool/GeoRocData/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GeoPyTool/GeoRocData/releases/tag/v0.1.0