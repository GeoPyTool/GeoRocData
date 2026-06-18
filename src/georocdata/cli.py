"""Command-line interface for georocdata."""

import argparse
import os
import sys
from datetime import datetime

from .downloader import GEOROCDownloader
from .processor import GEOROCProcessor


def main():
    parser = argparse.ArgumentParser(
        prog="georocdata",
        description="GEOROC Data Downloader and SQLite Converter",
    )
    parser.add_argument("--download-only", action="store_true",
                        help="Only download, don't process")
    parser.add_argument("--process-only", action="store_true",
                        help="Only process existing CSVs")
    parser.add_argument("--source", choices=["new", "legacy", "both"], default="new",
                        help="Data source (default: new)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for downloaded data")
    parser.add_argument("--db-name", default="GeoRoc.db",
                        help="SQLite database filename")
    parser.add_argument("--skip-fix", action="store_true",
                        help="Skip CSV quote fixing step")
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip CSV splitting step")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip database building step")
    parser.add_argument("-V", "--version", action="version",
                        version="%(prog)s 0.1.0")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d")
    if args.output_dir:
        data_dir = args.output_dir
    else:
        data_dir = os.path.join(os.getcwd(), f"georoc_data_{timestamp}")

    os.makedirs(data_dir, exist_ok=True)

    downloader = GEOROCDownloader(output_dir=data_dir)
    processor = GEOROCProcessor(data_dir=data_dir)

    if not args.process_only:
        print(f"\n{'='*60}")
        print(f"GEOROC Data Update - {timestamp}")
        print(f"Data directory: {data_dir}")
        print(f"{'='*60}\n")

        if args.source in ("new", "both"):
            downloader.download_new_source()
        if args.source in ("legacy", "both"):
            downloader.download_legacy_source()

    if args.download_only:
        print("\nDownload complete. Use --process-only to process downloaded files.")
        return

    print(f"\n{'='*60}")
    print("Processing downloaded CSV files")
    print(f"{'='*60}\n")

    if not args.skip_fix:
        print("Step 1: Fixing CSV quote errors...")
        processor.fix_all_csvs()
        print()

    if not args.skip_split:
        print("Step 2: Splitting CSV files...")
        split_dir = os.path.join(data_dir, "Split")
        processor.split_all_csvs(output_dir=split_dir)
        print()

    if not args.skip_db:
        print("Step 3: Building SQLite database...")
        data_source = os.path.join(data_dir, "Split", "Data") if not args.skip_split else data_dir
        db_path = os.path.join(data_dir, args.db_name)
        processor.build_database(data_dir=data_source, db_path=db_path)

        if not args.skip_split:
            ref_source = os.path.join(data_dir, "Split", "References")
            ref_db = os.path.join(data_dir, "Ref.db")
            processor.build_refs_database(ref_dir=ref_source, db_path=ref_db)

    print(f"\n{'='*60}")
    print("GEOROC data update complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()