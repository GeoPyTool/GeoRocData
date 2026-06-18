"""Download GEOROC precompiled datasets."""

import os
import re
import sys
import time
import urllib.parse
import urllib.request
from .compilations import GEOROC_COMPILATIONS, LEGACY_CATEGORIES

DATA_GOETTINGEN_BASE = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:"


class GEOROCDownloader:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "georoc_data")
        self.output_dir = output_dir

    def _download_progress(self, count, block_size, total_size):
        if total_size > 0:
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()

    def _download_file(self, url, dest_path, retries=3):
        for attempt in range(retries):
            try:
                if os.path.exists(dest_path):
                    size = os.path.getsize(dest_path)
                    if size > 0:
                        print(f"  Already exists: {os.path.basename(dest_path)} ({size} bytes)")
                        return True
                print(f"  Downloading: {os.path.basename(dest_path)}")
                urllib.request.urlretrieve(url, dest_path, reporthook=self._download_progress)
                print()
                return True
            except Exception as e:
                print(f"\n  Attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
        return False

    def _parse_metadata_page(self, html, doi):
        files = []
        pattern = r'''onclick="opendownload\('(https://data\.goettingen-research-online\.de/api/access/datafile/:persistentId\?persistentId=doi:[^']+)'\)"[^>]*>([^<]+)\.csv</a>'''
        for match in re.finditer(pattern, html):
            url = match.group(1)
            filename = match.group(2) + ".csv"
            files.append({"url": url, "filename": filename, "doi": doi})

        if not files:
            pattern2 = r"persistentId=doi:" + re.escape(doi) + r"/(\w+)['\"]"
            file_ids = re.findall(pattern2, html)
            for fid in file_ids:
                url = f"{DATA_GOETTINGEN_BASE}{doi}/{fid}"
                files.append({"url": url, "filename": None, "doi": doi, "file_id": fid})

        return files

    def download_new_source(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("Downloading GEOROC data from georoc.eu (new source)")
        print("=" * 60)

        all_files = []
        for comp_name, comp_info in GEOROC_COMPILATIONS.items():
            doi = comp_info["doi"]
            metadata_url = f"https://georoc.eu/georoc/precompiled/metadata.php?doi={doi}"
            print(f"\nFetching metadata for: {comp_name} (DOI: {doi})")

            try:
                req = urllib.request.Request(metadata_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                print(f"  Failed to fetch metadata: {e}")
                continue

            files = self._parse_metadata_page(html, doi)
            if not files:
                print(f"  No files found, trying direct DOI download...")
                url = f"{DATA_GOETTINGEN_BASE}{doi}"
                comp_dir = os.path.join(output_dir, comp_name)
                os.makedirs(comp_dir, exist_ok=True)
                self._download_file(url, os.path.join(comp_dir, f"{comp_name}_full.zip"))

            comp_dir = os.path.join(output_dir, comp_name)
            os.makedirs(comp_dir, exist_ok=True)

            for f in files:
                url = f["url"]
                if f.get("filename"):
                    dest = os.path.join(comp_dir, f["filename"])
                else:
                    dest = os.path.join(comp_dir, f.get("file_id", "unknown") + ".csv")
                self._download_file(url, dest)

            all_files.extend(files)
            print(f"  Found {len(files)} files for {comp_name}")

        print(f"\nTotal files found: {len(all_files)}")
        return all_files

    def download_legacy_source(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir

        print("=" * 60)
        print("Downloading GEOROC data from georoc.mpch-mainz.gwdg.de (legacy)")
        print("=" * 60)

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        links_file = os.path.join(base_dir, "links.sh")

        if os.path.exists(links_file):
            with open(links_file, "r") as f:
                content = f.read()
            urls = re.findall(r'wget\s+(http://georoc\.mpch-mainz\.gwdg\.de/georoc/Csv_Downloads/\S+)', content)
        else:
            urls = self._generate_legacy_urls()

        for url in urls:
            parsed = urllib.parse.urlparse(url)
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 3:
                category = path_parts[-2]
                filename = path_parts[-1]
            else:
                continue

            canonical = LEGACY_CATEGORIES.get(category, category)
            comp_dir = os.path.join(output_dir, canonical)
            os.makedirs(comp_dir, exist_ok=True)
            dest = os.path.join(comp_dir, filename)
            self._download_file(url, dest)

        return urls

    def _generate_legacy_urls(self):
        base = "http://georoc.mpch-mainz.gwdg.de/georoc/Csv_Downloads"
        categories = {
            "Archean_Cratons_comp": [
                "ALDAN_SHIELD_-_ARCHEAN", "AMAZONIA_CRATON", "BALTIC_SHIELD_-_ARCHEAN",
                "BASTAR_CRATON_ARCHEAN", "BUNDELKHAND_CRATON", "CHURCHILL_PROVINCE_ARCHEAN",
                "CONGO_CRATON", "DHARWAR_CRATON_ARCHEAN", "GAWLER_CRATON",
                "KAAPVAAL_CRATON_ARCHEAN", "LIMPOPO_BELT", "NORTH_ATLANTIC_CRATON_ARCHEAN",
                "NORTH_CHINA_CRATON", "OKHOTSK-OMOLON_CRATON", "RAE_CRATON_ARCHEAN",
                "SAO_FRANCISCO_CRATON_ARCHEAN", "SARMATIAN_CRATON_ARCHEAN",
                "SIBERIAN_CRATON_ARCHEAN", "SINGHBHUM_CRATON_ARCHEAN",
                "SLAVE_PROVINCE_ARCHEAN", "SUPERIOR_PROVINCE_ARCHEAN",
                "TANZANIA_CRATON_ARCHEAN", "UKRAINIAN_SHIELD_ARCHEAN",
                "WEST_AFRICAN_CRATON", "WEST_AUSTRALIAN_CRATON", "WYOMING_PROVINCE",
                "YANGTZE_BLOCK", "ZIMBABWE_CRATON_ARCHEAN",
            ],
        }
        urls = []
        for cat, names in categories.items():
            for name in names:
                urls.append(f"{base}/{cat}/{name}.csv")
        return urls