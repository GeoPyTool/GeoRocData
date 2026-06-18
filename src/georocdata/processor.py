"""Process GEOROC CSV files: fix, split, and build database."""

import csv
import os
import sqlite3

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class GEOROCProcessor:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.getcwd(), "georoc_data")
        self.data_dir = data_dir

    @staticmethod
    def fix_csv_line(line):
        pattern = r'(?<!")"(?!")'
        return re.sub(pattern, '', line) if 're' in dir() else line

    @staticmethod
    def fix_csv_file(file_path):
        import re as _re
        encodings = ['utf-8-sig', 'utf-8', 'ISO-8859-1', 'Windows-1252']
        rows = None
        used_enc = None

        for enc in encodings:
            try:
                with open(file_path, 'r', newline='', encoding=enc) as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                used_enc = enc
                break
            except (UnicodeDecodeError, Exception):
                continue

        if rows is None:
            print(f"  Cannot read: {file_path}")
            return

        has_extra_quotes = any('"' in field for row in rows for field in row)
        if not has_extra_quotes:
            return

        print(f"  Fixing quotes in: {os.path.basename(file_path)}")
        pattern = r'(?<!")"(?!")'
        with open(file_path, 'w', newline='', encoding=used_enc) as f:
            writer = csv.writer(f)
            for row in rows:
                fixed_row = [_re.sub(pattern, '', field) for field in row]
                writer.writerow(fixed_row)

    def fix_all_csvs(self, data_dir=None):
        if data_dir is None:
            data_dir = self.data_dir
        csv_files = self._find_csvs(data_dir)
        print(f"Fixing {len(csv_files)} CSV files...")
        for i, path in enumerate(csv_files):
            print(f"  [{i+1}/{len(csv_files)}] {os.path.basename(path)}")
            self.fix_csv_file(path)

    def split_csv_file(self, file_path, output_base_dir=None):
        if output_base_dir is None:
            output_base_dir = os.path.join(self.data_dir, "Split")

        data_dir = os.path.join(output_base_dir, "Data")
        abbrev_dir = os.path.join(output_base_dir, "Abbreviations")
        ref_dir = os.path.join(output_base_dir, "References")
        for d in [data_dir, abbrev_dir, ref_dir]:
            os.makedirs(d, exist_ok=True)

        dir_path, file_name = os.path.split(file_path)
        base_name = file_name.replace('.csv', '')

        encodings = ['utf-8-sig', 'utf-8', 'ISO-8859-1', 'Windows-1252']
        lines = None
        used_encoding = None
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    lines = f.readlines()
                used_encoding = enc
                break
            except (UnicodeDecodeError, Exception):
                continue

        if lines is None:
            print(f"  Cannot decode: {file_path}")
            return None, None, None

        abbrev_idx = None
        ref_idx = None
        for i, line in enumerate(lines):
            if 'Abbreviations:' in line:
                abbrev_idx = i
            if 'References:' in line:
                ref_idx = i

        if abbrev_idx is None or ref_idx is None:
            if HAS_PANDAS:
                try:
                    data = pd.read_csv(file_path, encoding=used_encoding, engine='python', on_bad_lines='warn')
                    data.insert(0, 'Type', base_name.replace('_data', ''))
                    out = os.path.join(data_dir, f"{base_name}_data.csv")
                    data.to_csv(out, index=False, encoding=used_encoding)
                    return out, None, None
                except Exception:
                    pass
            return file_path, None, None

        data_end = abbrev_idx - 1
        abbrev_end = ref_idx - 1

        data_out = os.path.join(data_dir, f"{base_name}_data.csv")
        abbrev_out = os.path.join(abbrev_dir, f"{base_name}_abbreviations.csv")
        ref_out = os.path.join(ref_dir, f"{base_name}_references.csv")

        if HAS_PANDAS:
            try:
                df_data = pd.read_csv(file_path, skiprows=0, nrows=data_end,
                                       encoding=used_encoding, engine='python', on_bad_lines='warn')
                df_data.insert(0, 'Type', base_name.replace('_data', ''))
                df_data.to_csv(data_out, index=False, encoding=used_encoding)

                df_abbrev = pd.read_csv(file_path, skiprows=abbrev_idx + 1,
                                         nrows=abbrev_end - abbrev_idx,
                                         encoding=used_encoding, engine='python', on_bad_lines='warn')
                df_abbrev.to_csv(abbrev_out, index=False, encoding=used_encoding)

                df_ref = pd.read_csv(file_path, skiprows=ref_idx + 1,
                                      encoding=used_encoding, engine='python', on_bad_lines='warn')
                df_ref.to_csv(ref_out, index=False, encoding=used_encoding)
                return data_out, abbrev_out, ref_out
            except Exception as e:
                print(f"  Pandas split failed for {file_path}: {e}")

        with open(data_out, 'w', encoding=used_encoding) as f:
            f.writelines(lines[:data_end])
        with open(abbrev_out, 'w', encoding=used_encoding) as f:
            f.writelines(lines[abbrev_idx + 1:abbrev_end])
        with open(ref_out, 'w', encoding=used_encoding) as f:
            f.writelines(lines[ref_idx + 1:])

        return data_out, abbrev_out, ref_out

    def split_all_csvs(self, data_dir=None, output_dir=None):
        if data_dir is None:
            data_dir = self.data_dir
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, "Split")

        csv_files = self._find_csvs(data_dir)
        print(f"Splitting {len(csv_files)} CSV files...")
        for i, path in enumerate(csv_files):
            print(f"  [{i+1}/{len(csv_files)}] {os.path.basename(path)}")
            self.split_csv_file(path, output_dir)

    def build_database(self, data_dir=None, db_path=None, table_name="georoc_data"):
        if data_dir is None:
            data_dir = os.path.join(self.data_dir, "Split", "Data")
        if db_path is None:
            db_path = os.path.join(self.data_dir, "GeoRoc.db")

        if not os.path.exists(data_dir):
            print(f"  Data directory not found: {data_dir}")
            return

        csv_files = sorted(
            f for f in os.listdir(data_dir) if f.lower().endswith('.csv')
        )
        if not csv_files:
            print("  No CSV files found")
            return

        all_columns = []
        file_dfs = []
        print("  Scanning CSV files for schema...")
        for i, file in enumerate(csv_files):
            file_path = os.path.join(data_dir, file)
            df = self._read_csv(file_path)
            if df is None:
                continue
            if HAS_PANDAS:
                df = df.drop_duplicates()
            all_columns.extend(c for c in df.columns if c not in all_columns)
            file_dfs.append((file, file_path, df))
            if (i + 1) % 50 == 0 or i + 1 == len(csv_files):
                print(f"    Scanned {i+1}/{len(csv_files)} files, {len(all_columns)} unique columns")

        print(f"  Total unique columns: {len(all_columns)}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        col_defs = ', '.join(f'[{c}] TEXT' for c in all_columns)
        cursor.execute(f'CREATE TABLE IF NOT EXISTS [{table_name}] ({col_defs})')
        conn.commit()

        imported = 0
        total = len(file_dfs)
        for i, (file, file_path, df) in enumerate(file_dfs):
            missing_cols = [c for c in all_columns if c not in df.columns]
            for c in missing_cols:
                df[c] = None
            df = df[all_columns]

            try:
                df.to_sql(table_name, conn, if_exists='append', index=False)
                imported += 1
            except Exception as e:
                print(f"    DB insert error for {file}: {e}")
                continue

            if (i + 1) % 50 == 0 or i + 1 == total:
                print(f"    Imported {i+1}/{total} files ({imported} successful)")

        conn.commit()
        conn.close()
        print(f"  Imported {imported}/{total} files into {db_path}")

    def build_refs_database(self, ref_dir=None, db_path=None, table_name="references"):
        if ref_dir is None:
            ref_dir = os.path.join(self.data_dir, "Split", "References")
        if db_path is None:
            db_path = os.path.join(self.data_dir, "Ref.db")

        if not os.path.exists(ref_dir):
            return

        all_columns = ['Reference']
        file_dfs = []
        for file in sorted(os.listdir(ref_dir)):
            if not file.lower().endswith('.csv'):
                continue
            file_path = os.path.join(ref_dir, file)
            lines = []
            for enc in ['utf-8-sig', 'utf-8', 'ISO-8859-1', 'Windows-1252']:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        lines = [line.strip().strip('"') for line in f if line.strip()]
                    break
                except Exception:
                    continue
            if not lines:
                continue
            if HAS_PANDAS:
                df = pd.DataFrame(lines, columns=all_columns)
            else:
                continue
            file_dfs.append((file, df))

        if not file_dfs:
            return

        conn = sqlite3.connect(db_path)
        col_defs = ', '.join(f'[{c}] TEXT' for c in all_columns)
        conn.execute(f'CREATE TABLE IF NOT EXISTS [{table_name}] ({col_defs})')

        imported = 0
        for file, df in file_dfs:
            try:
                df.to_sql('references', conn, if_exists='append', index=False)
                imported += 1
            except Exception as e:
                print(f"    Ref DB error for {file}: {e}")

        conn.commit()
        conn.close()
        print(f"  Imported {imported}/{len(file_dfs)} reference files")

    @staticmethod
    def _read_csv(file_path, skiprows=0, header='infer', **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'ISO-8859-1', 'Windows-1252']
        for enc in encodings:
            try:
                if HAS_PANDAS:
                    return pd.read_csv(file_path, encoding=enc, engine='python',
                                       on_bad_lines='warn', skiprows=skiprows, header=header, **kwargs)
                else:
                    rows = []
                    hdr = None
                    with open(file_path, 'r', encoding=enc) as f:
                        reader = csv.reader(f)
                        for i, row in enumerate(reader):
                            if i < skiprows:
                                continue
                            if hdr is None and header == 'infer':
                                hdr = row
                                continue
                            rows.append(row)
                    if hdr and rows:
                        return pd.DataFrame(rows, columns=hdr) if HAS_PANDAS else None
            except Exception:
                continue
        return None

    def _find_csvs(self, directory):
        csv_files = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, f))
        return sorted(csv_files)


import re