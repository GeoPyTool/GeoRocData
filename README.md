# GeoRocData

[![PyPI version](https://badge.fury.io/py/georocdata.svg)](https://pypi.org/project/georocdata/)
[![Python](https://img.shields.io/pypi/pyversions/georocdata.svg)](https://pypi.org/project/georocdata/)
[![License: GPLv3+](https://img.shields.io/badge/License-GPL%20v3%2B-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub](https://img.shields.io/badge/GitHub-GeoPyTool%2FGeoRocData-blue.svg)](https://github.com/GeoPyTool/GeoRocData)

GEOROC 数据库自动化下载、处理和 SQLite 数据库构建工具。

Automatically downloads, processes, and builds a SQLite database from
[GEOROC](https://georoc.eu/) precompiled geochemical datasets.

---

## 背景 / Background

[GEOROC](https://georoc.eu/) (Geochemistry of Rocks of the Oceans and Continents)
是由德国哥廷根大学 DIGIS 团队维护的全球最大的火成岩地球化学数据库，包含超过
100 万条来自约 20,000 篇文献的岩石样品分析数据。

GEOROC 提供了 14 类预编译数据集（Precompiled Files），涵盖太古代克拉通、汇聚边缘、
洋岛、岩石类型等，以 CSV 格式发布。这些 CSV 文件结构特殊（包含数据段、缩写段、
参考文献段），且存在引号错误等常见问题，需要处理后才能用于数据分析。

`georocdata` 工具实现了从下载到构建 SQLite 数据库的全流程自动化。

### 数据源变更 / Data Source Migration

| 版本 | 数据源 | 说明 |
|------|--------|------|
| 旧版 | `georoc.mpch-mainz.gwdg.de/Csv_Downloads/` | 逐个 CSV 下载，已于 2021 年停更 |
| **新版** | `georoc.eu` DOI 系统 | DOI 下载，数据更新至 2026-06-01 |

---

## 安装 / Installation

```bash
pip install georocdata
```

依赖: Python >= 3.8, pandas >= 1.3

---

## 快速开始 / Quick Start

### 命令行 / CLI

```bash
# 完整更新：下载 + 处理 + 建库
georocdata

# 仅下载
georocdata --download-only

# 仅处理已有 CSV + 建库
georocdata --process-only

# 使用旧数据源
georocdata --source legacy

# 同时使用新旧数据源
georocdata --source both

# 跳过某些步骤
georocdata --skip-fix       # 跳过CSV引号修复
georocdata --skip-split     # 跳过CSV段落拆分
georocdata --skip-db        # 跳过SQLite建库

# 自定义输出目录和数据库名
georocdata --output-dir ./my_data --db-name my_georoc.db

# 查看版本
georocdata -V
```

### Python API

```python
from georocdata import GEOROCDownloader, GEOROCProcessor

# 1. 下载
dl = GEOROCDownloader(output_dir="./georoc_data")
dl.download_new_source()   # 从 georoc.eu 下载
# dl.download_legacy_source()  # 或从旧源下载

# 2. 处理
proc = GEOROCProcessor(data_dir="./georoc_data")
proc.fix_all_csvs()                    # 修复CSV引号错误
proc.split_all_csvs()                  # 拆分CSV段落

# 3. 建库
proc.build_database(db_path="./GeoRoc.db")        # 数据库
proc.build_refs_database(db_path="./Ref.db")       # 参考文献库
```

---

## 数据集 / Datasets

GEOROC 提供 14 类预编译数据集：

| 编号 | 英文类别 | 中文类别 | DOI |
|------|----------|----------|-----|
| 1 | Archaean Cratons | 太古代克拉通 | 10.25625/1KRR1P |
| 2 | Complex Volcanic Settings | 复杂火山环境 | 10.25625/1VOFM5 |
| 3 | Continental Flood Basalts | 大陆溢流玄武岩 | 10.25625/WSTPOX |
| 4 | Convergent Margins | 板块汇聚边缘 | 10.25625/PVFZCE |
| 5 | Intraplate Volcanic Rocks | 板内火山岩 | 10.25625/RZZ9VM |
| 6 | Oceanic Plateaus | 洋底高原 | 10.25625/JRZIJF |
| 7 | Ocean Basin Flood Basalts | 洋底盆地溢流玄武岩 | 10.25625/AVLFC2 |
| 8 | Ocean Island Groups | 洋岛成分 | 10.25625/WFJZKY |
| 9 | Seamounts | 海山成分 | 10.25625/JUQK7N |
| 10 | Rift Volcanics | 裂谷火山 | 10.25625/KAIVCT |
| 11 | Rocks | 岩石成分 | 10.25625/2JETOA |
| 12 | Minerals | 矿物成分 | 10.25625/SGFTFN |
| 13 | Melt Inclusions | 熔体包裹体 | 10.25625/7JW6XU |
| 14 | Sample Metadata | 样品元数据 | 10.25625/4EZ7ID |

每类数据集包含若干 CSV 文件，例如 Archaean Cratons 包含 28 个克拉通的单独文件。

---

## 处理流程 / Processing Pipeline

```
原始 CSV 下载
     │
     ▼
┌──────────────────┐
│ 1. CSV 引号修复   │  修复 GEOROC CSV 中常见的孤立引号错误
└──────────────────┘
     │
     ▼
┌──────────────────┐
│ 2. CSV 段落拆分   │  将每个 CSV 拆分为:
│                   │    - 数据段 (Data) → *_data.csv
│                   │    - 缩写段 (Abbreviations) → *_abbreviations.csv
│                   │    - 参考文献段 (References) → *_references.csv
└──────────────────┘
     │
     ▼
┌──────────────────┐
│ 3. SQLite 建库    │  合并所有数据段到 GeoRoc.db，自动去重
│                   │  合并所有参考文献到 Ref.db
└──────────────────┘
```

---

## CSV 文件结构说明 / CSV Structure

每个 GEOROC CSV 文件由三个段落组成，以 `Abbreviations:` 和 `References:` 标记分隔：

```
Citation,Ref_ID,Name,Type,..."  ← 数据段（表头+数据）
Abbreviations:                   ← 缩写段开始
Abbreviation,Full_name,..."     ← 缩写段（表头+数据）
References:                      ← 参考文献段开始
Ref_ID,Reference,..."           ← 参考文献段（表头+数据）
```

工具自动识别并拆分这三个段落。

---

## 项目结构 / Project Structure

```
GeoRocData/
├── pyproject.toml                     # 构建/发布配置
├── README.md                          # 本文件
├── CHANGELOG.md                       # 变更日志
├── LICENSE                             # GPLv3+ 许可证
├── src/georocdata/                    # Python 包源码
│   ├── __init__.py                    # 包入口，导出核心类
│   ├── cli.py                         # CLI 命令入口
│   ├── compilations.py                # 14类数据集 DOI 映射配置
│   ├── downloader.py                  # GEOROCDownloader 下载器类
│   └── processor.py                   # GEOROCProcessor 处理器类
├── DB_Gen/                            # 旧版脚本（已弃用）
│   └── update_georoc.py              # 向后兼容入口
├── links.sh                           # 旧版下载链接列表
└── back.sh                            # 旧版下载脚本
```

---

## 从旧版迁移 / Migration from Legacy Scripts

| 旧版脚本 | 新版等效 |
|----------|---------|
| `bash back.sh` / `bash links.sh` | `georocdata --source legacy` |
| `0_Get_Data_Fix_Quota_Error.py` | `georocdata` (下载+修复，默认) |
| `1_Split_CSV.py` | `georocdata` (拆分，默认) |
| `2_Add_Type.py` | `georocdata` (Type列自动添加，默认) |
| `3_CSV_to_DB.py` | `georocdata` (建库，默认) |
| `4_REF_to_DB.py` | `georocdata` (参考文献建库，默认) |

---

## 开发 / Development

```bash
# 克隆仓库
git clone https://github.com/GeoPyTool/GeoRocData.git
cd GeoRocData

# 安装开发模式
pip install -e .

# 构建发布包
pip install build
python -m build

# 检查包质量
pip install twine
twine check dist/*
```

---

## 引用 / Citation

如果您在研究中使用了 GEOROC 数据，请引用：

> DIGIS Team (2026). GEOROC Compilation: [Dataset Name]. Göttingen Research Online / Data.
> https://doi.org/10.25625/[DOI]

---

## 许可证 / License

GPLv3+ - 详见 [LICENSE](LICENSE)

---

## 链接 / Links

- **GEOROC 数据库**: https://georoc.eu/
- **PyPI 包**: https://pypi.org/project/georocdata/
- **GitHub 仓库**: https://github.com/GeoPyTool/GeoRocData
- **GEOROC 预编译数据说明**: https://georoc.eu/georoc/CompRules.html