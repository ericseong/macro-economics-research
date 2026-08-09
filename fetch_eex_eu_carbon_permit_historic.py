#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EU ETS (EEX) auction data: fetch, combine, save CSV, and plot with Plotly.

What this does
- Fetch current-year XLSX (public EEX URL)
- Discover & download the historical ZIP from the EEX download page (typically 2020..prev-year)
- Crawl the public EEX index for pre-2020 loose files (.xls/.xlsx)
- Robust header detection & column aliasing
- Keep only: Date, Auction Price €/tCO2, Auction Volume tCO2, Country  (no Status)
- If Country is missing, derive from 'Auction Platform' / 'Auction Name' / 'Auction Details' (fallback 'EU')
- CSV output via --output (default: EU_carbon_permit_price.csv)
- Plot: top = price line, bottom = stacked volume; slider at bottom only
- Consistent per-country colors; single legend entry per country
- --years N shows the last N full calendar years (inclusive). If N exceeds history, clamps to earliest

Install deps:
    pip install pandas plotly requests beautifulsoup4 openpyxl xlrd
"""

import argparse
import io
import os
import re
import sys
import zipfile
from datetime import datetime
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="openpyxl.styles.stylesheet"
)

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import requests
from bs4 import BeautifulSoup

# -----------------------
# Config
# -----------------------
CURRENT_YEAR = datetime.now().year

CURRENT_XLSX_URL = (
    f"https://public.eex-group.com/eex/eua-auction-report/"
    f"emission-spot-primary-market-auction-report-{CURRENT_YEAR}-data.xlsx"
)

HIST_DOWNLOAD_PAGE = (
    "https://www.eex.com/en/market-data/market-data-hub/environmentals/"
    "eex-eua-primary-auction-spot-download"
)

PUBLIC_INDEX_URL = "https://public.eex-group.com/eex/eua-auction-report/"

# Optional fallback if page parsing ever fails (fill in if needed)
HIST_ZIP_FALLBACK = None
# HIST_ZIP_FALLBACK = "https://www.eex.com/fileadmin/EEX/Downloads/EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/emission-spot-primary-market-auction-report-2012-2024-data.zip"

REQUIRED_COLUMNS = [
    "Date",
    "Auction Price €/tCO2",
    "Auction Volume tCO2",
    "Country",
]

DEFAULT_OUTPUT_CSV = "EU_carbon_permit_price.csv"
WORKDIR = "data_eua"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EU-ETS-downloader/1.3)"
}


# -----------------------
# HTTP helpers
# -----------------------
def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _download_bytes(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
    resp.raise_for_status()
    return resp.content


def _discover_zip_url_from_page(page_html: bytes, base_url: str) -> Optional[str]:
    """Find the first .zip link on the EEX 'download' page."""
    soup = BeautifulSoup(page_html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"\.zip(\?.*)?$", href, re.IGNORECASE):
            return href if href.startswith(("http://", "https://")) else urljoin(base_url, href)
    return None


def _discover_loose_files_from_index(index_html: bytes, base_url: str) -> List[str]:
    """
    From the public EEX index, collect any file links that look like
    emission-spot-primary-market-auction-report-YYYY-data.(xls|xlsx).
    """
    soup = BeautifulSoup(index_html, "html.parser")
    links = []
    pat = re.compile(r"emission-spot-primary-market-auction-report-(\d{4})-data\.(xls|xlsx)$", re.I)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        m = pat.search(href)
        if m:
            url = href if href.startswith(("http://", "https://")) else urljoin(base_url, href)
            links.append(url)
    return sorted(set(links))


# -----------------------
# Excel parsing helpers
# -----------------------
def _normalize_colnames(cols: List[str]) -> List[str]:
    """Trim and collapse whitespace; keep accents intact."""
    return [re.sub(r"\s+", " ", str(c)).strip() for c in cols]


def _find_header_row(df_preview: pd.DataFrame) -> Optional[int]:
    """
    Detect the actual table header row by scanning early rows for a row
    containing 'Date' plus at least one known metric header.
    """
    must_have = {"Date"}
    any_of = {
        "Auction Price €/tCO2",
        "Auction Price \u20ac/tCO2",
        "Auction Price EUR/tCO2",
        "Auction Volume tCO2",
        "Auction volume tCO2",
        "Auction Volume (tCO2)",
        "Auction Volume",
        "Volume (tCO2)",
    }
    for i in range(min(80, len(df_preview))):  # scan generously
        row_vals = set(_normalize_colnames(df_preview.iloc[i].astype(str).tolist()))
        row_vals.discard("nan")
        if must_have.issubset(row_vals) and any(v in row_vals for v in any_of):
            return i
    return None


_COUNTRY_KEYWORDS = {
    "DE": ["DE", "Germany", "German"],
    "PL": ["PL", "Poland", "Polish"],
    "FR": ["FR", "France", "French"],
    "NL": ["NL", "Netherlands", "Dutch"],
    "UK": ["UK", "United Kingdom", "UKA", "U.K.", "Britain", "British"],
    "GB": ["GB", "Great Britain"],
    "IT": ["IT", "Italy", "Italian"],
    "ES": ["ES", "Spain", "Spanish"],
    "PT": ["PT", "Portugal", "Portuguese"],
    "IE": ["IE", "Ireland", "Irish"],
    "AT": ["AT", "Austria", "Austrian"],
    "BE": ["BE", "Belgium", "Belgian"],
    "LU": ["LU", "Luxembourg", "Luxembourgish"],
    "DK": ["DK", "Denmark", "Danish"],
    "SE": ["SE", "Sweden", "Swedish"],
    "FI": ["FI", "Finland", "Finnish"],
    "NO": ["NO", "Norway", "Norwegian"],
    "CZ": ["CZ", "Czech", "Czechia"],
    "SK": ["SK", "Slovakia", "Slovak"],
    "SI": ["SI", "Slovenia", "Slovenian"],
    "HU": ["HU", "Hungary", "Hungarian"],
    "RO": ["RO", "Romania", "Romanian"],
    "BG": ["BG", "Bulgaria", "Bulgarian"],
    "HR": ["HR", "Croatia", "Croatian"],
    "EE": ["EE", "Estonia", "Estonian"],
    "LV": ["LV", "Latvia", "Latvian"],
    "LT": ["LT", "Lithuania", "Lithuanian"],
    "GR": ["GR", "Greece", "Greek"],
    "CY": ["CY", "Cyprus", "Cypriot"],
    "MT": ["MT", "Malta", "Maltese"],
    "EU": ["EU", "European Union", "Common Auction Platform", "CAP", "EEX"],
}

def _infer_country(row: pd.Series) -> str:
    """Infer Country from text fields when missing."""
    fields = []
    for key in ["Auction Platform", "Auction Name", "Auction Details"]:
        if key in row and pd.notna(row[key]):
            fields.append(str(row[key]))
    text = " ".join(fields)
    if not text:
        return "EU"
    t = text.lower()
    for code, keys in _COUNTRY_KEYWORDS.items():
        for k in keys:
            if re.search(rf"\b{re.escape(k.lower())}\b", t):
                return code
    return "EU"


def _read_excel_force_columns(path: str, required_cols: List[str]) -> pd.DataFrame:
    """
    Robust Excel reader for both .xlsx (openpyxl) and .xls (xlrd):
      1) Choose sheet: 'Primary Market Auction' if present; else first sheet.
      2) Read a headerless preview; detect header row.
      3) Re-read with header row.
      4) Map aliases -> canonical names; if 'Country' missing, derive it.
      5) Coerce dtypes; keep required columns only.
    """
    ext = os.path.splitext(path)[1].lower()
    engine = "openpyxl" if ext == ".xlsx" else "xlrd"  # xlrd for legacy .xls

    xls = pd.ExcelFile(path, engine=engine)
    sheet_name = "Primary Market Auction" if "Primary Market Auction" in xls.sheet_names else xls.sheet_names[0]

    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, engine=engine)
    header_row = _find_header_row(preview)
    if header_row is None:
        raise ValueError(
            f"Could not detect a header row in {os.path.basename(path)} (sheet '{sheet_name}')."
        )

    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine=engine)
    df.columns = _normalize_colnames([str(c) for c in df.columns])

    # Aliases
    alias_price = [
        "Auction Price €/tCO2",
        "Auction Price \u20ac/tCO2",
        "Auction Price EUR/tCO2",
        "Auction price €/tCO2",
        "Auction Price (EUR/tCO2)",
        "Mean Price EUR/tCO2",
        "Median Price EUR/tCO2",
    ]
    alias_volume = [
        "Auction Volume tCO2",
        "Auction volume tCO2",
        "Auction Volume (tCO2)",
        "Volume (tCO2)",
        "Auction Volume",  # older files
    ]
    alias_country = [
        "Country",
        "Auction Platform",
        "Auction Location",
        "Auctioning State",
        "Auction Platform / Country",
    ]
    alias_date = ["Date", "Auction Date"]

    def _resolve_first(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_price = _resolve_first(alias_price)
    col_vol   = _resolve_first(alias_volume)
    col_ctry  = _resolve_first(alias_country)
    col_date  = _resolve_first(alias_date)

    if col_date is None:
        raise ValueError(f"Missing a date column in {os.path.basename(path)}. Available: {list(df.columns)}")
    if col_price is None and ("Mean Price EUR/tCO2" in df.columns or "Median Price EUR/tCO2" in df.columns):
        col_price = "Mean Price EUR/tCO2" if "Mean Price EUR/tCO2" in df.columns else "Median Price EUR/tCO2"
    if col_vol is None and "Total Amount of Bids" in df.columns:
        col_vol = "Total Amount of Bids"  # last resort (unit differs), only if nothing else exists

    keep_candidates = set([c for c in [col_date, col_price, col_vol, col_ctry] if c is not None])
    keep_candidates |= set([c for c in ["Auction Platform", "Auction Name", "Auction Details"] if c in df.columns])
    dfx = df[list(keep_candidates)].copy()

    if col_ctry is None:
        dfx["Country"] = dfx.apply(_infer_country, axis=1)
    else:
        dfx = dfx.rename(columns={col_ctry: "Country"})

    rename_map = {}
    if col_date is not None:  rename_map[col_date]  = "Date"
    if col_price is not None: rename_map[col_price] = "Auction Price €/tCO2"
    if col_vol is not None:   rename_map[col_vol]   = "Auction Volume tCO2"
    dfx = dfx.rename(columns=rename_map)

    missing = [c for c in required_cols if c not in dfx.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {os.path.basename(path)}: {missing}\n"
            f"Available columns: {list(dfx.columns)}"
        )

    # Coerce types
    dfx["Date"] = pd.to_datetime(dfx["Date"], errors="coerce")

    def to_num(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.replace("\xa0", " ", regex=False).str.replace(",", ".", regex=False)
        s = s.str.replace(r"[^\d\.\-]", "", regex=True)
        out = pd.to_numeric(s, errors="coerce")
        return out

    dfx["Auction Price €/tCO2"] = to_num(dfx["Auction Price €/tCO2"]).astype("float64")
    dfx["Auction Volume tCO2"] = to_num(dfx["Auction Volume tCO2"]).astype("float64")

    dfx["Country"] = dfx["Country"].astype(str).str.strip()
    dfx = dfx.dropna(subset=["Date"]).reset_index(drop=True)
    dfx = dfx[REQUIRED_COLUMNS]
    return dfx


# -----------------------
# Fetchers
# -----------------------
def fetch_current_year_file(local_files: List[str]):
    cur_path = os.path.join(
        WORKDIR,
        f"emission-spot-primary-market-auction-report-{CURRENT_YEAR}-data.xlsx",
    )
    print(f"Downloading current year XLSX: {CURRENT_XLSX_URL}")
    cur_bytes = _download_bytes(CURRENT_XLSX_URL)
    with open(cur_path, "wb") as f:
        f.write(cur_bytes)
    local_files.append(cur_path)


def fetch_archive_zip_files(local_files: List[str]):
    print(f"Discovering historical ZIP from: {HIST_DOWNLOAD_PAGE}")
    zip_url = None
    try:
        page = _download_bytes(HIST_DOWNLOAD_PAGE)
        zip_url = _discover_zip_url_from_page(page, HIST_DOWNLOAD_PAGE)
    except Exception as e:
        print(f"⚠️ Could not fetch or parse the download page: {e}")

    if not zip_url and HIST_ZIP_FALLBACK:
        print("No ZIP link discovered; using fallback ZIP URL.")
        zip_url = HIST_ZIP_FALLBACK

    if not zip_url:
        print("⚠️ No historical ZIP link found. Continuing without ZIP.")
        return

    print(f"Downloading historical ZIP: {zip_url}")
    zip_bytes = _download_bytes(zip_url)

    print("Extracting historical ZIP...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".xlsx", ".xls")):
                out_path = os.path.join(WORKDIR, os.path.basename(name))
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                local_files.append(out_path)


def fetch_public_index_files(local_files: List[str]):
    """Crawl the public directory index and download any pre-2020 annual .xls/.xlsx."""
    print(f"Crawling public index for loose files: {PUBLIC_INDEX_URL}")
    try:
        index_html = _download_bytes(PUBLIC_INDEX_URL)
        links = _discover_loose_files_from_index(index_html, PUBLIC_INDEX_URL)
        for url in links:
            m = re.search(r"(\d{4})-data\.(xls|xlsx)$", url)
            if not m:
                continue
            year = int(m.group(1))
            if year >= 2020:
                continue
            local_path = os.path.join(WORKDIR, os.path.basename(url))
            if not os.path.exists(local_path):
                print(f"Downloading pre-2020 file: {url}")
                with open(local_path, "wb") as f:
                    f.write(_download_bytes(url))
            local_files.append(local_path)
    except Exception as e:
        print(f"⚠️ Could not crawl public index: {e}")


def fetch_all_files() -> List[str]:
    """Fetch current-year XLSX, archive ZIP files, and pre-2020 loose files."""
    _ensure_dir(WORKDIR)
    local_files: List[str] = []

    fetch_current_year_file(local_files)
    fetch_archive_zip_files(local_files)
    fetch_public_index_files(local_files)

    local_files = sorted(set(local_files))
    print(f"Collected {len(local_files)} Excel file(s).")
    return local_files


# -----------------------
# Build, filter, save, plot
# -----------------------
def build_dataframe(xlsx_paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in xlsx_paths:
        try:
            df = _read_excel_force_columns(p, REQUIRED_COLUMNS)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {p}: {e}")
    if not dfs:
        raise RuntimeError("No valid Excel files parsed.")
    combined = pd.concat(dfs, ignore_index=True)

    combined = (
        combined.sort_values("Date")
        .drop_duplicates(subset=["Date", "Country", "Auction Volume tCO2"], keep="last")
        .reset_index(drop=True)
    )
    return combined


def filter_years_for_render(df: pd.DataFrame, years: Optional[int]) -> Tuple[pd.DataFrame, Optional[List[datetime]]]:
    """
    Return (df_filtered, x_range)
    - If years is None or <= 0: full df, x_range=None (auto)
    - If years is N: show the last N FULL calendar years (inclusive).
      If N exceeds history, clamp to the earliest available year.
    """
    if not years or years <= 0:
        return df, None

    min_year = int(df["Date"].dt.year.min())
    end_year = int(df["Date"].dt.year.max())
    start_year = max(min_year, end_year - years + 1)

    x0 = datetime(start_year, 1, 1)
    x1 = datetime(end_year, 12, 31, 23, 59, 59)

    mask = (df["Date"] >= pd.Timestamp(x0)) & (df["Date"] <= pd.Timestamp(x1))
    df2 = df.loc[mask].copy()
    return df2, [x0, x1]


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"✅ Saved CSV: {path}  (rows={len(df)})")


def _country_color_map(countries: List[str]) -> dict:
    """
    Assign a stable, high-contrast color per country.
    Force EU to orange.
    """
    palette = (
        pc.qualitative.Alphabet
        + pc.qualitative.Safe
        + pc.qualitative.Set2
        + pc.qualitative.Set3
        + pc.qualitative.Dark24
        + pc.qualitative.Light24
    )
    uniq = sorted([str(c) for c in countries if pd.notna(c)])
    mapping = {c: palette[i % len(palette)] for i, c in enumerate(uniq)}
    # Force EU to orange
    if "EU" in mapping:
        mapping["EU"] = "orange"
    return mapping


def render_plot(df: pd.DataFrame, title: str = "EU ETS Primary Auction (Price & Volume)", x_range=None):
    """
    Two-row subplot:
      - Row 1: Price line (€/tCO2), colored by Country
      - Row 2: Volume bars (tCO2), stacked by Country
    Range slider appears ONLY on the BOTTOM (volume) axis; top axis has no slider.
    """
    if df.empty:
        print("⚠️ No data to plot for the requested period.")
        return

    df = df.copy()
    df["Country"] = df["Country"].astype(str)

    countries = df["Country"].dropna().unique().tolist()
    color_map = _country_color_map(countries)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=("Auction Price (€/tCO₂)", "Auction Volume (tCO₂)"),
    )

    # Top: price line(s) by country (single legend entry per country)
    for country, g in df.groupby("Country", sort=True):
        fig.add_trace(
            go.Scatter(
                x=g["Date"],
                y=g["Auction Price €/tCO2"],
                mode="lines+markers",
                name=country,
                legendgroup=country,
                showlegend=True,  # legend only on price traces
                marker=dict(size=4, color=color_map[country]),
                line=dict(width=2, color=color_map[country]),
                hovertemplate="%{x|%Y-%m-%d}<br>Price: €%{y:.2f}<extra>" + country + "</extra>",
            ),
            row=1, col=1,
        )

    # Bottom: volume bars by country (stacked, no extra legend entries)
    for country, g in df.groupby("Country", sort=True):
        vols = pd.to_numeric(g["Auction Volume tCO2"], errors="coerce").astype(float).fillna(0.0)
        if (vols == 0).all():
            continue
        fig.add_trace(
            go.Bar(
                x=g["Date"],
                y=vols,
                legendgroup=country,
                showlegend=False,  # avoid duplicate legend entries
                marker=dict(color=color_map[country]),
                hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:,.0f} tCO₂<extra>" + country + "</extra>",
            ),
            row=2, col=1,
        )

    # If we got an explicit x_range (from --years), set bottom axis to it
    if x_range is not None:
        fig.update_xaxes(range=x_range, row=2, col=1)

    # Keep the top x-axis locked to the bottom (slider) axis
    fig.update_xaxes(matches="x2", row=1, col=1)

    # Layout & axes: slider only on bottom axis
    fig.update_layout(
        title=title,
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,      # top
            xanchor="right",
            x=1.0        # right-aligned
        ),
        margin=dict(l=50, r=20, t=70, b=40),
        hovermode="x unified",
    )

    # Top axis (no slider)
    fig.update_xaxes(
        type="date",
        rangeselector=None,
        rangeslider=dict(visible=False),
        row=1, col=1,
    )

    # Bottom axis (with range selector + slider at the very bottom)
    fig.update_xaxes(
        type="date",
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="€/tCO₂", row=1, col=1)
    fig.update_yaxes(title_text="tCO₂", row=2, col=1, rangemode="tozero")

    fig.show()


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="EU ETS primary auction price & volume (EEX).")
    parser.add_argument(
        "--years",
        type=int,
        help="Show the last N full calendar years (e.g. --years 10). If omitted, shows full history.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing files in data folder (debug).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV filename (default: {DEFAULT_OUTPUT_CSV})",
    )
    args = parser.parse_args()

    _ensure_dir(WORKDIR)

    if args.skip_download:
        xlsx_files = sorted(
            os.path.join(WORKDIR, f)
            for f in os.listdir(WORKDIR)
            if f.lower().endswith((".xlsx", ".xls"))
        )
        if not xlsx_files:
            print("No local Excel files found; remove --skip-download or run once without it.", file=sys.stderr)
            sys.exit(1)
    else:
        xlsx_files = fetch_all_files()

    df = build_dataframe(xlsx_files)
    save_csv(df, args.output)

    df_render, x_range = filter_years_for_render(df, args.years)
    render_plot(df_render, x_range=x_range)


if __name__ == "__main__":
    main()

