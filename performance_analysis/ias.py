#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IAS (Investment Attractiveness Score) calculator

- Company name: fetched ONLY via yfinance (Ticker(symbol).get_info()["longName"] or ["shortName"])
- All valuation/growth metrics: crawled from Yahoo Finance pages with Selenium (no yfinance)
- Horizon switch:
    --horizon current | next   (default: current)
  * current: uses Current-Year growths and Trailing P/E
  * next   : uses Next-Year growths and Forward P/E

Outputs:
  * Console table (markdown)
  * CSV file (default: ias_results_raw.csv)
  * Optional HTML dumps under ./debug_html/<SYMBOL>/ with --debug-html
"""

import os
import re
import json
import math
import time
import argparse
import pathlib
import pandas as pd
import chromedriver_autoinstaller
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ---------------------------
# Utilities
# ---------------------------

def clean_number(text):
    if text is None:
        return None
    t = str(text).strip()
    if t in ("", "--", "—", "N/A", "NaN"):
        return None
    t = re.sub(r"[^\d\.\-]", "", t)
    if t in ("", "-", ".", "-."):
        return None
    try:
        return float(t)
    except Exception:
        return None


def clean_percent(text):
    if text is None:
        return None
    t = str(text).strip()
    if t in ("", "--", "—", "N/A", "NaN"):
        return None
    sign = -1.0 if "(" in t and ")" in t else 1.0
    t = t.replace("(", "").replace(")", "")
    t = t.replace("per annum", "")
    m = re.search(r"(-?\d+(\.\d+)?)\s*%?", t)
    if not m:
        return None
    try:
        return float(m.group(1)) * sign
    except Exception:
        return None


def headless_driver():
    chromedriver_autoinstaller.install()
    options = Options()
    options.page_load_strategy = "eager"
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    return driver


def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def safe_get_html(driver, url, retries=4, sleep_s=1.0):
    """
    Load URL and return page_source, auto-retrying through interstitials.
    """
    last_err = None
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(0.8 + 0.4 * attempt)  # tiny backoff
            html = driver.page_source or ""
            # Interstitials or bot checks
            bad = (
                "Will be right back" in html or
                "please verify you are a human" in html.lower() or
                "Access Denied" in html or
                "Request blocked" in html
            )
            if bad:
                time.sleep(sleep_s)
                continue
            return html
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    print(f"⚠️ Failed to load url: {url}\n   Error: {last_err}")
    return None


def wait_visible(driver, xpath, timeout=12):
    WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))


def dump_table_html(table_elem, out_path: str):
    try:
        html = table_elem.get_attribute("outerHTML")
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        print(f"⚠️ Failed to save table HTML '{out_path}': {e}")


def try_dismiss_consent(driver):
    try:
        btn_texts = [
            "Accept all", "Accept", "I agree", "Agree",
            "동의", "승인", "확인", "모두 수락", "수락"
        ]
        for txt in btn_texts:
            btns = driver.find_elements(By.XPATH, f"//button[normalize-space(text())='{txt}']")
            if btns:
                btns[0].click()
                time.sleep(0.5)
                return
    except Exception:
        pass


# ---------------------------
# DOM helpers
# ---------------------------

def table_under_h3(driver, title_regex):
    headers = driver.find_elements(By.XPATH, "//h3")
    for h in headers:
        ttl = h.text.strip()
        if re.search(title_regex, ttl, re.I):
            try:
                section = h.find_element(By.XPATH, "./ancestor::section[1]")
                table = section.find_element(By.TAG_NAME, "table")
                return table
            except Exception:
                continue
    return None


# ---------------------------
# Company name via yfinance (ONLY)
# ---------------------------

def get_company_name_yf(symbol: str):
    """
    Fetches company name strictly via yfinance.
    Tries longName, then shortName. Returns None if unavailable.
    """
    try:
        t = yf.Ticker(symbol)
        # Newer yfinance prefers get_info() over .info (deprecated in some versions).
        info = {}
        try:
            info = t.get_info()
        except Exception:
            # fallback for older versions
            info = getattr(t, "info", {}) or {}
        name = info.get("longName") or info.get("shortName")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception as e:
        print(f"⚠️ yfinance company lookup failed for {symbol}: {e}")
    return None


# ---------------------------
# Key statistics / Analysis scrapers (Yahoo HTML)
# ---------------------------

def get_key_statistics(driver, symbol, debug_dir=None):
    url = f"https://finance.yahoo.com/quote/{symbol}/key-statistics?p={symbol}"
    html = safe_get_html(driver, url)
    if html is None:
        return {}
    if debug_dir:
        ensure_dir(debug_dir)
        with open(os.path.join(debug_dir, f"{symbol}_key_statistics.html"), "w", encoding="utf-8") as f:
            f.write(html)

    try_dismiss_consent(driver)

    trailing_pe = None
    forward_pe = None
    opm_ttm = None

    tables = driver.find_elements(By.XPATH, "//section//table")
    for i, t in enumerate(tables):
        if debug_dir:
            dump_table_html(t, os.path.join(debug_dir, f"{symbol}_key_stats_table_{i}.html"))
        try:
            rows = t.find_elements(By.TAG_NAME, "tr")
            for r in rows:
                tds = r.find_elements(By.TAG_NAME, "td")
                if len(tds) < 2:
                    continue
                label = tds[0].text.strip()
                value = tds[1].text.strip()

                if re.search(r"Trailing\s+P/?E", label, re.I):
                    if trailing_pe is None:
                        trailing_pe = clean_number(value)
                if re.search(r"Forward\s+P/?E", label, re.I):
                    if forward_pe is None:
                        forward_pe = clean_number(value)
                if re.search(r"Operating\s+Margin\s*\(ttm\)", label, re.I):
                    if opm_ttm is None:
                        opm_ttm = clean_percent(value)
        except Exception:
            continue

    return {"trailing_pe": trailing_pe, "forward_pe": forward_pe, "operating_margin_ttm": opm_ttm}


def get_analysis_growth(driver, symbol, horizon="current", debug_dir=None):
    assert horizon in ("current", "next")

    url = f"https://finance.yahoo.com/quote/{symbol}/analysis?p={symbol}"
    html = safe_get_html(driver, url)
    if html is None:
        return {}

    if debug_dir:
        ensure_dir(debug_dir)
        with open(os.path.join(debug_dir, f"{symbol}_analysis.html"), "w", encoding="utf-8") as f:
            f.write(html)

    try_dismiss_consent(driver)

    # Revenue Growth
    revenue_growth = None
    sales_est_cy = sales_est_ny = sales_est_py = None

    try:
        rev_table = table_under_h3(driver, r"Revenue\s+Estimate")
        if rev_table:
            if debug_dir:
                dump_table_html(rev_table, os.path.join(debug_dir, f"{symbol}_analysis_revenue_estimate.html"))

            thead = rev_table.find_element(By.TAG_NAME, "thead")
            ths = thead.find_elements(By.TAG_NAME, "th")
            cols = [th.text.strip() for th in ths]
            cy_idx = next((i for i, c in enumerate(cols) if re.search(r"Current\s+Year", c, re.I)), None)
            ny_idx = next((i for i, c in enumerate(cols) if re.search(r"Next\s+Year", c, re.I)), None)

            tbody = rev_table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            for r in rows:
                tds = r.find_elements(By.TAG_NAME, "td")
                if not tds:
                    continue
                row_label = tds[0].text.strip()

                if re.search(r"Sales\s+Estimate", row_label, re.I):
                    if cy_idx is not None and cy_idx < len(tds):
                        sales_est_cy = clean_number(tds[cy_idx].text)
                    if ny_idx is not None and ny_idx < len(tds):
                        sales_est_ny = clean_number(tds[ny_idx].text)

                if re.search(r"Year\s+Ago\s+Sales", row_label, re.I):
                    if cy_idx is not None and cy_idx < len(tds):
                        sales_est_py = clean_number(tds[cy_idx].text)

                if re.search(r"Sales\s*Growth", row_label, re.I):
                    if horizon == "current" and cy_idx is not None and cy_idx < len(tds):
                        revenue_growth = clean_percent(tds[cy_idx].text)
                    elif horizon == "next" and ny_idx is not None and ny_idx < len(tds):
                        revenue_growth = clean_percent(tds[ny_idx].text)

            if revenue_growth is None:
                if horizon == "current":
                    if sales_est_cy is not None and sales_est_py not in (None, 0.0):
                        revenue_growth = (sales_est_cy - sales_est_py) / abs(sales_est_py) * 100.0
                else:
                    if sales_est_ny is not None and sales_est_cy not in (None, 0.0):
                        revenue_growth = (sales_est_ny - sales_est_cy) / abs(sales_est_cy) * 100.0
    except Exception:
        pass

    # Earnings Growth
    earnings_growth = None
    try:
        ge_table = table_under_h3(driver, r"Growth\s+Estimates")
        if ge_table:
            if debug_dir:
                dump_table_html(ge_table, os.path.join(debug_dir, f"{symbol}_analysis_growth_estimates.html"))

            thead = ge_table.find_element(By.TAG_NAME, "thead")
            ths = thead.find_elements(By.TAG_NAME, "th")
            cols = [th.text.strip() for th in ths]

            if any(re.search(r"Company", c, re.I) for c in cols):
                comp_idx = next((i for i, c in enumerate(cols) if re.search(r"Company", c, re.I)), None)
                tbody = ge_table.find_element(By.TAG_NAME, "tbody")
                for r in tbody.find_elements(By.TAG_NAME, "tr"):
                    tds = r.find_elements(By.TAG_NAME, "td")
                    if not tds:
                        continue
                    row_label = tds[0].text.strip()
                    if horizon == "current" and re.search(r"Current\s+Year", row_label, re.I):
                        if comp_idx is not None and comp_idx < len(tds):
                            earnings_growth = clean_percent(tds[comp_idx].text)
                            break
                    if horizon == "next" and re.search(r"Next\s+Year", row_label, re.I):
                        if comp_idx is not None and comp_idx < len(tds):
                            earnings_growth = clean_percent(tds[comp_idx].text)
                            break

            elif any(re.search(r"Symbol", c, re.I) for c in cols):
                cy_idx = next((i for i, c in enumerate(cols) if re.search(r"Current\s+Year", c, re.I)), None)
                ny_idx = next((i for i, c in enumerate(cols) if re.search(r"Next\s+Year", c, re.I)), None)
                sym_upper = symbol.upper()
                target_idx = cy_idx if horizon == "current" else ny_idx

                tbody = ge_table.find_element(By.TAG_NAME, "tbody")
                for r in tbody.find_elements(By.TAG_NAME, "tr"):
                    tds = r.find_elements(By.TAG_NAME, "td")
                    if not tds:
                        continue
                    row_key = tds[0].text.strip().upper()
                    if row_key == sym_upper and target_idx is not None and target_idx < len(tds):
                        earnings_growth = clean_percent(tds[target_idx].text)
                        break
    except Exception:
        pass

    # Fallback via Earnings Estimate math
    if earnings_growth is None:
        try:
            ee_table = table_under_h3(driver, r"Earnings\s+Estimate")
            if ee_table:
                if debug_dir:
                    dump_table_html(ee_table, os.path.join(debug_dir, f"{symbol}_analysis_earnings_estimate.html"))

                thead = ee_table.find_element(By.TAG_NAME, "thead")
                ths = thead.find_elements(By.TAG_NAME, "th")
                cols = [th.text.strip() for th in ths]
                cy_idx = next((i for i, c in enumerate(cols) if re.search(r"Current\s+Year", c, re.I)), None)
                ny_idx = next((i for i, c in enumerate(cols) if re.search(r"Next\s+Year", c, re.I)), None)

                tbody = ee_table.find_element(By.TAG_NAME, "tbody")
                rows = tbody.find_elements(By.TAG_NAME, "tr")

                avg_est_cy = avg_est_ny = year_ago_eps_cy = None
                for r in rows:
                    tds = r.find_elements(By.TAG_NAME, "td")
                    if not tds:
                        continue
                    row_label = tds[0].text.strip()
                    if re.search(r"Avg\.?\s*Estimate", row_label, re.I):
                        if cy_idx is not None and cy_idx < len(tds):
                            avg_est_cy = clean_number(tds[cy_idx].text)
                        if ny_idx is not None and ny_idx < len(tds):
                            avg_est_ny = clean_number(tds[ny_idx].text)
                    if re.search(r"Year\s+Ago\s+EPS", row_label, re.I):
                        if cy_idx is not None and cy_idx < len(tds):
                            year_ago_eps_cy = clean_number(tds[cy_idx].text)

                if horizon == "current":
                    if avg_est_cy is not None and year_ago_eps_cy not in (None, 0.0):
                        earnings_growth = (avg_est_cy - year_ago_eps_cy) / abs(year_ago_eps_cy) * 100.0
                else:
                    if avg_est_ny is not None and avg_est_cy not in (None, 0.0):
                        earnings_growth = (avg_est_ny - avg_est_cy) / abs(avg_est_cy) * 100.0
        except Exception:
            pass

    return {"revenue_growth": revenue_growth, "earnings_growth": earnings_growth}


# ---------------------------
# IAS computation
# ---------------------------

def compute_ias(pe_used, revenue_growth, earnings_growth, opm_ttm, w1=0.6, w2=0.4):
    if pe_used is None or pe_used <= 0:
        return None, None

    vals = [v for v in (revenue_growth, earnings_growth) if v is not None]
    if len(vals) == 2:
        G = sum(vals) / 2.0
    elif len(vals) == 1:
        G = vals[0]
    else:
        G = 0.0

    opm = opm_ttm if opm_ttm is not None else 0.0
    ias = (w1 * G + w2 * opm) / pe_used
    return G, ias


# ---------------------------
# Orchestration
# ---------------------------

def process_symbol(driver, symbol, horizon="current", debug_html=False):
    debug_dir = os.path.join("debug_html", symbol) if debug_html else None

    company = get_company_name_yf(symbol)  # <- yfinance ONLY
    stats   = get_key_statistics(driver, symbol, debug_dir=debug_dir)
    growth  = get_analysis_growth(driver, symbol, horizon=horizon, debug_dir=debug_dir)

    trailing_pe = stats.get("trailing_pe")
    forward_pe  = stats.get("forward_pe")
    opm_ttm     = stats.get("operating_margin_ttm")
    rev_g       = growth.get("revenue_growth")
    earn_g      = growth.get("earnings_growth")

    pe_used = trailing_pe if horizon == "current" else forward_pe
    G, ias = compute_ias(pe_used, rev_g, earn_g, opm_ttm)

    return {
        "Company": company,
        "Symbol": symbol,
        "PE used": ("Trailing P/E" if horizon == "current" else "Forward P/E"),
        "Trailing P/E": trailing_pe,
        "Forward P/E": forward_pe,
        f"Revenue Growth ({'CY' if horizon=='current' else 'NY'} %)": rev_g,
        f"Earnings Growth ({'CY' if horizon=='current' else 'NY'} %)": earn_g,
        f"Combined Growth G ({'CY' if horizon=='current' else 'NY'} %)": G,
        "Operating Margin (ttm %)": opm_ttm,
        "IAS (0.6·G+0.4·OPM)/PE_used": ias
    }


def main():
    parser = argparse.ArgumentParser(description="Compute IAS from Yahoo Finance (crawling) with company via yfinance.")
    parser.add_argument("--symbol", required=True, nargs="+",
                        help="Ticker symbols. Comma- or space-separated (e.g., MSFT,AAPL or MSFT AAPL).")
    parser.add_argument("--horizon", choices=["current", "next"], default="current",
                        help="Use Current Year or Next Year growth. Also switches Trailing vs Forward P/E. Default: current.")
    parser.add_argument("--debug-html", action="store_true",
                        help="Dump raw Yahoo HTML to ./debug_html/<SYMBOL>/")
    parser.add_argument("--out", default="ias_results_raw.csv", help="CSV output filename (default: ias_results_raw.csv)")
    args = parser.parse_args()

    # flatten comma-separated inputs
    raw_list = []
    for item in args.symbol:
        raw_list.extend([s for s in re.split(r"[,\s]+", item) if s])
    symbols = [s.upper() for s in raw_list]

    driver = headless_driver()
    results = []
    try:
        for sym in symbols:
            print(f"\n🚀 Processing {sym} ...")
            row = process_symbol(driver, sym, horizon=args.horizon, debug_html=args.debug_html)
            results.append(row)
            time.sleep(0.6)
    finally:
        driver.quit()

    df = pd.DataFrame(results)

    # Console view (rounded)
    print("\n📊 IAS Inputs & Score")
    view = df.copy()
    for col in view.columns:
        if col in ("Company", "Symbol", "PE used"):
            continue
        view[col] = view[col].apply(
            lambda x: None if x is None or (isinstance(x, float) and math.isnan(x))
            else (round(x, 3) if isinstance(x, (int, float)) else x)
        )
    preferred = ["Company", "Symbol", "PE used"]
    ordered_cols = [c for c in preferred if c in view.columns] + [c for c in view.columns if c not in preferred]
    view = view[ordered_cols]
    print(view.to_markdown(index=False))

    # Save CSV in the same column order
    df = df[ordered_cols]
    df.to_csv(args.out, index=False)
    print(f"\n💾 Saved raw results to {args.out}")
    if args.debug_html:
        print("🧪 Debug HTML saved under ./debug_html/<SYMBOL>/")


if __name__ == "__main__":
    main()

