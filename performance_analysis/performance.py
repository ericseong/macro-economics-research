import time
import re
import argparse
import hashlib
import pandas as pd
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def clean_numeric(text):
    return float(re.sub(r"[^\d.-]", "", text)) if text not in {"", "--"} else None

def setup_driver():
    chromedriver_autoinstaller.install()
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15)
    return driver

def switch_tab(driver, tab_name):
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f"//button[text()='{tab_name}']"))
        )
        btn = driver.find_element(By.XPATH, f"//button[text()='{tab_name}']")
        if "selected" not in btn.get_attribute("class"):
            btn.click()
            time.sleep(3)
    except Exception as e:
        print(f"⚠️ Failed to switch to {tab_name} tab:", str(e))

def scrape_financial_table(driver, label):
    print(f"\n📊 Yahoo Finance - {label} Financials")
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'rowTitle') and text()='Total Revenue']"))
        )
    except Exception as e:
        print("⚠️ Timed out waiting for financial data to load.", str(e))
        return None

    try:
        header_row = driver.find_element(By.XPATH, "//div[contains(@class, 'tableHeader')]//div[contains(@class, 'row')]")
        header_cells = header_row.find_elements(By.XPATH, ".//div[contains(@class, 'column')]")
        columns = [cell.text.strip() for cell in header_cells[1:]]
    except Exception:
        columns = [f"Col{i+1}" for i in range(6)]

    target_items = {
        "Total Revenue": "total_revenue",
        "Operating Income": "operating_income",
        "Net Income Common Stockholders": "net_income",
        "Basic EPS": "basic_eps",
        "Diluted EPS": "diluted_eps",
        "Net Interest Income": "net_interest_income"
    }

    data = {}
    rows = driver.find_elements(By.XPATH, "//div[contains(@class, 'row') and contains(@class, 'lv-0')]")
    for row in rows:
        try:
            title_div = row.find_element(By.XPATH, ".//div[contains(@class, 'rowTitle')]")
            label = title_div.get_attribute("title").strip()
            if label in target_items:
                value_divs = row.find_elements(By.XPATH, ".//div[contains(@class, 'column') and not(contains(@class, 'sticky'))]")
                values = [v.text.strip() for v in value_divs]
                data[target_items[label]] = values
        except Exception:
            continue

    if data:
        df = pd.DataFrame(data).transpose()
        df.columns = columns[:len(df.columns)]
        print(df.to_markdown())
        return df
    else:
        print("❌ No financial data extracted.")
        return None

def scrape_statistics(driver):
    print("\n📊 Yahoo Finance - Valuation Statistics")
    url = driver.current_url.replace("/financials", "/key-statistics")
    driver.get(url)
    time.sleep(3)

    try:
        section = driver.find_element(By.XPATH, "//section[@data-testid='qsp-statistics']")
        table = section.find_element(By.TAG_NAME, "table")
        header_cells = table.find_element(By.TAG_NAME, "thead").find_elements(By.TAG_NAME, "th")
        columns = [cell.text.strip() for cell in header_cells[1:]]  # skip label column

        rows = table.find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr")
        data = {}
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 2:
                label = cells[0].text.strip()
                values = [c.text.strip() for c in cells[1:len(columns)+1]]
                data[label] = values

        df = pd.DataFrame(data, index=columns).T
        print(df.to_markdown())
        return df

    except Exception as e:
        print("⚠️ Failed to parse valuation statistics:", str(e))
        return None


def hash_table(headers, records):
    """Generate a stable hash from headers and records."""
    raw = str(headers) + str(records)
    return hashlib.md5(raw.encode()).hexdigest()


def fetch_all_financials(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}/financials?p={symbol}"
    driver = setup_driver()
    driver.get(url)

    print("🔍 Current URL:", driver.current_url)
    print("📝 Page title:", driver.title)

    # Annual
    switch_tab(driver, "Annual")
    annual_df = scrape_financial_table(driver, "Annual")

    # Quarterly
    switch_tab(driver, "Quarterly")
    quarterly_df = scrape_financial_table(driver, "Quarterly")

    # Statistics (with quarterly trend)
    statistics_df = scrape_statistics(driver)

    # Analysis
    analysis_dfs = scrape_analysis_sections(driver)

    driver.quit()
    return annual_df, quarterly_df, statistics_df, analysis_dfs

def scrape_analysis_sections(driver):
    print("\n📊 Yahoo Finance - Analyst Estimates")
    url = driver.current_url.replace("/financials", "/analysis")
    driver.get(url)
    time.sleep(3)

    target_titles = {
        "Earnings Estimate",
        "Revenue Estimate",
        "Earnings History",
        "EPS Trend",
        "Growth Estimates"
    }

    results = {}

    try:
        sections = driver.find_elements(By.XPATH, "//section[contains(@data-test,'qsp-analyst')]")
        for section in sections:
            try:
                title_elem = section.find_element(By.XPATH, ".//h3")
                title = title_elem.text.strip()
                if title in target_titles:
                    table = section.find_element(By.TAG_NAME, "table")
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    headers = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]
                    data = []
                    for row in rows[1:]:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if cells:
                            label = cells[0].text.strip()
                            values = [c.text.strip() for c in cells[1:]]
                            data.append([label] + values)

                    df = pd.DataFrame(data, columns=headers)
                    results[title] = df
                    print(f"\n🔹 {title}")
                    print(df.to_markdown(index=False))
            except Exception:
                continue

    except Exception as e:
        print("⚠️ Failed to parse analysis section:", str(e))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance financials")
    parser.add_argument("symbol", help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--csv", help="Base filename to save all data")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    annual_df, quarterly_df, statistics_df, analysis_dfs = fetch_all_financials(symbol)

    if args.csv:
        if annual_df is not None:
            annual_df.to_csv(f"{args.csv}_annual.csv")
        if quarterly_df is not None:
            quarterly_df.to_csv(f"{args.csv}_quarterly.csv")
        if statistics_df is not None:
            statistics_df.to_csv(f"{args.csv}_valuation.csv")
        for key, df in analysis_dfs.items():
            safe_key = key.lower().replace(" ", "_").replace("/", "_")
            df.to_csv(f"{args.csv}_analysis_{safe_key}.csv", index=False)
        print(f"\n💾 Saved to {args.csv}_*.csv")
