import sys
import requests
from tabulate import tabulate

# IMF API URLs
IMF_GDP_API_URL = "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH"
IMF_COUNTRIES_API_URL = "https://www.imf.org/external/datamapper/api/v1/countries"

def fetch_gdp_data():
    response = requests.get(IMF_GDP_API_URL)
    response.raise_for_status()
    data = response.json()
    return data.get("values", {}).get("NGDP_RPCH", {})

def fetch_country_names():
    response = requests.get(IMF_COUNTRIES_API_URL)
    response.raise_for_status()
    data = response.json()
    return {code: info["label"] for code, info in data.get("countries", {}).items()}

def analyze_gdp_growth(data, country_names, target_year):
    result = []
    investigated_countries = 0

    for country_code, growth_data in data.items():
        if target_year not in growth_data:
            continue

        investigated_countries += 1

        # Get the target year's growth rate
        target_growth = growth_data.get(target_year)

        # Get the last three years' growth rates
        try:
            prev_years = [str(int(target_year) - i) for i in range(1, 4)]
            prev_growth_rates = [growth_data[year] for year in prev_years if year in growth_data]

            if len(prev_growth_rates) < 3:
                continue  # Skip countries with incomplete data

            avg_prev_growth = sum(prev_growth_rates) / 3

            if target_growth > avg_prev_growth:
                country_name = country_names.get(country_code, country_code)
                result.append((country_name, avg_prev_growth, target_growth))
        except Exception as e:
            print(f"Error processing country {country_code}: {e}")

    return sorted(result, key=lambda x: x[0]), investigated_countries

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_gdp_growth.py <target_year>")
        sys.exit(1)

    target_year = sys.argv[1]

    try:
        gdp_data = fetch_gdp_data()
        country_names = fetch_country_names()
        results, investigated_countries = analyze_gdp_growth(gdp_data, country_names, target_year)

        print("\n===== GDP Growth Analysis =====\n")
        print(f"Target Year: {target_year}\n")
        print(f"Number of countries investigated: {investigated_countries}")

        criteria_description = "Countries where the target year's growth rate is higher than the average of the previous three years"
        print(f"Criteria: {criteria_description}\n")

        print(f"Number of countries meeting the criteria: {len(results)}\n")

        if results:
            headers = ["Country", "Three-Year Average Growth (%)", "Target Year Growth (%)"]
            table = [(country, f"{avg_growth:.2f}", f"{target_growth:.2f}") for country, avg_growth, target_growth in results]
            print(tabulate(table, headers=headers, tablefmt="grid"))
        else:
            print("No countries meet the criteria.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

