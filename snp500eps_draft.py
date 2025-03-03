from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import shutil
import os

# URL of the Excel file
url = "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx"
output_filename = "sp-500-eps-est.xlsx"

# Set up Chrome options for headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run without GUI
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36")

# Path to your ChromeDriver (download from https://chromedriver.chromium.org/)
# Replace with your actual ChromeDriver path
chromedriver_path = "/path/to/chromedriver"  # e.g., "C:/chromedriver/chromedriver.exe" on Windows

# Set download directory to current working directory
download_dir = os.getcwd()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

# Initialize the driver
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Navigate to the URL
    driver.get(url)
    print("Request sent, waiting for file download...")

    # Wait for the file to download (adjust time based on file size/network)
    timeout = 30  # seconds
    elapsed = 0
    while elapsed < timeout:
        if os.path.exists("sp-500-eps-est.xlsx"):
            print("File detected in default download name.")
            # Rename/move the file to desired name
            shutil.move("sp-500-eps-est.xlsx", output_filename)
            print(f"File downloaded and saved as {output_filename}")
            break
        time.sleep(1)
        elapsed += 1
    else:
        print("Download timed out. File not found in expected location.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up
    driver.quit()
