from pathlib import Path
from urllib.parse import unquote, urlparse
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


SEARCH_URLS = [
    "https://www.tripadvisor.com/Restaurant_Review-g304556-d27106440-Reviews-Geetham_Veg_Restaurant-Chennai_Madras_Chennai_District_Tamil_Nadu.html"
]
OUTPUT_CSV = Path(__file__).with_name("geetham_reviews_tripadvisor.csv")


def build_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def derive_branch_name(place):
    try:
        path = urlparse(place).path
        parts = [p for p in path.split("/") if p]
        branch_name = None
        if "place" in parts:
            idx = parts.index("place")
            if idx + 1 < len(parts):
                branch_name = parts[idx + 1]
        if not branch_name:
            branch_name = parts[-1] if parts else place
        branch_name = unquote(branch_name)
        branch_name = branch_name.replace("+", " ").replace("-", " ").strip()
        return branch_name or "Unknown Branch"
    except Exception:
        return place


def extract_reviews(driver, place):
    try:
        driver.get(place)
        time.sleep(6)
    except Exception as exc:
        return [("Geetham", derive_branch_name(place), f"Page load failed: {exc}")]

    try:
        cookie_button = driver.find_element(
            By.XPATH,
            "//button[contains(translate(normalize-space(.), 'ACCEPT', 'accept'), 'accept') or contains(translate(normalize-space(.), 'AGREE', 'agree'), 'agree')]"
        )
        cookie_button.click()
        time.sleep(2)
    except Exception:
        pass

    try:
        review_tab = driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Reviews') or contains(@aria-label, 'reviews')]")
        review_tab.click()
        time.sleep(5)
    except Exception:
        pass

    for _ in range(10):
        try:
            scroll_box = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_box)
            time.sleep(2)
        except Exception:
            break

    selectors = [
        "div[data-test-target='review-text']",
        "span[data-test-target='review-text']",
        "q",
        "[class*='reviewText']",
        "[class*='review-text']",
    ]

    reviews = []
    for selector in selectors:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if elements:
            for element in elements:
                text = " ".join(element.text.split())
                if len(text) > 20:
                    reviews.append(("Geetham", derive_branch_name(place), text))
            if reviews:
                break

    if not reviews:
        page_text = " ".join(driver.page_source.split())
        if "captcha" in page_text.lower() or "verify you are human" in page_text.lower():
            reviews.append(("Geetham", derive_branch_name(place), "TripAdvisor blocked access with a captcha or bot-check page."))
        else:
            reviews.append(("Geetham", derive_branch_name(place), "No review text was found with the current selectors or page structure."))

    return reviews


def main():
    driver = build_driver()
    all_data = []

    try:
        for url in SEARCH_URLS:
            print(f"Opening: {url}")
            all_data.extend(extract_reviews(driver, url))
    finally:
        driver.quit()

    df = pd.DataFrame(all_data, columns=["Restaurant", "Branch", "Reviews"])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"SCRAPING COMPLETED! Total rows: {len(df)}")
    print(f"Saved output to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
