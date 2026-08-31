from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd


# =========================
# STEP 1: SEARCH LINKS (Tamil Nadu)
# =========================
search_urls = [
    "https://www.google.com/maps/search/Adyar+Ananda+Bhavan+Chennai",
    "https://www.google.com/maps/search/Adyar+Ananda+Bhavan+Coimbatore",
    "https://www.google.com/maps/search/Adyar+Ananda+Bhavan+Madurai",
    "https://www.google.com/maps/search/Adyar+Ananda+Bhavan+Salem",
    "https://www.google.com/maps/search/Adyar+Ananda+Bhavan+Tiruchirappalli",
]


# =========================
# STEP 2: DRIVER SETUP
# =========================
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


# =========================
# STEP 3: GET ALL PLACE LINKS
# =========================
place_links = set()

for url in search_urls:
    driver.get(url)
    time.sleep(5)

    links = driver.find_elements(By.XPATH, "//a[contains(@href, '/place/')]")

    for link in links:
        href = link.get_attribute("href")
        if href:
            place_links.add(href)

print("Branches found:", len(place_links))


# =========================
# STEP 4: SCRAPE REVIEWS
# =========================
all_data = []

for place in place_links:

    driver.get(place)
    time.sleep(5)

    # Click reviews tab
    try:
        driver.find_element(By.XPATH, "//button[contains(@aria-label,'Reviews')]").click()
        time.sleep(5)
    except:
        pass

    # Scroll reviews panel (IMPORTANT FIX)
    for _ in range(15):
        try:
            scroll_box = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_box)
            time.sleep(2)
        except:
            pass

    # Extract reviews safely
    review_elements = driver.find_elements(By.CSS_SELECTOR, "span.wiI7pd")

    branch_name = place.split("/")[5].replace("-", " ")

    count = 0

    for review in review_elements:

        text = review.text

        if text.strip() != "" and count < 50:

            all_data.append({
                "branch": branch_name,
                "review": text,
                "restaurant": "A2B"
            })

            count += 1

    print(f"Done: {branch_name}")


# =========================
# STEP 5: SAVE DATA
# =========================
df = pd.DataFrame(all_data)

df.to_csv("a2b_reviews_tamilnadu.csv", index=False, encoding="utf-8")

print("SCRAPING COMPLETED! Total reviews:", len(df))

driver.quit()