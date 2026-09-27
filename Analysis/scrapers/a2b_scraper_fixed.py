from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from datetime import datetime, timedelta
import pandas as pd

# =========================
# HELPER: PARSE GOOGLE'S RELATIVE REVIEW DATE
# =========================
def parse_relative_date(text):
    text = text.lower().replace("edited", "").strip()
    match = re.search(r"(a|an|\d+)\s+(day|week|month|year)s?\s+ago", text)
    if not match:
        return None

    qty = 1 if match.group(1) in ("a", "an") else int(match.group(1))
    unit = match.group(2)

    if unit == "day":
        delta = timedelta(days=qty)
    elif unit == "week":
        delta = timedelta(weeks=qty)
    elif unit == "month":
        delta = timedelta(days=qty * 30)
    else:
        delta = timedelta(days=qty * 365)

    return datetime.now() - delta

# =========================
# STEP 1: SEARCH LINKS
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

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# =========================
# STEP 3: COLLECT PLACE LINKS
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

  try:
    driver.get(place)
    time.sleep(5)

    # =========================
    # FIXED BRANCH NAME (IMPORTANT)
    # =========================
    try:
        branch_name = driver.title.replace(" - Google Maps", "").strip()
    except:
        branch_name = "Unknown Branch"

    # Open reviews tab
    try:
        driver.find_element(By.XPATH, "//button[contains(@aria-label,'Reviews')]").click()
        time.sleep(5)
    except:
        pass

    # Sort by Newest so the 30-day filter reflects recent reviews
    try:
        driver.find_element(By.XPATH, "//button[contains(@aria-label,'Sort reviews')]").click()
        time.sleep(2)
        driver.find_element(By.XPATH, "//div[@role='menuitemradio' and contains(.,'Newest')]").click()
        time.sleep(3)
    except:
        pass

    # Scroll reviews
    for _ in range(12):
        try:
            scroll_box = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_box)
            time.sleep(2)
        except:
            pass

    # Extract reviews
    review_blocks = driver.find_elements(By.CSS_SELECTOR, "div.jftiEf")

    count = 0
    cutoff_date = datetime.now() - timedelta(days=30)

    for block in review_blocks:

        if count >= 50:
            break

        try:
            text = block.find_element(By.CSS_SELECTOR, "span.wiI7pd").text
        except:
            text = ""

        try:
            date_text = block.find_element(By.CSS_SELECTOR, "span.rsqaWe").text
        except:
            date_text = ""

        review_date = parse_relative_date(date_text)

        # reviews are sorted newest-first, so once we pass the cutoff the rest are older too
        if review_date is not None and review_date < cutoff_date:
            break

        if text.strip() == "" or review_date is None:
            continue

        all_data.append({
            "restaurant": "A2B",
            "branch": branch_name,
            "review": text,
            "review_date": review_date.strftime("%Y-%m-%d")
        })

        count += 1

    print(f"Done: {branch_name}")

  except Exception as e:
    # keep already-collected data even if this branch's session/page fails
    print(f"Skipped a branch due to error: {e}")
    continue

# =========================
# STEP 5: SAVE CSV
# =========================
df = pd.DataFrame(all_data)

df.to_csv("a2b_reviews_tamilnadu.csv", index=False, encoding="utf-8")

print("\nSCRAPING COMPLETED!")
print("Total reviews:", len(df))

try:
    driver.quit()
except:
    pass