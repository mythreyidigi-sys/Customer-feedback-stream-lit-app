from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd


# =========================
# CONFIG
# =========================
search_url = "https://www.google.com/maps/search/Sangeetha+Restaurant+Chennai"

MAX_REVIEWS = 100


# =========================
# DRIVER SETUP
# =========================
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


# =========================
# STEP 1: GET PLACE LINKS
# =========================
driver.get(search_url)
time.sleep(6)

place_links = set()

links = driver.find_elements(By.XPATH, "//a[contains(@href, '/place/')]")

for l in links:
    href = l.get_attribute("href")
    if href:
        place_links.add(href.split("?")[0])

print("Places found:", len(place_links))


# =========================
# STEP 2: SCRAPE REVIEWS
# =========================
all_reviews = []

for place in list(place_links):

    driver.get(place)
    time.sleep(6)

    # scroll a bit to load page
    driver.execute_script("window.scrollBy(0, 800);")
    time.sleep(3)

    # click reviews button safely
    try:
        review_btn = driver.find_element(
            By.XPATH,
            "//button[contains(@aria-label,'reviews') or contains(text(),'Reviews')]"
        )
        review_btn.click()
        time.sleep(5)
    except:
        pass

    # scroll review panel (IMPORTANT FIX)
    for _ in range(10):
        try:
            scroll_box = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_box)
            time.sleep(2)
        except:
            pass

    # =========================
    # REVIEW EXTRACTION (STABLE METHOD)
    # =========================
    from selenium.webdriver.common.by import By
import time


# click reviews button (IMPORTANT)
try:
    driver.execute_script("window.scrollBy(0, 1000)")
    time.sleep(3)

    review_button = driver.find_element(
        By.XPATH,
        "//button[contains(@aria-label,'Reviews')]"
    )
    review_button.click()
    time.sleep(5)

except:
    pass


# scroll review panel properly
for _ in range(15):
    try:
        scroll_box = driver.find_element(
            By.CSS_SELECTOR,
            "div.m6QErb.DxyBCb.kA9KIf.dS8AEf"
        )
        driver.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollHeight",
            scroll_box
        )
        time.sleep(2)
    except:
        pass


# ⭐ NEW WORKING SELECTOR (CRITICAL FIX)
reviews = driver.find_elements(
    By.XPATH,
    "//span[contains(@class,'wiI7pd')]"
)

count = 0
for r in reviews:
    text = r.text

    if text and len(text) > 15 and count < 50:
        all_reviews.append({
            "restaurant": "Sangeetha",
            "review": text
        })
        count += 1

# =========================
# SAVE DATA
# =========================
df = pd.DataFrame(all_reviews)
df.to_csv("sangeetha_reviews_clean.csv", index=False, encoding="utf-8")

print("DONE! Total reviews collected:", len(df))

driver.quit()