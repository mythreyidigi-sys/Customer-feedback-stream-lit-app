from playwright.sync_api import sync_playwright
import time
import pandas as pd


search_url = "https://www.google.com/maps/search/Sangeetha+Restaurant+Chennai"

all_reviews = []


with sync_playwright() as p:

    # =========================
    # 1. LAUNCH BROWSER
    # =========================
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto(search_url)
    time.sleep(6)

    # =========================
    # 2. GET PLACE LINKS
    # =========================
    links = page.query_selector_all("a")

    place_links = set()

    for l in links:
        href = l.get_attribute("href")
        if href and "/place/" in href:
            place_links.add(href.split("?")[0])

    print("Places found:", len(place_links))


    # =========================
    # 3. SCRAPE EACH PLACE
    # =========================
    for place in place_links:

        page.goto(place)
        time.sleep(6)

        # scroll page a bit
        page.mouse.wheel(0, 1500)
        time.sleep(2)

        # click reviews button
        try:
            page.click("button:has-text('Reviews')")
            time.sleep(5)
        except:
            pass

        # scroll reviews section
        for _ in range(10):
            page.mouse.wheel(0, 2000)
            time.sleep(2)

        # =========================
        # 4. EXTRACT REVIEWS
        # =========================
        reviews = page.query_selector_all("span[lang]")

        count = 0

        for r in reviews:

            text = r.inner_text().strip()

            if text and len(text) > 20 and count < 50:

                all_reviews.append({
                    "restaurant": "Sangeetha",
                    "review": text,
                    "place_url": place
                })

                count += 1

        print("Scraped:", place.split("/")[-1])


    # =========================
    # 5. SAVE DATA
    # =========================
    df = pd.DataFrame(all_reviews)
    df.to_csv("sangeetha_reviews_final.csv", index=False, encoding="utf-8")

    print("DONE! Total reviews collected:", len(all_reviews))

    # IMPORTANT: safe close INSIDE context
    browser.close()