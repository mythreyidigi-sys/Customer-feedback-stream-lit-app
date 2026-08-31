from playwright.sync_api import sync_playwright
import pandas as pd
import time

search_urls = [
    "https://www.google.com/maps/search/Sree+Annapoorna+Coimbatore",
    "https://www.google.com/maps/search/Sree+Annapoorna+Chennai",
    "https://www.google.com/maps/search/Sree+Annapoorna+Madurai",
    "https://www.google.com/maps/search/Sree+Annapoorna+Salem",
    "https://www.google.com/maps/search/Sree+Annapoorna+Erode",
    "https://www.google.com/maps/search/Sree+Annapoorna+Tiruppur",
    "https://www.google.com/maps/search/Sree+Annapoorna+Trichy",
    "https://www.google.com/maps/search/Sree+Annapoorna+Karur",
    "https://www.google.com/maps/search/Sree+Annapoorna+Namakkal",
    "https://www.google.com/maps/search/Sree+Annapoorna+Vellore"
]

reviews_data = []

with sync_playwright() as p:

    browser = p.chromium.launch(headless=False)

    page = browser.new_page()

    place_links = set()

    # ======================
    # FIND BRANCHES
    # ======================
    for url in search_urls:

        print(f"Searching: {url}")

        page.goto(url, timeout=60000)

        page.wait_for_timeout(5000)

        for _ in range(8):
            page.mouse.wheel(0, 5000)
            page.wait_for_timeout(1500)

        links = page.locator("a[href*='/place/']").all()

        for link in links:
            href = link.get_attribute("href")

            if href:
                place_links.add(href)

    print("\nBranches found:", len(place_links))

    # ======================
    # SCRAPE REVIEWS
    # ======================
    for place_url in place_links:

        try:
            page.goto(place_url, timeout=60000)

            page.wait_for_timeout(5000)

            branch_name = page.locator("h1").inner_text()

            print("Scraping:", branch_name)

            review_button = page.locator('button:has-text("Reviews")')

            if review_button.count() > 0:
                review_button.first.click()
                page.wait_for_timeout(5000)

            for _ in range(20):
                page.mouse.wheel(0, 6000)
                page.wait_for_timeout(2000)

            reviews = page.locator("span.wiI7pd").all_inner_texts()

            count = 0

            for review in reviews:

                if len(review.strip()) > 20:

                    reviews_data.append({
                        "restaurant": "Sree Annapoorna",
                        "branch": branch_name,
                        "review": review
                    })

                    count += 1

                if count >= 50:
                    break

            print("Collected:", count)

        except Exception as e:
            print("Error:", e)

    browser.close()

# ======================
# SAVE CSV
# ======================

df = pd.DataFrame(reviews_data)

df.drop_duplicates(inplace=True)

df.to_csv(
    "sree_annapoorna_reviews_tamilnadu.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTOTAL REVIEWS:", len(df))
print("Saved: sree_annapoorna_reviews_tamilnadu.csv")







