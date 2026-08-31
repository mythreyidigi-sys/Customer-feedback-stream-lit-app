from playwright.sync_api import sync_playwright
import pandas as pd
import time

search_urls = [
    "https://www.google.com/maps/search/Saravana+Bhavan+Chennai",
    "https://www.google.com/maps/search/Saravana+Bhavan+Coimbatore",
    "https://www.google.com/maps/search/Saravana+Bhavan+Madurai",
    "https://www.google.com/maps/search/Saravana+Bhavan+Salem",
    "https://www.google.com/maps/search/Saravana+Bhavan+Trichy"
]

reviews_data = []

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False,
        slow_mo=1000
    )

    page = browser.new_page()

    place_links = set()

    # Collect branch URLs
    for url in search_urls:

        page.goto(url, timeout=60000)
        page.wait_for_timeout(5000)

        links = page.locator("a[href*='/place/']").all()

        for link in links:
            href = link.get_attribute("href")

            if href and "/place/" in href:
                place_links.add(href)

    print("Branches found:", len(place_links))

    # Visit each branch
    for place_url in place_links:

        try:
            print("\nOpening:", place_url)

            page.goto(place_url, timeout=60000)
            page.wait_for_timeout(5000)

            # Click Reviews button
            try:
                page.locator("button:has-text('Reviews')").click(timeout=5000)
                page.wait_for_timeout(4000)
            except:
                pass

            # Scroll review section
            for _ in range(15):
                page.mouse.wheel(0, 5000)
                page.wait_for_timeout(2000)

            reviews = page.locator("span.wiI7pd").all_text_contents()

            branch_name = "Saravana Bhavan"

            count = 0

            for review in reviews:

                review = review.strip()

                if len(review) > 20:

                    reviews_data.append({
                        "restaurant": "Saravana Bhavan",
                        "branch": branch_name,
                        "review": review
                    })

                    count += 1

                if count >= 40:
                    break

            print("Collected:", count)

        except Exception as e:
            print("Error:", e)

    browser.close()

df = pd.DataFrame(reviews_data)

df.to_csv(
    "saravana_bhavan_reviews_tamilnadu.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTOTAL REVIEWS:", len(df))