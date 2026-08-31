from playwright.sync_api import sync_playwright
import pandas as pd
import time

search_urls = [

    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Chennai",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Tambaram",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Vadapalani",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Mylapore",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Velachery",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Medavakkam",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Anna+Nagar",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Egmore",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Maduravoyal",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Kanchipuram",
    "https://www.google.com/maps/search/Namma+Veedu+Vasanta+Bhavan+Hosur"

]

reviews_data = []

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    page = browser.new_page()

    place_links = set()

    for search_url in search_urls:

        print("\nSearching:", search_url)

        page.goto(search_url, timeout=60000)

        time.sleep(5)

        links = page.query_selector_all("a")

        for link in links:

            href = link.get_attribute("href")

            if href and "/maps/place/" in href:
                place_links.add(href)

    print("\nBranches found:", len(place_links))

    for place_url in place_links:

        try:

            print("\nOpening:", place_url)

            page.goto(place_url, timeout=60000)

            time.sleep(5)

            review_buttons = page.query_selector_all(
                'button[role="tab"]'
            )

            for btn in review_buttons:

                try:
                    text = btn.inner_text().lower()

                    if "review" in text:
                        btn.click()
                        time.sleep(5)
                        break

                except:
                    pass

            for _ in range(15):

                page.mouse.wheel(0, 5000)

                time.sleep(2)

            review_elements = page.query_selector_all(
                'span.wiI7pd'
            )

            branch_name = place_url.split("/place/")[1].split("/")[0]
            branch_name = branch_name.replace("+", " ")

            count = 0

            for review in review_elements:

                review_text = review.inner_text().strip()

                if len(review_text) > 20:

                    reviews_data.append({
                        "restaurant": "Namma Veedu Vasanta Bhavan",
                        "branch": branch_name,
                        "review": review_text
                    })

                    count += 1

                    if count >= 50:
                        break

            print("Collected:", count)

        except Exception as e:

            print("Error:", e)

    browser.close()

df = pd.DataFrame(reviews_data)

df.to_csv(
    "vasantha_bhavan_reviews_tamilnadu.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTOTAL REVIEWS:", len(df))