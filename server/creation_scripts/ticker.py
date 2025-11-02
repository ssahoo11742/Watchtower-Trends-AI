import requests
from bs4 import BeautifulSoup
import time
import string

base_url = "https://stock-screener.org/stock-list.aspx?alpha={}"
all_tickers = []

for letter in string.ascii_uppercase:
    url = base_url.format(letter)
    print(f"Scraping tickers starting with {letter}...")
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to load {url}")
        continue
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find the table
    table = soup.find("table", {"class": "table"})  # the stock table has class "table"
    if not table:
        print(f"No table found for {letter}")
        continue

    # Grab all rows except the header
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 1:
            ticker = cols[0].text.strip()
            if ticker:
                all_tickers.append(ticker)
    
    time.sleep(1)  # polite delay

# Remove duplicates and sort
all_tickers = sorted(list(set(all_tickers)))
print(f"Collected {len(all_tickers)} tickers")

# Save to file
with open("tickers.txt", "w") as f:
    f.write("\n".join(all_tickers))
